from azureml.pipeline.core.graph import PipelineParameter
from azureml.pipeline.steps import PythonScriptStep
from azureml.pipeline.core import Pipeline, PipelineData
from azureml.core import Workspace, Dataset, Datastore
from azureml.core.runconfig import RunConfiguration
from ml_service.util.attach_compute import get_compute
from ml_service.util.env_variables import Env
from ml_service.util.manage_environment import get_environment
import os
from azureml.core import Experiment
from ml_service.pipelines.load_sample_data import create_sample_data_csv

def main():
    a = 12
    e = Env()
    # Get Azure machine learning workspace
    aml_workspace = Workspace.get(
        name=e.workspace_name,
        subscription_id=e.subscription_id,
        resource_group=e.resource_group,
    )
    print("get_workspace:")
    print(aml_workspace)

    # Get Azure machine learning cluster
    aml_compute = get_compute(aml_workspace, e.compute_name, e.vm_size)
    if aml_compute is not None:
        print("aml_compute:")
        print(aml_compute)

    # Create a reusable Azure ML environment
    environment = get_environment(
        aml_workspace,
        e.aml_env_name,
        conda_dependencies_file=e.aml_env_train_conda_dep_file,
        create_new=e.rebuild_env,
    )  #
    run_config = RunConfiguration()
    run_config.environment = environment

    if e.datastore_name:
        datastore_name = e.datastore_name
    else:
        datastore_name = aml_workspace.get_default_datastore().name
    run_config.environment.environment_variables[
        "DATASTORE_NAME"
    ] = datastore_name  # NOQA: E501
    
    model_name_param = PipelineParameter(name="model_name", default_value=e.model_name)  # NOQA: E501
    # dataset_version_param = PipelineParameter(
    #     name="dataset_version", default_value=e.dataset_version
    # )
    data_file_path_param = PipelineParameter(
        name="data_file_path", default_value="none"
    )
    caller_run_id_param = PipelineParameter(name="caller_run_id", default_value="none")  # NOQA: E501

# Get dataset name
    dataset_name = e.dataset_name

    # Check to see if dataset exists
    if dataset_name not in aml_workspace.datasets:
        # This call creates an example CSV from sklearn sample data. If you
        # have already bootstrapped your project, you can comment this line
        # out and use your own CSV.
        create_sample_data_csv()

        # Use a CSV to read in the data set.
        file_name = "diabetes.csv"

        if not os.path.exists(file_name):
            raise Exception(
                'Could not find CSV dataset at "%s". If you have bootstrapped your project, you will need to provide a CSV.'  # NOQA: E501
                % file_name
            )  # NOQA: E501

        # Upload file to default datastore in workspace
        datatstore = Datastore.get(aml_workspace, datastore_name)
        target_path = "training-data/"
        datatstore.upload_files(
            files=[file_name],
            target_path=target_path,
            overwrite=True,
            show_progress=False,
        )

        # Register dataset
        path_on_datastore = os.path.join(target_path, file_name)
        dataset = Dataset.Tabular.from_delimited_files(
            path=(datatstore, path_on_datastore)
        )
        dataset = dataset.register(
            workspace=aml_workspace,
            name=dataset_name,
            description="diabetes training data",
            tags={"format": "CSV"},
            create_new_version=True,
        )





   
    # Create a PipelineData to pass data between steps
   

    train_step = PythonScriptStep(
        name="Train Model",
        script_name=e.train_script_path,
        compute_target=aml_compute,
        source_directory=e.sources_directory_train,
        outputs=None,
        arguments=None,
        runconfig=run_config,
        allow_reuse=True,
    )
    print("Step Train created")

    
    steps = [train_step]

    train_pipeline = Pipeline(workspace=aml_workspace, steps=steps)
    # train_pipeline._set_experiment_name
    train_pipeline.validate()
    

    # Submit the pipeline to be run
    pipeline_run1 = Experiment(aml_workspace, e.experiment_name).submit(train_pipeline)
    pipeline_run1.wait_for_completion()

    print("train pipeline submitted")


if __name__ == "__main__":
    main()
 