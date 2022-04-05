from azureml.pipeline.core.graph import PipelineParameter
from azureml.pipeline.steps import PythonScriptStep
from azureml.pipeline.core import Pipeline, PipelineData
from azureml.core import Workspace, Dataset, Datastore
from azureml.core.runconfig import RunConfiguration
from ml_service.pipelines.load_sample_data import create_sample_data_csv
from ml_service.util.attach_compute import get_compute
from ml_service.util.env_variables import Env
from ml_service.util.manage_environment import get_environment
import os
from azureml.core import Experiment


def main():
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

    

    model_name_param = PipelineParameter(name="model_name", default_value=e.model_name)  # NOQA: E501
    
    caller_run_id_param = PipelineParameter(name="caller_run_id", default_value="none")  # NOQA: E501

   
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
 