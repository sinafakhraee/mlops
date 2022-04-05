
from azureml.core.run import Run
from azureml.core import Dataset, Datastore, Workspace
import os
import argparse
import joblib
import json





def main():
    print("Running train_aml.py")

    
    run = Run.get_context()

    print("Getting training parameters")

    # Load the training parameters from the parameters file
    with open("parameters.json") as f:
        pars = json.load(f)
    try:
        train_args = pars["training"]
    except KeyError:
        print("Could not load training values from file")
        train_args = {}

    # Log the training parameters
    print(f"Parameters: {train_args}")
    for (k, v) in train_args.items():
        run.log(k, v)
        run.parent.log(k, v)

    
    # Link dataset to the step run so it is trackable in the UI
    run.input_datasets['training_data'] = 'empty dataset'
    run.parent.tag("dataset_id", value='dataset.id')

    
    run.tag("run_type", value="train")
    print(f"tags now present for run: {run.tags}")

    run.complete()


if __name__ == '__main__':
    main()
