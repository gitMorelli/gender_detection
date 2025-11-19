from multiprocessing import Pool
import subprocess
import yaml
import os
import sys
import tempfile
import copy
from datetime import datetime
import pandas as pd
import json

source_path=os.path.abspath(os.path.join(os.path.dirname(__file__), '..','..'))
sys.path.append(source_path)
import utils.file_IO as file_IO

class DotDict:
    def __init__(self, **entries):
        self.__dict__.update(entries)

    def __setitem__(self, key, value):
        setattr(self, key, value)

    def __getitem__(self, key):
        return getattr(self, key)

    def __repr__(self):
        return f"{self.__dict__}"

def load_config(path):
    with open(path, 'r') as f:
        config = yaml.safe_load(f)
        return DotDict(**config)

def run_experiment(try_args, script_name):
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.yaml') as tmp:
        yaml.dump(try_args.__dict__, tmp)
        tmp_path = tmp.name
    print(f"Starting experiment: file={try_args.input_file_name}, model={try_args.selected_model}")
    result = subprocess.run(['python', script_name, '--config', tmp_path],
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    print(f"Experiment finished with return code: {result.returncode}")
    print("=== STDOUT ===")
    print(result.stdout)
    print("=== STDERR ===")
    print(result.stderr)
    if result.returncode != 0:
        print(f"Error output:\n{result.stderr}")
    return

def generate_experiments(config_path, experiment_csv,experiment):
    #df = pd.read_csv(experiment_csv)
    df = pd.read_pickle(experiment_csv)  # Use read_csv if the file is in CSV format
    args = load_config(config_path)
    args.experiment_name = experiment
    #if int(experiment.split('_')[1]) <=11:

    script_name = 'cv_evaluation.py'
    try_args = []
    i = 0

    args.with_pca = False
    args.n_splits = 10
    args.selected_model = 'logreg'
    args.with_feature_scaling = False

    if experiment == 'standard_patches':
        try_files = df['standard_patches'].dropna().tolist()
        #compare all models on standard patches
        for file in try_files:
            i += 1
            args.input_file_name = file
            args.n_job=i
            try_args.append(copy.deepcopy(args))
    elif experiment == 'standard_patches_single':
        try_files = df['standard_patches'].dropna().tolist()
        #compare all models on standard patches
        args.n_patches = 1
        for file in try_files:
            i += 1
            args.input_file_name = file
            args.n_job=i
            try_args.append(copy.deepcopy(args))
    elif experiment == 'standard_body':
        try_files = df['standard_body'].dropna().tolist()
        #compare all models on standard body
        for file in try_files:
            i += 1
            args.input_file_name = file
            args.n_job=i
            try_args.append(copy.deepcopy(args))
    elif experiment == 'standard_patches_generalization': 
        try_files = df['standard_patches'].dropna().tolist()
        #generalization experiment on body and standard patches
        try_task = ['arabic','english']
        for file in try_files:
            for language in try_task:
                i += 1
                args.input_file_name = file
                args.n_job=i
                args.train_on_language = language
                try_args.append(copy.deepcopy(args))

    return try_args, i, script_name 

if __name__ == "__main__":
    experiment_csv = os.path.join('experiment_tables', "experiment_table_20251112_164941.pkl")
    experiments = ['standard_patches','standard_patches_single','standard_body','standard_patches_generalization']
    config_path = 'configs/exp_patch_overfitting1.yaml'

    for experiment in experiments:
        print(f"Running experiment: {experiment}")
        try_args, total_jobs, script_name = generate_experiments(config_path, experiment_csv,experiment)
        print(f"Total experiments to run: {total_jobs}")
        results_path = os.path.join('outputs', f"results_{experiment}.json")
        # Load the JSON data
        if not os.path.exists(results_path):
            print(f"Still to be created: {results_path}")
            unique_fe_models = []
        else:
            with open(results_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            # Extract all unique FE model values
            unique_fe_models = list({entry.get("FE model") for entry in data.values() if "FE model" in entry})
            # Print them
            print('unique models already trained: ',len(unique_fe_models))
        for i, arg in enumerate(try_args):
            FE_model=file_IO.get_selected_model_from_input_filename(source_path,arg.input_file_name)
            if FE_model in unique_fe_models:
                print(f"Skipping experiment {i+1}/{total_jobs}: {arg.input_file_name}, Model: {FE_model} (already done)")
            else:
                print(f"Running experiment {i}/{total_jobs}: {arg.input_file_name}, Model: {FE_model}")
                run_experiment(arg,script_name)  # Test a single run first