from multiprocessing import Pool
import subprocess
import yaml
import os
import sys
import tempfile
import copy
from datetime import datetime
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
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

def run_experiment(try_args):
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.yaml') as tmp:
        yaml.dump(try_args.__dict__, tmp)
        tmp_path = tmp.name
    print(f"Starting experiment: file={try_args.input_file_name}, model={try_args.selected_model}, PCA={try_args.with_pca}")
    result=subprocess.run(['python', 'feature_extraction_script.py', '--config', tmp_path], 
                   stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    print(f"Experiment finished with return code: {result.returncode}")
    if result.returncode != 0:
        print(f"Error output:\n{result.stderr}")
    return 

def generate_experiments(config_path, log_file_path):
    df = file_IO.assemble_csv_from_log(log_file_path)
    args = load_config(config_path)

    try_files = ['icdar_EXTRACTED_train_df_clip-vit-large-patch14_20250604_201425.csv','icdar_EXTRACTED_train_df_resnet50_20250523_221556.csv']
    #try_files = df[df['experiment'].str.contains('icdar_EXTRACTED_train_df_')]['experiment'].tolist()
    try_pca = [False]
    try_models = ['logreg']#, 'lgbm']
    n = 15  # or any desired value
    try_n_patches = list(range(1, n + 1))

    try_args = []
    i = 0
    for file in try_files:
        for model in try_models:
            for pca in try_pca:
                for n_patches in try_n_patches:
                    i += 1
                    args.selected_model = model
                    args.with_pca = pca
                    args.input_file_name = file
                    args.n_job=i
                    args.n_patches = n_patches
                    try_args.append(copy.deepcopy(args))
    return try_args, i

if __name__ == "__main__":
    output_dir = os.path.join("..", "outputs", "preprocessed_data")
    log_file = os.path.join(output_dir, "file_metadata_log.json")
    config_path = 'feature_extraction_configs/exp_patch_overfitting1.yaml'
    exclusion=[]

    try_args, total_jobs = generate_experiments(config_path, log_file)
    print(f"Total experiments to run: {total_jobs}")

    for i, arg in enumerate(try_args):
        if not(i in exclusion):
            print(f"Experiment {i}/{total_jobs}: {arg.input_file_name}, Model: {arg.selected_model}, Patches: {arg.n_patches}")
            run_experiment(arg)  
