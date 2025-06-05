from multiprocessing import Pool
import subprocess
import yaml
import os
import sys
import tempfile
import copy

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

    try_files = df[df['experiment'].str.contains('icdar_EXTRACTED_train_df_')]['experiment'].tolist()
    try_pca = [True, False]
    try_models = ['logreg', 'mlp']#, 'lgbm']

    try_args = []
    i = 0
    for file in try_files:
        for model in try_models:
            for pca in try_pca:
                i += 1
                args.selected_model = model
                args.with_pca = pca
                args.input_file_name = file
                args.n_job=i
                try_args.append(copy.deepcopy(args))
    return try_args, i

if __name__ == "__main__":
    output_dir = os.path.join("..", "outputs", "preprocessed_data")
    log_file = os.path.join(output_dir, "file_metadata_log.json")
    config_path = 'feature_extraction_configs/exp_patch_overfitting1.yaml'

    '''exclusion = [1,  2,  3,  7,  8,  9, 10, 16, 22, 25, 26, 28, 31, 32, 34, 37,
       38, 40, 43, 44, 46, 52, 55, 56, 57, 58, 61, 62, 63, 64, 67, 68, 69,
       70, 73, 74, 76, 79, 80]'''
    exclusion=[]

    try_args, total_jobs = generate_experiments(config_path, log_file)
    print(f"Total experiments to run: {total_jobs}")

    #print(try_args[:3])  # Print first 3 for verification
    # Run in parallel
    for i, arg in enumerate(try_args):
        if not(i in exclusion):
            print(f"Experiment {i}/{total_jobs}: {arg.input_file_name}, Model: {arg.selected_model}, PCA: {arg.with_pca}")
            run_experiment(arg)  # Test a single run first
    '''with Pool(processes=os.cpu_count()) as pool:  # Adjust the number of processes as needed
        pool.map(run_experiment, try_args[3:])  # Use the full list or a subset for testing'''


'''with Pool(processes=3) as pool:  # Use os.cpu_count() or limit by RAM/GPU
    pool.map(run_experiment, configs)'''

'''
from apscheduler.schedulers.blocking import BlockingScheduler

def run_job():
    subprocess.run(['python', 'train.py', '--config', 'config.yaml'])

scheduler = BlockingScheduler()
scheduler.add_job(run_job, 'cron', hour=2)
scheduler.start()'''