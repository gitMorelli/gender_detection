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

def run_experiment(try_args, script_name):
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.yaml') as tmp:
        yaml.dump(try_args.__dict__, tmp)
        tmp_path = tmp.name
    print(f"Starting experiment: file={try_args.input_file_name}, model={try_args.selected_model}, PCA={try_args.with_pca}")
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

'''def run_experiment(try_args,script_name):
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.yaml') as tmp:
        yaml.dump(try_args.__dict__, tmp)
        tmp_path = tmp.name
    process = subprocess.Popen(
        ['python', script_name, '--config', tmp_path],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    # Print stdout in real time
    while True:
        output = process.stdout.readline()
        if output == '' and process.poll() is not None:
            break
        if output:
            print(output, end='')
    # Print any remaining stderr
    stderr = process.stderr.read()
    if stderr:
        print(stderr, end='')
    result = process.wait()
    #print(result.stdout)
    #print(result.stderr)
    print(f"Experiment finished with return code: {result}")
    if result != 0:
        print(f"Error output:\n{result}")
    return '''

def generate_experiments(config_path, experiment_csv,experiment):
    #df = pd.read_csv(experiment_csv)
    df = pd.read_pickle(experiment_csv)  # Use read_csv if the file is in CSV format
    args = load_config(config_path)
    args.experiment_name = experiment
    #if int(experiment.split('_')[1]) <=11:
    try_files = df[experiment].dropna().tolist()

    try_args = []
    i = 0
    args.selected_model = 'logreg'
    args.with_pca = False
    script_name = 'feature_extraction_script.py'
    if experiment == 'experiment_1':
        try_pca = [True,False]
        try_models = ['logreg', 'lgbm','mlp']
        for file in try_files:
            for model in try_models:
                for pca in try_pca:
                    i += 1
                    args.selected_model = model
                    args.with_pca = pca
                    args.input_file_name = file
                    args.n_job=i
                    try_args.append(copy.deepcopy(args))
    elif experiment == 'experiment_2':
        for file in try_files:
            i += 1
            args.n_patches = 5
            args.input_file_name = file
            args.n_job=i
            try_args.append(copy.deepcopy(args))
    elif experiment == 'experiment_3':
        for file in try_files:
            i += 1
            args.input_file_name = file
            args.n_job=i
            try_args.append(copy.deepcopy(args))
    elif experiment == 'experiment_4':
        try_n_patches = [list(range(1, 15 + 1)),list(range(1, 15 + 1)),list(range(1, 50 + 1))]
        try_models = ['logreg','mlp']
        for model in try_models:
            for j,file in enumerate(try_files):
                for n_patches in try_n_patches[j]:
                    i += 1
                    args.input_file_name = file
                    args.n_job=i
                    args.n_patches = n_patches
                    args.selected_model = model
                    try_args.append(copy.deepcopy(args))
    elif experiment == 'experiment_5':
        #linear
        n_test=20
        f_0 = 1/n_test
        try_n_writers = [i*f_0 for i in range(1, n_test+1)]
        #print(f"Number of writers to try: {try_n_writers}")
        for file in try_files:
            for n_writers in try_n_writers:
                i += 1
                args.input_file_name = file
                args.n_job=i
                args.n_writers = n_writers
                try_args.append(copy.deepcopy(args))
    elif experiment == 'experiment_6':
        try_n_patches = [-1,1]
        for file in try_files:
            for n_patches in try_n_patches:
                i += 1
                args.input_file_name = file
                args.n_job=i
                args.n_patches = n_patches
                try_args.append(copy.deepcopy(args))
    elif experiment == 'experiment_7': #generalization experiment
        try_pca = [True,False]
        try_models = ['logreg', 'mlp']
        try_task = ['arabic','english']
        for file in try_files:
            for model in try_models:
                for pca in try_pca:
                    for language in try_task:
                        i += 1
                        args.selected_model = model
                        args.with_pca = pca
                        args.input_file_name = file
                        args.n_job=i
                        args.train_on_language = language
                        try_args.append(copy.deepcopy(args))
    elif experiment == 'experiment_8': #combining features on same dataset experiments
        try_pca = [True,False]
        try_models = ['logreg', 'mlp']
        script_name = 'script_2_models.py'
        for j in range(len(try_files)):
            try_files[j]=list(try_files[j])
        for file in try_files:
            for model in try_models:
                for pca in try_pca:
                    i += 1
                    args.selected_model = model
                    args.with_pca = pca
                    args.input_file_name = file
                    args.n_job=i
                    try_args.append(copy.deepcopy(args))
    elif experiment == 'experiment_9': #combining features on different datasets experiments
        script_name = 'script_2_datasets.py'
        args.type_of_ensembling = 'averaging'  # Set the type of ensembling
        for j in range(len(try_files)):
            try_files[j]=list(try_files[j])
        for file in try_files:
            i += 1
            args.with_pca = False
            args.input_file_name = file
            args.n_job=i
            try_args.append(copy.deepcopy(args))
    elif experiment == 'experiment_10': #combining features on different datasets experiments but averaging ll patches
        try_models = ['logreg', 'mlp']
        try_pca = [True,False]
        script_name = 'script_2_datasets_join.py'
        for j in range(len(try_files)):
            try_files[j]=list(try_files[j])
        for file in try_files:
            for model in try_models:
                for pca in try_pca:
                    i += 1
                    args.selected_model = model
                    args.with_pca = pca
                    args.input_file_name = file
                    args.n_job=i
                    try_args.append(copy.deepcopy(args))
    elif experiment == 'experiment_11': #combining features on different datasets experiments
        try_models = ['logreg', 'mlp']
        try_pca = [True,False]
        args.type_of_ensembling = 'concatenation'  # Set the type of ensembling
        script_name = 'script_2_datasets_join.py'
        for j in range(len(try_files)):
            try_files[j]=list(try_files[j])
        for file in try_files:
            for model in try_models:
                for pca in try_pca:
                    i += 1
                    args.with_pca = pca
                    args.selected_model = model
                    args.input_file_name = file
                    args.n_job=i
                    try_args.append(copy.deepcopy(args))
    elif experiment == 'experiment_12': #combining features on different datasets experiments
        try_models = ['logreg', 'mlp']
        try_combine = ['max', 'average']
        args.with_pca = False
        for file in try_files:
            for model in try_models:
                for combine in try_combine:
                    i += 1
                    args.selected_model = model
                    args.input_file_name = file
                    args.n_job=i
                    args.patch_merging = combine
                    try_args.append(copy.deepcopy(args))
    '''elif experiment == 'experiment_12': #combining features on different datasets experiments
        args.selected_model = 'mlp'
        args.with_pca = False
        args.script_mode = 'explainability_pipeline'
        args.input_file_name = 'icdar_EXTRACTED_train_df_clip-vit-large-patch14_20250517_144404.csv'
        args.n_job = 1
        try_args.append(copy.deepcopy(args))'''
    return try_args, i, script_name

if __name__ == "__main__":
    output_dir = os.path.join("..", "outputs", "preprocessed_data")
    #experiment_csv = os.path.join(output_dir, "experiment_table.csv")
    #experiment_csv = os.path.join(output_dir, "experiment_table.pkl")  # Change to 'experiment_table.csv' if needed
    #experiment_csv = os.path.join(output_dir, "experiment_table_20250702_193304.pkl")  # Change to 'experiment_table.csv' if needed
    #20250711_154537
    #experiment_csv = os.path.join(output_dir, "experiment_table_20250711_154537.pkl")
    experiment_csv = os.path.join(output_dir, "experiment_table_20250715_154509.pkl")
    #experiments = [f'experiment_{i}' for i in range(4,5)]  # Change to 'experiment_2' if needed
    experiments = ['experiment_2','experiment_6']
    print(experiments)
    config_path = 'feature_extraction_configs/exp_patch_overfitting1.yaml'

    for experiment in experiments:
        print(f"Running experiment: {experiment}")
        try_args, total_jobs, script_name = generate_experiments(config_path, experiment_csv,experiment)
        print(f"Total experiments to run: {total_jobs}")

        #print(try_args[:3])  # Print first 3 for verification
        # Run in parallel
        for i, arg in enumerate(try_args):
            print(f"Experiment {i}/{total_jobs}: {arg.input_file_name}, Model: {arg.selected_model}, PCA: {arg.with_pca}")
            run_experiment(arg,script_name)  # Test a single run first
    
    
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