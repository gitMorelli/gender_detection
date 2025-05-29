from multiprocessing import Pool
import subprocess
import yaml
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import utils.file_IO as file_IO
import tempfile
import copy

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
        config=yaml.safe_load(f)
        args = DotDict()
        for key, value in config.items():
            args[key] = value
        return args

def run_experiment(try_args):
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.yaml') as tmp:
        yaml.dump(try_args.__dict__, tmp)
        tmp_path = tmp.name
    return subprocess.run(['python', 'feature_extraction_script.py', '--config', tmp_path])

output_dir = "..\\outputs\\preprocessed_data\\"
LOG_FILE = output_dir+"file_metadata_log.json"
df = file_IO.assemble_csv_from_log(LOG_FILE)


configs = ['feature_extraction_configs/exp_patch_overfitting1.yaml']
args = load_config(configs[0])
#function to get the name of all the feature extraction files to use
try_files = df[df['experiment'].str.contains('icdar_EXTRACTED_train_df_')]['experiment'].tolist()

try_pca = [True,False] # Example of PCA usage

try_models = ['logreg','mlp','lgbm']

try_args = []

i=0
for file in try_files:
    for model in try_models:
        for pca in try_pca:
            i+=1
            args.selected_model = model
            args.with_pca = pca
            args.input_file_name = file
            try_args.append(copy.deepcopy(args))
print(i)

'''for arg in try_args:
    #print(arg)
    run_experiment(arg)'''

#run_experiment(args)

if __name__ == "__main__":
    with Pool(processes=3) as pool:  # Use os.cpu_count() or limit by RAM/GPU
        pool.map(run_experiment, try_args[:3])

'''with Pool(processes=3) as pool:  # Use os.cpu_count() or limit by RAM/GPU
    pool.map(run_experiment, configs)'''

'''
from apscheduler.schedulers.blocking import BlockingScheduler

def run_job():
    subprocess.run(['python', 'train.py', '--config', 'config.yaml'])

scheduler = BlockingScheduler()
scheduler.add_job(run_job, 'cron', hour=2)
scheduler.start()'''