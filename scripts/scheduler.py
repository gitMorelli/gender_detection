from multiprocessing import Pool
import subprocess

def run_experiment(config_file):
    return subprocess.run(['python', 'feature_extraction_script.py', '--config', config_file])

configs = ['feature_extraction_configs/config1.yaml', 
           'feature_extraction_configs/exp_patch_overfitting1.yaml']

run_experiment(configs[1])  # Example of running a single experiment

'''with Pool(processes=3) as pool:  # Use os.cpu_count() or limit by RAM/GPU
    pool.map(run_experiment, configs)'''

'''
from apscheduler.schedulers.blocking import BlockingScheduler

def run_job():
    subprocess.run(['python', 'train.py', '--config', 'config.yaml'])

scheduler = BlockingScheduler()
scheduler.add_job(run_job, 'cron', hour=2)
scheduler.start()'''