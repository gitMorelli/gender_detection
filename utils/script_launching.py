import yaml
import tempfile
import subprocess
import threading

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

'''
def load_config(path):
    with open(path, 'r') as f:
        config=yaml.safe_load(f)
        args = DotDict()
        for key, value in config.items():
            args[key] = value
        return args
'''

def run_experiment(try_args,script_name):
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
    return 

def run_experiment_threaded(try_args, script_name):
    def stream_output(pipe, name):
        for line in iter(pipe.readline, ''):
            if line:
                print(f"[{name}] {line}", end='')

    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.yaml') as tmp:
        yaml.dump(try_args.__dict__, tmp)
        tmp_path = tmp.name

    print(f"Starting experiment:")

    process = subprocess.Popen(
        ['python', script_name, '--config', tmp_path],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

    # Start threads for live output
    stdout_thread = threading.Thread(target=stream_output, args=(process.stdout, 'STDOUT'))
    stderr_thread = threading.Thread(target=stream_output, args=(process.stderr, 'STDERR'))
    stdout_thread.start()
    stderr_thread.start()

    process.wait()
    stdout_thread.join()
    stderr_thread.join()

    print(f"Experiment finished with return code: {process.returncode}")

    return 