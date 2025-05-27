import json
from datetime import datetime
import os

def get_base_metadata(filepath):
    stats = os.stat(filepath)
    return {
        "full_path": os.path.abspath(filepath),
        "size_bytes": stats.st_size,
        "created": datetime.fromtimestamp(stats.st_ctime).isoformat(),
        "modified": datetime.fromtimestamp(stats.st_mtime).isoformat(),
        "accessed": datetime.fromtimestamp(stats.st_atime).isoformat()
    }

def load_log(path):
    if os.path.exists(path):
        with open(path, 'r') as f:
            return json.load(f)
    return {}

def save_log(data, path):
    with open(path, 'w') as f:
        json.dump(data, f, indent=4)

def add_or_update_file(filepath, log_path, custom_metadata=None):
    """
    Adds or updates a file's metadata entry, including custom metadata.
    """
    if not os.path.isfile(filepath):
        print(f"File not found: {filepath}")
        return
    
    filename = os.path.basename(filepath)
    log = load_log(log_path)

    base_meta = get_base_metadata(filepath)
    entry = log.get(filename, {})

    # Combine existing metadata, new base, and new custom metadata
    entry.update(base_meta)
    if custom_metadata:
        entry.update(custom_metadata)

    log[filename] = entry
    save_log(log, log_path)
    print(f"Updated log for {filename}")


def read_metadata(filepath, log_path):
    """
    Adds or updates a file's metadata entry, including custom metadata.
    """
    if not os.path.isfile(filepath):
        print(f"File not found: {filepath}")
        return
    
    filename = os.path.basename(filepath)
    log = load_log(log_path)

    entry = log.get(filename, None)
    if entry:
        print(f"Metadata for {filename}:")
        for key, value in entry.items():
            print(f"{key}: {value}")
    else:
        print(f"No metadata found for {filename}")

def show_model_instances(log_path,keys=None, source_file=None):
    """
    Returns a list of model instances from the specified directory.
    """

    log = load_log(log_path)

    for filename, metadata in log.items():
        '''if 'EXTRACTED' in filename:
            #this is the result of applying a model to a preproccessed file
            pass
        elif 'KAGGLE' in filename:
            continue
        elif 'representation' in filename:
            continue
        else:
            #this is a preprocessed file
            pass'''
        metadata_source_file= metadata.get('source_file', None)
        if metadata_source_file == source_file:
            print(f"Metadata for {filename}:")
            #print(metadata)
            for key in keys:
                if key in metadata:
                    print(f"{key}: {metadata[key]}")
            print("------------------------------------------")
        else:
            continue
        '''
        if 'model' in metadata.get('model', []):
            model_instances.append({
                "filename": filename,
                "metadata": metadata
            })'''
    return 

def get_file_name(log_path,key_values=None, source_file=None):
    log = load_log(log_path)
    for filename, metadata in log.items():
        metadata_source_file= metadata.get('source_file', None)
        if metadata_source_file == source_file:
            truth_value=True
            for key in key_values:
                if key in metadata and metadata[key]==key_values[key]:
                    pass
                else:
                    truth_value=False
                    break
            if truth_value:
                print(filename)
        else:
            continue
    return

def change_filename_from_to(df, fr="old-laptop", to="new-laptop"):
    '''
    this function takes the dataframe (with the file_name column) and changes it so 
    that it can be used on the new computer if it was saved on the old an viceversa
    '''
    source_old = 'D:\\download\\PD project\\datasets'
    source_new = 'C:\\Users\\andre\\PhD\\Datasets'
    if fr == 'old-laptop':
        remove=source_old
    elif fr == 'new-laptop':
        remove=source_new
    else:
        raise ValueError("Invalid value for 'which'. Use 'New' or 'Old'.")
    if to == 'new-laptop':
        add=source_new
    elif to == 'old-laptop':
        add=source_old
    else:
        raise ValueError("Invalid value for 'which'. Use 'New' or 'Old'.")
    df['file_name'] = df['file_name'].str.replace(remove, add, regex=False)
    return df

def add_or_update_experiment(experiment_id, log_path, custom_metadata=None):
    """
    Adds or updates an experiment's metadata entry in the log.

    Parameters:
        experiment_id (str): A unique identifier for the experiment (e.g., a timestamp).
        log_path (str): Path to the JSON or pickle log file.
        base_metadata (dict): Core metadata for the experiment.
        custom_metadata (dict): Any additional metadata to attach.
    """
    log = load_log(log_path)
    
    # Start with any existing entry
    entry = log.get(experiment_id, {})

    if custom_metadata:
        entry.update(custom_metadata)

    log[experiment_id] = entry
    save_log(log, log_path)
    print(f"Updated log for experiment {experiment_id}")

def read_experiment_metadata(experiment_id, log_path):
    """
    Reads and prints metadata for a specific experiment.

    Parameters:
        experiment_id (str): The unique identifier for the experiment.
        log_path (str): Path to the JSON or pickle log file.
    """
    log = load_log(log_path)
    
    entry = log.get(experiment_id, None)
    if entry:
        print(f"Metadata for experiment {experiment_id}:")
        for key, value in entry.items():
            print(f"{key}: {value}")
    else:
        print(f"No metadata found for experiment {experiment_id}")