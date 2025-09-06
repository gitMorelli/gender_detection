import json
from datetime import datetime
import os
import pandas as pd
import glob
import shutil
import matplotlib.pyplot as plt


def assemble_file_name(input_string):
    file_parts = input_string.split("\\")
    last_two = file_parts[-2:] if len(file_parts) >= 2 else file_parts
    last_two[-1] = os.path.splitext(last_two[-1])[0]
    return os.path.join(*last_two)

def access_or_create_dir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path,exist_ok=True)

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
    print(filename)
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

def read_experiment_metadata(experiment_id, log_path, keys=None):
    """
    Reads and prints metadata for a specific experiment.

    Parameters:
        experiment_id (str): The unique identifier for the experiment.
        log_path (str): Path to the JSON or pickle log file.
    """
    log = load_log(log_path)
    
    entry = log.get(experiment_id, None)
    metadata = log[entry]
    for key in keys:
        if key in metadata:
            print(f"{key}: {metadata[key]}")
        else:
            print(f"{key} not found in metadata for experiment {experiment_id}")

def assemble_csv_from_log(log_path):
    """
    Assembles a CSV file from the log data.

    Parameters:
        log_path (str): Path to the JSON or pickle log file.
        output_csv (str): Path where the CSV file will be saved.
    """
    
    log = load_log(log_path)
    df = pd.DataFrame.from_dict(log, orient='index')
    df = df.reset_index()
    # Rename the new column for clarity (e.g., "timestamp" if keys represent time)
    df.rename(columns={'index': 'experiment'}, inplace=True)

    return df

def save_args(args, filepath):
    filepath = os.path.join(filepath, "args.txt")  # or any filename you prefer
    with open(filepath, 'w') as f:
        for key, value in args.__dict__.items():
            f.write(f"{key} = {value}\n")

def get_csv_filename(folder):
    files = glob.glob(os.path.join(folder, "*.csv"))
    return os.path.basename(files[0]) if files else None

def load_input_files(source_path,selected_FE,kind,suffix,custom_pretrained='original'):
    if custom_pretrained=='original':
        input_dir=source_path+f'\\outputs\\online_deep_feature_extraction\\{selected_FE}\\representation_extraction\\extracted_representation'
    else:
        input_dir=source_path+f'\\outputs\\online_deep_feature_extraction\\{selected_FE}\\{custom_pretrained}\\extracted_representation'
    if kind == 'patches_224':
        source_file_train='icdar_train_df_patches_20250716_113702'
        source_file_val='icdar_train_df_patches_20250716_120511'
        source_file_test='icdar_train_df_patches_20250716_115204'
        train_filename = input_dir+f'\\train{suffix}\\{selected_FE}_features_{source_file_train}.csv'
        val_filename = input_dir+f'\\val\\{selected_FE}_features_{source_file_val}.csv'
        test_filename = input_dir+f'\\test\\{selected_FE}_features_{source_file_test}.csv'
    if kind == 'patches_448':
        source_file_train='icdar_train_df_patches_20250811_193545'
        source_file_val='icdar_train_df_patches_20250716_120511'
        source_file_test='icdar_train_df_patches_20250716_115204'
        train_filename = input_dir+f'\\train{suffix}\\{selected_FE}_features_{source_file_train}.csv'
        val_filename = input_dir+f'\\val\\{selected_FE}_features_{source_file_val}.csv'
        test_filename = input_dir+f'\\test\\{selected_FE}_features_{source_file_test}.csv'
    if kind == 'body':
        extra_file_train = 'icdar_train_df_body_20250523_181312'
        extra_file_val = 'icdar_train_df_body_20250716_123551' 
        train_filename = input_dir+'\\extra_view'+f'\\train{suffix}\\{selected_FE}_features_{extra_file_train}.csv' 
        val_filename = input_dir+'\\extra_view'+f'\\val\\{selected_FE}_features_{extra_file_val}.csv' 
        test_filename = None
    return train_filename, val_filename, test_filename
def load_preprocessed_files(kind, mode):
    if kind == 'patches_224':
        if mode == 'train':
            output = 'icdar_train_df_patches_20250716_113702.csv'
        elif mode == 'val':
            output = 'icdar_train_df_patches_20250716_120511.csv'
        elif mode == 'test':
            output = 'icdar_train_df_patches_20250716_115204.csv'
        elif mode == 'contrastive':
            output = 'icdar_train_df_iam_rimes_patches_20250724_200130.csv'
    if kind == 'patches_448':
        if mode == 'train':
            output = 'icdar_train_df_patches_20250811_193545.csv'
        elif mode == 'val':
            raise ValueError("No validation set for patches_448")
        elif mode == 'test':
            raise ValueError("No test set for patches_448")
    elif kind == 'body':
        if mode == 'train':
            output = 'icdar_train_df_body_20250523_181312.csv'
        elif mode == 'val':
            output = 'icdar_train_df_body_20250716_123551.csv'
    elif kind == 'words_sentences':
        if mode == 'train':
            output = 'icdar_train_df_words_sentences_20250522_230307.csv'
    elif kind == 'patches_standard':
        if mode == 'train':
            output = 'icdar_train_df_patches_20250515_164130.csv'
    return output

def clear_folder(folder_path):
    """
    Deletes all files and subdirectories in the specified folder.
    """
    if not os.path.exists(folder_path):
        print(f"Folder not found: {folder_path}")
        return
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print(f"Failed to delete {file_path}. Reason: {e}")

def init_expl_dirs(search_dir,clear=False):
    explanation_dir = search_dir + 'explanations'
    cam_dir = explanation_dir + '\\cam'
    attention_dir = explanation_dir + '\\attention'
    original_dir = search_dir + 'original'
    transformed_dir = search_dir + 'preprocessed'
    augmented_dir = search_dir + 'augmentation'
    dirs= [explanation_dir, cam_dir, attention_dir, original_dir, transformed_dir, augmented_dir]
    for dir in dirs:
        if clear:
            access_or_create_dir(dir)
            clear_folder(dir)
        else:
            access_or_create_dir(dir)
    return explanation_dir, cam_dir, attention_dir, original_dir, transformed_dir, augmented_dir

'''def dumb_init(search_dir,clear=False):
    explanation_dir = search_dir + 'explanations'
    string = '\\cam\\attention'
    for i in range(10):
        explanation_dir += string
        os.makedirs(explanation_dir, exist_ok=True)
        os.chmod(explanation_dir, 0o555)  if i%3==0 else os.chmod(explanation_dir, 0o755)'''
def save_dict_to_txt(my_dict, file_path="output.txt"):
    with open(file_path, "w") as f:
        for key, value in my_dict.items():
            f.write(f"{key}: {value}\n")

def save_df_to_png(display_df, save_dir):
    # Include index as a column for display
    display_df_with_index = display_df.copy()
    display_df_with_index.insert(0, display_df_with_index.index.name or "index", display_df_with_index.index)
    
    fig, ax = plt.subplots()
    ax.axis('tight')
    ax.axis('off')
    table = ax.table(
        cellText=display_df_with_index.values,
        colLabels=display_df_with_index.columns,
        loc='center'
    )

    plt.savefig(save_dir, dpi=300)
    plt.close()