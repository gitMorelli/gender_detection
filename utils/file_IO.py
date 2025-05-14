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