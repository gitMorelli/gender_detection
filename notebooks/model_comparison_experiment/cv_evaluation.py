from xgboost import XGBClassifier
import lightgbm as lgb
from lightgbm import early_stopping, log_evaluation
#from catboost import CatBoostClassifier

import matplotlib.pyplot as plt
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
import sys
import argparse
import time
from collections import Counter
import io

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GroupKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

import torch
import torch.nn as nn
import torch.optim as optim
import yaml
import warnings

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


def main(args):
    """
    Main function to run the feature extraction and classification pipeline.
    """
    # Reproducibility: set a global random seed (can be overridden in the config under "seed" or via env SEED)
    seed = 42
    # Set deterministic seeds for Python, numpy, torch and environment
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["OMP_NUM_THREADS"] = os.environ.get("OMP_NUM_THREADS", "1")
    os.environ["MKL_NUM_THREADS"] = os.environ.get("MKL_NUM_THREADS", "1")
    random.seed(seed)
    np.random.seed(seed)
    try:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # Prefer to enable deterministic algorithms when available (may raise on some ops)
        if hasattr(torch, "use_deterministic_algorithms"):
            torch.use_deterministic_algorithms(True)
    except Exception:
        pass

    warnings.filterwarnings("ignore")
    print("Running feature extraction script...")
    args = load_config(args.config)
    script_mode = args.script_mode
    patch_mode = args.patch_merging 

    output_dir = source_path + "\\outputs\\preprocessed_data\\"
    LOG_FILE = output_dir+"file_metadata_log.json"
    df_log = file_IO.assemble_csv_from_log(LOG_FILE)

    experiment_name = args.experiment_name
    n_patches = args.n_patches
    n_writers = args.n_writers
    input_file_name=args.input_file_name
    if script_mode == 'explainability_pipeline':
        pass
    else:
        row=df_log[df_log['experiment']==input_file_name]
        source_data = row['source_file'].values[0]
        model_used = row['model'].values[0]
        row=df_log[df_log['experiment']==source_data]
        extracted_from = row['type'].values[0]
    #args for training
    selected_model = args.selected_model
    is_kaggle = args.is_kaggle
    with_pca=args.with_pca
    n_components = args.n_components
    with_cross_validation=args.with_cross_validation
    task = args.task
    train_on_language = args.train_on_language
    train_on_same = args.train_on_same
    n_splits = args.n_splits
    with_feature_scaling = args.with_feature_scaling
    if task == 'language detection' and train_on_language != 'all':
        raise ValueError("For language detection, 'train_on' must be 'all'.")
    if task=='gender_detection':
        target_label='male'
    else:
        target_label='isEng'
    
    if script_mode == 'standalone':
        input_file=source_path+'\\outputs\\preprocessed_data\\'+input_file_name
    else:
        input_file = source_path+'\\outputs\\online_deep_feature_extraction\\'+input_file_name
    train_FE = pd.read_csv(input_file)
    if is_kaggle:
        cols_to_drop = ['writer', 'same_text', 'train','page_id','isEng','train','index','male']
    else:
        cols_to_drop = [c for c in train_FE.columns if not(c.startswith('f') and len(c) > 1 and c[1].isdigit())]
    if script_mode == 'standalone':
        train_FE = file_IO.change_filename_from_to(train_FE, fr="old-laptop", to="new-laptop")
    train_FE['page'] = train_FE.groupby(['writer', 'isEng', 'same_text']).ngroup()
    if n_patches > 0:
        #print(f"Selecting {n_patches} patches per page...")
        train_FE = select_n_patches(train_FE, n_patches=n_patches).reset_index(drop=True)
    if n_writers > 0:
        unique_writers = train_FE['writer'].unique()
        n_selected = int(len(unique_writers) * n_writers)
        selected_writers = np.random.choice(unique_writers, n_selected, replace=False)
        train_FE = train_FE[train_FE['writer'].isin(selected_writers)].reset_index(drop=True)
    cols_to_drop = [c for c in train_FE.columns if not(c.startswith('f') and len(c) > 1 and c[1].isdigit())]
    cols_to_keep = [c for c in train_FE.columns if c.startswith('f') and len(c) > 1 and c[1].isdigit()]

    if patch_mode == 'average':
        agg_dict = {col: 'mean' for col in cols_to_keep}
        agg_dict.update({col: 'first' for col in cols_to_drop})
        # Group by 'page' and average the feature columns
        train_FE = train_FE.groupby('page',as_index=False).agg(agg_dict)
    elif patch_mode == 'max':
        agg_dict = {col: 'max' for col in cols_to_keep}
        agg_dict.update({col: 'first' for col in cols_to_drop})
        # Group by 'page' and average the feature columns
        train_FE = train_FE.groupby('page',as_index=False).agg(agg_dict)


    train_FE=select_groups(train_FE,select_column='train', 
                        train_on_language=train_on_language, train_on_same=train_on_same)
    train_FE = train_FE.sort_values('page').reset_index(drop=True)
    train_FE_selected = train_FE.copy()

    X = train_FE[train_FE['train']==1].drop(columns=cols_to_drop)
    repr_size = X.shape[1]
    y = train_FE[train_FE['train']==1][target_label]

    writers = train_FE[train_FE['train']==1]['writer']
    pages = train_FE[train_FE['train']==1]['page']

    gkf = GroupKFold(n_splits=n_splits)

    model=get_sklearn_model(name='logreg')
    
    if with_pca:
        from sklearn.decomposition import PCA
        pca = PCA(n_components=n_components)  #384
        if with_feature_scaling:
            pipeline = Pipeline([
                ('scaler', StandardScaler()),  # Normalize features
                ('pca', pca),  # Apply PCA
                (selected_model, model)  
            ])
        else:
            pipeline = Pipeline([
                ('pca', pca),  # Apply PCA
                (selected_model, model)  
            ])
    else:    
        if with_feature_scaling:
            pipeline = Pipeline([
                ('scaler', StandardScaler()),  # Normalize features
                (selected_model, model)  
            ])
        else:
            pipeline = Pipeline([
                (selected_model, model)  
            ])

    # Measure the start time
    start_time = time.time()
    print(f"Starting model cross-val...")
    cross_val_accuracies = {"IF": [], "OOF": []}
    cross_val_subgroup_accuracies = []
    for train_idx, val_idx in gkf.split(X, y, groups=writers):
        #print(train_idx)
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        X_train, y_train, writers_train, pages_train = shuffle(
            X_train, y_train, writers.iloc[train_idx], pages.iloc[train_idx], random_state=42
        )
        # Fit the model on training data
        pipeline.fit(X_train.values, y_train)
        y_prob= pipeline.predict_proba(X_train.values)[:,1]
        #y_pred = pipeline.predict(X_train.values)
        y_pred=(y_prob>= 0.5).astype(int)
        accuracies = compute_accuracies(y_train, y_pred, y_prob,pages_train,writers_train)
        cross_val_accuracies["IF"].append(accuracies)
        y_prob= pipeline.predict_proba(X_val.values)[:,1]
        #y_pred = pipeline.predict(X_val.values)
        y_pred=(y_prob >= 0.5).astype(int)
        accuracies = compute_accuracies(y_val, y_pred, y_prob,pages.iloc[val_idx], writers.iloc[val_idx])
        cross_val_accuracies["OOF"].append(accuracies)
        # Filter train_FE_selected to only include rows with train==1 and index in val_idx
        train_FE_temp = train_FE_selected[train_FE_selected['writer'].isin(writers.iloc[val_idx].unique())]
        cross_val_subgroup_accuracies.append(compute_subgroup_accuracies(pipeline, train_FE_temp, cols_to_drop, target_label))
        if script_mode == 'explainability_pipeline': # i onli need one fold (simulate grouped train test split) if i am evaluating explainability
            print("Cross-validation completed for explainability pipeline mode.")
            break
    
    # Measure the end time
    end_time = time.time()
    # Calculate the time taken
    time_taken_cross_val = end_time - start_time
    print(f"Time taken to cross-validate the model: {time_taken_cross_val:.2f} seconds")

    if with_pca:
        pca = pipeline.named_steps['pca']
        print(f"Number of features used after PCA: {pca.n_components_}")

    if script_mode == 'standalone':
        print('saving to log file...')
        #experiment = datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = 'outputs'
        # Example usage:
        LOG_OUT_FILE = out_dir+f"/results_{experiment_name}.json"
        print(f"Log file path: {LOG_OUT_FILE}")
        file_IO.add_or_update_experiment(
            f"{args.n_job}_{experiment}", LOG_OUT_FILE,
            custom_metadata={
                "type of preprocessing": extracted_from,
                "original raw file": source_data,
                "input file": input_file_name,
                "FE model": model_used,
                "classifier model": selected_model,
                "model_params": model.get_params(),
                "n_splits": n_splits,
                "train_on_language": train_on_language,
                "train_on_same": train_on_same,
                "task": task,
                "with cross validation": with_cross_validation,
                "with PCA": with_pca,
                "n_components": n_components,
                "training time for cross-validation": time_taken_cross_val,
                "cross_val_accuracies": cross_val_accuracies,
                "cross_val_subgroup_accuracies": cross_val_subgroup_accuracies,
                "is_kaggle": is_kaggle,
                "test": 'this is a test column',
                "n_sub_patches": n_patches,
                "n_writers": n_writers,
                "patch_mode": patch_mode,
                "representation size": repr_size,
                "description": ''' I am training a classifier on the feature vectors extracted by a deep model
                I am evaluating the results on subsets of the training data, based on language and same/different text.''' 
            }
        )
    elif script_mode == 'explainability_pipeline':
        val_writers= writers.iloc[val_idx].unique() 
        #train_writers = writers.iloc[train_idx].unique()
        writer_train_df = pd.DataFrame({
            'writer': writers.unique()
        })
        writer_train_df['train'] = writer_train_df['writer'].apply(lambda w: 0 if w in val_writers else 1)
        writer_train_df.to_csv(input_file.split('.')[0]+'_writers.csv', index=False)
        # Save the trained pipeline to a file for later use
        joblib.dump(pipeline, input_file.split('.')[0]+'_pipeline.joblib')
        print(f"Model pipeline saved to file")

def parse_args():
    parser = argparse.ArgumentParser(description="ML experiments!")
    parser.add_argument("--config", type=str, required=True, help="The cofig file to pass in input to the script")
    return parser.parse_args()

if __name__ == "__main__":
    # Add the root of the project to the path
    source_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..','..',))
    sys.path.append(source_path)
    import utils.file_IO as file_IO
    from utils.evaluation_utils import select_n_patches,compute_subgroup_accuracies
    from utils.evaluation_utils import select_groups,compute_accuracies,group_labels,ensembled_predictions
    from utils.model_utils import get_sklearn_model
    import random
    args = parse_args()
    main(args)