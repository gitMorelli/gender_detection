import json
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

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GroupKFold
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, ClassifierMixin

import torch
import torch.nn as nn
import torch.optim as optim
import yaml
import warnings


def main(args):
    """
    Main function to run the feature extraction and classification pipeline.
    """
    warnings.filterwarnings("ignore")
    print("Running feature extraction script...")
    args = load_config(args.config)
    patch_mode = args.patch_merging
    n_patches = args.n_patches
    n_writers = args.n_writers
    train_filename=args.train_filename
    val_filename = args.val_filename
    selected_model = args.selected_model
    is_kaggle = args.is_kaggle
    with_pca=args.with_pca
    n_components = args.n_components
    validation_mode=args.validation_mode
    task = args.task
    train_on_language = args.train_on_language
    train_on_same = args.train_on_same
    n_splits = args.n_splits
    save_path = args.save_path
    extra_view=args.extra_view
    extra_integration_mode = args.extra_integration_mode
    extra_train_filename = args.extra_train_filename
    extra_val_filename = args.extra_val_filename
    aggregation_mode = args.aggregation_mode
    sklearn_model_parameters = args.sklearn_model_parameters

    if task == 'language detection' and train_on_language != 'all':
        raise ValueError("For language detection, 'train_on' must be 'all'.")
    if task=='gender_detection':
        target_label='male'
    else:
        target_label='isEng'
    
    train_FE = pd.read_csv(train_filename)
    if validation_mode=='val_only':
        val_df = pd.read_csv(val_filename)
        if extra_view:
            train_df_extra = pd.read_csv(extra_train_filename)
            val_df_extra = pd.read_csv(extra_val_filename)
            train_FE = dataframes.merge_dfs(train_FE, train_df_extra, mode=extra_integration_mode)
            val_df = dataframes.merge_dfs(val_df, val_df_extra, mode=extra_integration_mode) 

        train_FE = dataframes.aggregate_dfs(train_FE,mode=aggregation_mode)
        val_df = dataframes.aggregate_dfs(val_df,mode=aggregation_mode)

        train_FE = pd.concat([train_FE, val_df], ignore_index=True)
    
    if is_kaggle:
        cols_to_drop = ['writer', 'same_text', 'train','page_id','isEng','train','index','male']
    else:
        cols_to_drop = [c for c in train_FE.columns if not(c.startswith('f') and len(c) > 1 and c[1].isdigit())]
    train_FE['page'] = train_FE.groupby(['writer', 'isEng', 'same_text']).ngroup()
    if n_patches > 0:
        train_FE = select_n_patches(train_FE, n_patches=n_patches).reset_index(drop=True)
    if n_writers > 0:
        unique_writers = train_FE['writer'].unique()
        n_selected = int(len(unique_writers) * n_writers)
        selected_writers = np.random.choice(unique_writers, n_selected, replace=False)
        train_FE = train_FE[train_FE['writer'].isin(selected_writers)].reset_index(drop=True)
    cols_to_drop = [c for c in train_FE.columns if not(c.startswith('f') and len(c) > 1 and c[1].isdigit())]
    cols_to_keep = [c for c in train_FE.columns if c.startswith('f') and len(c) > 1 and c[1].isdigit()]


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

    model = get_sklearn_model(name=selected_model, **sklearn_model_parameters)

    if with_pca:
        from sklearn.decomposition import PCA
        pca = PCA(n_components=n_components)  #384
        pipeline = Pipeline([
            ('scaler', StandardScaler()),  # Normalize features
            ('pca', pca),  # Apply PCA
            (selected_model, model)  # Train GBM classifier
        ])
    else:    
        pipeline = Pipeline([
            ('scaler', StandardScaler()),  # Normalize features
            (selected_model, model)  # Train GBM classifier
        ])

    # Measure the start time
    start_time = time.time()
    print(f"Starting model cross-val...")
    cross_val_accuracies = {"IF": [], "OOF": []}
    cross_val_subgroup_accuracies = []
    average_ensembled_weighted = 0
    average_individual = 0
    for train_idx, val_idx in gkf.split(X, y, groups=writers):
        if validation_mode=='val_only':
            val_writers = val_df['writer'].unique()
            val_idx = train_FE[train_FE['writer'].isin(val_writers)].index
            train_idx = train_FE[~train_FE['writer'].isin(val_writers)].index
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
        print(f"Fold {len(cross_val_accuracies['IF'])} - IF Accuracy: {accuracies['individual']:.4f}, IF Accuracy: {accuracies['ensembled_weighted']:.4f}")
        y_prob= pipeline.predict_proba(X_val.values)[:,1]
        #y_pred = pipeline.predict(X_val.values)
        y_pred=(y_prob >= 0.5).astype(int)
        accuracies = compute_accuracies(y_val, y_pred, y_prob,pages.iloc[val_idx], writers.iloc[val_idx])
        cross_val_accuracies["OOF"].append(accuracies)
        print(f"Fold {len(cross_val_accuracies['IF'])} - OOF Accuracy: {accuracies['individual']:.4f}, OOF Accuracy: {accuracies['ensembled_weighted']:.4f}")
        average_ensembled_weighted += accuracies['ensembled_weighted']
        average_individual += accuracies['individual']
        # Filter train_FE_selected to only include rows with train==1 and index in val_idx
        train_FE_temp = train_FE_selected[train_FE_selected['writer'].isin(writers.iloc[val_idx].unique())]
        cross_val_subgroup_accuracies.append(compute_subgroup_accuracies(pipeline, train_FE_temp, cols_to_drop, target_label))
        if validation_mode=='val_only':
            n_splits=1
            break
        if validation_mode=='1fold_train_only':
            n_splits=1
            val_writers= writers.iloc[val_idx].unique() 
            writer_train_df = pd.DataFrame({
                'writer': writers.unique()
            })
            writer_train_df['train'] = writer_train_df['writer'].apply(lambda w: 0 if w in val_writers else 1)
            file_name = '1fold_train_only_writers.csv'
            writer_train_df.to_csv(os.path.join(save_path, file_name), index=False)
            break
    print(f"Average ensembled weighted accuracy: {average_ensembled_weighted / n_splits:.4f}")
    print(f"Average individual accuracy: {average_individual / n_splits:.4f}")
    
    # Measure the end time
    end_time = time.time()
    # Calculate the time taken
    time_taken_cross_val = end_time - start_time
    print(f"Time taken to cross-validate the model: {time_taken_cross_val:.2f} seconds")

    if with_pca: 
        pca = pipeline.named_steps['pca']
        print(f"Number of features used after PCA: {pca.n_components_}")

    # Save the trained pipeline to a file for later use
    file_name = validation_mode+'_pipeline.joblib'
    full_save_path = os.path.join(save_path, file_name)
    if not os.path.exists(save_path):
        print('error: save path does not exist')
    joblib.dump(pipeline, full_save_path)
    print(f"Model pipeline saved to file")

    # Save best model performance to JSON as a dict
    best_model_performance = {
        "average_ensembled_weighted_accuracy": average_ensembled_weighted / n_splits,
        "average_individual_accuracy": average_individual / n_splits,
        "cross_val_accuracies": cross_val_accuracies,
        "cross_val_subgroup_accuracies": cross_val_subgroup_accuracies,
        "time_taken_cross_val_seconds": time_taken_cross_val
    }
    performance_json_path = os.path.join(save_path, "best_model_performance_temp.json")
    with open(performance_json_path, "w") as f:
        json.dump(best_model_performance, f, indent=4)
    print(f"Best model performance saved to {performance_json_path}")

def parse_args():
    parser = argparse.ArgumentParser(description="ML experiments!")
    parser.add_argument("--config", type=str, required=True, help="The cofig file to pass in input to the script")
    return parser.parse_args()

if __name__ == "__main__":
    # Add the root of the project to the path
    source_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    sys.path.append(source_path)
    import utils.file_IO as file_IO
    from utils.script_launching import load_config
    from utils.script_launching import DotDict
    from utils.train_on_rep_utils import select_n_patches
    from utils.evaluation_utils import ensembled_predictions, compute_accuracies, group_labels, select_groups, compute_subgroup_accuracies
    from utils.model_utils import get_sklearn_model
    import utils.dataframes as dataframes
    args = parse_args()
    main(args)