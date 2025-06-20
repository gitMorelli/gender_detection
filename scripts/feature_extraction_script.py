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

class SimpleNN(nn.Module):
    def __init__(self, input_dim, hidden_dim=32, output_dim=2, dropout_rate=0.5):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class TorchClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, input_dim=None, hidden_dim=32, output_dim=2, dropout_rate=0.5,
                 epochs=50, lr=0.001, batch_size=32, patience=5, device=None):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.dropout_rate = dropout_rate
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.patience = patience
        self.device = device 

    def fit(self, X, y):
        device = torch.device(self.device)
        X = torch.tensor(X, dtype=torch.float32)
        y = torch.tensor(y.values if hasattr(y, "values") else y, dtype=torch.long)

        # Split for early stopping
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=42)

        train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
        val_dataset = torch.utils.data.TensorDataset(X_val, y_val)

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=self.batch_size)

        if self.input_dim is None:
            self.input_dim = X.shape[1]

        self.model_ = SimpleNN(self.input_dim, self.hidden_dim, self.output_dim, self.dropout_rate).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model_.parameters(), lr=self.lr)

        best_loss = float('inf')
        patience_counter = 0

        for epoch in range(self.epochs):
            self.model_.train()
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(device)
                optimizer.zero_grad()
                outputs = self.model_(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()

            # Early stopping evaluation
            self.model_.eval()
            val_loss = 0.0
            with torch.no_grad():
                for val_X, val_y in val_loader:
                    val_X, val_y = val_X.to(self.device), val_y.to(device)
                    val_outputs = self.model_(val_X)
                    val_loss += criterion(val_outputs, val_y).item()
            val_loss /= len(val_loader)

            if val_loss < best_loss:
                best_loss = val_loss
                patience_counter = 0
                best_model_state = self.model_.state_dict()
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    print(f"Early stopping at epoch {epoch + 1}")
                    break

        self.model_.load_state_dict(best_model_state)
        return self

    def predict(self, X):
        device = torch.device(self.device)
        X = torch.tensor(X, dtype=torch.float32).to(device)
        self.model_.eval()
        with torch.no_grad():
            outputs = self.model_(X)
            _, preds = torch.max(outputs, 1)
        return preds.cpu().numpy()


def ensembled_predictions(base_preds,writers,mode='majority_vote',probs=None):
    pred_df = pd.DataFrame({
            'writer': writers,
            'pred': base_preds
        })
    if probs is not None:
        probs = np.abs(probs - 0.5) / 0.5
        pred_df['prob'] = probs
    if mode == 'majority_vote':
        writer_preds = pred_df.groupby('writer')['pred'].agg(
            lambda x: Counter(x).most_common(1)[0][0]
        )
    elif mode == 'weighted_vote':
        if probs is None:
            raise ValueError("For 'weighted_vote', 'probs' must be provided.")
        writer_preds = pred_df.groupby('writer', group_keys=False).apply(
            lambda x: pd.Series({
                'writer_pred': round((x['pred'] * x['prob']).sum() / x['prob'].sum())
            })
        )['writer_pred'].astype(int)
    elif mode == 'most_probable':
        if probs is None:
            raise ValueError("For 'most_probable', 'probs' must be provided.")
        writer_preds = pred_df.groupby('writer', group_keys=False).apply(
            lambda x: x.loc[x['prob'].idxmax(), 'pred']
        )

    # Step 5: Map writer-level prediction back to each sample
    #final_preds = writers.map(writer_preds)
    return writer_preds#final_preds.values

def group_labels(y, writers):
    """
    Groups labels by writer.
    """
    grouped_labels = y.groupby(writers).agg(lambda x: Counter(x).most_common(1)[0][0])
    return grouped_labels

def compute_accuracies(y_true, y_pred, y_prob, pages, writers):
    """
    Computes accuracy for each writer.
    """
    accuracies = {}
    accuracies['individual'] = accuracy_score(y_true, y_pred)

    grouped_true = group_labels(y_true, pages)
    grouped_pred = ensembled_predictions(y_pred, pages)
    accuracies['ensembled'] = accuracy_score(grouped_true, grouped_pred)

    grouped_pred = ensembled_predictions(y_pred, pages, mode='weighted_vote',probs=y_prob)
    accuracies['ensembled_weighted'] = accuracy_score(grouped_true, grouped_pred)

    grouped_pred = ensembled_predictions(y_pred, pages, mode='most_probable',probs=y_prob)
    accuracies['ensembled_most_probable'] = accuracy_score(grouped_true, grouped_pred)

    grouped_true_writers = group_labels(y_true, writers)
    grouped_pred_writers = ensembled_predictions(y_pred, writers)
    accuracies['ensembled_writers'] = accuracy_score(grouped_true_writers, grouped_pred_writers)
    
    return accuracies


def select_groups(train_FE,select_column='train', train_on_language='arabic', train_on_same='same'):
        train_FE[select_column] = 1
        if train_on_language=='arabic':
            train_FE.loc[train_FE['isEng'] == 1, select_column] = 0 #remove english
        elif train_on_language=='english':
            train_FE.loc[train_FE['isEng'] == 0, select_column] = 0 #remove arabic
        else:
            pass
        if train_on_same=='same':
            train_FE.loc[train_FE['same_text'] == 0, select_column] = 0 #remove different texts
        elif train_on_same=='different':
            train_FE.loc[train_FE['same_text'] == 1, select_column] = 0 #remove same texts
        else:
            pass
        return train_FE


def compute_subgroup_accuracies(pipeline, train_df,cols_to_drop,target_label):
    subgroup_accuracies = {}
    groups = [('english','different'), ('english','same'), ('arabic','different'), ('arabic','same'),
              ('english','all'), ('arabic','all'), ('all','different'), ('all','same')]
    group_sizes = []
    acc_keys=None
    for group in groups:
        train_df=select_groups(train_df,select_column='train', 
                            train_on_language=group[0], train_on_same=group[1])
        X_s = train_df[train_df['train']==1].drop(columns=cols_to_drop)
        y_s = train_df[train_df['train']==1][target_label]
        writers_s = train_df[train_df['train']==1]['writer']-1
        pages_s = train_df[train_df['train']==1]['page']
        group_sizes.append(len(X_s))

        y_prob= pipeline.predict_proba(X_s.values)[:,1]
        #y_pred = pipeline.predict(X_s.values)
        y_pred=(y_prob >= 0.5).astype(int)
        accuracies = compute_accuracies(y_s, y_pred, y_prob, pages_s,writers_s)
        subgroup_accuracies[f'{group[0]},{group[1]}'] = accuracies

    return subgroup_accuracies

def select_n_patches(train_df, n_patches=10):
    grouped_sorted = train_df.groupby('page', group_keys=False).apply(lambda x: x.sort_values('black_ratio', ascending=False))
    grouped_sorted = grouped_sorted.groupby('page', group_keys=False).head(n_patches)
    return grouped_sorted

def main(args):
    """
    Main function to run the feature extraction and classification pipeline.
    """
    print("Running feature extraction script...")
    args = load_config(args.config)
    output_dir = source_path + "\\outputs\\preprocessed_data\\"
    LOG_FILE = output_dir+"file_metadata_log.json"
    df_log = file_IO.assemble_csv_from_log(LOG_FILE)

    n_patches = args.n_patches
    n_writers = args.n_writers
    input_file_name=args.input_file_name
    df_log = file_IO.assemble_csv_from_log(LOG_FILE)
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
    if task == 'language detection' and train_on_language != 'all':
        raise ValueError("For language detection, 'train_on' must be 'all'.")
    if task=='gender_detection':
        target_label='male'
    else:
        target_label='isEng'
    
    input_file=source_path+'\\outputs\\preprocessed_data\\'+input_file_name
    train_FE = pd.read_csv(input_file)
    if is_kaggle:
        cols_to_drop = ['writer', 'same_text', 'train','page_id','isEng','train','index','male']
    else:
        cols_to_drop = [c for c in train_FE.columns if not(c.startswith('f') and len(c) > 1 and c[1].isdigit())]
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


    train_FE=select_groups(train_FE,select_column='train', 
                        train_on_language=train_on_language, train_on_same=train_on_same)
    train_FE_selected = train_FE[train_FE['train']==1].copy()

    X = train_FE[train_FE['train']==1].drop(columns=cols_to_drop)
    y = train_FE[train_FE['train']==1][target_label]

    writers = train_FE[train_FE['train']==1]['writer']-1
    pages = train_FE[train_FE['train']==1]['page']

    gkf = GroupKFold(n_splits=n_splits)

    svm = SVC(kernel='rbf', C=0.1, gamma='scale', probability=True, random_state=42)
    # Define the models
    gbm_classifier = GradientBoostingClassifier(
        n_estimators=100, #100 is standard 
        learning_rate=0.1,  
        max_depth=3,  
        random_state=42
    )
    #cat = CatBoostClassifier(iterations=100, learning_rate=0.1, depth=6, verbose=0, random_state=42)
    #lgbm = lgb.LGBMClassifier(n_estimators=10, learning_rate=0.1, max_depth=-1, random_state=42)
    lgbm = lgb.LGBMClassifier(
        n_estimators=1000,
        learning_rate=0.05,
        max_depth=5,
        num_leaves=20,
        min_child_samples=30,#Minimum number of data samples per leaf
        subsample=0.8, #Randomness in row 
        colsample_bytree=0.8, #and feature sampling respectively.
        reg_alpha=1.0, # L1 regularization
        reg_lambda=1.0, # L2 regularization
        random_state=42,
        n_jobs=-1,
        min_split_gain=0.01,  # Minimum gain to make a split
    )
    xgb = XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
    #rf = RandomForestClassifier(n_estimators=100, max_depth=None, random_state=42)
    rf = RandomForestClassifier(
        n_estimators=200,            # More trees = more stable
        max_depth=10,                # Limits tree depth (main regularizer)
        min_samples_split=10,        # Minimum samples to split a node
        min_samples_leaf=5,          # Minimum samples at a leaf node
        max_features='sqrt',         # Random feature selection at each split
        bootstrap=True,              # Use bootstrapped samples (default)
        oob_score=True,              # Out-of-bag error estimate
        random_state=42,
        n_jobs=-1
    )

    mlp = MLPClassifier(hidden_layer_sizes=(128), activation='relu', solver='adam',
                        max_iter=200, random_state=42, early_stopping=True, validation_fraction=0.1, n_iter_no_change=10)
    dt = DecisionTreeClassifier(max_depth=3, min_samples_split=5, min_samples_leaf=2, ccp_alpha=0.01, random_state=42)
    logreg = LogisticRegression(max_iter=1000, random_state=42)
    feature_extraction_model = {'gbm': gbm_classifier, 'lgbm': lgbm, 'xgb': xgb, 'rf':rf, 'mlp':mlp, 'dt':dt,'logreg':logreg, 'torch': None, 'svm': svm}

    if selected_model == 'torch':
        # We'll infer input dimension dynamically just before training
        input_dim = X.shape[1]
        output_dim = len(np.unique(y))
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print("Device is: ",device)
        torch_model_wrapper = TorchClassifier(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dim=512,
            dropout_rate=0.5,
            epochs=200,
            patience=15,
            lr=0.001,
            device=device,
        )
        feature_extraction_model['torch'] = torch_model_wrapper

    if with_pca:
        from sklearn.decomposition import PCA
        pca = PCA(n_components=n_components)  #384
        pipeline = Pipeline([
            ('scaler', StandardScaler()),  # Normalize features
            ('pca', pca),  # Apply PCA
            (selected_model, feature_extraction_model[selected_model])  # Train GBM classifier
        ])
    else:    
        pipeline = Pipeline([
            ('scaler', StandardScaler()),  # Normalize features
            (selected_model, feature_extraction_model[selected_model])  # Train GBM classifier
        ])

    # Measure the start time
    start_time = time.time()
    print(f"Starting model cross-val...")
    cross_val_accuracies = {"IF": [], "OOF": []}
    cross_val_subgroup_accuracies = []
    for train_idx, val_idx in gkf.split(X, y, groups=writers):
        #print(train_idx)
        X_train, X_val = X.loc[train_idx], X.loc[val_idx]
        y_train, y_val = y.loc[train_idx], y.loc[val_idx]
        X_train, y_train, writers_train, pages_train = shuffle(
            X_train, y_train, writers.loc[train_idx], pages.loc[train_idx], random_state=42
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
        accuracies = compute_accuracies(y_val, y_pred, y_prob,pages.loc[val_idx], writers.loc[val_idx])
        cross_val_accuracies["OOF"].append(accuracies)

        cross_val_subgroup_accuracies.append(compute_subgroup_accuracies(pipeline, train_FE_selected.loc[val_idx],cols_to_drop,target_label))
    
    # Measure the end time
    end_time = time.time()
    # Calculate the time taken
    time_taken_cross_val = end_time - start_time
    print(f"Time taken to cross-validate the model: {time_taken_cross_val:.2f} seconds")

    if with_pca:
        pca = pipeline.named_steps['pca']
        print(f"Number of features used after PCA: {pca.n_components_}")


    print('saving to log file...')
    #experiment = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(source_path, "outputs", "logs")
    # Example usage:
    LOG_OUT_FILE = out_dir+"\\feature_extraction_metadata_log.json"
    print(f"Log file path: {LOG_OUT_FILE}")
    file_IO.add_or_update_experiment(
        f"{args.n_job}_{experiment}", LOG_OUT_FILE,
        custom_metadata={
            "type of preprocessing": extracted_from,
            "original raw file": source_data,
            "input file": input_file_name,
            "FE model": model_used,
            "classifier model": selected_model,
            "model_params": feature_extraction_model[selected_model].get_params(),
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
            "description": ''' I am training a classifier on the feature vectors extracted by a deep model
            I am evaluating the results on subsets of the training data, based on language and same/different text.''' 
        }
    )

def parse_args():
    parser = argparse.ArgumentParser(description="ML experiments!")
    parser.add_argument("--config", type=str, required=True, help="The cofig file to pass in input to the script")
    '''parser.add_argument("--input_file_name", type=str, required=True, help="The input training file")
    parser.add_argument("--selected_model", type=str, required=True, help="The name of the model to use.")
    parser.add_argument("--is_kaggle", default=False, action="store_true", help="Flag to indicate if running on Kaggle.")
    parser.add_argument("--with_pca", required=True, action="store_true", help="Flag to apply PCA.")
    parser.add_argument("--n_components", default=0.95, help="Number of PCA components if PCA is applied.")
    parser.add_argument("--with_cross_validation", default=True, action="store_true", help="Flag to enable cross-validation.")
    parser.add_argument("--task", type=str, required=True, choices=['gender detection', 'language detection'], help="Type of task.")
    parser.add_argument("--train_on_language", type=str, default='all' , choices=['all', 'english','arabic'],help="Language subset to train on (e.g., 'en', 'fr').")
    parser.add_argument("--train_on_same", type=str, default='all', choices=['all', 'same','different'], help="Train only on samples with the same attribute.")
    parser.add_argument("--n_splits", type=int, default=5, help="Number of splits for cross-validation.")'''
    return parser.parse_args()

if __name__ == "__main__":
    # Add the root of the project to the path
    source_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    sys.path.append(source_path)
    import utils.file_IO as file_IO
    args = parse_args()
    main(args)