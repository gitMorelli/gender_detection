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


def ensembled_predictions(base_preds,writers):
    pred_df = pd.DataFrame({
            'writer': writers,
            'pred': base_preds
        })

    # Step 4: Compute majority vote for each writer
    writer_preds = pred_df.groupby('writer')['pred'].agg(
        lambda x: Counter(x).most_common(1)[0][0]
    )

    # Step 5: Map writer-level prediction back to each sample
    final_preds = writers.map(writer_preds)

    return writer_preds#final_preds.values

def group_labels(y, writers):
    """
    Groups labels by writer.
    """
    grouped_labels = y.groupby(writers).agg(lambda x: Counter(x).most_common(1)[0][0])
    return grouped_labels

def compute_accuracies(y_true, y_pred, writers):
    """
    Computes accuracy for each writer.
    """
    grouped_true = group_labels(y_true, writers)
    grouped_pred = ensembled_predictions(y_pred, writers)
    
    accuracies = {}
    accuracies['ensembled'] = accuracy_score(grouped_true, grouped_pred)
    accuracies['individual'] = accuracy_score(y_true, y_pred)
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


def main(args):
    """
    Main function to run the feature extraction and classification pipeline.
    """
    print("Running feature extraction script...")
    args = load_config(args.config)
    output_dir = source_path + "\\outputs\\preprocessed_data\\"
    LOG_FILE = output_dir+"file_metadata_log.json"
    df_log = file_IO.assemble_csv_from_log(LOG_FILE)

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
    cols_to_drop = [c for c in train_FE.columns if not(c.startswith('f') and len(c) > 1 and c[1].isdigit())]


    train_FE=select_groups(train_FE,select_column='train', 
                        train_on_language=train_on_language, train_on_same=train_on_same)

    X = train_FE[train_FE['train']==1].drop(columns=cols_to_drop)
    y = train_FE[train_FE['train']==1][target_label]

    writers = train_FE[train_FE['train']==1]['writer']-1

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
    for train_idx, val_idx in gkf.split(X, y, groups=writers):
        #print(train_idx)
        X_train, X_val = X.loc[train_idx], X.loc[val_idx]
        y_train, y_val = y.loc[train_idx], y.loc[val_idx]
        X_train, y_train, writers_train = shuffle(
            X_train, y_train, writers.loc[train_idx], random_state=42
        )
        '''X_train_sub, X_val_sub, y_train_sub, y_val_sub = train_test_split(
            X_train, y_train, test_size=0.1, random_state=42
        )
        # Fit preprocessing steps only
        pipeline[:-1].fit(X_train_sub.values, y_train_sub)

        # Transform training and validation sets
        X_train_trans = pipeline[:-1].transform(X_train_sub.values)
        X_val_trans = pipeline[:-1].transform(X_val_sub.values)
        pipeline.named_steps['lgbm'].fit(
            X_train_trans, y_train_sub,
            eval_set=[(X_val_trans, y_val_sub)],
            callbacks=[early_stopping(stopping_rounds=20), log_evaluation(0)]
        )'''
        # Fit the model on training data
        pipeline.fit(X_train.values, y_train)
        y_pred = pipeline.predict(X_train.values)
        accuracies = compute_accuracies(y_train, y_pred, writers_train)
        cross_val_accuracies["IF"].append(accuracies)
        y_pred = pipeline.predict(X_val.values)
        accuracies = compute_accuracies(y_val, y_pred, writers.loc[val_idx])
        cross_val_accuracies["OOF"].append(accuracies)
    IF_values = cross_val_accuracies['IF']
    ensembled_accuracies_IF = []
    individual_accuracies_IF = []
    for value in IF_values:
        ensembled_accuracies_IF.append(value['ensembled'])
        individual_accuracies_IF.append(value['individual'])
    avg_ensembled_IF = np.mean(ensembled_accuracies_IF)
    avg_individual_IF = np.mean(individual_accuracies_IF)
    print(f"Average ensembled accuracy on training data (IF): {avg_ensembled_IF:.3f}")
    print(f"Average individual accuracy on training data (IF): {avg_individual_IF:.3f}")
    OOF_values = cross_val_accuracies['OOF']
    ensembled_accuracies_OOF = []
    individual_accuracies_OOF = []
    for value in OOF_values:
        ensembled_accuracies_OOF.append(value['ensembled'])
        individual_accuracies_OOF.append(value['individual'])
    avg_ensembled_OOF = np.mean(ensembled_accuracies_OOF)
    avg_individual_OOF = np.mean(individual_accuracies_OOF)
    print(f"Average ensembled accuracy on validation data (OOF): {avg_ensembled_OOF:.3f}")
    print(f"Average individual accuracy on validation data (OOF): {avg_individual_OOF:.3f}")
    # Measure the end time
    end_time = time.time()
    # Calculate the time taken
    time_taken_cross_val = end_time - start_time
    print(f"Time taken to cross-validate the model: {time_taken_cross_val:.2f} seconds")


    start_time = time.time()
    print(f"Retraining on all training data...")
    pipeline.fit(X, y)
    end_time = time.time()
    # Calculate the time taken
    time_taken = end_time - start_time
    print(f"Time taken to fit the model: {time_taken:.2f} seconds")
    if with_pca:
        pca = pipeline.named_steps['pca']
        print(f"Number of features used after PCA: {pca.n_components_}")

    subgroup_accuracies = {}
    # Predict on train data
    y_pred = pipeline.predict(X)
    accuracies = compute_accuracies(y, y_pred, writers)
    subgroup_accuracies['english+arabic,different+same'] = accuracies
    print(f"Training Accuracy on all training data -> Ensembled accuracy: {accuracies['ensembled']:.3f}, Individual accuracy: {accuracies['individual']:.3f}")
    print('----------------------------------------')
    groups = [('english','different'), ('english','same'), ('arabic','different'), ('arabic','same')]
    group_sizes = []
    for group in groups:
        train_FE=select_groups(train_FE,select_column='train', 
                            train_on_language=group[0], train_on_same=group[1])
        X_s = train_FE[train_FE['train']==1].drop(columns=cols_to_drop)
        y_s = train_FE[train_FE['train']==1][target_label]
        writers_s = train_FE[train_FE['train']==1]['writer']-1
        group_sizes.append(len(X_s))
        y_pred = pipeline.predict(X_s)
        accuracies = compute_accuracies(y_s, y_pred, writers_s)
        subgroup_accuracies[f'{group[0]},{group[1]}'] = accuracies
    groups_joined = [(0,1),(2,3),(0,2),(1,3)]
    for group in groups_joined:
        language_1,same_1 = groups[group[0]]
        language_2,same_2 = groups[group[1]]
        if language_1 == language_2:
            first= f'{language_1}'
        else:
            first = f'{language_1}+{language_2}'
        if same_1 == same_2:
            second = f'{same_1}'
        else:
            second = f'{same_1}+{same_2}'
        group_name= f'{first},{second}'
        sub_group_sizes = [group_sizes[group[0]], group_sizes[group[1]]]
        sub_group_accuracies = [subgroup_accuracies[f'{language_1},{same_1}'], subgroup_accuracies[f'{language_2},{same_2}']]
        accuracies['ensembled'] = np.average([g['ensembled'] for g in sub_group_accuracies], weights=sub_group_sizes)
        accuracies['individual'] = np.average([g['individual'] for g in sub_group_accuracies], weights=sub_group_sizes)
        subgroup_accuracies[f'{group_name}'] = accuracies

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
            "training time for final model": time_taken,
            "cross_val_accuracies": cross_val_accuracies,
            "subgroup_accuracies": subgroup_accuracies,
            "is_kaggle": is_kaggle,
            "test": 'this is a test column',
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