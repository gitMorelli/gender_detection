import torch
from torch.cuda.amp import autocast
from tqdm import tqdm
import random
from torch.utils.data import DataLoader, Subset
import pandas as pd
import numpy as np
from collections import Counter
from sklearn.metrics import accuracy_score

def test_model(model, test_loader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for batch in test_loader:
            inputs, labels = batch['image'], batch['label']
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = outputs.max(1)
            correct += preds.eq(labels).sum().item()
            total += labels.size(0)

    test_acc = correct / total
    print(f"Test Accuracy: {test_acc:.4f}")


def perform_validation(model, val_dataloader, device, val_percentage, epoch, total_epochs, use_amp, loss_fn):
    """Perform validation using linear evaluation"""
    model.eval()
    reps, labels_list = [], []
    
    val_dataset = val_dataloader.dataset
    n_val = max(1, int(len(val_dataset) * val_percentage))
    subset_indices = random.sample(range(len(val_dataset)), n_val)

    subset = Subset(val_dataset, subset_indices)
    selected_val_batches = DataLoader(
        subset,
        batch_size=val_dataloader.batch_size,
        shuffle=False,
        num_workers=val_dataloader.num_workers,
        pin_memory=val_dataloader.pin_memory,
    )
    
    # Extract representations
    val_loss = 0.0
    correct = 0
    total = 0

    if use_amp:
        with torch.no_grad(), autocast(device_type='cuda'):
            for batch in tqdm(selected_val_batches, desc=f"Epoch {epoch+1}/{total_epochs} [Val]"):
                inputs, labels = batch['image'], batch['label']
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = loss_fn(outputs, labels)
                val_loss += loss.item() * labels.size(0)
                _, preds = outputs.max(1)
                correct += preds.eq(labels).sum().item()
                total += labels.size(0)
    else:
        with torch.no_grad():
            for batch in tqdm(selected_val_batches, desc=f"Epoch {epoch+1}/{total_epochs} [Val]"):
                inputs, labels = batch['image'], batch['label']
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = loss_fn(outputs, labels)
                val_loss += loss.item() * labels.size(0)
                _, preds = outputs.max(1)
                correct += preds.eq(labels).sum().item()
                total += labels.size(0)

    val_loss = val_loss / total if total > 0 else 0
    val_acc = correct / total if total > 0 else 0
    

    return val_loss, val_acc

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
    return writer_preds#final_preds.value

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
        #print(group)
        train_df=select_groups(train_df,select_column='train', 
                            train_on_language=group[0], train_on_same=group[1])
        X_s = train_df[train_df['train']==1].drop(columns=cols_to_drop)
        y_s = train_df[train_df['train']==1][target_label]
        writers_s = train_df[train_df['train']==1]['writer']
        pages_s = train_df[train_df['train']==1]['page']
        group_sizes.append(len(X_s))

        y_prob= pipeline.predict_proba(X_s.values)[:,1]
        #y_pred = pipeline.predict(X_s.values)
        y_pred=(y_prob >= 0.5).astype(int)
        accuracies = compute_accuracies(y_s, y_pred, y_prob, pages_s,writers_s)
        subgroup_accuracies[f'{group[0]},{group[1]}'] = accuracies

    return subgroup_accuracies