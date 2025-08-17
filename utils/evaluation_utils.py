import torch
from torch.cuda.amp import autocast
from tqdm import tqdm
import random
from torch.utils.data import DataLoader, Subset
import pandas as pd
import numpy as np
from collections import Counter
from sklearn.metrics import accuracy_score
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss

list_of_metrics = ['majority_vote', 'weighted_vote', 'most_probable']

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

def perform_validation_contrastive(model, val_dataloader, device, val_percentage, epoch, total_epochs, use_amp):
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
    if use_amp:
        with torch.no_grad(), autocast(device_type='cuda'):
            for batch in tqdm(selected_val_batches, desc=f"Epoch {epoch+1}/{total_epochs} [Val]"):
                x, labels = batch['image'].to(device), batch['label'].to(device)
                z = model(x)
                reps.append(z.cpu().numpy())
                labels_list.append(labels.cpu().numpy())
    else:
        with torch.no_grad():
            for batch in tqdm(selected_val_batches, desc=f"Epoch {epoch+1}/{total_epochs} [Val]"):
                x, labels = batch['image'].to(device), batch['label'].to(device)
                z = model(x)
                reps.append(z.cpu().numpy())
                labels_list.append(labels.cpu().numpy())
    
    # Linear evaluation
    reps = np.concatenate(reps, axis=0)
    labels_all = np.concatenate(labels_list, axis=0)

    clf = LogisticRegression(max_iter=1000, solver='lbfgs', multi_class='auto')
    clf.fit(reps, labels_all)
    y_proba = clf.predict_proba(reps)
    val_loss = log_loss(labels_all, y_proba)
    
    if y_proba.shape[1] == 2:
        y_pred = (y_proba[:, 1] >= 0.5).astype(int)
    else:
        raise ValueError("Multiclass classification is not supported in this implementation.")
    val_acc = np.mean(y_pred == labels_all)
    
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

def ensembled_predictions_w_uncertainty(base_preds,writers,mode='majority_vote',probs=None):
    pred_df = pd.DataFrame({
            'writer': writers,
            'pred': base_preds
        })
    if probs is not None:
        probs = np.abs(probs - 0.5) / 0.5
        pred_df['prob'] = probs
    if mode == 'majority_vote':
        def most_common_with_diff(x):
            counts = Counter(x)
            most_common_count = counts.most_common(1)[0][1]
            least_common_count = counts.most_common()[-1][1]
            return pd.Series({
            'writer_pred': counts.most_common(1)[0][0],
            'count_diff': most_common_count - least_common_count
            })
        def most_common_with_diff_with_fallback(group):
            preds = group['pred']
            counts = Counter(preds)

            # Check for tie
            common = counts.most_common()
            if len(common) > 1 and common[0][1] == common[1][1]:
                # Tie fallback: choose label with highest summed probability
                total_prob = group.groupby(group['pred'])['prob'].sum()
                fallback_label = total_prob.idxmax()
                most_common_label = fallback_label
                count_diff = 0
            else:
                most_common_label = common[0][0]
                count_diff = common[0][1] - common[-1][1]

            return pd.Series({
                'writer_pred': most_common_label,
                'count_diff': count_diff
            })
        writer_stats = pred_df.groupby('writer').apply(lambda df: most_common_with_diff(df['pred'])).reset_index()
        #writer_stats = pred_df.groupby('writer').apply(most_common_with_diff_with_fallback).reset_index()
        writer_preds = writer_stats['writer_pred']
        uncertainty = writer_stats['count_diff']
    elif mode == 'weighted_vote':
        if probs is None:
            raise ValueError("For 'weighted_vote', 'probs' must be provided.")
        writer_preds = pred_df.groupby('writer', group_keys=False).apply(
            lambda x: pd.Series({
                'writer_pred': round((x['pred'] * x['prob']).sum() / x['prob'].sum())
            })
        )['writer_pred'].astype(int)
        uncertainty = pred_df.groupby('writer', group_keys=False).apply(
            lambda x: pd.Series({
                'writer_pred': np.abs((x['pred'] * x['prob']).sum() / x['prob'].sum() - 0.5) / 0.5
            })
        )['writer_pred']
    elif mode == 'most_probable':
        if probs is None:
            raise ValueError("For 'most_probable', 'probs' must be provided.")
        writer_preds = pred_df.groupby('writer', group_keys=False).apply(
            lambda x: x.loc[x['prob'].idxmax(), 'pred']
        )
        uncertainty = pred_df.groupby('writer', group_keys=False).apply(
            lambda x: x.loc[x['prob'].idxmax(), 'prob']
        )
        

    # Step 5: Map writer-level prediction back to each sample
    #final_preds = writers.map(writer_preds)
    return writer_preds,uncertainty

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

def compute_predictions_and_uncertainties(model, train_df, head_type,calibrate=False,threshold=0.5):
    cols_to_drop = [c for c in train_df.columns if not(c.startswith('f') and len(c) > 1 and c[1].isdigit())]
    if head_type == 'pytorch':
        model.eval()
        with torch.no_grad():
            X_train = torch.tensor(train_df.drop(columns=cols_to_drop).values, dtype=torch.float32)
            logits = model(X_train)  # shape: [N, 2]
            probs = F.softmax(logits, dim=1)  # shape: [N, 2]
            y_prob = probs[:, 1].numpy()  # probability of class 1
    else:
        y_prob = model.predict_proba(train_df.drop(columns=cols_to_drop).values)[:,1]
    if calibrate:
        from sklearn.isotonic import IsotonicRegression
        iso = IsotonicRegression(out_of_bounds='clip')
        iso.fit(y_prob, train_df['male'])
        y_prob = iso.predict(y_prob)
    #y_pred = pipeline.predict(X_train.values)
    y_pred =(y_prob>= threshold).astype(int)
    train_df['y_prob'] = y_prob
    train_df['y_pred'] = y_pred
    print(f"Accuracy for individual patches: {accuracy_score(train_df[train_df['train']==0]['male'], train_df[train_df['train']==0]['y_pred'])}")
    train_df['page'] = train_df.groupby(['writer', 'isEng', 'same_text']).ngroup()
    train_df = train_df.sort_values('page').reset_index(drop=True)
    grouped_true = group_labels(train_df['male'], train_df['page'])
    train_df['grouped_true'] = train_df['page'].map(grouped_true)
    #When you use pandas.Series.groupby(), the result is sorted by group keys (i.e., the unique values in writers) unless you tell it not to.
    #train_df=train_df.merge(['grouped_true'] = grouped_true.values
    for metric in list_of_metrics:
        grouped_predictions,uncertainty = ensembled_predictions_w_uncertainty(train_df['y_pred'], train_df['page'], mode=metric,probs=train_df['y_prob'])
        train_df[metric]= train_df['page'].map(grouped_predictions)
        train_df[metric+'_uncertainty'] = train_df['page'].map(uncertainty)
        print(f"Accuracy for {metric}: {accuracy_score(train_df[train_df['train']==0]['grouped_true'], train_df[train_df['train']==0][metric])}")
    # Save the predictions to a CSV file
    #cross_val_subgroup_accuracies.append(compute_subgroup_accuracies(pipeline, train_FE_temp, cols_to_drop, target_label))
    n_patches = int(len(train_df)/len(train_df['page'].unique()))
    train_df['majority_vote_uncertainty'] /= n_patches #-> 1 if all patches classified the same, 0 if all patches classified differently
    return train_df

def select_instances(train_df, selected_metric='majority_vote'):
    grouped = train_df[train_df['train'] == 0].groupby('page').first()[['grouped_true'] + list_of_metrics +[metric + '_uncertainty' for metric in list_of_metrics]].reset_index()
    for metric in list_of_metrics:

        percentile_25 = grouped[metric + '_uncertainty'].quantile(0.25) #threshold for unsure
        percentile_75 = grouped[metric + '_uncertainty'].quantile(0.75) #threshold for sure
        #print(f"Percentiles for {metric}: 25th = {percentile_25}, 75th = {percentile_75}")
        grouped[metric+'_selected'] = 0
        grouped[metric+'_sure'] = (grouped[metric+'_uncertainty'] >= percentile_75).astype(int)
        grouped[metric+'_unsure'] = (grouped[metric+'_uncertainty'] <= percentile_25).astype(int)
        grouped[metric+'_ok'] = (grouped[metric] == grouped['grouped_true']).astype(int)
        # Group by sure, unsure, ok, and grouped_true, assign a unique number to each group
        class_grouped=grouped.groupby([metric + '_sure', metric + '_unsure', metric + '_ok', 'grouped_true'])
        for group_id, group_df in class_grouped:
            idx = group_df.sample(1, random_state=42).index if not group_df.empty else group_df.sample(1, random_state=42).index
            grouped.loc[idx, metric + '_selected'] = 1
        print(f"Number of unique groups for {metric}: {(class_grouped.ngroup()).nunique()}")

        '''#select one sure and ok and 0
        conditions = (grouped[metric+'_uncertainty'] >= percentile_75) & (train_df[metric] == train_df['grouped_true']) & (1 == train_df['grouped_true'])
        selected = grouped[conditions].sample(1)
        grouped.loc[selected.index, metric+'_selected'] = 1'''
    cols_to_drop = ['grouped_true'] + [metric + '_uncertainty' for metric in list_of_metrics]+list_of_metrics
    grouped = grouped.drop(columns=cols_to_drop)
    train_df = train_df.merge(grouped, on='page', how='left')

    selected = train_df[train_df[selected_metric + '_selected'] == 1]
    grouping = ['sure', 'unsure', 'ok']
    grouping = [selected_metric + '_' + group for group in grouping] + ['grouped_true']
    print(grouping)
    display(selected[grouping+['page']].groupby('page').first().sort_values(grouping))
    cols_to_drop = [c for c in selected.columns if c.startswith('f') and len(c) > 1 and c[1].isdigit()]
    selected = selected.drop(columns=cols_to_drop)
    return selected.sort_values(grouping)