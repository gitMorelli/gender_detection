import sys
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from datetime import datetime
from PIL import Image
import pandas as pd
#import matplotlib.pyplot as plt
import io
import h5py
from collections import defaultdict, OrderedDict
import threading
from functools import lru_cache
import cv2
import zarr
import time
import yaml
import argparse
from torch.amp import GradScaler, autocast
import random
import numpy as np
import torch.nn as nn
import json


def compute_output_gpu(model, device, batch):
    model.eval()
    with torch.no_grad():
        images = batch['image'].to(device)
        outputs = model(images)
    return outputs

def compute_output(model, device, transform, t, huggingface, patches):
    image_file = t['file_name']
    image = Image.open(image_file).convert("RGB")
    if patches:
        x1 = t['x']
        y1 = t['y']
        x2 = t['x2']
        y2 = t['y2']
        patch = image.crop((x1, y1, x2, y2))
    else:
        patch = image.copy()
    if huggingface:
        # the transform is actually an huggingface processor in this case
        inputs = transform(images=patch, return_tensors="pt")
        # Remove batch dimension from inputs
        patch = inputs['pixel_values'].squeeze()
    else:
        patch = transform(patch)
    patch = patch.to(device)
    '''with torch.no_grad(), autocast(device_type='cuda'):
        output = model(patch.unsqueeze(0))'''
    with torch.no_grad():
        output = model(patch.unsqueeze(0))
    return output


def main(args):
    #parameters
    args = load_config(args.config)
    N_max = args.N_max
    patches = args.patches
    input_filename = args.input_filename
    val_filename = args.val_filename
    huggingface = args.huggingface
    pooling = args.pooling  # if true in transformer models use pooling, if false only the cls token
    custom_transform = args.custom_transform
    transform_mode = args.transform_mode
    save_h5 = args.save_h5
    selected_model = args.selected_model  # googlenet, alexnet
    selected_classifier = args.selected_classifier  # 'logreg', 'svm', 'rf', 'gbc', 'mlp', 'dt'
    truncation = args.truncation
    running = args.running
    saved = args.saved
    model_mode = args.model_mode  # 'truncated' or 'truncation'
    batch_size = args.batch_size
    select_cls = args.select_cls
    num_workers = args.num_workers
    pin_memory = args.pin_memory
    show_image = args.show_image
    checkpoint_path = args.checkpoint_path
    save_path = args.save_path
    #hyperpar suggests
    total_epochs = args.total_epochs    
    log_grad_norm = args.log_grad_norm
    use_profiler = args.use_profiler
    run_epochs = args.run_epochs
    plot_every = args.plot_every
    patience = args.patience
    use_amp = args.use_amp  # mixed precision training
    val_percentage = args.val_percentage  # percentage of validation data used for linear evaluation
    n_splits = args.n_splits
    loss_criterion = args.loss_criterion  # loss function to use, e.g., 'cross_entropy', 'nt_xent'
    selected_classifier = args.selected_classifier  # 'logreg', 'svm', 'rf', 'gbc', 'mlp', 'dt'
    optim_config = args.optim_config  # e.g., 'Adam', 'SGD', 'AdamW'
    use_augmentation = args.use_augmentation  # Set to True for data augmentation
    n_patches = args.n_patches
    contrastive_mode = args.contrastive_mode  # Set to True for contrastive learning
    load_contrastive = args.load_contrastive  # Set to True if loading a trained contrastive model for fine tuning
    nn_parameters = args.nn_parameters
    load_data_from = args.load_data_from  # 'zarr' or 'folder'

    profiler_config = {
        'profile_epochs': list(range(0, total_epochs, 10)),  # Profile every 10 epochs
        'profile_batches': 30,
        'output_dir': save_path+'detailed_profiler_logs',
        'profile_memory': True,
        'profile_shapes': True,
        'with_stack': True,
        'with_flops': True,
        'export_chrome_trace': True,
        'export_stacks': True,
    }

    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    #Initialization
    transform = u_transforms.get_transform(selected_model, use_patches=patches, custom=custom_transform, mode=transform_mode)
    model = model_utils.get_model(name=selected_model, mode=model_mode, 
                                  pretrained=True, truncation=truncation, 
                                  contrastive=contrastive_mode, load_contrastive=load_contrastive)

    # Define model
    train_df = pd.read_csv(f"{source_path}\\outputs\\preprocessed_data\\{input_filename}")
    train_df=file_IO.change_filename_from_to(train_df, fr=saved, to=running)
    if contrastive_mode:
        train_df['page'] = train_df.groupby(['file_name']).ngroup()
    else:
        train_df['page'] = train_df.groupby(['writer', 'isEng', 'same_text']).ngroup()
    if n_patches > 0:
        #print(f"Selecting {n_patches} patches per page...")
        train_df = select_n_patches(train_df, n_patches=n_patches).reset_index(drop=True)

    i=0
    output=compute_output(model, 'cpu', transform, train_df.iloc[i], huggingface, patches)
    print("Output shape: ", output.shape)
    in_features = output.shape[1]  # Number of features from the model output
    if contrastive_mode == False:
        classificaton_head = model_utils.get_classification_head(name=selected_classifier, in_features=in_features, num_classes=2,
                                                dropout=nn_parameters['dropout'], n_neurons=nn_parameters['n_neurons'],
                                                activation=nn_parameters['activation'])
        model = model_utils.JoinedModels(model, classificaton_head)
        output=compute_output(model, 'cpu', transform, train_df.iloc[i], huggingface, patches)
    print(output)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device is: ",device)

    #split in train and validation
    train_df['train']=1
    if contrastive_mode:
        if N_max == -1:
            N_max = len(train_df)
        train_df=train_df.iloc[:N_max]
        zarr_path_train = "C:\\Users\\andre\PhD\Datasets\ICDAR 2013 - Gender Identification Competition Dataset\\contrastive.zarr"
    else:
        if N_max == -1:
            N_max = train_df['writer'].max()
        train_df=train_df[train_df['writer']<=N_max]
        writers = train_df['writer']
        zarr_path_train = "C:\\Users\\andre\PhD\Datasets\ICDAR 2013 - Gender Identification Competition Dataset\\train_writers.zarr"
    if val_filename is not None:
        val_df = pd.read_csv(f"{source_path}\\outputs\\preprocessed_data\\{val_filename}")
        val_df=file_IO.change_filename_from_to(val_df, fr=saved, to=running)
        val_df['page'] = val_df.groupby(['writer', 'isEng', 'same_text']).ngroup()
        if n_patches > 0:
            #print(f"Selecting {n_patches} patches per page...")
            val_df = select_n_patches(val_df, n_patches=n_patches).reset_index(drop=True)
        val_df['train'] = 0
        if contrastive_mode:
            zarr_path_val = "C:\\Users\\andre\PhD\Datasets\ICDAR 2013 - Gender Identification Competition Dataset\\train_writers.zarr"
        else:
            zarr_path_val = "C:\\Users\\andre\PhD\Datasets\ICDAR 2013 - Gender Identification Competition Dataset\\test_public_writers.zarr"
    else:
        val_writers = set(random.sample(list(writers.unique()), max(1, len(writers.unique()) // n_splits)))
        train_df.loc[train_df['writer'].isin(val_writers), 'train'] = 0
        print("val writers are: ", val_writers)
        writer_train_df = pd.DataFrame({
            'writer': writers.unique()
        })
        writer_train_df['train'] = writer_train_df['writer'].apply(lambda w: 0 if w in val_writers else 1)
        writer_train_df.to_csv(save_path+selected_model+'_'+input_filename+'_writers.csv', index=False)
        val_df = train_df[train_df['train'] == 0]
        zarr_path_val = zarr_path_train  # Use the same zarr path for validation if not provided


    '''print(len(train_df[train_df['train']==1]), len(train_df[train_df['train']==0]))
    assert set(train_df[train_df['train'] == 1]['writer']).isdisjoint(set(train_df[train_df['train'] == 0]['writer'])), "Train and validation writers overlap!"
    return 0'''
    if contrastive_mode:
        contrastive_transform = u_transforms.get_contrastive_transform('simclr')
        if load_data_from == 'zarr':
            train_dataset = ZarrContrastive(train_df[train_df['train']==1], zarr_path_train, transform=transform, 
                                                huggingface=huggingface, contrastive_transform=contrastive_transform)
        elif load_data_from == 'pre-processed':
            train_dataset = PreProcessedDataset_contrastive(train_df[train_df['train']==1], transform=transform, 
                                                 huggingface=huggingface, contrastive_transform=contrastive_transform)
    else:
        if load_data_from == 'zarr':
            train_dataset = ZarrImageCropDataset_resize(train_df[train_df['train']==1], zarr_path_train, transform=transform, 
                                                        huggingface=huggingface, use_augmentation=use_augmentation)
        elif load_data_from == 'pre-processed':
            train_dataset = PreProcessedDataset(train_df[train_df['train']==1], 'male' ,transform=transform, 
                                                huggingface=huggingface, use_augmentation=use_augmentation)
    
    #dataset = ZarrImageCropDataset_resize_workers(train_df[:1000], zarr_path, transform=transform, huggingface=huggingface)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers,pin_memory=pin_memory)
    if load_data_from == 'zarr':
        val_dataset = ZarrImageCropDataset_resize(val_df, zarr_path_val, transform=transform, 
                                                huggingface=huggingface, use_augmentation=False)
    elif load_data_from == 'pre-processed':
        val_dataset = PreProcessedDataset(val_df, 'male' ,transform=transform, 
                                                huggingface=huggingface, use_augmentation=False)
    #dataset = ZarrImageCropDataset_resize_workers(train_df[:1000], zarr_path, transform=transform, huggingface=huggingface)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers,pin_memory=pin_memory)
    
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")
    if show_image:
        print("Saving debug images...")
        display_debug_images(train_dataloader,save_path, save_name='train', contrastive_mode=contrastive_mode)
        display_debug_images(val_dataloader,save_path, save_name='val')
        print("Debug images saved.")
    
    print(f"[GPU Memory] Allocated: {torch.cuda.memory_allocated() / 1e6:.2f} MB | Reserved: {torch.cuda.memory_reserved() / 1e6:.2f} MB")

    loss_fn = get_criterion(name=loss_criterion)

    best_model_performance=train_fine(
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        device=device,
        total_epochs=total_epochs,
        loss_fn=loss_fn,
        use_profiler=use_profiler,
        profiler_config=profiler_config,
        save_path=save_path,
        plot_every=plot_every,
        early_stopping_patience=patience,
        checkpoint_path=checkpoint_path,
        log_grad_norm=log_grad_norm,
        run_epochs=run_epochs,
        use_amp=use_amp,
        val_percentage=val_percentage,  # Use 10% of validation data for linear evaluation
        optim_config=optim_config,  # e.g., 'Adam', 'SGD', 'AdamW'
        contrastive_mode=contrastive_mode,  # Set to True for contrastive learning
        # ... other parameters
    )
    
    # Save best model performance to JSON as a dict
    performance_json_path = os.path.join(save_path, "best_model_performance_temp.json")
    with open(performance_json_path, "w") as f:
        json.dump(best_model_performance, f, indent=4)
    print(f"Best model performance saved to {performance_json_path}")

    return best_model_performance
    # Get the current timestamp
    #timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Save the DataFrame to a CSV file
    #output_filename = f"{source_path}\\outputs\\online_deep_feature_extraction\\{selected_model}_features_{timestamp}.csv"
    '''output_filename = f"{source_path}\\outputs\\online_deep_feature_extraction\\{selected_model}_features_{input_filename.split('.')[0]}.csv"
    train_df.to_csv(output_filename, index=False)'''

def parse_args():
    parser = argparse.ArgumentParser(description="ML experiments!")
    parser.add_argument("--config", type=str, required=True, help="The cofig file to pass in input to the script")
    return parser.parse_args()

if __name__ == "__main__":
    # Add the root of the project to the path
    source_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    sys.path.append(source_path)
    import utils.file_IO as file_IO
    import utils.utils_transforms as u_transforms
    import utils.model_utils as model_utils
    from utils.training_utils import train_fine, get_criterion
    from utils.visualization import display_debug_images
    from utils.script_launching import load_config
    from utils.script_launching import DotDict
    from utils.train_on_rep_utils import select_n_patches
    from utils.dataframes import ZarrImageCropDataset_resize,ZarrContrastive, PreProcessedDataset_contrastive, PreProcessedDataset
    args = parse_args()
    main(args)