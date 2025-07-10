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

class ZarrImageCropDataset_resize_workers(Dataset):
    def __init__(self, df, zarr_path, transform=None, huggingface=False):
        """
        df: DataFrame with columns ['file_name', 'x', 'y', 'x2', 'y2']
        zarr_path: path to directory-based Zarr store
        transform: Optional transform applied to the cropped patch
        """
        self.df = df.reset_index(drop=True)
        self.zarr_path = zarr_path
        self.transform = transform
        self.huggingface = huggingface

        # Do NOT open Zarr store or load filenames here!
        self.zarr_store = None
        self.file_to_idx = None

    def _init_zarr(self):
        """Helper to lazily open the zarr store and load filename mapping."""
        if self.zarr_store is None:
            self.zarr_store = zarr.open(self.zarr_path, mode='r')
            filenames = list(self.zarr_store['filenames'][:])
            # decode bytes if needed (sometimes filenames are bytes)
            self.file_to_idx = {
                fn.decode('utf-8') if isinstance(fn, bytes) else fn: i
                for i, fn in enumerate(filenames)
            }

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        self._init_zarr()

        row = self.df.iloc[idx]
        file_name = row['file_name']
        x1, y1, x2, y2 = row['x'], row['y'], row['x2'], row['y2']

        img_idx = self.file_to_idx[file_name]
        full_img = self.zarr_store['images'][img_idx]  # numpy array HWC

        patch = full_img[y1:y2, x1:x2, :]
        patch = Image.fromarray(patch)

        if self.huggingface:
            inputs = self.transform(images=patch, return_tensors="pt")
            patch_tensor = inputs['pixel_values'][0]  # shape: (C, H, W)
        elif self.transform:
            patch_tensor = self.transform(patch)
        else:
            patch_tensor = patch

        return {'image': patch_tensor}

    def __del__(self):
        self.zarr_store = None  # Allow GC to clean up
class ZarrImageCropDataset_resize(Dataset):
    def __init__(self, df, zarr_path, transform=None, huggingface=False):
        """
        df: DataFrame with columns ['file_name', 'x', 'y', 'x2', 'y2']
        zarr_path: path to directory-based Zarr store
        transform: Optional transform applied to the cropped patch
        """
        self.df = df.reset_index(drop=True)
        self.zarr_path = zarr_path
        self.transform = transform
        self.huggingface = huggingface
        self.zarr_store = None  # will be lazily opened

        # Load filenames and create mapping: file_name -> index
        z = zarr.open(self.zarr_path, mode='r')
        filenames = list(z['filenames'][:])
        self.file_to_idx = {fn: i for i, fn in enumerate(filenames)}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        '''times = []
        time_names = ['opening zarr', 'reading image', 'cropping patch', 'converting to PIL', 'transforming patch']
        times.append(datetime.now())'''

        if self.zarr_store is None:
            self.zarr_store = zarr.open(self.zarr_path, mode='r')
        #times.append(datetime.now())

        row = self.df.iloc[idx]
        file_name = row['file_name']
        x1, y1, x2, y2 = row['x'], row['y'], row['x2'], row['y2']

        img_idx = self.file_to_idx[file_name]
        full_img = self.zarr_store['images'][img_idx]  # numpy array HWC
        #times.append(datetime.now())

        patch = full_img[y1:y2, x1:x2, :]
        #times.append(datetime.now())

        patch = Image.fromarray(patch)
        if hasattr(patch, "size"):
            width, height = patch.size
            if width == 0 or height == 0:
                raise ValueError(f"Invalid patch size: {patch.size} at index {idx}; x1={x1}, y1={y1}, x2={x2}, y2={y2} in file {file_name}")
        #times.append(datetime.now())

        if self.huggingface:
            inputs = self.transform(images=patch, return_tensors="pt")
            patch_tensor = inputs['pixel_values'][0]  # shape: (C, H, W)
        elif self.transform:
            patch_tensor = self.transform(patch)
        else:
            patch_tensor = patch
        #times.append(datetime.now())

        '''for i, name in enumerate(time_names):
            print(f"Time for {name}: {(times[i+1] - times[i]).total_seconds() * 1000:.2f} ms")
        raise RuntimeError("Debug: error raised after timing print statements")'''

        return {'image': patch_tensor}

    def __del__(self):
        self.zarr_store = None  # optional: let GC handle closure

class FeatureExtractorWithLogReg(nn.Module):
    def __init__(self, backbone, in_features, num_classes):
        super().__init__()
        self.backbone = backbone
        self.classifier = nn.Linear(in_features, num_classes)

    def forward(self, x):
        features = self.backbone(x)
        return self.classifier(features)


def main(args):
    #parameters
    args = load_config(args.config)
    N_max = args.N_max
    patches = args.patches
    input_filename = args.input_filename
    huggingface = args.huggingface
    pooling = args.pooling  # if true in transformer models use pooling, if false only the cls token
    custom_transform = args.custom_transform
    transform_mode = args.transform_mode
    save_h5 = args.save_h5
    selected_model = args.selected_model  # googlenet, alexnet
    truncation = args.truncation
    running = args.running
    saved = args.saved
    model_mode = args.model_mode  # 'truncated' or 'truncation'
    batch_size = args.batch_size
    select_cls = args.select_cls
    num_workers = args.num_workers
    pin_memory = args.pin_memory
    show_image = args.show_image
    is_progressive = args.is_progressive
    checkpoint_path = args.checkpoint_path
    save_path = args.save_path
    #hyperpar suggests
    base_lr = args.base_lr
    total_epochs = args.total_epochs    
    temperature = args.temperature
    log_grad_norm = args.log_grad_norm
    use_profiler = args.use_profiler
    run_epochs = args.run_epochs
    plot_every = args.plot_every
    patience = args.patience
    warmup_epochs = args.warmup_epochs
    use_amp = args.use_amp  # mixed precision training
    val_percentage = args.val_percentage  # percentage of validation data used for linear evaluation
    n_splits = args.n_splits
    loss_criterion = args.loss_criterion  # loss function to use, e.g., 'cross_entropy', 'nt_xent'
    scheduler_name = args.scheduler_name  # e.g., 'CosineAnnealingLR', 'StepLR', etc.
    selected_classifier = args.selected_classifier  # 'logreg', 'svm', 'rf', 'gbc', 'mlp', 'dt'
    optim_name = args.optim_name  # e.g., 'Adam', 'SGD', 'AdamW'

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
    model = model_utils.get_model(name=selected_model, mode=model_mode, pretrained=True, truncation=truncation)

    # Define model
    train_df = pd.read_csv(f"{source_path}\\outputs\\preprocessed_data\\{input_filename}")
    train_df=file_IO.change_filename_from_to(train_df, fr=saved, to=running)
    i=0
    output=compute_output(model, 'cpu', transform, train_df.iloc[i], huggingface, patches)
    print("Output shape: ", output.shape)
    in_features = output.shape[1]  # Number of features from the model output
    if selected_classifier == 'logreg':
        model = FeatureExtractorWithLogReg(
            backbone=model,
            in_features=in_features,
            num_classes=2) # Assuming binary classification
    else:
        raise ValueError(f"Unsupported classifier: {selected_classifier}. Supported classifiers: 'logreg'.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device is: ",device)

    #split in train and validation
    train_df['train']=1
    writers = train_df['writer']
    val_writers = set(random.sample(list(writers.unique()), max(1, len(writers.unique()) // n_splits)))
    train_df.loc[train_df['writer'].isin(val_writers), 'train'] = 0
    print("val writers are: ", val_writers)
    writer_train_df = pd.DataFrame({
        'writer': writers.unique()
    })
    writer_train_df['train'] = writer_train_df['writer'].apply(lambda w: 0 if w in val_writers else 1)
    writer_train_df.to_csv(save_path+selected_model+'_'+input_filename+'_writers.csv', index=False)

    zarr_path = "C:\\Users\\andre\PhD\Datasets\ICDAR 2013 - Gender Identification Competition Dataset\\train_writers.zarr"
    train_dataset = ZarrImageCropDataset_resize(train_df[train_df['train']==1], zarr_path, transform=transform, huggingface=huggingface)
    #dataset = ZarrImageCropDataset_resize_workers(train_df[:1000], zarr_path, transform=transform, huggingface=huggingface)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers,pin_memory=pin_memory)
    val_dataset = ZarrImageCropDataset_resize(train_df[train_df['train']==0], zarr_path, transform=transform, huggingface=huggingface)
    #dataset = ZarrImageCropDataset_resize_workers(train_df[:1000], zarr_path, transform=transform, huggingface=huggingface)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers,pin_memory=pin_memory)
    
    if show_image:
        # Display one image from the dataloader
        import matplotlib.pyplot as plt
        batch = next(iter(dataloader))
        img = batch['image'][0]
        # Define unnormalize transform (example for ImageNet)
        mean = torch.tensor([0.485, 0.456, 0.406])
        std = torch.tensor([0.229, 0.224, 0.225])

        if isinstance(img, torch.Tensor):
            if img.dim() == 4:
                img = img[0]
            img = img.detach().cpu()

            # Unnormalize
            if img.shape[0] == 3:
                img = img * std[:, None, None] + mean[:, None, None]  # Unnormalize only RGB images

            if img.shape[0] == 1:
                img = img.squeeze(0)
            elif img.shape[0] == 3:
                img = img.permute(1, 2, 0)

            img = img.clamp(0, 1).numpy()  # Clamp values to valid range
        plt.imshow(img)
        plt.axis('off')
        plt.show()
        # Save the image to file
        plt.imsave(f"{source_path}/outputs/online_deep_feature_extraction/sample_image.png", img)
        return 0
    
    print(f"[GPU Memory] Allocated: {torch.cuda.memory_allocated() / 1e6:.2f} MB | Reserved: {torch.cuda.memory_reserved() / 1e6:.2f} MB")

    loss_fn = get_criterion(name=loss_criterion)

    if is_progressive:
        raise NotImplementedError("Progressive training is not implemented yet.")
    else:
        # Call training function
        train_fine(
            model=model,
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            device=device,
            base_lr=base_lr,
            total_epochs=total_epochs,
            loss_fn=loss_fn,
            use_profiler=use_profiler,
            profiler_config=profiler_config,
            save_path=save_path,
            plot_every=plot_every,
            early_stopping_patience=patience,
            warmup_epochs=warmup_epochs,
            checkpoint_path=checkpoint_path,
            log_grad_norm=log_grad_norm,
            run_epochs=run_epochs,
            use_amp=use_amp,
            val_percentage=val_percentage,  # Use 10% of validation data for linear evaluation
            scheduler=scheduler_name,
            optim_name=optim_name,  # e.g., 'Adam', 'SGD', 'AdamW'
            # ... other parameters
        )
        

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
    args = parse_args()
    main(args)