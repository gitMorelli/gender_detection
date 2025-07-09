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
    output = model(patch.unsqueeze(0))
    return output

class CustomPatchDataset(Dataset):
    def __init__(self, df,transform=None,huggingface=False):
        """
        Args:
            image_dirs (list of str): List of directories to load images from.
            labels_df (DataFrame): DataFrame containing labeled images.
            transform (callable, optional): Optional transform to be applied on an image.
        """
        self.image_files = df['file_name'].tolist()
        self.img_writers = df['writer'].tolist()
        self.x1 = df['x'].tolist()
        self.y1 = df['y'].tolist()
        self.x2 = df['x2'].tolist()
        self.y2 = df['y2'].tolist()
        self.transform = transform
        self.huggingface = huggingface

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        times=[]
        time_names=['opening image', 'cropping patch', 'transforming patch']
        times.append(datetime.now())
        img_path = self.image_files[idx]
        x1=self.x1[idx]
        y1=self.y1[idx]
        x2=self.x2[idx]
        y2=self.y2[idx]
        image = Image.open(img_path).convert("RGB")
        times.append(datetime.now())
        patch = image.crop((x1, y1, x2, y2))
        times.append(datetime.now())

        if self.huggingface:
            # the transform is actually an huggingface processor in this case
            inputs = self.transform(images=patch, return_tensors="pt")
            # Remove batch dimension from inputs
            patch = inputs['pixel_values'].squeeze()
        else:
            if self.transform:
                patch = self.transform(patch)
        times.append(datetime.now())

        for i, name in enumerate(time_names):
            print(f"Time for {name}: {(times[i+1]-times[i]).total_seconds()*1000:.2f} ms")
        raise RuntimeError("Debug: error raised after timing print statements")
        
        return {
            'image': patch
        }
class CachedPatchDataset(Dataset):
    def __init__(self, df, transform=None, huggingface=False):
        self.transform = transform
        self.huggingface = huggingface
        self.data = []  # Will store all patches in memory

        for idx, row in df.iterrows():
            img_path = row['file_name']
            x1, y1, x2, y2 = row['x'], row['y'], row['x2'], row['y2']

            image = Image.open(img_path).convert("RGB")
            patch = image.crop((x1, y1, x2, y2))

            if huggingface:
                inputs = transform(images=patch, return_tensors="pt")
                patch_tensor = inputs['pixel_values'][0]  # shape: (C, H, W)
            elif transform:
                patch_tensor = transform(patch)
            else:
                patch_tensor = patch  # raw PIL.Image

            self.data.append(patch_tensor)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return {'image': self.data[idx]}
class LazyPatchDataset(Dataset):
    def __init__(self, df, transform=None, huggingface=False):
        self.transform = transform
        self.images = []
        self.huggingface = huggingface

        for idx, row in df.iterrows():
            img_path = row['file_name']
            x1, y1, x2, y2 = row['x'], row['y'], row['x2'], row['y2']
            image = Image.open(img_path).convert("RGB")
            patch = image.crop((x1, y1, x2, y2))
            self.images.append(patch)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]
        if self.huggingface:
            inputs = self.transform(images=img, return_tensors="pt")
            patch_tensor = inputs['pixel_values'][0]  # shape: (C, H, W)
        elif transform:
            patch_tensor = self.transform(img)
        else:
            patch_tensor = img  # raw PIL.Image
        return {'image': patch_tensor}
class HDF5ImageCropDataset_resize(Dataset):
    def __init__(self, df, hdf5_path, transform=None,huggingface=False):
        """
        df: DataFrame with columns ['file_name', 'x', 'y', 'x2', 'y2']
        hdf5_path: HDF5 file path with 'images' dataset
        transform: Optional transform applied to the cropped patch
        """
        self.df = df.reset_index(drop=True)
        self.hdf5_path = hdf5_path
        self.transform = transform
        self.hdf5_file = None
        self.huggingface = huggingface
        
        # Create a mapping from file_name to index in HDF5 dataset
        with h5py.File(hdf5_path, 'r') as f:
            filenames = list(f['filenames'])
        self.file_to_idx = {fn.decode('utf-8') if isinstance(fn, bytes) else fn: i for i, fn in enumerate(filenames)}
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        #times=[]
        #time_names=['opening file','opening image', 'cropping patch', 'converting to pil','transforming patch']
        #times.append(datetime.now())
        if self.hdf5_file is None:
            self.hdf5_file = h5py.File(self.hdf5_path, 'r')
        #times.append(datetime.now())
        
        row = self.df.iloc[idx]
        file_name = row['file_name']
        x1, y1, x2, y2 = row['x'], row['y'], row['x2'], row['y2']
        
        img_idx = self.file_to_idx[file_name]
        full_img = self.hdf5_file['images'][img_idx]  # numpy array HWC
        #times.append(datetime.now())
        
        # Crop patch (remember numpy uses [y1:y2, x1:x2])
        patch = full_img[y1:y2, x1:x2, :]
        #times.append(datetime.now())
        patch = Image.fromarray(patch)
        #times.append(datetime.now())

        if self.huggingface:
            inputs = self.transform(images=patch, return_tensors="pt")
            patch_tensor = inputs['pixel_values'][0]  # shape: (C, H, W)
        elif self.transform:
            patch_tensor = self.transform(patch)
        else:
            patch_tensor = patch  # raw PIL.Image
        #times.append(datetime.now())

        '''for i, name in enumerate(time_names):
            print(f"Time for {name}: {(times[i+1]-times[i]).total_seconds()*1000:.2f} ms")
        raise RuntimeError("Debug: error raised after timing print statements")'''
        
        return {'image': patch_tensor}
    
    def __del__(self):
        if self.hdf5_file is not None:
            self.hdf5_file.close()
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

class HDF5ImageCropDataset(Dataset):
    def __init__(self, df, hdf5_path, transform=None, huggingface=False):
        """
        df: DataFrame with columns ['file_name', 'x', 'y', 'x2', 'y2']
        hdf5_path: HDF5 file path with 'images' dataset
        transform: Optional transform applied to the cropped patch
        """
        self.df = df.reset_index(drop=True)
        self.hdf5_path = hdf5_path
        self.transform = transform
        self.hdf5_file = None
        self.huggingface = huggingface
        
        # Create a mapping from file_name to index in HDF5 dataset
        with h5py.File(hdf5_path, 'r') as f:
            filenames = list(f['filenames'])
        #self.file_to_idx = {fn: i for i, fn in enumerate(filenames)}
        self.file_to_idx = {fn.decode('utf-8') if isinstance(fn, bytes) else fn: i for i, fn in enumerate(filenames)}
        #print(filenames[:10])
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        times=[]
        time_names=['opening image', 'conversion', 'cropping','transofrmation']
        times.append(datetime.now())
        if self.hdf5_file is None:
            self.hdf5_file = h5py.File(self.hdf5_path, 'r')
        
        row = self.df.iloc[idx]
        file_name = row['file_name']
        x1, y1, x2, y2 = row['x'], row['y'], row['x2'], row['y2']
        
        img_idx = self.file_to_idx[file_name]
        img_bytes = self.hdf5_file['images'][img_idx]  # numpy array HWC
        times.append(datetime.now())
        #img_bytes = full_img.tobytes()  # Convert back to bytes
        img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
        times.append(datetime.now())
        
        # Crop patch (remember numpy uses [y1:y2, x1:x2])
        #patch = img_bytes[y1:y2, x1:x2, :]
        patch=img.crop((x1, y1, x2, y2))
        times.append(datetime.now())

        if self.huggingface:
            inputs = self.transform(images=patch, return_tensors="pt")
            patch_tensor = inputs['pixel_values'][0]  # shape: (C, H, W)
        elif transform:
            patch_tensor = self.transform(patch)
        else:
            patch_tensor = patch  # raw PIL.Image
        
        times.append(datetime.now())
        print(f'''Times for {time_names[0]}: {(times[1]-times[0]).total_seconds()*1000:.2f} ms; 
              {time_names[1]}: {(times[2]-times[1]).total_seconds()*1000:.2f} ms; 
              {time_names[2]}: {(times[3]-times[2]).total_seconds()*1000:.2f} ms;
              {time_names[3]}: {(times[4]-times[3]).total_seconds()*1000:.2f} ms''')
        
        return {'image': patch_tensor}
    
    def __del__(self):
        if self.hdf5_file is not None:
            self.hdf5_file.close()
class FastOnTheFlyDataset(Dataset):
    def __init__(self, df, transform=None, huggingface=True,image_cache_size=100):
        self.df = df
        self.transform = transform 
        self.huggingface = huggingface
        # Thread-safe LRU cache for loaded images
        self.image_cache = OrderedDict()
        self.cache_size = image_cache_size
        self.cache_lock = threading.Lock()
        
        # Group crops by image for batch processing opportunities
        self.image_groups = defaultdict(list)
        for idx, row in self.df.iterrows():
            self.image_groups[row['file_name']].append({
                'idx': idx,
                'coords': (int(row['x']), int(row['y']), int(row['x2']), int(row['y2']))
            })
        
        # Create index mapping
        self.idx_to_info = {}
        for image_path, crops in self.image_groups.items():
            for crop_info in crops:
                self.idx_to_info[crop_info['idx']] = {
                    'image_path': image_path,
                    'coords': crop_info['coords']
                }
    
    def _load_image_cached(self, image_path):
        """Thread-safe cached image loading"""
        with self.cache_lock:
            if image_path in self.image_cache:
                # Move to end (most recently used)
                image_tensor = self.image_cache.pop(image_path)
                self.image_cache[image_path] = image_tensor
                return image_tensor
            
            # Load new image
            # OpenCV is faster for loading
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image_tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
            
            # Add to cache
            self.image_cache[image_path] = image_tensor
            
            # Remove oldest if cache is full
            if len(self.image_cache) > self.cache_size:
                self.image_cache.popitem(last=False)
            
            return image_tensor
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        info = self.idx_to_info[idx]
        image_path = info['image_path']
        x1, y1, x2, y2 = info['coords']
        
        # Load image (cached)
        image_tensor = self._load_image_cached(image_path)
        
        # Fast tensor cropping
        crop = image_tensor[:, y1:y2, x1:x2]
        
        if self.huggingface:
            # the transform is actually an huggingface processor in this case
            inputs = self.transform(images=crop, return_tensors="pt")
            # Remove batch dimension from inputs
            patch = inputs['pixel_values'].squeeze()
        else:
            if self.transform:
                patch = self.transform(crop)

        return {
            'image': patch
        }
        
        return crop

def main(args):
    #parameters
    '''
    N_max=282
    patches=True
    input_filename='icdar_train_df_patches_20250515_164130.csv'
    huggingface=True
    pooling=False # if true in transformer mdoels use pooling, if false only the cls token
    custom_transform=False
    transform_mode='resize'
    save_h5=False
    selected_model = 'clip-vit-large-patch14' #googlenet, alexnet
    truncation = 'remove head'
    running = 'new-laptop'
    saved = 'old-laptop'
    model_mode = 'truncated' #'truncation
    batching = True
    batch_size = 256
    select_cls=False
    num_workers=4
    pin_memory=True'''
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
    batching = args.batching
    batch_size = args.batch_size
    select_cls = args.select_cls
    num_workers = args.num_workers
    pin_memory = args.pin_memory
    show_image = args.show_image

    #Initialization
    transform = u_transforms.get_transform(selected_model, use_patches=patches, custom=custom_transform, mode=transform_mode)
    model = model_utils.get_model(name=selected_model, mode=model_mode, pretrained=True, truncation=truncation)
    # Define loss function and optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device is: ",device)
    model = model.to(device)
    train_df = pd.read_csv(f"{source_path}\\outputs\\preprocessed_data\\{input_filename}")
    train_df=file_IO.change_filename_from_to(train_df, fr=saved, to=running)
    i=0
    output=compute_output(model, device, transform, train_df.iloc[i], huggingface, patches)
    print("Output shape: ", output.shape)

    #dataloading
    if batching:
        #dataset = CustomPatchDataset(train_df[:1000], transform, huggingface=huggingface)
        #dataset = CachedPatchDataset(train_df[:1000], transform, huggingface=huggingface)
        #dataset = LazyPatchDataset(train_df[:1000], transform, huggingface=huggingface)
        #hdf5_path = "C:\\Users\\andre\PhD\Datasets\ICDAR 2013 - Gender Identification Competition Dataset\\hdf5_train_writers_2.h5"
        zarr_path = "C:\\Users\\andre\PhD\Datasets\ICDAR 2013 - Gender Identification Competition Dataset\\train_writers.zarr"
        #dataset = HDF5ImageCropDataset(train_df[:1000], hdf5_path,transform, huggingface=huggingface)
        #dataset=FastOnTheFlyDataset(train_df[:1000], transform, huggingface=huggingface, image_cache_size=100)
        #dataset = HDF5ImageCropDataset_resize(train_df[:1000], hdf5_path, transform=transform, huggingface=huggingface)#, patches=patches, select_cls=select_cls)
        dataset = ZarrImageCropDataset_resize(train_df, zarr_path, transform=transform, huggingface=huggingface)
        #dataset = ZarrImageCropDataset_resize_workers(train_df[:1000], zarr_path, transform=transform, huggingface=huggingface)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers,pin_memory=pin_memory)
    
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
    
    if batching:
        new_features = []
        for i, batch in enumerate(dataloader):
            output = compute_output_gpu(model, device, batch)
            if select_cls:
                output = output[:, 0, :]
            for vec in output:
                new_features.append(vec.cpu().numpy())
            if i == 0:
                start_time = time.time()
            else:
                elapsed = time.time() - start_time
                images_processed = len(new_features)
                print(f'''Elapsed time {elapsed:.2f};Processed batch {i} out of {len(dataloader)}; Speed: {images_processed / elapsed:.2f} images/sec; 
                    Projected time: {(len(dataloader) - i) * (elapsed / (i+1)):.2f} seconds''')
        # Create the DataFrame
        df_out = pd.DataFrame(new_features)
        df_out.columns = [f'f{i+1}' for i in range(df_out.shape[1])]
        train_df = pd.concat([train_df.reset_index(drop=True), df_out.reset_index(drop=True)], axis=1)
    if batching==False:
        model.eval()

        if save_h5:
            import h5py
            import numpy as np
            import shutil
            icdar_path=train_df['file_name'][0]
            icdar_path = icdar_path[:icdar_path.lower().find("unzipped")].rsplit("\\", 1)[0] + "\\"
            # Define the directory and file paths
            h5_directory_name = "extracted_representations_full"
            h5_save_path=icdar_path+h5_directory_name
            # Open the file in append mode
            #from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            h5_file_name = h5_save_path+f"\\representations_{selected_model}_{truncation}_{timestamp}.h5"
            with h5py.File(h5_file_name, "a") as f:
                model.eval()
                for index,t in train_df.iterrows():
                    if huggingface:
                        output = compute_output(model, device, transform, t, huggingface, patches)
                    else:
                        output = compute_output(model, device, transform, t, huggingface, patches)
                    #print(output)
                    # Convert index to string key (e.g., "0001")
                    key = f"{index:06d}"
                    # Store with compression (optional)
                    rep_np = output.squeeze(0).detach().cpu().numpy()
                    f.create_dataset(key, data=rep_np, compression="gzip")
                    if index % 100 == 0:
                        print(f"Processed {index} images, out of {len(train_df)}")
            #close the file
            f.close()
        # Initialize a dictionary to store new feature columns
        else:
            new_features = {}

            for index,t in train_df.iterrows():
                if huggingface:
                    if pooling:
                        print("Pooling is not implemented yet")
                        break
                    else:
                        output = compute_output(model, device, transform, t, huggingface, patches)#[:,0,:]
                else:
                    output = compute_output(model, device, transform, t, huggingface, patches)
                for i, value in enumerate(output.squeeze().tolist()):
                    column_name = f"f{i+1}"
                    if column_name not in new_features:
                        new_features[column_name] = []
                    new_features[column_name].append(value)
                if index % 100 == 0:
                    print(f"Processed {index} images, out of {len(train_df)}")

                
            # Add the new features to the DataFrame in one operation
            new_features_df = pd.DataFrame(new_features)
            train_df = pd.concat([train_df.reset_index(drop=True), new_features_df], axis=1)

    # Get the current timestamp
    #timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Save the DataFrame to a CSV file
    #output_filename = f"{source_path}\\outputs\\online_deep_feature_extraction\\{selected_model}_features_{timestamp}.csv"
    output_filename = f"{source_path}\\outputs\\online_deep_feature_extraction\\{selected_model}_features_{input_filename.split('.')[0]}.csv"
    train_df.to_csv(output_filename, index=False)

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
    args = parse_args()
    main(args)