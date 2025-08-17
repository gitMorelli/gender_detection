import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import cv2
from PIL import Image
import h5py
import numpy as np
import io
from datetime import datetime
from collections import OrderedDict, defaultdict
import threading
import zarr
import utils.utils_transforms as u_transforms
import pandas as pd

def pad_collate_fn(batch):
    """
    Pads images in a batch to match the largest image dimensions.
    """
    images = [item['image'] for item in batch]
    max_height = max(img.shape[1] for img in images)  # Find max height
    max_width = max(img.shape[2] for img in images)   # Find max width

    # Pad all images to the max dimensions in the batch
    padded_images = [
        F.pad(img, (0, 0, max_width - img.shape[2], max_height - img.shape[1])) for img in images
    ]

    # Stack images into a batch tensor
    batch_images = torch.stack(padded_images)

    # Keep other data (index, version, etc.)
    writers = torch.tensor([item['writer'] for item in batch])
    labels = torch.tensor([item['label'] for item in batch])  # Keeping labels in a list

    return {
        'image': batch_images,
        'writer': writers,
        'label': labels
    }


class CustomImageDataset(Dataset):
    def __init__(self, df,label_column, transform=None, huggingface=False):
        """
        Args:
            image_dirs (list of str): List of directories to load images from.
            labels_df (DataFrame): DataFrame containing labeled images.
            transform (callable, optional): Optional transform to be applied on an image.
        """
        #self.label_column=label_column
        self.image_files = df['file_name'].tolist()
        self.img_labels = df[label_column].tolist()
        self.img_writers = df['writer'].tolist()
        self.transform = transform
        self.huggingface = huggingface

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        image = Image.open(img_path).convert("RGB")
        writer=self.img_writers[idx]
        label = self.img_labels[idx]

        if self.huggingface:
            # the transform is actually an huggingface processor in this case
            inputs = self.transform(images=image, return_tensors="pt")
            # Remove batch dimension from inputs
            image = inputs['pixel_values'].squeeze()
        else:
            if self.transform:
                image = self.transform(image)

        return {
            'image': image,
            'writer': int(writer),
            'label': label
        }

class CustomPatchDataset(Dataset):
    def __init__(self, df, label_column,transform=None,huggingface=False):
        """
        Args:
            image_dirs (list of str): List of directories to load images from.
            labels_df (DataFrame): DataFrame containing labeled images.
            transform (callable, optional): Optional transform to be applied on an image.
        """
        self.image_files = df['file_name'].tolist()
        self.img_labels = df[label_column].tolist()
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
        img_path = self.image_files[idx]
        x1=self.x1[idx]
        y1=self.y1[idx]
        x2=self.x2[idx]
        y2=self.y2[idx]
        image = Image.open(img_path).convert("RGB")
        patch = image.crop((x1, y1, x2, y2))
        writer=self.img_writers[idx]
        label = self.img_labels[idx]

        if self.huggingface:
            # the transform is actually an huggingface processor in this case
            inputs = self.transform(images=patch, return_tensors="pt")
            # Remove batch dimension from inputs
            patch = inputs['pixel_values'].squeeze()
        else:
            if self.transform:
                patch = self.transform(patch)

        return {
            'image': patch,
            'writer': int(writer),
            'label': label
        }

class CustomExtractedDataset(Dataset):
    def __init__(self, df,label_column):
        """
        Args:
            image_dirs (list of str): List of directories to load images from.
            labels_df (DataFrame): DataFrame containing labeled images.
            transform (callable, optional): Optional transform to be applied on an image.
        """
        #self.label_column=label_column
        self.feature_columns = [col for col in df.columns if col.startswith('f') and col[1].isdigit()]
        print(f'Extracted {len(self.feature_columns)} feature columns:')
        self.features_df = df[self.feature_columns]
        self.img_labels = df[label_column].tolist()

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        features = torch.tensor(self.features_df.iloc[idx].values, dtype=torch.float32)
        label = self.img_labels[idx]

        return {
            'image': features,
            'label': torch.tensor(label, dtype=torch.long)
        }

class CustomHdf5ExtractedDataset(Dataset):
    def __init__(self, df,label_column,filepath):
        """
        Args:
            image_dirs (list of str): List of directories to load images from.
            labels_df (DataFrame): DataFrame containing labeled images.
            transform (callable, optional): Optional transform to be applied on an image.
        """
        #self.label_column=label_column
        self.img_labels = df[label_column].tolist()
        self.img_writers = df['writer'].tolist()
        self.indices = df.index
        self.filepath=filepath

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        index=self.indices[idx]
        with h5py.File(self.filepath, "r") as f:
            key = f"{index:06d}"
            rep = f[key][:]  
        features = torch.tensor(rep, dtype=torch.float32)
        #features = torch.tensor(self.features_df.iloc[idx].values, dtype=torch.float32)
        writer=self.img_writers[idx]
        label = self.img_labels[idx]

        return {
            'features': features,
            'writer': int(writer),
            'label': label
        }

'''class CustomPatchDataset(Dataset):
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
        }'''
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
class ZarrImageCropDataset_resize_augmentation(Dataset):
    def __init__(self, df, zarr_path, transform=None, huggingface=False,use_augmentation=False,num_views=5):
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
        self.use_augmentation = use_augmentation
        self.augmentation_transform = u_transforms.get_augmentation_transform() if use_augmentation else None
        self.num_views = num_views

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
        if self.use_augmentation:
            patch_list=[patch]
            for i in range(self.num_views):
                augmented_patch = self.augmentation_transform(patch)
                patch_list.append(augmented_patch)
            for i in range(len(patch_list)):
                if self.huggingface:
                    inputs = self.transform(images=patch_list[i], return_tensors="pt")
                    patch_list[i] = inputs['pixel_values'][0]  # shape: (C, H, W)
                elif self.transform:
                    patch_list[i] = self.transform(patch_list[i])
                else:
                    pass
            return patch_list
        else:
            if self.huggingface:
                inputs = self.transform(images=patch, return_tensors="pt")
                patch_tensor = inputs['pixel_values'][0]
            elif self.transform:
                patch_tensor = self.transform(patch)
            else:
                patch_tensor = patch
            return {'image': patch_tensor}
        #times.append(datetime.now())

    def __del__(self):
        self.zarr_store = None  # optional: let GC handle closure

class ZarrImageCropDataset_resize(Dataset):
    def __init__(self, df, zarr_path, transform=None, huggingface=False, use_augmentation=False, code='simclr'):
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
        self.use_augmentation = use_augmentation
        self.augmentation_transform = u_transforms.get_augmentation_transform(code=code) if use_augmentation else None

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
        label = row['male']

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
        if self.use_augmentation:
            patch = self.augmentation_transform(patch)
        if self.huggingface:
            inputs = self.transform(images=patch, return_tensors="pt")
            patch_tensor = inputs['pixel_values'][0]  # shape: (C, H, W)
        elif self.transform:
            patch_tensor = self.transform(patch)
        else:
            patch_tensor = patch
        '''assert isinstance(patch_tensor, torch.Tensor), f"Expected torch.Tensor but got {type(patch_tensor)}"
        assert not patch_tensor.is_cuda, "Expected patch_tensor to be on CPU, but it's on CUDA"
        assert patch_tensor.device.type == "cpu", f"Expected CPU tensor but got device: {patch_tensor.device}"'''
        #times.append(datetime.now())

        '''for i, name in enumerate(time_names):
            print(f"Time for {name}: {(times[i+1] - times[i]).total_seconds() * 1000:.2f} ms")
        raise RuntimeError("Debug: error raised after timing print statements")'''

        return {'image': patch_tensor,'label': torch.tensor(label, dtype=torch.long)}

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

class ContrastivePatchDataset(Dataset):
    def __init__(self, df,transform):
        """
        Args:
            image_dirs (list of str): List of directories to load images from.
            labels_df (DataFrame): DataFrame containing labeled images.
            transform (callable, optional): Optional transform to be applied on an image.
        """
        self.image_files = df['file_name'].tolist()
        self.x1 = df['x'].tolist()
        self.y1 = df['y'].tolist()
        self.x2 = df['x2'].tolist()
        self.y2 = df['y2'].tolist()
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        x1=self.x1[idx]
        y1=self.y1[idx]
        x2=self.x2[idx]
        y2=self.y2[idx]
        image = Image.open(img_path).convert("RGB")
        patch = image.crop((x1, y1, x2, y2))

        if self.transform:
            patch1 = self.transform(patch)
            patch2 = self.transform(patch)

        return {
            'image1': patch1,
            'image2': patch2,
        }

class ZarrContrastive(Dataset):
    def __init__(self, df, zarr_path, transform=None, huggingface=False, contrastive_transform=None):
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
        self.contrastive_transform = contrastive_transform

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
        #label = row['male']

        img_idx = self.file_to_idx[file_name]
        full_img = self.zarr_store['images'][img_idx]  # numpy array HWC
        #times.append(datetime.now())

        patch = full_img[y1:y2, x1:x2, :]
        #times.append(datetime.now())

        patch_source = Image.fromarray(patch)
        if hasattr(patch_source, "size"):
            width, height = patch_source.size
            if width == 0 or height == 0:
                raise ValueError(f"Invalid patch size: {patch_source.size} at index {idx}; x1={x1}, y1={y1}, x2={x2}, y2={y2} in file {file_name}")
        #times.append(datetime.now())
        patches=[]
        for i in range(2):
            patch = self.contrastive_transform(patch_source)
            if self.huggingface:
                inputs = self.transform(images=patch, return_tensors="pt")
                patch_tensor = inputs['pixel_values'][0]  # shape: (C, H, W)
            elif self.transform:
                patch_tensor = self.transform(patch)
            else:
                patch_tensor = patch
            patches.append(patch_tensor)
        #times.append(datetime.now())

        return {
            'image1': patches[0],
            'image2': patches[1],
        }

    def __del__(self):
        self.zarr_store = None  # optional: let GC handle closure

def merge_dfs(train_1, train_2, mode):
    print('merging dfs train_1 and train_2 with lengths:', len(train_1), len(train_2))
    if mode=='concat':
        train_1.drop(columns=['page'], inplace=True) if 'page' in train_1.columns else None
        train_2.drop(columns=['page'], inplace=True) if 'page' in train_2.columns else None
        # Concatenate both datasets to build a unified group mapping
        combined = pd.concat([train_1[['writer', 'isEng', 'same_text']],
                            train_2[['writer', 'isEng', 'same_text']]]).drop_duplicates()
        # Create consistent group ids
        combined['page'] = combined.groupby(['writer', 'isEng', 'same_text']).ngroup()
        # Merge group ids back to each original DataFrame
        train_1 = train_1.merge(combined, on=['writer', 'isEng', 'same_text'], how='left')
        train_2 = train_2.merge(combined, on=['writer', 'isEng', 'same_text'], how='left')
        cols_to_drop_1 = [c for c in train_1.columns if not(c.startswith('f') and len(c) > 1 and c[1].isdigit())]
        cols_to_drop_2 = [c for c in train_2.columns if not(c.startswith('f') and len(c) > 1 and c[1].isdigit())]
        #print(train_2[cols_to_drop_2].head())
        common_cols = list(set(cols_to_drop_1) & set(cols_to_drop_2))
        # Remove 'page' from common_cols if present
        if 'page' in common_cols:
            common_cols.remove('page')
        train_2.drop(columns=common_cols, inplace=True, errors='ignore')
        num_pages = combined['page'].nunique()
        patch_1_per_page = int(len(train_1) / num_pages)
        patch_2_per_page = int(len(train_2) / num_pages)
        # Repeat train_2 so it matches the number of patches per page in train_1
        # Add a 'patch_num' column to train_1: unique number per row within each 'page' group
        train_1['patch_num'] = train_1.groupby('page').cumcount()
        train_2['patch_num'] = train_2.groupby('page').cumcount()
        train_1['patch_num'] = train_1['patch_num'] % patch_2_per_page
        merged_df = pd.merge(train_1, train_2, on=['page','patch_num'], suffixes=('_1', '_2'))
        print(f'Merged DataFrame length: {len(merged_df)}')
    else:
        raise ValueError("Invalid mode. Use 'concat' to merge DataFrames.")
    return merged_df

def aggregate_dfs(train_FE, mode=None):
    print('aggregating patches, length before:', len(train_FE))
    cols_to_drop = [c for c in train_FE.columns if not(c.startswith('f') and len(c) > 1 and c[1].isdigit())]
    cols_to_keep = [c for c in train_FE.columns if c.startswith('f') and len(c) > 1 and c[1].isdigit()]

    if mode == 'mean':
        agg_dict = {col: 'mean' for col in cols_to_keep}
        agg_dict.update({col: 'first' for col in cols_to_drop})
        # Group by 'page' and average the feature columns
        train_FE = train_FE.groupby('page',as_index=False).agg(agg_dict)
    elif mode == 'max':
        agg_dict = {col: 'max' for col in cols_to_keep}
        agg_dict.update({col: 'first' for col in cols_to_drop})
        # Group by 'page' and average the feature columns
        train_FE = train_FE.groupby('page',as_index=False).agg(agg_dict)
    elif mode == None:
        pass
    print('length after:', len(train_FE))
    return train_FE