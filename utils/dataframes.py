import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import cv2
from PIL import Image

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