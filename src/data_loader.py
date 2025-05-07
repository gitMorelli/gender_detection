import os
import pandas as pd
import torch
from torchvision import datasets, transforms
from torch.utils.data import Dataset,DataLoader
from PIL import Image
import os
import torchvision.transforms.functional as F

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
    indices = torch.tensor([item['index'] for item in batch])
    versions = torch.tensor([item['version'] for item in batch])
    filenames = [item['filename'] for item in batch]
    labels = torch.tensor([item['label'] for item in batch])  # Keeping labels in a list

    return {
        'image': batch_images,
        'index': indices,
        'version': versions,
        'filename': filenames,
        'label': labels
    }


class CustomImageDataset(Dataset):
    def __init__(self, image_dirs, labels_df, transform=None):
        """
        Args:
            image_dirs (list of str): List of directories to load images from.
            labels_df (DataFrame): DataFrame containing labeled images.
            transform (callable, optional): Optional transform to be applied on an image.
        """
        self.image_files = []
        self.img_labels = labels_df
        self.transform = transform

        # Loop through each directory and collect image file paths for labeled images only
        for image_dir in image_dirs:
            for f in os.listdir(image_dir):
                if f.endswith('.jpg'):
                    base_name = os.path.splitext(f)[0]  # Remove extension
                    parts = base_name.split('_')

                    if len(parts) != 2:
                        continue  # Skip files that don't follow the expected pattern

                    index, version = parts

                    # Keep only images that exist in labels_df
                    if int(index) in self.img_labels.index:
                        self.image_files.append(os.path.join(image_dir, f))

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        # Extract index and version from filename
        img_name = os.path.basename(img_path)
        base_name = os.path.splitext(img_name)[0]
        index, version = base_name.split('_')

        # Fetch the label using the pre-set index
        label = int(self.img_labels.loc[int(index), 'male'])

        return {
            'image': image,
            'index': int(index),
            'version': int(version),
            'filename': img_name,
            'label': label
        }

def load_csv_sex(data_PATH):
    """
    Load the data from the CSV files and return the DataFrames.
    """
    image_PATH=data_PATH+"\\unzipped"
    #load the file with the writer names and version numbers
    train_df = pd.read_csv(os.path.join(data_PATH, "train\\train.csv"))
    #load the file with the training labels
    sex_df = pd.read_csv(os.path.join(data_PATH, "train_answers.csv"),delimiter=',')
    # Merge train_df with sex_df on the "writer" column to add the label column to the train_df DataFrame
    train_df = train_df.merge(sex_df[['writer', 'male']], on='writer', how='left')
    # Rename the "male" column to "sex"
    train_df.rename(columns={'male': 'sex'}, inplace=True)
    #I transform the sex_df so that it is the label vector (where the index column will be the writer column)
    sex_df = sex_df.drop(columns=['writer'])
    sex_df.index = sex_df.index + 1
    #i remove some columns from the train_df DataFrame to make it easier to work with
    selected_columns = ['writer', 'page_id', 'language', 'same_text','sex']
    df_selected = train_df[selected_columns].copy()
    # Load the file with the writer names and version numbers

def load_csv_arab(data_PATH):
    """
    Load the data from the CSV files and return the DataFrames.
    """

    # Load the file with the writer names and version numbers

def load_data(data_PATH,sex_or_arab):
    """
    Load the data from the CSV files and return the DataFrames. If sex_or_arab=0 i load the sex as label
    
    """

    image_PATH=data_PATH+"\\unzipped"
    #I create a list of the folder names in the image_PATH directory
    folder_names = [folder for folder in os.listdir(image_PATH) if os.path.isdir(os.path.join(image_PATH, folder))]
    # Extract the X part from the folder names
    x_values = [int(folder.split('_')[0]) for folder in folder_names]
    # Sort both lists based on the X values
    sorted_indices = sorted(range(len(x_values)), key=lambda k: x_values[k])
    folder_names = [folder_names[i] for i in sorted_indices]
    x_values = [x_values[i] for i in sorted_indices]

    # Define the transformations for the images
    transform = transforms.Compose([
        transforms.Resize(256, max_size=512), # Resize the images from HxW to 256xY where H is the smallest dimension
        transforms.ToTensor()           # Convert the images to tensors
    ])
    # Create the list of directories containing images
    image_dirs = [os.path.join(image_PATH, folder_names[i]) for i in range(len(folder_names))]
    # Initialize the dataset and dataloader
    dataset = CustomImageDataset(image_dirs=image_dirs, labels_df=sex_df, transform=transform)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True,collate_fn=pad_collate_fn)
    return dataset,dataloader 


if __name__ == "__main__":
    data_PATH="D:\\download\\PD project\\datasets\\ICDAR 2013 - Gender Identification Competition Dataset"
    load_data(data_PATH)
    print("Data loaded successfully.")