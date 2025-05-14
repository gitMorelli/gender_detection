import matplotlib.pyplot as plt
import os
import cv2
import random
import matplotlib.pyplot as plt

def plot_cropped_sections(train_df_expanded):
    # Select 10 random rows from the dataset
    random_rows = train_df_expanded.sample(n=10)

    # Plot the cropped sections
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    axes = axes.flatten()

    for i, (_, row) in enumerate(random_rows.iterrows()):
        # Load the image
        image = cv2.imread(row['file_name'])
        if image is None:
            print(f"Could not load image: {row['file_name']}")
            continue

        # Convert BGR to RGB for plotting
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Crop the section
        cropped_image = image[row['y']:row['y2'], row['x']:row['x2']]

        # Plot the cropped section
        axes[i].imshow(cropped_image)
        axes[i].axis('off')
        axes[i].set_title(f"Row {row['index']}")

    plt.tight_layout()
    plt.show()
# Function to show a batch of images
def show_images(images, writers, labels, save_path=None):
    fig, axes = plt.subplots(4, 4, figsize=(12, 12))
    axes = axes.flatten()
    
    for img, ax, writer, label in zip(images, axes, writers, labels):
        img = img.permute(1, 2, 0)  # Change the order of dimensions to (H, W, C)
        ax.imshow(img)
        ax.set_title(str(int(writer))+'_'+str(int(label)))
        ax.axis('off')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()


def show_single_image(dataset,index=0, save_path=None, from_dataloader=False):
    if from_dataloader:
        sample = dataset.dataset[index]  # Access the dataset from the DataLoader
    else:
        sample = dataset[index]
    image = sample['image']
    writer = sample['writer']
    label = sample['label']
    img = image.permute(1, 2, 0)  # Change the order of dimensions to (H, W, C)
    plt.imshow(img)
    plt.title(f'Writer: {writer}, Label: {label}')
    plt.axis('off')
    if save_path:
        plt.savefig(save_path)
    plt.show()

def plot_loss(train_losses, val_losses):
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, len(train_losses) + 1), train_losses, label='Train Loss', marker='o')
    plt.plot(range(1, len(val_losses) + 1), val_losses, label='Val Loss', marker='o')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss Evolution')
    plt.legend()
    plt.show()