import matplotlib.pyplot as plt
import os
import cv2
import random
import matplotlib.pyplot as plt
from PIL import Image
from PIL import ImageDraw, ImageFont
import zarr
import utils.utils_transforms as u_transforms

class ZarrVisualizer():
    def __init__(self, selected, zarr_path,selected_metric, save_path, transform=None, huggingface=False, use_augmentation=False):
        """
        df: DataFrame with columns ['file_name', 'x', 'y', 'x2', 'y2']
        zarr_path: path to directory-based Zarr store
        transform: Optional transform applied to the cropped patch
        """
        self.selected = selected.reset_index(drop=True)
        self.zarr_path = zarr_path
        self.transform = transform
        self.huggingface = huggingface
        self.zarr_store = None  # will be lazily opened
        self.use_augmentation = use_augmentation
        self.augmentation_transform = u_transforms.get_augmentation_transform() if use_augmentation else None
        self.save_path = save_path
        self.selected_metric = selected_metric

        # Load filenames and create mapping: file_name -> index
        z = zarr.open(self.zarr_path, mode='r')
        filenames = list(z['filenames'][:])
        self.file_to_idx = {fn: i for i, fn in enumerate(filenames)}

        self.pages = self.selected['page'].unique()
        self.types = ['sure', 'unsure', 'ok']
        self.grouping = [selected_metric + '_' + g for g in self.types] 
        self.types = ['sure', 'unsure', 'ok']

    def save_images(self,mode):
        if self.zarr_store is None:
            self.zarr_store = zarr.open(self.zarr_path, mode='r')
        pages = self.selected['page'].unique()
        grouping = [self.selected_metric + '_' + g for g in self.types] 
        for page in pages:
            image_list=[]
            group = self.selected[self.selected['page'] == page].reset_index(drop=True)
            for idx, row in group.iterrows():
                file_name = row['file_name']
                x1, y1, x2, y2 = row['x'], row['y'], row['x2'], row['y2']
                label = row['male']
                img_idx = self.file_to_idx[file_name]
                full_img = self.zarr_store['images'][img_idx]  # numpy array HWC
                patch = full_img[y1:y2, x1:x2, :]
                patch = Image.fromarray(patch)
                if mode == 'original':
                    pass
                elif mode == 'augmentation':
                    #times.append(datetime.now())
                    if self.use_augmentation:
                        patch = self.augmentation_transform(patch)
                elif mode == 'preprocessed':
                    if self.huggingface:
                        inputs = self.transform(images=patch, return_tensors="pt")
                        patch_tensor = inputs['pixel_values'][0]  # shape: (C, H, W)
                        # Convert tensor to numpy and scale to [0, 1] for visualization
                    elif self.transform:
                        patch_tensor = self.transform(patch)
                    img_np = patch_tensor.permute(1, 2, 0).cpu().numpy()
                    img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min())
                    patch = Image.fromarray((img_np * 255).astype('uint8'))
                image_list.append(patch)
            # Create a grid of images
            n = len(image_list)
            cols = 4
            rows = (n + cols - 1) // cols  # Round up to the nearest whole number
            fig, axes = plt.subplots(rows, cols, figsize=(15, 4 * rows))
            axes = axes.flatten() if rows > 1 else [axes]
            for i, img in enumerate(image_list):
                axes[i].imshow(img)
                axes[i].axis('off')
                axes[i].set_title(f"Pred: {group['y_pred'].iloc[i]}; Prob: {group['y_prob'].iloc[i]}")
                #i can use i because I have sorted the group before iterating
            for j in range(i + 1, len(axes)):
                axes[j].axis('off')
            plt.tight_layout()
            if self.save_path:
                plt.savefig(os.path.join(self.save_path, mode,f"page_{page}.png"))
                txt_info = (
                    f"Page: {page}; IsMale: {group['male'].iloc[0]}; Predicted page value: {group[self.selected_metric].iloc[0]}; Explanation for: Predicted_value\n"
                    f"{self.types[0]} = {group[grouping[0]].iloc[0]}; {self.types[1]} = {group[grouping[1]].iloc[0]}; {self.types[2]} = {group[grouping[2]].iloc[0]}"
                )
                with open(self.save_path + f"{mode}\\page_{page}_info.txt", "w", encoding="utf-8") as f:
                    f.write(txt_info)

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

def show_preprocessed_from_csv(train_df,transform,hugging=False,index=0,patches=True):
    if patches:
        # Select an image row from train_df
        sample_row = train_df.iloc[index]
        image_file = sample_row['file_name']
        image = Image.open(image_file).convert("RGB")

        # If using patches, crop the patch, else use the whole image
        if patches:
            x1, y1, x2, y2 = sample_row['x'], sample_row['y'], sample_row['x2'], sample_row['y2']
            patch = image.crop((x1, y1, x2, y2))
        else:
            patch = image.copy()

        if hugging:
            # the transform is actually an huggingface processor in this case
            inputs = transform(images=patch, return_tensors="pt")
            # Remove batch dimension from inputs
            patch = inputs['pixel_values'].squeeze()
        else:
            patch = transform(patch)

        # Convert tensor to numpy for plotting
        img_np = patch.permute(1, 2, 0).cpu().numpy()
        img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min())

        plt.imshow(img_np)
        plt.title("Preprocessed Image")
        plt.axis('off')
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

def display_gradcam_vis(visualizations,df,ind,selected_metric='weighted_vote'):
    pages = df['page'].unique()
    page = pages[ind]
    group = df[df['page'] == page]
    types = ['sure', 'unsure', 'ok']
    grouping = [selected_metric + '_' + g for g in types] 
    n = len(visualizations[ind])
    plt.figure(figsize=(4 * n, 4))
    for i, visualization in enumerate(visualizations[ind]):
        plt.subplot(1, n, i + 1)
        plt.imshow(visualization)
        #print(group[selected_metric].iloc[0])
        plt.suptitle(f"Page: {page}; IsMale: {group['male'].iloc[i]}; Predicted page value: {group[selected_metric].iloc[i]}; Explanation for: Predicted_value\n"
                  f"{types[0]} = {group[grouping[0]].iloc[i]}; {types[1]} = {group[grouping[1]].iloc[i]}; {types[2]} = {group[grouping[2]].iloc[i]}")
        plt.title(f"Predicted value: {group['y_pred'].iloc[i]};")
        #plt.xlabel(f"{types[0]} = {group[grouping[0]].iloc[i]}; {types[1]} = {group[grouping[1]].iloc[i]}; {types[2]} = {group[grouping[2]].iloc[i]}", fontsize=16, y=1.05)
        
        plt.axis('off')
    plt.tight_layout()
    plt.show()

def display_vis_on_background(visualizations,df,selected_metric='weighted_vote',blank_background=False,save_path=None):
    for ind in range(len(visualizations)):
        #only works with transforms that first resize and then crop and that work on square images
        pages = df['page'].unique()
        page = pages[ind]
        group = df[df['page'] == page]
        types = ['sure', 'unsure', 'ok']
        grouping = [selected_metric + '_' + g for g in types] 
        n = len(visualizations[ind])
        # Open the background image
        if blank_background:
            background = Image.open(group['file_name'].iloc[0]).convert("RGBA")
            background = Image.new("RGBA", (background.width, background.height), (255, 255, 255, 0))
        else:
            background = Image.open(group['file_name'].iloc[0]).convert("RGBA")
        x_scale = 1
        y_scale = 1
        # Example font (use a TTF font file if available)
        try:
            font = ImageFont.truetype("arial.ttf", size=16)
        except IOError:
            font = ImageFont.load_default()
        for i, visualization in enumerate(visualizations[ind]):
            # Open the image to overlay
            x1= group['x'].iloc[i]
            y1= group['y'].iloc[i]
            x2= group['x2'].iloc[i]
            y2= group['y2'].iloc[i]
            rgb_image = Image.fromarray(visualization, mode='RGB')
            # Convert to RGBA (adds alpha channel = fully opaque)
            overlay = rgb_image.convert('RGBA')

            if i==0:
                width, height = overlay.size
                x_scale = (x2 - x1) / width
                y_scale = (y2 - y1) / height
                # Calculate new size
                new_width = int(background.width / x_scale)
                new_height = int(background.height / y_scale)
                # Resize image
                background = background.resize((new_width, new_height), resample=Image.BILINEAR)

            # Draw text on the overlay
            draw = ImageDraw.Draw(overlay)
            text = f"{group['y_pred'].iloc[i]};{group['y_prob'].iloc[i]:.2f}"  # Change to whatever text you want, e.g. grouping[i] or metric values
            text_position = (5, 5)  # top-left corner, adjust as needed
            text_color = (255, 255, 255, 255)  # white text with full opacity
            # Optionally add a shadow or outline for better visibility
            # Shadow
            shadow_color = (0, 0, 0, 255)
            shadow_offset = (1, 1)
            # Draw shadow
            draw.text((text_position[0] + shadow_offset[0], text_position[1] + shadow_offset[1]), text, font=font, fill=shadow_color)
            # Draw main text
            draw.text(text_position, text, font=font, fill=text_color)

            # Choose position for overlay (x, y)
            position = (int(x1/x_scale), int(y1/y_scale))

            # Make the overlay semi-transparent
            #alpha = overlay.split()[3].point(lambda p: int(p * 0.8))
            alpha = overlay.split()[3].point(lambda p: 255)
            overlay.putalpha(alpha)
            # Paste the overlay image onto the background
            background.paste(overlay, position, overlay)  # third argument is the mask for transparency

        # Save or show the result
        background.save(save_path+f"\\page_{page}.png")
        txt_info = (
            f"Page: {page}; IsMale: {group['male'].iloc[i]}; Predicted page value: {group[selected_metric].iloc[i]}; Explanation for: Predicted_value\n"
            f"{types[0]} = {group[grouping[0]].iloc[i]}; {types[1]} = {group[grouping[1]].iloc[i]}; {types[2]} = {group[grouping[2]].iloc[i]}"
        )
        with open(save_path + f"\\page_{page}_info.txt", "w", encoding="utf-8") as f:
            f.write(txt_info)

def display_debug_images(dataloader,path,save_name):
    # Display one batch of images from the dataloader in a grid
    for batch in dataloader:
        images = batch['image']
        n_samples = min(8, len(images))
        n_cols = 4
        n_rows = (n_samples + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
        axes = axes.flatten()
        for i in range(n_samples):
            img = images[i].permute(1, 2, 0).cpu().numpy()
            img = (img - img.min()) / (img.max() - img.min())
            img = Image.fromarray((img * 255).astype('uint8'))
            axes[i].imshow(img)
            axes[i].axis('off')
        # Hide any unused subplots
        for j in range(n_samples, len(axes)):
            axes[j].axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(path, f"{save_name}_sample_image.png"))
        plt.close()
        break
    return 0