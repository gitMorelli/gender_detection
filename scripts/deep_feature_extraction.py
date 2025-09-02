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
    save_to_log = args.save_to_log
    data_augmentation = args.data_augmentation
    num_views= args.num_views
    save_dir = args.save_dir
    n_patches = args.n_patches
    zarr_path = args.zarr_path
    contrastive_mode = args.contrastive_mode
    augmentation_code = args.augmentation_code
    custom_pretrained = args.custom_pretrained  # 'original', 'contrastive', 'fine-tune'

    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()

    #Initialization
    transform = u_transforms.get_transform(selected_model, use_patches=patches, custom=custom_transform, mode=transform_mode)
    model = model_utils.get_model(name=selected_model, mode=model_mode, pretrained=True, truncation=truncation, 
                                                    contrastive=contrastive_mode,custom_pretrained=custom_pretrained)
    # Define loss function and optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device is: ",device)
    model = model.to(device)
    train_df = pd.read_csv(f"{source_path}\\outputs\\preprocessed_data\\{input_filename}")
    train_df=file_IO.change_filename_from_to(train_df, fr=saved, to=running)
    train_df['page'] = train_df.groupby(['writer', 'isEng', 'same_text']).ngroup()
    if n_patches > 0:
        #print(f"Selecting {n_patches} patches per page...")
        train_df = select_n_patches(train_df, n_patches=n_patches).reset_index(drop=True)

    i=0
    output=compute_output(model, device, transform, train_df.iloc[i], huggingface, patches)
    print("Output shape: ", output.shape)
    if output.dim() == 3:
        select_cls = True  # If output is 3D, we assume it's a transformer model with CLS token
    else:
        select_cls = False
    
    #dataloading
    if batching:
        #dataset = CustomPatchDataset(train_df[:1000], transform, huggingface=huggingface)
        #dataset = CachedPatchDataset(train_df[:1000], transform, huggingface=huggingface)
        #dataset = LazyPatchDataset(train_df[:1000], transform, huggingface=huggingface)
        #hdf5_path = "C:\\Users\\andre\PhD\Datasets\ICDAR 2013 - Gender Identification Competition Dataset\\hdf5_train_writers_2.h5"
        #dataset = HDF5ImageCropDataset(train_df[:1000], hdf5_path,transform, huggingface=huggingface)
        #dataset=FastOnTheFlyDataset(train_df[:1000], transform, huggingface=huggingface, image_cache_size=100)
        #dataset = HDF5ImageCropDataset_resize(train_df[:1000], hdf5_path, transform=transform, huggingface=huggingface)#, patches=patches, select_cls=select_cls)
        dataset = ZarrImageCropDataset_resize(train_df, zarr_path, transform=transform, 
                                              huggingface=huggingface,use_augmentation=data_augmentation,code=augmentation_code)
        #dataset = ZarrImageCropDataset_resize_workers(train_df[:1000], zarr_path, transform=transform, huggingface=huggingface)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers,pin_memory=pin_memory)
    
    if show_image:
        print("Saving debug images...")
        display_debug_images(dataloader,save_dir, save_name='train')
        print("Debug images saved.")
    
    if not(data_augmentation):
        num_views=1
    if batching:
        model.eval()
        df_list=[]
        for i in range(num_views):
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
            df_list.append(pd.concat([train_df.reset_index(drop=True), df_out.reset_index(drop=True)], axis=1))
        # Concatenate all DataFrames
        if num_views > 1:
            train_df = pd.concat(df_list, axis=0, ignore_index=True)
        else:
            train_df = df_list[0]
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

    if save_to_log:
        output_dir = os.path.join(source_path, "outputs", "preprocessed_data")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join(output_dir, f"icdar_EXTRACTED_train_df_{selected_model}_{timestamp}.csv")
        train_df.to_csv(output_file, index=False)
        print(f"Dataframe saved to {output_file}")

        LOG_FILE = output_dir+"\\file_metadata_log.json"
        print(f"Log file path: {LOG_FILE}")
        print(f"Output file path: {output_file}")

        file_IO.add_or_update_file(
            output_file, LOG_FILE,
            custom_metadata={
                #"seed": seed,
                "source_file": input_filename,
                "model": selected_model,
                "pooling": pooling,
                "custom transform": custom_transform,
                "save_h5": save_h5,
                "truncation": truncation,
                "transform_mode": transform_mode,
                "description": ''' ''' 
            }
        )
    else:
        output_filename = f"{save_dir}\\{selected_model}_features_{input_filename.split('.')[0]}.csv"
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
    from utils.train_on_rep_utils import select_n_patches
    from utils.script_launching import load_config
    from utils.script_launching import DotDict
    from utils.dataframes import ZarrImageCropDataset_resize
    from utils.visualization import display_debug_images
    from model_utils import compute_output_gpu
    
    args = parse_args()
    main(args)