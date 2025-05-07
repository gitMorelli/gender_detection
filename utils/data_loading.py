
from utils.dataframes import CustomImageDataset, CustomPatchDataset
import pandas as pd
import os
from torch.utils.data import Dataset,DataLoader, random_split

def get_dataloaders(transform, batch_size=16, N_max=282,file_name='icdar_train_df_patches_cc.csv', 
                    source='D:\\burtm\\Visual_studio_code\\PD_related_projects\\outputs\\preprocessed_data'
                    ,hugging=False):
    file_path = os.path.join(source, file_name)
    train_df = pd.read_csv(file_path)
    print('loaded train_df from:', file_path)
    if file_name=='icdar_train_df_w_image_paths.csv':
        #train_dataset = CustomImageDataset(train_df[train_df['train']==1], transform=model_transforms[selected_model])
        train_dataset = CustomImageDataset(train_df[(train_df['train']==1) & (train_df['writer']<=N_max)] ,
                                        label_column='male', transform=transform, huggingface=hugging)
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        #val_dataset = CustomImageDataset(train_df[train_df['train']==0], transform=model_transforms[selected_model])
        val_dataset = CustomImageDataset(train_df[(train_df['train']==0) & (train_df['writer']<=N_max)] ,
                                        label_column='male', transform=transform, huggingface=hugging)
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    elif file_name == 'icdar_train_df_patches_cc.csv':
        train_dataset = CustomPatchDataset(train_df[(train_df['train']==1) & (train_df['writer']<=N_max)] ,
                                        label_column='male', transform=transform, huggingface=hugging)
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        val_dataset = CustomPatchDataset(train_df[(train_df['train']==0) & (train_df['writer']<=N_max)] ,
                                        label_column='male', transform=transform, huggingface=hugging)
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    else:
        #print(file_name)
        raise ValueError(f"Unknown file name: {file_name}. Please provide a valid file name.")
    return train_dataloader, val_dataloader