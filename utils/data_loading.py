
from utils.dataframes import CustomExtractedDataset, CustomPatchDataset, CustomImageDataset, CustomHdf5ExtractedDataset
import pandas as pd
import os
from torch.utils.data import Dataset,DataLoader, random_split

def get_dataloaders(transform, batch_size=16, N_max=282,file_name='icdar_train_df_patches_cc.csv', 
                    source='D:\\burtm\\Visual_studio_code\\PD_related_projects\\outputs\\preprocessed_data'
                    ,hugging=False, from_df=False, df=None, h5=False):
    if from_df:
        train_df = df.copy()
    else:
        file_path = os.path.join(source, file_name)
        train_df = pd.read_csv(file_path)
        print('loaded train_df from:', file_path)
    if h5:
        file_path_h5='D:\\download\\PD project\\datasets\\ICDAR 2013 - Gender Identification Competition Dataset\\extracted_representations_full\\representations.h5'
        train_dataset = CustomHdf5ExtractedDataset(train_df[(train_df['train']==1) & (train_df['writer']<=N_max)] ,
                                        label_column='male',filepath=file_path_h5)
        
        val_dataset = CustomHdf5ExtractedDataset(train_df[(train_df['train']==0) & (train_df['writer']<=N_max)] ,
                                        label_column='male',filepath=file_path_h5)
    else:
        if file_name=='icdar_train_df_w_image_paths.csv':
            train_dataset = CustomImageDataset(train_df[(train_df['train']==1) & (train_df['writer']<=N_max)] ,
                                            label_column='male', transform=transform, huggingface=hugging)
            val_dataset = CustomImageDataset(train_df[(train_df['train']==0) & (train_df['writer']<=N_max)] ,
                                            label_column='male', transform=transform, huggingface=hugging)
        elif 'EXTRACTED' in file_name:
            train_dataset = CustomExtractedDataset(train_df[(train_df['train']==1) & (train_df['writer']<=N_max)] ,
                                            label_column='male')
            
            val_dataset = CustomExtractedDataset(train_df[(train_df['train']==0) & (train_df['writer']<=N_max)] ,
                                            label_column='male')
            
        elif 'patches' in file_name:
            train_dataset = CustomPatchDataset(train_df[(train_df['train']==1) & (train_df['writer']<=N_max)] ,
                                            label_column='male', transform=transform, huggingface=hugging)
            

            val_dataset = CustomPatchDataset(train_df[(train_df['train']==0) & (train_df['writer']<=N_max)] ,
                                            label_column='male', transform=transform, huggingface=hugging)
        else:
            #print(file_name)
            raise ValueError(f"Unknown file name: {file_name}. Please provide a valid file name.")
    
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

    return train_dataloader, val_dataloader