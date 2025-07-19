
from utils.dataframes import CustomExtractedDataset, CustomPatchDataset, CustomImageDataset, CustomHdf5ExtractedDataset
import pandas as pd
import os
from torch.utils.data import Dataset,DataLoader, random_split
import torch
import utils.model_utils as model_utils 
from joblib import load

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

def load_classification_head(selected_FE,selected_classifier,classifier_type,validation_mode,train_filename,source_path):

    train_df = pd.read_csv(train_filename)
    cols_to_keep = [c for c in train_df.columns if c.startswith('f') and len(c) > 1 and c[1].isdigit()]
    in_features = len(cols_to_keep)  # Number of features from the model output

    base_dir=source_path + f'/outputs/online_deep_feature_extraction/{selected_FE}/representation_extraction/'
    if classifier_type == 'sklearn':
        search_dir = base_dir+f'sklearn_model_trained_on_rep/{selected_classifier}/'
        pipeline = load(os.path.join(search_dir, validation_mode+'_pipeline.joblib'))
        if validation_mode == '1fold_train_only':
            val_writers_path = os.path.join(search_dir, '1fold_train_only_writers.csv')
            df_writers = pd.read_csv(val_writers_path)
            train_df = train_df.drop(columns=['train'])
            train_df = train_df.merge(df_writers, on='writer', how='left')
        return pipeline, train_df
    else:
        search_dir = base_dir+f'torch_model_trained_on_rep/{selected_classifier}/'
        checkpoint_path = os.path.join(search_dir, 'checkpoint_best.pt')
        model = model_utils.get_classification_head(name=selected_classifier, in_features=in_features, num_classes=2)
        checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'), weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        return model, train_df

def load_selected_instances(source_path, selected_FE, selected_classification_head, head_type, selected_metric='weighted_vote'):
    base_dir=source_path + f'/outputs/online_deep_feature_extraction/{selected_FE}/representation_extraction/'
    if head_type == 'sklearn':
        search_dir = base_dir+f'sklearn_model_trained_on_rep/{selected_classification_head}/'
    else:
        search_dir = base_dir+f'torch_model_trained_on_rep/{selected_classification_head}/'
    selected=pd.read_csv(search_dir+f'selected_instances_{selected_metric}.csv')
    return selected