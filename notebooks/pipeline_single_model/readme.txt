To train and evaluate a model follwos these steps:
    - decide which dataset you want to use: icdar, iam
        - if you use icdar run the prepare_icdar_table, the patches_cc and the save_as_zarr notebooks in this order
        - if you use iam ...
    - run the deep_feature_extraction_pipeline notebook on the train,val and test set (for the dataset you choose earlier)
    - run the torch_model_on_rep (specifying the correct dataset)
    - run evaluation on public set (validation) to select the best patch aggregation metric
    - run evaluation on private set to get the results for the model on the selected dataset 
To use a model for prediction do the following steps:
    - 