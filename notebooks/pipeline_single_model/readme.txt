To train and evaluate a model follwos these steps:
    - decide which dataset you want to use: icdar, iam
        - if you use icdar run the prepare_icdar_table, the patches_cc and the save_as_zarr notebooks in this order
        - if you use iam run the prepare_iam_table, the patches_cc and the save_as_zarr notebooks in this order
    - run the deep_feature_extraction_pipeline notebook on the train,val and test set (for the dataset you choose earlier)
        -> this will save the extracted features in a model/{model_name} folder
    - run the torch_model_on_rep (specifying the correct dataset)
        -> this will save the mlp weights and training info in the model/{model_name}/{name_of_the_classification_head}/{dataset_name}/checkpoints folder
    - run evaluation on public set (validation) to select the best patch aggregation metric
    - run evaluation on private set to get the results for the model on the selected dataset
        -> this will save model's results in the  model/{model_name}/{name_of_the_classification_head}/{dataset_name}/current folder
    - compute explanation maps running the explainability_standalone script (tested only on the clip-vit models)
        -> this will save the explanation maps in subfolders of the model/{model_name}/{name_of_the_classification_head}/{dataset_name}/current folder folder
To use a trained model for prediction do the following steps:
    - put your image in custom. Number it followin the NNNN_X format where NNNN will be the writer index and X=0 for female and 1 for model_name
    - run prediction_script.ipynb with appropriate settings