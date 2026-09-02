# Comparing Vision Transformers and CNNs for Sex Classification from Handwriting [![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![Conda Environment](https://img.shields.io/badge/environment-conda:yaml-green.svg)](#environment)
[![Hardware](https://img.shields.io/badge/hardware-NVIDIA_RTX_4060-orange.svg)](#environment)

## 📝 Description

This repository contains the official code implementation for the paper:  
🔬 **"Comparing Vision Transformers and CNNs for Sex Classification from Handwriting"**

> 🏆 **Key Finding:** Vision Transformers (ViTs) consistently outperform CNN architectures as feature extractors for handwriting classification. By leveraging features from `clip-vit-large`, this approach achieves a page-level classification accuracy of **80.1%**—matching specialized, task-specific state-of-the-art architectures in current literature.

---

### 🔗 Project Resources

* 📄 **Manuscript:** [Download Full Paper PDF](./docs/paper-pre-submission.pdf) *(Pre-submission draft / Local Copy)*
* 📊 **Dataset:** [ICDAR2013 "Gender Prediction from Handwriting" Dataset (QUWI)](https://www.kaggle.com/competitions/icdar2013-gender-prediction-from-handwriting/data)  
  * *Note: The ICDAR2013 handwriting dataset is a standard multilingual baseline compiled from 475 unique writers, featuring both Arabic and English text samples.*
  * The dataset can be also found here: https://tc11.cvc.uab.es/datasets/GenderIdentifify2013_1

---

### 📄 Abstract

Handwriting is a biometric marker, encoding information regarding an individual's demographic traits, psychological state, and neurological health. Advances in machine learning enable the automatic analysis of handwriting, with potential applications in fields like forensics and healthcare. 

Because data for handwriting classification tasks is often limited, pre-trained deep learning vision models are frequently used instead of training from scratch. While Vision Transformers (ViTs) are increasingly adopted as pre-trained baselines across various domains, their application to handwriting classification remains under-explored. 

To address this, we present a **comparative analysis of ViTs and CNNs architectures** as feature extractors for the task of sex classification, utilizing the publicly available **ICDAR2013 "Gender Prediction from Handwriting"** dataset. We explore a wide array of pre-trained models, differing in architecture, pre-training data, and pre-training objectives. 

Our results demonstrate that **models based on ViT features consistently outperform those based on CNN features**. Notably, exploiting features extracted from `clip-vit-large`, we achieve a page-level classification accuracy of **80.1%**, which is highly competitive with the best results in the literature achieved via more task-specific approaches.

---

## 🚀 Using the Code

Note that when using ipynb files you usually should run the "easy_acces" or "reload" group of cells (last heading in each notebook)

### 1. Training a specific pipeline from scratch
You can use the notebooks in the notebooks/pipeline_single_model folder to train a pipeline on a specific dataset (IAM or ICDAR) with a specific vision extractor.
A readme.txt contains more detailed information
The available models are the one in utils/model_utils.py -> get_model function.
You first have to download the ICDAR13 or IAM dataset yourself to use the code (and use the same directory structure assumed in the notebook)
You can find the IAM (online) dataset at: https://fki.tic.heia-fr.ch/databases/iam-on-line-handwriting-database 

### 2. Reproducing paper experiments
a) Generating the patch datasets: 
Use the notebooks in notebooks/add_new_dataset for creating a new patch dataset (a csv with a row for each patch and columns that define the position of the patch in the page, the full image path, the page metadata and the patch properties) and its .zarr version
(.zarr format makes the deep feature extraction from the patch dataset fast) -> this saves the csv in the outputs\\preprocessed_data folder 
and the zarr in a folder of choice (you can set the path, eg save it with the icdar image dataset); This also saves metadata about the csv file in an
output/preprocessed_data/file_metadata_log.json file.

b) Extact features with the vision encoders:
Use the notebooks/key_notebooks/deep_feature_extraction.ipynb notebook to extract representations for the desired folders
When file_metadata_log.json is read, a unique name is associated to each patches dataset (eg 'squares_gw5.0_m5.0_idx2' is the standard squares set 
discussed in the paper and 'body_gwnan_m1.0_idx1' is the standard body) -> identify correctly the name of the dataset you want to use
-> this saves a csv file for each specified vision encoder in output/preprocessed_data/ (containing the same patch data as before and for each patch the extracted features, one column per feature)
The name of the dataset and its properties are logged in file_metadata_log.json

c) Run full model comparison
- Use model_comparison_experiment/model_comparison_experiment_prepararion.ipynb notebook to prepare the list of experiments to run ->
file_metadata_log.json is read, as before choose the patch datasets you want to consider (eg 'squares_gw5.0_m5.0_idx2' and 'body_gwnan_m1.0_idx1').
All the files generated at the step b from this dataset will be put in the experiment (experiment is saved as a pkl file inside notebooks/model_comaprison../experiment_tables)
- Run the experiments_scheduler.py script in the model_comparison_experiment/ folder to run the comparison (select the correct filepath for the experiment)  -> for each model the cross validaiton results are saved in the usual file_metadata_log.json file.
- Use the experiment_analysis notebook to inspect results (to select the correct experiments you should specify the name of the experiment file
same as first step in this section c)

d) Run selected model comparison using the notebooks/key_notebooks/sklearn_model_on_rep notebook (probably you will have to change the filenames
in the file_IO.load_input_files function, now it is set to load the file saved at step c for the selected models on my last experiment file)

e) Similarly as 1 you can train the final models with the notebooks in notebooks/key_notebooks using torch_model_on_rep.ipynb

### Note
- Much of the code is not useful anymore (residuals from experiments not reported in the paper, e.g. fine-tuning) and in general is a bit messy. 
I will take time to clean the repository soon :-)

---

## 🧪 Reproducibility

### Environment & Hardware

* **OS/Hardware:** Developed and trained on a **Dell Inspiron 16 Plus**
* **CPU:** Intel(R) Core(TM) Ultra 7 155H @ 3.80 GHz
* **GPU:** NVIDIA GeForce RTX 4060 Laptop GPU (8 GB VRAM)


* **Software Environment:** Package versions and dependencies are explicitly managed via Conda. You can recreate the exact environment using:
```bash
conda env create -f environment.yml
conda activate GeneralPurposeML

```

### Global Hyperparameters (Table 3 Best Models)

Unless specified otherwise per model, all classifier heads share the following training configuration:

| Hyperparameter | Default Value | Notes / Details |
| --- | --- | --- |
| **Data Split** | Train / Val / Test | Calibrated on Val set (if applicable). See paper for details. |
| **Batch Size** | `64` |  |
| **Max Epochs** | `100` | With Early Stopping |
| **Early Stopping** | `2` epochs | Monitors Validation Loss |
| **Loss Function** | Cross-Entropy |  |
| **Optimizer** | Adam | Standard parameters |
| **Learning Rate** | Fixed | Maintained at initial value throughout training |
| **Calibration** | Isotonic Regression | `sklearn.isotonic.IsotonicRegression` (Standard params on male class) |
| **Classification** | Threshold = `0.5` | Output `1` if $p > 0.5$, else `0` (except Majority Voting) |

---

### 🧠 Model-Specific Configurations

Below are the unique architectural configurations and feature extraction strategies utilized for the best-performing models in the paper.

#### 1. `clip-vit-large-patch14-inter`

* **Feature Extractor:** `clip-vit-large-patch14`
* *Target:* Extracted features correspond to the **CLS token of the 12th transformer block** of the vision encoder.
* *Encoder definition:* The feature extractor is defined by the following module with the specified arguments: `src.utils.model_utils.get_clip_vit(pretrained=True, mode='truncated', truncation='remove head', name='clip-vit-large-patch14-inter')`


* **Classification Head:** MLP
* The classifier used on top of the feature extractor is an MLP with one hidden layer and two output neurons. The hidden layer has 256 neurons, relu activation functions are used, after the hidden layer a dropout layer with dropout=0.9 is used, a BatchNorm1d layer is used before the hidden layer to normalize the input data, weighted-ensembling is used as aggregation strategy during prediction. Gradient clipping with a max norm of 1.0 on the loss is used, a learning rate of lr=1e-3 is used -> 
* *Classifier definition:* The feature extractor is defined by the following module with the specified arguments: `src.utils.model_utils.get_classification_head(name='MLPClassifier1', num_classes=2, activation='relu', hidden_sizes=[256], dropout=0.9, batchnorm=False, with_input_norm='batch_norm')`
* **Optimization:** Learning Rate = `1e-3` | Gradient Clipping Max Norm = `1.0`
* **Aggregation Strategy:** Weighted Ensembling

#### 2. `clip-vit-large-patch14-inter (cal)`

* Same configuration as `clip-vit-large-patch14-inter` but with post-training **Isotonic Regression calibration** applied.

#### 3. `clip-vit-large-patch14 (cal)`

* **Feature Extractor:** `clip-vit-large-patch14-un`
* *Target:* **CLS token of the last transformer block**.
* *Encoder definition:* `src.utils.model_utils.get_clip_vit(pretrained=True, mode='truncated', truncation='remove head', name='clip-vit-large-patch14-un')`


* **Classification Head:** MLP 
* *Classifier definition:* `src.utils.model_utils.get_classification_head(name='MLPClassifier1', num_classes=2, activation='relu', hidden_sizes=[128], dropout=0.1, batchnorm=False, with_input_norm='batch_norm')`


* **Optimization:** Learning Rate = `1e-4` | Gradient Clipping enabled
* **Aggregation Strategy:** Weighted Ensembling + Calibration

#### 4. `BEiT-Large-inter (cal)`

* **Feature Extractor:** `BEiT-Large-inter`
* *Target:* **CLS token of the 12th transformer block** of the vision encoder.
* *Encoder definition:* `src.utils.model_utils.get_clip_vit(pretrained=True, mode='truncated', truncation='remove head', name='BEiT-Large-inter')`


* **Classification Head:** MLP 
* *Classifier definition:* `src.utils.model_utils.get_classification_head(name='MLPClassifier1', num_classes=2, activation='relu', hidden_sizes=[16], dropout=0.1, batchnorm=False, with_input_norm='batch_norm')`

* **Optimization:** Learning Rate = `1e-3` | Gradient Clipping enabled
* **Aggregation Strategy:** Majority Voting + Calibration

#### 5. `ResNet50 (cal)`

* **Feature Extractor:** `resnet50`
* *Target:* Output of the last layer before the pre-trained classification head (average pooled and flattened output of the final convolutional block).
* *Encoder definition:* `src.utils.model_utils.get_clip_vit(pretrained=True, mode='truncated', truncation='remove head', name='resnet50')`


* **Classification Head:** MLP 
* *Classifier definition:* `src.utils.model_utils.get_classification_head(name='MLPClassifier1', num_classes=2, activation='relu', hidden_sizes=[16], dropout=0.1, batchnorm=False, with_input_norm='batch_norm')`

* **Optimization:** Learning Rate = `1e-3` | Gradient Clipping enabled
* **Aggregation Strategy:** Majority Voting + Calibration
