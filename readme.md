# Project Name Placeholder [![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![Conda Environment](https://img.shields.io/badge/environment-conda:yaml-green.svg)](#environment)
[![Hardware](https://img.shields.io/badge/hardware-NVIDIA_RTX_4060-orange.svg)](#environment)

## 📝 Description
Provide a concise summary of your research project, the dataset used, and the core purpose of these machine learning models here.

---

## 🚀 Using the Code

Follow these steps to process the data, extract features, and train/evaluate the models.

### 1. Data Pre-processing
```bash
# Add the command or script used for pre-processing
python src/preprocess.py --data_dir ./data

```

### 2. Feature Extraction

```bash
# Add the command used for extracting representations
python src/extract_features.py --model clip-vit-large-patch14-inter

```

### 3. Model Training

```bash
# Add the command used for training the model
python src/train.py --config config/default.yaml

```

### 4. Evaluation

```bash
# Add the command used for evaluating the model
python src/evaluate.py --model_path ./models/best_model.pt

```

---

## 🧪 Reproducibility

### Environment & Hardware

* **OS/Hardware:** Developed and trained on a **Dell Inspiron 16 Plus**
* **CPU:** Intel(R) Core(TM) Ultra 7 155H @ 3.80 GHz
* **GPU:** NVIDIA GeForce RTX 4060 Laptop GPU (8 GB VRAM)


* **Software Environment:** Package versions and dependencies are explicitly managed via Conda. You can recreate the exact environment using:
```bash
conda env create -f environment.yml
conda activate <env_name>

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
* *Target:* **CLS token of the 12th transformer block** of the vision encoder.
* *Module Path:* `src.utils.model_utils.get_clip_vit(pretrained=True, mode='truncated', truncation='remove head', name='clip-vit-large-patch14-inter')`


* **Classification Head:** MLP (`src.utils.model_utils.get_classification_head`)
* *Parameters:* `name='MLPClassifier1'`, `num_classes=2`, `activation='relu'`, `hidden_sizes=[256]`, `dropout=0.9`, `batchnorm=False`, `with_input_norm='batch_norm'`


* **Optimization:** Learning Rate = `1e-3` | Gradient Clipping Max Norm = `1.0`
* **Aggregation Strategy:** Weighted Ensembling

#### 2. `clip-vit-large-patch14-inter (cal)`

* Same configuration as `clip-vit-large-patch14-inter` but with post-training **Isotonic Regression calibration** applied.

#### 3. `clip-vit-large-patch14 (cal)`

* **Feature Extractor:** `clip-vit-large-patch14-un`
* *Target:* **CLS token of the last transformer block**.
* *Module Path:* `src.utils.model_utils.get_clip_vit(pretrained=True, mode='truncated', truncation='remove head', name='clip-vit-large-patch14-un')`


* **Classification Head:** MLP (`src.utils.model_utils.get_classification_head`)
* *Parameters:* `name='MLPClassifier1'`, `num_classes=2`, `activation='relu'`, `hidden_sizes=[128]`, `dropout=0.1`, `batchnorm=False`, `with_input_norm='batch_norm'`


* **Optimization:** Learning Rate = `1e-4` | Gradient Clipping enabled
* **Aggregation Strategy:** Weighted Ensembling + Calibration

#### 4. `BEiT-Large-inter (cal)`

* **Feature Extractor:** `BEiT-Large-inter`
* *Target:* **CLS token of the 12th transformer block** of the vision encoder.
* *Module Path:* `src.utils.model_utils.get_clip_vit(pretrained=True, mode='truncated', truncation='remove head', name='BEiT-Large-inter')`


* **Classification Head:** MLP (`src.utils.model_utils.get_classification_head`)
* *Parameters:* `name='MLPClassifier1'`, `num_classes=2`, `activation='relu'`, `hidden_sizes=[16]`, `dropout=0.1`, `batchnorm=False`, `with_input_norm='batch_norm'`


* **Optimization:** Learning Rate = `1e-3` | Gradient Clipping enabled
* **Aggregation Strategy:** Majority Voting + Calibration

#### 5. `ResNet50 (cal)`

* **Feature Extractor:** `resnet50`
* *Target:* Output of the last layer before the pre-trained classification head (average pooled and flattened output of the final convolutional block).
* *Module Path:* `src.utils.model_utils.get_clip_vit(pretrained=True, mode='truncated', truncation='remove head', name='resnet50')`


* **Classification Head:** MLP (`src.utils.model_utils.get_classification_head`)
* *Parameters:* `name='MLPClassifier1'`, `num_classes=2`, `activation='relu'`, `hidden_sizes=[16]`, `dropout=0.1`, `batchnorm=False`, `with_input_norm='batch_norm'`


* **Optimization:** Learning Rate = `1e-3` | Gradient Clipping enabled
* **Aggregation Strategy:** Majority Voting + Calibration
