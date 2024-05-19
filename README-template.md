---
license: apache-2.0
base_model: google/vit-base-patch16-224
tags:
- Image Regression
datasets:
- "-"
metrics:
- accuracy
model-index:
- name: "-"
  results: []
---

# Title
## Image Regression Model

This model was trained with [Image Regression Model Trainer](https://github.com/TonyAssi/ImageRegression/tree/main). It takes an image as input and outputs a float value.

```python
from ImageRegression import predict
predict(repo_id='-',image_path='image.jpg')
```

---

## Dataset
Dataset:\
Value Column:\
Train Test Split:

---

## Training
Base Model: [google/vit-base-patch16-224](https://huggingface.co/google/vit-base-patch16-224)\
Epochs:\
Learning Rate:

---

## Usage

### Download
```bash
git clone https://github.com/TonyAssi/ImageRegression.git
cd ImageRegression
```

### Installation
```bash
pip install -r requirements.txt
```

### Import 
```python
from ImageRegression import train_model, upload_model, predict
```

### Inference (Prediction)
- **repo_id** 🤗 repo id of the model
- **image_path** path to image
```python
predict(repo_id='-',
        image_path='image.jpg')
```
The first time this function is called it'll download the safetensor model. Subsequent function calls will run faster.

### Train Model
- **dataset_id** 🤗 dataset id
- **value_column_name** column name of prediction values in dataset
- **test_split** test split of the train/test split
- **output_dir** the directory where the checkpoints will be saved
- **num_train_epochs** training epochs
- **learning_rate** learning rate
```python
train_model(dataset_id='-',
            value_column_name='-',
            test_split=-,
            output_dir='./results',
            num_train_epochs=-,
            learning_rate=-)

```
The trainer will save the checkpoints in the output_dir location. The model.safetensors are the trained weights you'll use for inference (predicton).

### Upload Model
This function will upload your model to the 🤗 Hub.
- **model_id** the name of the model id
- **token** go [here](https://huggingface.co/settings/tokens) to create a new 🤗 token
- **checkpoint_dir** checkpoint folder that will be uploaded
```python
upload_model(model_id='-',
             token='YOUR_HF_TOKEN',
             checkpoint_dir='./results/checkpoint-940')
```