# Image Regression

![diagram2](https://github.com/TonyAssi/ImageRegression/assets/42156881/650a6b86-4d40-4bd0-b8e8-fa1e0e89388e)

by [Tony Assi](https://www.tonyassi.com/)

Image Regression model training and inference. The trainer fine-tunes [google/vit-base-patch16-224](https://huggingface.co/google/vit-base-patch16-224) with a custom PyTorch model. You can upload the model to 🤗 Hub and use a simple function for inference. Built with 🤗 and PyTorch.

## Installation
```bash
pip install -r requirements.txt
```

## Usage

### Import 
```python
from ImageRegression import train_model, upload_model, predict
```

### Train Model
- **dataset_id** 🤗 dataset id
- **value_column_name** column name of the dataset. these are the prediction values
- **test_split** test split of the train/test split
- **output_dir** the directory where the checkpoints will be saved
- **num_train_epochs** training epochs
- **learning_rate** learning rate
```python
train_model(dataset_id='tonyassi/sales1',
            value_column_name='sales',
            test_split=0.2,
            output_dir='./results',
            num_train_epochs=10,
            learning_rate=1e-4)

```
The trainer will save the checkpoints in the output_dir location. The "model.safetensors" are the trained weights you'll use for inference (predicton).

### Upload Model
This function will upload your model to the 🤗 Hub, which will be useful for inference.
```python
upload_model(model_id='sales-prediction1',
             token='YOUR_HF_TOKEN',
             checkpoint_dir='./results/checkpoint-940')
```

## Dataset

The model trainer takes a 🤗 dataset id as input so your dataset must be uploaded to 🤗. It should have a column of images and a column of values (floats or ints). Check out [Create an image dataset](https://huggingface.co/docs/datasets/en/image_dataset) if you need help creating a 🤗 dataset. Your dataset should look like [tonyassi/sales1](https://huggingface.co/datasets/tonyassi/sales1) (the values column can be named whatever you'd like).

<img width="868" alt="Screenshot 2024-05-18 at 12 11 32 PM" src="https://github.com/TonyAssi/ImageRegression/assets/42156881/06ed6954-de6f-45ab-84a3-57781d39722b">
