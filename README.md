# Image Regression

by [Tony Assi](https://www.tonyassi.com/)

Image Regression model training and inference. Built with 🤗 and PyTorch.

## Installation
```bash
pip install -r requirements.txt
```

## Usage

Import 
```python
from ImageRegression import train_model, upload_model, predict
```

Train model
- **dataset_id** 🤗 Dataset ID
- **value_column_name** column name of the dataset. these are the prediction values
- **test_split** test split of the train/test split
- **output_dir** the directory where the checkpoints will be saved
- **num_train_epochs**
```python
train_model(dataset_id='tonyassi/sales1',
            value_column_name='sales',
            test_split=0.2,
            output_dir='./results',
            num_train_epochs=10,
            learning_rate=1e-4)

```
- **model_ckpt** if this parameter is not specified then [google/vit-base-patch16-224](https://huggingface.co/google/vit-base-patch16-224) will be used by default
```python
create_dataset_embeddings(input_dataset='tonyassi/fashion-decade-images-1',
                          output_dataset='tonyassi/fashion-decade-images-1-embeddings',
                          token='YOUR_TOKEN',
                          model_ckpt='tonyassi/fashion-clothing-decade')

```
