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
