# Embed Images in 🤗 Dataset
Generates image embeddings for a 🤗 Dataset then uploads it to the 🤗 Hub

## Installation
```bash
pip install -r requirements.txt
```

## Usage

Import module
```python
from Embed import create_dataset_embeddings
```

Generates image embeddings and upload to 🤗
- **input_dataset** the source dataset
- **out_dataset** the name of dataset that will be created and uploaded to
- **token** HuggingFace write access token can be created [here](https://huggingface.co/settings/tokens)
```python
create_dataset_embeddings(input_dataset='tonyassi/fashion-decade-images-1',
                          output_dataset='tonyassi/fashion-decade-images-1-embeddings',
                          token='YOUR_TOKEN')

```
- **model_ckpt** if this parameter is not specified then [google/vit-base-patch16-224](https://huggingface.co/google/vit-base-patch16-224) will be used by default
```python
create_dataset_embeddings(input_dataset='tonyassi/fashion-decade-images-1',
                          output_dataset='tonyassi/fashion-decade-images-1-embeddings',
                          token='YOUR_TOKEN',
                          model_ckpt='tonyassi/fashion-clothing-decade')

```
