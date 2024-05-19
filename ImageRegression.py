from datasets import load_dataset, DatasetDict
from PIL import Image
import torch
from torchvision import transforms
from transformers import ViTModel, TrainingArguments, Trainer
from torch import nn
from torch.utils.data import DataLoader
from safetensors.torch import load_file as safetensors_load_file
from huggingface_hub import create_repo, HfApi
import logging
import os
import json
import shutil

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ViTRegressionModel(nn.Module):
    def __init__(self):
        super(ViTRegressionModel, self).__init__()
        self.vit = ViTModel.from_pretrained('google/vit-base-patch16-224')
        self.classifier = nn.Linear(self.vit.config.hidden_size, 1)

    def forward(self, pixel_values, labels=None):
        outputs = self.vit(pixel_values=pixel_values)
        cls_output = outputs.last_hidden_state[:, 0, :]  # Take the [CLS] token
        values = self.classifier(cls_output)
        loss = None
        if labels is not None:
            loss_fct = nn.MSELoss()
            loss = loss_fct(values.view(-1), labels.view(-1))
        return (loss, values) if loss is not None else values



def train_model(dataset_id, value_column_name, test_split, output_dir, num_train_epochs, learning_rate):
    # Load the dataset
    dataset = load_dataset(dataset_id)

    # Split the dataset into train and test
    train_test_split = dataset['train'].train_test_split(test_size=test_split)
    dataset = DatasetDict({
        'train': train_test_split['train'],
        'test': train_test_split['test']
    })

    # Define a transform to convert PIL images to tensors
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    # Get max value
    train_values = dataset['train'][value_column_name]
    test_values = dataset['test'][value_column_name]
    max_value = max(train_values + test_values)
    print('Max Value:', max_value)


    def preprocess(example):
        example['image'] = transform(example['image'])
        example[value_column_name] = example[value_column_name] / max_value  # Normalize values
        return example

    # Apply the preprocessing with normalization
    dataset = dataset.map(preprocess, batched=False)


    def collate_fn(batch):
        # Ensure that each item['image'] is a tensor
        pixel_values = torch.stack([torch.tensor(item['image']) for item in batch])
        labels = torch.tensor([item[value_column_name] for item in batch], dtype=torch.float).unsqueeze(1)
        return {'pixel_values': pixel_values, 'labels': labels}


    model = ViTRegressionModel()

    training_args = TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="epoch",
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=num_train_epochs,
        learning_rate=learning_rate,
        save_steps=10,
        save_total_limit=2,
        logging_steps=10,
        remove_unused_columns=False,
        resume_from_checkpoint=True,
    )

    train_dataloader = DataLoader(dataset['train'], batch_size=8, shuffle=True, collate_fn=collate_fn)
    eval_dataloader = DataLoader(dataset['test'], batch_size=8, shuffle=False, collate_fn=collate_fn)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset['train'],
        eval_dataset=dataset['test'],
        data_collator=collate_fn,
    )

    # Add logging to inspect the model outputs and labels
    def compute_metrics(p):
        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        labels = p.label_ids
        logger.info(f"Predictions: {preds[:5]}")
        logger.info(f"Labels: {labels[:5]}")
        mse = ((preds - labels) ** 2).mean().item()
        return {"mse": mse}

    trainer.compute_metrics = compute_metrics

    trainer.train()
    eval_results = trainer.evaluate()
    print(f"Evaluation results: {eval_results}")

    # Write jSON file
    data = {
        "dataset_id": dataset_id,
        "value_column_name": value_column_name,
        "test_split": test_split,
        "num_train_epochs": num_train_epochs,
        "learning_rate": learning_rate,
        "max_value": max_value,
    }
    filename = 'metadata.json'
    # Traverse the directory tree starting from the current directory
    for root, dirs, files in os.walk(output_dir):
        for dir_name in dirs:
            if 'checkpoint' in dir_name:
                # Construct the full path to the target directory
                dir_path = os.path.join(root, dir_name)
                # Construct the full path to the JSON file in the target directory
                file_path = os.path.join(dir_path, filename)
                # Write the JSON data to the file
                with open(file_path, 'w') as file:
                    json.dump(data, file, indent=4)
                print(f'Data successfully written to {file_path}')


def upload_model(model_id, token, checkpoint_dir):
    # Create a new repo
    repo_url = create_repo(model_id, token=token, repo_type='model')
    print(repo_url)
    repo_id = "/".join(repo_url.split("/")[3:])
    print(repo_id)

    # Copy README template to checkpoint folder
    readme_path = checkpoint_dir + '/README.md'
    shutil.copy('README-template.md',  readme_path)

    # Read metadata.json
    with open(checkpoint_dir + '/metadata.json', 'r') as f:
        metadata = json.load(f)

    # Read README-template.md file
    with open(readme_path, 'r') as f:
        readme_content = f.read()

    # Replace values of README file
    updated_readme_content = readme_content.replace('- \"-\"', f'- \"{metadata.get("dataset_id", "")}\"')
    updated_readme_content = updated_readme_content.replace('- name: "-"', f'- name: \"{model_id}\"')
    updated_readme_content = updated_readme_content.replace('# Title', f'# {model_id}')
    updated_readme_content = updated_readme_content.replace("repo_id='-'", f"repo_id=\'{repo_id}\'")
    updated_readme_content = updated_readme_content.replace("Dataset:", f"Dataset: {metadata.get('dataset_id', '')}")
    updated_readme_content = updated_readme_content.replace("Value Column:", f"Value Column: \'{metadata.get('value_column_name', '')}\'")
    updated_readme_content = updated_readme_content.replace("Train Test Split:", f"Train Test Split: {metadata.get('test_split', '')}")
    updated_readme_content = updated_readme_content.replace("Epochs:", f"Epochs: {metadata.get('num_train_epochs', '')}")
    updated_readme_content = updated_readme_content.replace("Learning Rate:", f"Learning Rate: {metadata.get('learning_rate', '')}")
    updated_readme_content = updated_readme_content.replace("dataset_id='-'", f"dataset_id=\'{metadata.get('dataset_id', '')}\'")
    updated_readme_content = updated_readme_content.replace("value_column_name='-'", f"value_column_name=\'{metadata.get('value_column_name', '')}\'")
    updated_readme_content = updated_readme_content.replace("test_split=-", f"test_split={metadata.get('test_split', '')}")
    updated_readme_content = updated_readme_content.replace("num_train_epochs=-", f"num_train_epochs={metadata.get('num_train_epochs', '')}")
    updated_readme_content = updated_readme_content.replace("learning_rate=-", f"learning_rate={metadata.get('learning_rate', '')}")
    updated_readme_content = updated_readme_content.replace("model_id='-'", f"model_id=\'{model_id}\'")


    # Write the updated content back to README.md file
    with open(readme_path, 'w') as f:
        f.write(updated_readme_content)

    # Upload files to hub
    api = HfApi()
    api.upload_folder(
        folder_path=checkpoint_dir,
        repo_id=repo_id,
        repo_type="model",
        token=token,
    )

def predict(repo_id, image_path):
    # If not already downloaded, download model and metadata
    if(not os.path.exists('./model.safetensors')):
        api = HfApi()
        api.hf_hub_download(repo_id=repo_id, local_dir='.', filename="model.safetensors")
    if(not os.path.exists('./metadata.json')):
        api = HfApi()
        api.hf_hub_download(repo_id=repo_id, local_dir='.', filename="metadata.json")
    
    model = ViTRegressionModel()

    # Load the saved model checkpoint
    checkpoint_path = "./model.safetensors"
    state_dict = safetensors_load_file(checkpoint_path)
    model.load_state_dict(state_dict)
    model.eval()

    # get max value
    with open('./metadata.json', 'r') as file:
        data = json.load(file)
    max_value = data.get('max_value')

    # Define a transform to convert PIL images to tensors
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    # Load and preprocess the image
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0)  # Add batch dimension

    with torch.no_grad():
        # Run the model
        prediction = model(image)

    # De-normalize the prediction
    prediction = prediction.item() * max_value
    return prediction