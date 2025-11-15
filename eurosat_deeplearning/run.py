# %% [markdown]
# # EuroSat Image Classification using Vision Transformer (ViT)
# 
# Hector Becerra  
# Juan Terven  
# 2023
# 
# The **Vision Transformer (ViT)** is a pioneering neural network architecture...

# %%
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
import numpy as np 
import torch
import torchvision.transforms as transforms
import torch.nn as nn
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import wandb
from torch.utils.data import DataLoader
import json
from torchvision.models import vit_l_32,ViT_L_32_Weights
from torchvision.models import vit_b_32, vit_l_32, ViT_B_32_Weights, ViT_L_32_Weights

from train_utils import get_predictions,compute_val_loss,EarlyStopping,train_model
from eval_utils import remove_wandb_hooks,evaluate_test_set,precision_recall_analysis
# 确保您已经按照新脚本的指示更新了 EuroSAT.py 文件
from EuroSAT import EurosatDataset,visualize_classes

# %% [markdown]
# ## Login to Wandb to log experiment

# %%
os.environ['WANDB_NOTEBOOK_NAME'] = 'EuroSat_ViT_Classifier.ipynb'

wandb.login()

# %% [markdown]
# # Data
# The [EuroSat Dataset]...

# %% [markdown]
# ### **--- 代码修改区域 1: 更新数据路径 ---**
# 根据您的 `data_preprocessing_new.ipynb` 脚本，我们将路径分为 `data_path`（指向图片文件夹）和 `metadata_path`（指向包含CSV和JSON文件的文件夹）。

# %%
# 图片所在的根文件夹
# 注意：根据您的新脚本，这里应该指向 EuroSAT_RGB 文件夹
data_path = "/data1/wwx/EuroSat_classification/dataset/EuroSAT_RGB"

# CSV和JSON文件所在的文件夹
# 这是由您的预处理脚本生成的
metadata_path = "/data1/wwx/EuroSat_classification/dataset/split_info"

# %% [markdown]
# ### **--- 代码修改区域 2: 更新标签映射文件的加载路径 ---**

# %%
# Load the labels from the new metadata_path
with open(os.path.join(metadata_path, "label_map.json"), "r") as f:
    label_to_index = json.load(f)

# %%
index_to_label = {v: k for k, v in label_to_index.items()}
print(index_to_label)

# %% [markdown]
# ## Data Transformations

# %%
# Check the transformations used in the pre-trained model
weights = ViT_L_32_Weights.DEFAULT
preprocess = weights.transforms()
print(preprocess)

# %%
# Define a sequence of transformations to be applied to images
transformToTensor = transforms.Compose([
    transforms.Resize((256,)),
    transforms.CenterCrop((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# %% [markdown]
# ### **--- 代码修改区域 3: 更新数据集的创建方式 ---**
# 现在我们使用 `csv_path` 参数来指定 `train.csv`, `valid.csv` 和 `test.csv` 的位置，而不是使用 `_type`。

# %%
# Create dataset instances
# Instantiate the EurosatDataset class using the new csv_path argument.
train_dataset = EurosatDataset(
    csv_path=os.path.join(metadata_path, 'train.csv'),
    transform=transformToTensor,
    data_path=data_path
)

valid_dataset = EurosatDataset(
    csv_path=os.path.join(metadata_path, 'valid.csv'),
    transform=transformToTensor,
    data_path=data_path
)

test_dataset = EurosatDataset(
    csv_path=os.path.join(metadata_path, 'test.csv'),
    transform=transformToTensor,
    data_path=data_path
)

# Define batch sizes for training and validation data.
train_batch = 1024
val_batch = 1024 

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=train_batch, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=val_batch, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# %%
visualize_classes(train_loader, index_to_label)

# %%
dataset_managers = {
    'valid': valid_loader,
    'train': train_loader,
    'test': test_loader
}

print(f"Train set batches: {len(train_loader)}")
print(f"Validation set batches: {len(valid_loader)}")
print(f"Test set examples: {len(test_dataset)}")

# %% [markdown]
# ## Labels distribution

# %%
# Make sure to use the .data attribute from your updated EurosatDataset class
labels = train_dataset.data['Label'].values

# Convert numerical labels to string names
label_names = [index_to_label[label] for label in labels]

sns.set(style="whitegrid")
plt.figure(figsize=(12, 6))

ax = sns.countplot(x=label_names, hue=label_names, palette="viridis", legend=False)

plt.ylabel('Number of images', fontsize=12)
plt.title('Distribution of Classes in the Training Set', fontsize=16)
plt.xticks(rotation=45, fontsize=10)

for p in ax.patches:
    ax.annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='center', fontsize=10, color='black', xytext=(0, 5),
                textcoords='offset points')

plt.tight_layout()
plt.show()

# %% [markdown]
# # Model

# %%
# Use un-pretrained model
model = vit_l_32()

# Replace the model head with the number of classes
num_classes = 10
model.heads.head = nn.Linear(1024, num_classes) 
model = model.cuda() 
# model.to(DEVICE)
if torch.cuda.device_count() > 1:
    print(f"使用 {torch.cuda.device_count()} 个GPU")
    model = nn.DataParallel(model)

# # 使用更小的 vit_b_32 (基础版) 模型，而不是 vit_l_32 (大型版)
model = vit_b_32()

# Replace the model head with the number of classes
num_classes = 10
# 注意：基础版(Base)的 head 输入维度是 768，不是大型版(Large)的 1024
model.heads.head = nn.Linear(768, num_classes) 
# model.to(DEVICE)

# %% [markdown]
# # Training model from Scratch

# %%
project_name = "EuroSAT"
model_name = "ViT32"
models_path = "./models/" # 建议使用相对路径

# #add by wky
project_name = "EuroSAT"
model_name = "ViT32-Base-Scratch" # 增加了 "-Base-Scratch" 以表明是基础版且从零开始
models_path = "./models/" # 保持相对路径

# %%
epochs = 20
lr = 0.0001
patience = 10

run = wandb.init(
    project=project_name,
    name=model_name,
    notes="ViT32 from scratch",
    config={
        "learning_rate": lr,
        "epochs": epochs,
        "batch_size": train_batch,
        "patience": patience
    })



loss_i, loss_val_i = train_model(model=model, epochs=epochs, 
                                 train_loader=train_loader,
                                 valid_loader=valid_loader,
                                 lr=lr, patience=patience, 
                                 model_name=model_name)

# %% [markdown]
# ## Save model

# %%
# os.makedirs(models_path, exist_ok=True)
# model_save_path = os.path.join(models_path, model_name + ".pth")
# print(f"Saving model to: {model_save_path}")
# torch.save(obj=model.state_dict(), f=model_save_path)

# %%
# with open(f"{models_path}/loss_{model_name}.txt", "w") as file:
#     file.write(f"loss_i: {loss_i}\n")
#     file.write(f"loss_val_i: {loss_val_i}\n")

# %%
# plt.figure(figsize=(7,5))
# plt.plot(loss_i, label="Train Loss", linestyle='--', alpha=1.0)
# plt.plot(loss_val_i, label="Validation Loss", linestyle='-', alpha=1.0)
# plt.legend()
# plt.show()

# %% [markdown]
# ## (以下代码块保持不变，继续进行评估和后续训练)...

# %% [markdown]
# # Evaluate model

# %%
# Call this function before evaluating your model to remove any wandb hook
#remove_wandb_hooks(model)

# %% [markdown]
# ## Load trained model

# %%
# Load the state_dict of our saved model (this will update the new instance of our model with trained weights)

model_save_path = models_path + f"{model_name}.pth"
print(f"Loading model {model_save_path}")
model.load_state_dict(torch.load(f = model_save_path))

# %%
DEVICE = torch.device('cuda') 
evaluate_test_set(model, test_loader, DEVICE, index_to_label=index_to_label)

# %%
precision_recall_analysis(model, test_loader, DEVICE,
                          output_path=models_path,
                          model_name=model_name,
                          index_to_label=index_to_label)

# %%
wandb.finish()

# %% [markdown]
# ## Load the precision/recall curve and plot it

# %%
filename = f"{model_name}_precision_recall_values.json"
file_path = os.path.join(models_path, filename)

with open(file_path, 'r') as file:
    precision_recall_data = json.load(file)
    
all_classes_data = precision_recall_data['All Classes']

# Extract precision and recall values
precisions = all_classes_data.get('precision', [])
recalls = all_classes_data.get('recall', [])

# Create a DataFrame for plotting
df = pd.DataFrame({
    'Precision': precisions,
    'Recall': recalls
})

# Plotting the precision/recall curve
plt.figure(figsize=(10, 6))
sns.lineplot(data=df, x='Recall', y='Precision')
plt.title('Precision/Recall Curve')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.grid(True)
plt.show()

# %% [markdown]
# ## Clean-up

# %%
import gc

del model
torch.cuda.empty_cache()
gc.collect()


# %% [markdown]
# # Now train the model with pre-trained weights

# %%
# Use pretrained model
model2 = vit_l_32(weights=ViT_L_32_Weights.IMAGENET1K_V1)
model2 = model2.cuda()
# Replace the model head with the number of classes
num_classes = 10
model2.heads.head = nn.Linear(1024, num_classes) 

model2.to(DEVICE)
if torch.cuda.device_count() > 1:
    print(f"使用 {torch.cuda.device_count()} 个GPU")
    model2 = nn.DataParallel(model2)

# %% [markdown]
# ## Train with pre-trained weights

# %%
model_name = "ViT32-Pretrained"

# %%
epochs = 10
lr = 0.0001
patience = 10

run = wandb.init(
    # Set the project where this run will be logged
    project=project_name,
    name=model_name,
    notes="ViT32 pre-trained on ImageNet",
    # Track hyperparameters and run metadata
    config={
        "learning_rate": lr,
        "epochs": epochs,
        "batch_size": train_batch,
        "patience": patience
    })

loss_i, loss_val_i = train_model(model=model2, epochs=epochs, 
                                 train_loader=train_loader,
                                 valid_loader=valid_loader,
                                 lr=lr, patience=patience,
                                 model_name=model_name)

# %% [markdown]
# ## Save model

# %%
# Create models directory
os.makedirs(models_path, exist_ok=True)

# 2. Create model save path
model_save_path = os.path.join(models_path, model_name + ".pth")

# 3. Save the model state dict
print(f"Saving model to: {model_save_path}")
torch.save(obj=model2.state_dict(), # only saving the state_dict() only saves the models learned parameters
           f=model_save_path)

# %%
# Save the loss data in a file
with open(f"{models_path}/loss_{model_name}.txt", "w") as archivo:
    archivo.write(f"loss_i: {loss_i}\n")
    archivo.write(f"loss_val_i: {loss_val_i}\n")

# %%
plt.figure(figsize=(7,5))
# Use a dashed line for the training loss, and add markers
plt.plot(loss_i, label="Train Loss", linestyle='--', alpha=1.0)

# Use a solid line for the validation loss
plt.plot(loss_val_i, label="Validation Loss", linestyle='-', alpha=1.0)
plt.legend()
plt.show()

# %% [markdown]
# # Evaluate model

# %%
# Call this function before evaluating your model to remove any wandb hook
#remove_wandb_hooks(model)

# %% [markdown]
# ## Load trained model

# %%
# Load the state_dict of our saved model (this will update the new instance of our model with trained weights)
model_save_path = models_path + f"{model_name}.pth"
print(f"Loading model {model_save_path}")
model2.load_state_dict(torch.load(f = model_save_path))

# %%
evaluate_test_set(model2, test_loader, DEVICE, index_to_label=index_to_label)

# %%
precision_recall_analysis(model2, test_loader, DEVICE,
                          output_path=models_path,
                          model_name=model_name,
                          index_to_label=index_to_label)

# %%
wandb.finish()

# %%
del model2
torch.cuda.empty_cache()
gc.collect()

# %%



