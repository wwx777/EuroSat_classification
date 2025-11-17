
import torch
from torch.utils.data import Dataset,DataLoader
from PIL import Image
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from torch.utils.data import Dataset

#原作者的類
# Define a class EurosatDataset that inherits from Dataset.
# class EurosatDataset(Dataset):
#     # Constructor method for the class.
#     def __init__(self, _type, transform, data_path):
        
#         # A dictionary mapping dataset types (valid, test, train) to their respective file paths.
#         type_to_folder = {
#             "valid": f"{data_path}validation.csv",
#             "test": f"{data_path}test.csv",
#             "train": f"{data_path}train.csv"
#         }
        
#         # Read the CSV file for the given type (valid, test, or train) using pandas and store the data.
#         self.data = pd.read_csv(type_to_folder[_type])

#         # Store the base path to the folder containing the data.
#         self.folder_base = data_path
        
#         self.transform = transform

#     # Method to return the length of the dataset.
#     def __len__(self):
#         # Return the number of rows in the data.
#         return len(self.data)

#     # Method to get an item at a specific index from the dataset.
#     def __getitem__(self, idx):
#         # Convert the index to a list if it's a tensor.
#         if torch.is_tensor(idx):
#             idx = idx.tolist()

#         # Extract the filename (X) and label (Y) from the data at the given index.
#         X, Y = self.data.iloc[idx].Filename, self.data.iloc[idx].Label

#         # Convert the image path to a tensor.
#         X = self.path_to_tensor(self.folder_base + X)

#         # Convert the label to a long tensor.
#         Y = torch.tensor(Y, dtype=torch.long)

#         # Return the image tensor and its corresponding label.
#         return X, Y

#     # Method to convert an image path to a tensor.
#     def path_to_tensor(self, path):
#         # Open the image file.
#         img = Image.open(path)

#         # Apply the predefined transformations to the image.
#         img_transformed = self.transform(img)

#         # Permute the dimensions of the tensor for compatibility with PyTorch.
#         return img_transformed.permute(1, 2, 0)

#add by wky 修改类 直接跑VIT
# class EurosatDataset(Dataset):
#     def __init__(self, csv_path, data_path, transform=None):
#         self.data = pd.read_csv(csv_path)
#         self.data_path = data_path
#         self.transform = transform

#     def __len__(self):
#         return len(self.data)

#     def __getitem__(self, idx):
#         img_rel_path = self.data.iloc[idx, 0]
#         img_abs_path = os.path.join(self.data_path, img_rel_path)
#         label = self.data.iloc[idx, 1]
#         image = Image.open(img_abs_path).convert("RGB")
#         if self.transform:
#             image = self.transform(image)
#         return image, label

#add by wky 跑rgb+光譜融合才需要這樣修改
class EurosatDataset:
    def __init__(self, csv_path, data_path, transform=None):
        self.data = pd.read_csv(csv_path)
        self.data_path = data_path
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # 1. 将路径中的 .jpg 替换为 .pt
        img_rel_path = self.data.loc[idx, 'ImagePath'].replace('.jpg', '.pt')
        img_path = os.path.join(self.data_path, img_rel_path)
        
        # 2. 直接使用 torch.load() 加载 Tensor
        image = torch.load(img_path)
        
        label = self.data.loc[idx, 'Label']

        if self.transform:
            # 3. 此时的 transform 只用于归一化 (Normalize)
            image = self.transform(image)
            
        return image, label

# (visualize_classes 函数如果还需要使用，也需要相应修改，但对于训练不是必需的)

def visualize_classes(data_loader, index_to_label):
    # Set Seaborn's aesthetic parameters for a more polished look.
    sns.set(style="whitegrid", context="notebook")

    # Create a 2x5 grid of subplots with a suitable figure size.
    fig, axs = plt.subplots(2, 5, figsize=(10, 5))

    # Adjust the space between the plots for better visibility.
    fig.subplots_adjust(hspace=0.3, wspace=0.3)

    # Dictionary to keep track of whether an image of each class has been found.
    found_classes = {}

    # Iterate over the batches in the data loader.
    for images, labels in data_loader:
        # Iterate over each image and its label in the current batch.
        for img, label in zip(images, labels):
            label_item = label.item()
            # If the image of this class hasn't been found yet, store it.
            if label_item not in found_classes:
                found_classes[label_item] = img

            # If images for all 10 classes have been found, break out of the loop.
            if len(found_classes) == 10:
                break

        # Check again outside the inner loop to break from the outer loop.
        if len(found_classes) == 10:
            break

    # Now that we have one image for each class, display them on the grid.
    for i, (label, img) in enumerate(found_classes.items()):
        if img.shape[0] == 3:
            img = img.permute(1, 2, 0)
        elif img.shape[0] == 1:
            img = img.squeeze(0)
        img = img.numpy()
        axs[i // 5, i % 5].imshow(img, cmap='gray' if img.ndim == 2 else None)
        axs[i // 5, i % 5].set_title(index_to_label[label])
        axs[i // 5, i % 5].axis('off')
        sns.despine(ax=axs[i // 5, i % 5], left=True, bottom=True)  # Remove spines for a cleaner look

    plt.tight_layout()
    plt.show()

def visualize_fused_data(data_loader, index_to_label):
    """
    可视化用于数据融合的6通道数据。
    它将分别显示RGB图像和三个光谱指数图。
    """
    # 获取一个批次的数据
    try:
        data_iter = iter(data_loader)
        images, labels = next(data_iter)
    except StopIteration:
        print("数据加载器为空。")
        return

    # 定义用于反归一化的均值和标准差 (必须与你训练脚本中的一致)
    mean = torch.tensor([0.485, 0.456, 0.406, 0.5, 0.5, 0.5]).view(6, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225, 0.5, 0.5, 0.5]).view(6, 1, 1)

    # 我们只可视化几张图就足够了
    num_to_show = min(3, len(images)) # 最多显示3行

    for i in range(num_to_show):
        img_tensor = images[i]
        label = labels[i].item()

        # --- 核心步骤：反归一化，以便于可视化 ---
        # 变换的逆运算: img = img * std + mean
        denorm_img = img_tensor * std + mean
        denorm_img = torch.clamp(denorm_img, 0, 1) # 将值限制在[0, 1]范围

        # 创建一个1x4的子图布局来显示一张图的4个方面
        fig, axs = plt.subplots(1, 4, figsize=(15, 4))
        fig.suptitle(f'类别: {index_to_label[label]}', fontsize=16)

        # 1. 显示RGB部分 (前3个通道)
        rgb_img = denorm_img[:3, :, :].permute(1, 2, 0).numpy()
        axs[0].imshow(rgb_img)
        axs[0].set_title('RGB 图像')
        axs[0].axis('off')

        # 2. 显示NDVI指数图 (第4个通道, 索引为3)
        ndvi_img = denorm_img[3, :, :].numpy()
        im1 = axs[1].imshow(ndvi_img, cmap='viridis') # 使用色图来显示单通道数据
        axs[1].set_title('NDVI 指数')
        axs[1].axis('off')
        fig.colorbar(im1, ax=axs[1], fraction=0.046, pad=0.04)

        # 3. 显示NDWI指数图 (第5个通道, 索引为4)
        ndwi_img = denorm_img[4, :, :].numpy()
        im2 = axs[2].imshow(ndwi_img, cmap='viridis')
        axs[2].set_title('NDWI 指数')
        axs[2].axis('off')
        fig.colorbar(im2, ax=axs[2], fraction=0.046, pad=0.04)

        # 4. 显示NDBI指数图 (第6个通道, 索引为5)
        ndbi_img = denorm_img[5, :, :].numpy()
        im3 = axs[3].imshow(ndbi_img, cmap='viridis')
        axs[3].set_title('NDBI 指数')
        axs[3].axis('off')
        fig.colorbar(im3, ax=axs[3], fraction=0.046, pad=0.04)

        plt.tight_layout(rect=[0, 0, 1, 0.96]) # 调整布局以适应主标题
        plt.show()
        
        
def visualize_fused_data(data_loader, index_to_label):
    """
    可视化用于数据融合的6通道数据。
    它将分别显示RGB图像和三个光谱指数图。
    """
    # 获取一个批次的数据
    try:
        data_iter = iter(data_loader)
        images, labels = next(data_iter)
    except StopIteration:
        print("数据加载器为空。")
        return

    # 定义用于反归一化的均值和标准差 (必须与你训练脚本中的一致)
    mean = torch.tensor([0.485, 0.456, 0.406, 0.5, 0.5, 0.5]).view(6, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225, 0.5, 0.5, 0.5]).view(6, 1, 1)

    # 我们只可视化几张图就足够了
    num_to_show = min(3, len(images)) # 最多显示3行

    for i in range(num_to_show):
        img_tensor = images[i]
        label = labels[i].item()

        # --- 核心步骤：反归一化，以便于可视化 ---
        # 变换的逆运算: img = img * std + mean
        denorm_img = img_tensor * std + mean
        denorm_img = torch.clamp(denorm_img, 0, 1) # 将值限制在[0, 1]范围

        # 创建一个1x4的子图布局来显示一张图的4个方面
        fig, axs = plt.subplots(1, 4, figsize=(15, 4))
        fig.suptitle(f'类别: {index_to_label[label]}', fontsize=16)

        # 1. 显示RGB部分 (前3个通道)
        rgb_img = denorm_img[:3, :, :].permute(1, 2, 0).numpy()
        axs[0].imshow(rgb_img)
        axs[0].set_title('RGB 图像')
        axs[0].axis('off')

        # 2. 显示NDVI指数图 (第4个通道, 索引为3)
        ndvi_img = denorm_img[3, :, :].numpy()
        im1 = axs[1].imshow(ndvi_img, cmap='viridis') # 使用色图来显示单通道数据
        axs[1].set_title('NDVI 指数')
        axs[1].axis('off')
        fig.colorbar(im1, ax=axs[1], fraction=0.046, pad=0.04)

        # 3. 显示NDWI指数图 (第5个通道, 索引为4)
        ndwi_img = denorm_img[4, :, :].numpy()
        im2 = axs[2].imshow(ndwi_img, cmap='viridis')
        axs[2].set_title('NDWI 指数')
        axs[2].axis('off')
        fig.colorbar(im2, ax=axs[2], fraction=0.046, pad=0.04)

        # 4. 显示NDBI指数图 (第6个通道, 索引为5)
        ndbi_img = denorm_img[5, :, :].numpy()
        im3 = axs[3].imshow(ndbi_img, cmap='viridis')
        axs[3].set_title('NDBI 指数')
        axs[3].axis('off')
        fig.colorbar(im3, ax=axs[3], fraction=0.046, pad=0.04)

        plt.tight_layout(rect=[0, 0, 1, 0.96]) # 调整布局以适应主标题
        plt.show()