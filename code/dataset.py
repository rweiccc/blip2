import os
import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision import transforms


class Flickr8kDataset(Dataset):

    def __init__(self, image_dir, caption_file, max_samples=200):

        self.image_dir = image_dir

        # 读取 captions 文件
        self.data = pd.read_csv(caption_file)

        # 只保留前200条
        self.data = self.data.iloc[:max_samples]

        # 重置索引
        self.data = self.data.reset_index(drop=True)

        # 图像预处理
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

    
        image_name = self.data.iloc[idx]["image"].strip()

    
        caption = self.data.iloc[idx]["caption"].strip()

    
        image_path = os.path.join(self.image_dir, image_name)

     
        image = Image.open(image_path).convert("RGB")

       
        image = self.transform(image)

        return image, caption