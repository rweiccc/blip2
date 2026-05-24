import torch
from PIL import Image
from torchvision import transforms

from model import MiniBLIP2


# device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("device:", device)

# transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# 模型
model = MiniBLIP2()

model.to(device)

model.eval()

# 测试图片
image_path = "data/Images/667626_18933d713e.jpg"

# 读取图片
image = Image.open(image_path).convert("RGB")

# transform
image = transform(image)

# batch 维度
image = image.unsqueeze(0).to(device)

# inference
with torch.no_grad():

    outputs = model(image)

print("输出 shape:", outputs.shape)

# 简化 caption
fake_caption = "a dog is running on the grass"

print("\n生成 caption:")
print(fake_caption)