from PIL import Image
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration

# 设备
device = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", device)

# 加载模型
processor = BlipProcessor.from_pretrained(
    "Salesforce/blip-image-captioning-base"
)

model = BlipForConditionalGeneration.from_pretrained(
    "Salesforce/blip-image-captioning-base"
).to(device)

# ====== 这里换图片 ======
image_path = "data/Images/86542183_5e312ae4d4.jpg"

# 读取图片
image = Image.open(image_path).convert("RGB")

# 处理输入
inputs = processor(images=image, return_tensors="pt").to(device)

# 生成 caption
out = model.generate(**inputs, max_new_tokens=30)

caption = processor.decode(out[0], skip_special_tokens=True)

# 输出结果
print("\n图片路径:")
print(image_path)

print("\n生成 caption:")
print(caption)