import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from tqdm import tqdm

from dataset import Flickr8kDataset
from model import MiniBLIP2


# 设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device)

# 数据集
dataset = Flickr8kDataset(
    image_dir="data/Images",
    caption_file="data/captions.txt"
)

print("数据集数量:", len(dataset))

# dataloader
dataloader = DataLoader(
    dataset,
    batch_size=2,
    shuffle=True
)

# tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    "facebook/opt-125m"
)

tokenizer.pad_token = tokenizer.eos_token

# 模型
model = MiniBLIP2()
model.to(device)

# projection 到 vocab
vocab_projection = nn.Linear(
    768,
    tokenizer.vocab_size
).to(device)

# optimizer
optimizer = torch.optim.AdamW(
    list(model.parameters()) +
    list(vocab_projection.parameters()),
    lr=1e-4
)

# loss
criterion = nn.CrossEntropyLoss(
    ignore_index=tokenizer.pad_token_id
)

# epoch
epochs = 1

# 开始训练
for epoch in range(epochs):

    model.train()

    total_loss = 0

    progress_bar = tqdm(dataloader)

    for images, captions in progress_bar:

        images = images.to(device)

        # tokenizer
        tokens = tokenizer(
            list(captions),
            padding=True,
            truncation=True,
            return_tensors="pt"
        )

        input_ids = tokens.input_ids.to(device)

        # forward
        outputs = model(images)

        # outputs:
        # [batch, query_num, hidden_size]

        # 平均 pooling
        outputs = outputs.mean(dim=1)

        # vocab logits
        logits = vocab_projection(outputs)

        # 扩展 seq_len
        seq_len = input_ids.shape[1]

        logits = logits.unsqueeze(1).repeat(
            1,
            seq_len,
            1
        )

        # loss
        loss = criterion(
            logits.view(-1, tokenizer.vocab_size),
            input_ids.view(-1)
        )

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

        progress_bar.set_description(
            f"epoch {epoch+1} loss {loss.item():.4f}"
        )

    avg_loss = total_loss / len(dataloader)

    print(f"\nepoch {epoch+1} avg loss: {avg_loss:.4f}")

print("\n训练完成")