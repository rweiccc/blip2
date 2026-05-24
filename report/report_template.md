# Mini-BLIP2 图像描述生成复现实验报告

## 1. 论文信息

- 论文名称：BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models
- 论文地址：https://arxiv.org/abs/2301.12597

## 2. 任务说明

本实验复现的任务是图像描述生成 Image Captioning。

输入：图片  
输出：英文 caption

## 3. 数据集

- 数据集名称：Flickr8k
- 数据集地址：https://www.kaggle.com/datasets/adityajn105/flickr8k
- 实际使用数据量：前 200 张图片

## 4. 模型结构

Image
→ Frozen Vision Encoder
→ Mini Q-Former
→ Projection Layer
→ Frozen Language Decoder
→ Caption

## 4.1 视觉编码器

openai/clip-vit-base-patch32

## 4.2 Mini Q-Former

- 查询向量数量：32
- 隐藏层维度：768
- Transformer 层数：2
- 是否使用 cross-attention：是

## 4.3 语言解码器

facebook/opt-125m

## 5. 训练设置

请填写：

- 训练数据量：200
- epoch：3
- batch size：4
- learning rate：le-4
- optimizer：Adamw
- loss function：CrossEntropyLoss
- 冻结的模块： 
  - Vision Encoder
  - Language Model
- 训练的模块：
  - Q-Former
  - Projection Layer
## 6. 训练过程

部分训练如下：
```text
10.843851089477539
10.673324584960938
10.59214973449707
10.371302604675293
10.270759582519531
9.8764066696167
10.259981155395508
9.94402027130127
9.650736808776855
9.445172309875488
8.967204093933105
8.819290161132812
9.716012001037598
8.588641166687012
8.378031730651855
8.316567420959473
8.537890434265137
8.16748046875
8.2960844039917
8.42436408996582
8.92316722869873
7.490298748016357
7.473545551300049
8.65912914276123
7.3162970542907715
7.37550163269043
6.960870265960693
7.210042953491211
```

## 7. 生成结果展示

至少展示 3—5 个例子。

| 图片编号 | 真实 Caption | 模型生成 Caption |
|---|---|---|
| 1 |  |  |
| 2 |  |  |
| 3 |  |  |

如果方便，可以把图片也插入报告中。

## 8. 总结

请简要说明：

- 是否成功跑通训练；
- 生成效果如何；
- 遇到了什么问题；
- 如果继续改进，可以怎么做。

## 9. AI 对话过程记录

请填写本次复现过程中与 AI 工具的对话记录（对应 requirements.md 第 9.1 节）。

- 录制工具：例如 entir.io
- 对话链接：
- 使用的 AI 模型：例如 Claude / ChatGPT / Gemini
- 累计对话时长 / 会话数：

简要说明 AI 在哪些环节给了帮助、哪些地方是自己独立完成或推翻了 AI 的建议（2—4 句话即可）：

```text
（在这里写）
```

## 10. Git 提交记录

请填写本次复现的代码仓库与提交历史（对应 requirements.md 第 9.2 节）。

- 仓库地址：
- 总 commit 数：

粘贴 `git log --oneline` 输出（或截图）：

```text
（在这里粘贴 git log --oneline）
```
