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
| 667626_18933d713e.jpg | A girl is stretched out in shallow water | a woman in a bikini on a surfboard |
| 3124838157_7ef96745b7.jpg | Three people stand in front of a store window and smile . | two people standing in front of a store |
| 86542183_5e312ae4d4.jpg | Two children are playing ice hockey on frozen ground outside . | two children playing in the snow |

如果方便，可以把图片也插入报告中。

## 8. 总结

请简要说明：

本实验成功复现了 Mini-BLIP2 图像描述生成流程。

实验中使用 CLIP 作为视觉编码器，OPT-125M 作为语言解码器，并实现了简化版 Q-Former 结构。

模型能够成功生成简单英文 caption。

实验过程中主要问题包括：

- Windows 环境 Git 配置问题
- HuggingFace 模型下载速度较慢
- 图片路径读取错误

通过逐步调试，最终完成训练与推理。

## 9. AI 对话过程记录

请填写本次复现过程中与 AI 工具的对话记录（对应 requirements.md 第 9.1 节）。

- 录制工具：
- 对话链接：
- 使用的 AI 模型：ChatGPT 
- 累计对话时长 / 会话数：6h

简要说明 AI 在哪些环节给了帮助、哪些地方是自己独立完成或推翻了 AI 的建议（2—4 句话即可）：

```text
AI帮助我完成了数据集整理、项目结构搭建、代码调试与训练流程实现。
在inference.py路径错误、train.py显存问题等部分，我根据报错进行了手动修改与测试。
最终成功完成
```

## 10. Git 提交记录


- 仓库地址：https://github.com/rweiccc/blip2.git
- 总 commit 数：13

粘贴 `git log --oneline` 输出（或截图）：

```text
077de0c (HEAD -> main, origin/main, origin/HEAD) feat: add inference module
5c9868a feat: add inference pipeline
a86aa23 feat: add training pipeline
5f6d65e feat: implement mini blip2 model
5bfb6bf feat: implement dataset loader
6e750e0 feat: add flickr8k dataset
d2b11c3 feat: add flickr8k dataset
b6011ba Translate README to Chinese and document data/ folder
edac08a Add empty data/ placeholder
674cc47 Add code/ placeholder so the directory appears in the repo
54a0028 Add anti-cheat requirements: AI chat log and granular git commits
d6cb42d Add Mini-BLIP2 reproduction brief
```
