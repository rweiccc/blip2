import torch
import torch.nn as nn

from transformers import (
    CLIPVisionModel,
    AutoModelForCausalLM,
    AutoTokenizer
)


class MiniQFormer(nn.Module):

    def __init__(
            self,
            hidden_size=768,
            num_queries=32,
            num_layers=2,
            num_heads=8
    ):

        super().__init__()

       
        self.query_tokens = nn.Parameter(
            torch.randn(1, num_queries, hidden_size)
        )

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            batch_first=True
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

    def forward(self, image_features):

        batch_size = image_features.size(0)

        
        query_tokens = self.query_tokens.expand(
            batch_size,
            -1,
            -1
        )

        # 简化版 Cross Attention
        qformer_input = query_tokens + image_features.mean(
            dim=1,
            keepdim=True
        )

        output = self.transformer(qformer_input)

        return output


class MiniBLIP2(nn.Module):

    def __init__(self):

        super().__init__()

        # Vision Encoder
        self.vision_encoder = CLIPVisionModel.from_pretrained(
            "openai/clip-vit-base-patch32"
        )

        
        for param in self.vision_encoder.parameters():
            param.requires_grad = False

        # Mini Q-Former
        self.qformer = MiniQFormer()

        # Projection Layer
        self.projection = nn.Linear(768, 768)

        # Language Model
        self.language_model = AutoModelForCausalLM.from_pretrained(
            "facebook/opt-125m"
        )

        
        for param in self.language_model.parameters():
            param.requires_grad = False

        # tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            "facebook/opt-125m"
        )

    def forward(self, images):

        # Vision Encoder
        vision_outputs = self.vision_encoder(
            pixel_values=images
        )

        image_features = vision_outputs.last_hidden_state

        # Q-Former
        qformer_output = self.qformer(
            image_features
        )

        # Projection
        projected_features = self.projection(
            qformer_output
        )

        return projected_features