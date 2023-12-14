import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import MultiheadAttention
from transformers import ViTFeatureExtractor, ViTForImageSequenceFeatureExtractor, AutoModel, AutoTokenizer
import pytorch_lightning as pl

class AttentionBlock(nn.Module):
    def __init__(self, input_dim, output_dim, num_heads, dropout):
        super(AttentionBlock, self).__init__()
        self.attention = MultiheadAttention(input_dim, num_heads=num_heads, dropout=dropout)
        self.norm1 = nn.LayerNorm(input_dim)
        self.norm2 = nn.LayerNorm(input_dim)
        self.dropout = nn.Dropout(dropout)
        self.feedforward = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, input_dim)
        )

    def forward(self, text_embedding, image_embedding):
        # Attend over text and image embeddings
        attended_text, _ = self.attention(text_embedding, image_embedding, image_embedding)
        attended_image, _ = self.attention(image_embedding, text_embedding, text_embedding)

        # Apply normalization and dropout
        attended_text = self.dropout(self.norm1(attended_text + text_embedding))
        attended_image = self.dropout(self.norm2(attended_image + image_embedding))

        # Feedforward network
        text_output = self.feedforward(attended_text)
        image_output = self.feedforward(attended_image)

        # Combine text and image outputs
        final_output = torch.cat([text_output, image_output], dim=1)

        return final_output

class EnglishVQA(pl.LightningModule):
    def __init__(self, opt):
        super(EnglishVQA, self).__init__()
        self.opt = opt

        # Load or create required models
        self.vision_transformers = ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-224")
        self.sentence_bert = AutoModel.from_pretrained("sentence-transformers/paraphrase-MiniLM-L6-v2")
        self.tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/paraphrase-MiniLM-L6-v2")

        # Define attention block
        self.attention_block = AttentionBlock(
            input_dim=self.sentence_bert.config.hidden_size + self.vision_transformers.config.hidden_size,
            output_dim=self.sentence_bert.config.hidden_size,
            num_heads=self.opt.get('num_heads', 8),
            dropout=self.opt.get('dropout', 0.1)
        )

        # Fully connected layer for output
        self.fc = nn.Linear(self.sentence_bert.config.hidden_size, opt['num_labels'])

    def forward(self, text_input, image_input):
        # Get embeddings for text and image
        text_embedding = self.get_text_embedding(text_input)
        image_embedding = self.get_image_embedding(image_input)

        # Apply attention block
        output = self.attention_block(text_embedding, image_embedding)

        # Fully connected layer for output
        output = self.fc(output)

        return output

    def get_text_embedding(self, text_input):
        # Tokenize and get embeddings for the text
        input_ids = self.tokenizer(text_input, return_tensors="pt")["input_ids"]
        text_embedding = self.sentence_bert(input_ids)[0][:, 0, :]  # Take the [CLS] token embedding
        return text_embedding

    def get_image_embedding(self, image_input):
        # Extract image features using Vision Transformers
        image_features = self.vision_transformers(images=image_input, return_tensors="pt").pixel_values
        return image_features

    def inference(self, text_input, image_input):
        # Get embeddings for text and image
        text_embedding = self.get_text_embedding(text_input)
        image_embedding = self.get_image_embedding(image_input)

        # Apply attention block for inference
        output = self.attention_block(text_embedding, image_embedding)

        # Fully connected layer for output
        output = self.fc(output)

        return output

# Example usage:
opt = {'num_labels': 10, 'num_heads': 12, 'dropout': 0.2}  # You can customize the options here
english_vqa_model = EnglishVQA(opt)

# Example inference:
text_input = "What is in the image?"
image_input = torch.randn((1, 3, 224, 224))  # Example image input
output = english_vqa_model.inference(text_input, image_input)
print("Inference Output Shape:", output.shape)

