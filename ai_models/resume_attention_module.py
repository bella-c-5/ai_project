import torch
import torch.nn as nn

# A basic PyTorch multi-head attention module - computes token importance weights for resume text
# Not currently called in app.py but can be used for visualization or advanced features

class ResumeAttention(nn.Module):
    def __init__(self, embed_dim=64):
        super().__init__()
        # Converts token IDs into embeddings
        self.embedding = nn.Embedding(5000, embed_dim)
        # Multi-head self-attention - 4 heads
        self.attn = nn.MultiheadAttention(embed_dim, num_heads=4)

    def forward(self, tokens):
        # Apply embedding layer
        embeddings = self.embedding(tokens)
        # Compute attention and get weights
        attn_output, attn_weights = self.attn(embeddings, embeddings, embeddings)
        # Returns averaged attention across heads and query positions
        return attn_weights.mean(dim=1).mean(dim=0)
