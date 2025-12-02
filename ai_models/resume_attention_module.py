import torch
import torch.nn as nn

# uses multi-head self-attention - gpt - to determine which parts of the resume text are important
# outputs attention weights for token importance

# embeds token IDs into vectors using an embedding layer
# runs them through nn.MultiheadAttention
# runs an importance vector by averaging attention scores
# results show which words the model focuses on

class ResumeAttention(nn.Module):
    def __init__(self, embed_dim=64):
        super().__init__()
        self.embedding = nn.Embedding(5000, embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads=4)

    def forward(self, tokens):
        embeddings = self.embedding(tokens)
        attn_output, attn_weights = self.attn(embeddings, embeddings, embeddings)
        # return average importance per token
        return attn_weights.mean(dim=1).mean(dim=0)
