import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from models_train.layers.bvae import VAE

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# Parameter reset for initializing weights
def reset_parameters(named_parameters):
    for name, p in named_parameters:
        if "weight" in name:
            nn.init.xavier_normal_(p)
        if "bias" in name:
            nn.init.constant_(p, 0.0)

# Scaled dot-product attention scores
def scaled_dot_product_attention(q, k, v, dk, attention_mask=None, dropout=0.2, training=True):
    
    # dot product of queries and keys
    scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(dk)
    
    #  the attention mask
    if attention_mask is not None:
        scores = scores + ((1 - attention_mask) * -1e5) 
    
    #  softmax to get attention probabilities
    scores = F.softmax(scores, dim=-1)
    

    scores = F.dropout(scores, p=dropout, training=training)
    
    return scores, torch.matmul(scores, v)

# Multi-Headed Self-Attention Module
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, attention_size, num_heads=8, input_size=None, dropout=0.1):
        '''
        Multi-Headed Dot-product Self-Attention Module.
        '''
        super(MultiHeadSelfAttention, self).__init__()
        if input_size is None:
            input_size = attention_size
        
        self.dk = input_size // num_heads  
        self.num_heads = num_heads
        
        # Linear layer to project input into query, key, and value (Q, K, V)
        self.kqv = nn.Linear(input_size, 3 * attention_size, bias=False)  # Combined projection
        
        # Dropout for regularization
        self.dropout_layer = nn.Dropout(dropout)
        
        # combine attention heads 
        self.head_projection = nn.Linear(attention_size, attention_size)
        
        # compute the final scalar (importance score)
        self.outt = nn.Sequential(nn.Linear(attention_size, 1), nn.Sigmoid())  
        
        # Reset parameters
        reset_parameters(self.named_parameters())

    def forward(self, x, attention_mask=None):
        '''
        Args:
            x: [seq_len, batch_size, input_size] Input tensor (e.g., feature map from CNN)
            attention_mask: Optional attention mask (to ignore certain positions, like padding)
        Returns:
            out: Attention-weighted output values
            scores: Attention scores
        '''
        if attention_mask is not None and len(attention_mask.size()) == 2:
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(1) 
        
        # Project input tensor to Q, K, V using the combined linear layer
        k, q, v = self.kqv(x).chunk(3, dim=-1)  # (seq_len, batch_size, attention_size)

        # Reshape the query, key, value tensors to handle multi-head attention
        k = k.view(k.shape[0], self.num_heads, -1, self.dk)  # [seq_len, num_heads, batch_size, head_dim]
        q = q.view(q.shape[0], self.num_heads, -1, self.dk)
        v = v.view(v.shape[0], self.num_heads, -1, self.dk)

        # scaled dot-product attention function
        scores, out = scaled_dot_product_attention(q, k, v, self.dk, attention_mask=attention_mask, dropout=self.dropout_layer.p, training=self.training)
        
        # dropout to attention scores
        scores = self.dropout_layer(scores)
        
        # Concatenate the outputs from all heads and project back
        out = out.view(out.shape[0], -1)  # [seq_len, attention_size]
        out = self.head_projection(out)

        out = self.outt(out)  # Output score for each frame
        return out, scores

class Summarizer(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=2):
        super().__init__()
        self.attn = MultiHeadSelfAttention(hidden_size)  
        self.vae = VAE(input_size, hidden_size, num_layers)

    def forward(self, image_features, uniform=False):
        if not uniform:
            scores, weighted_features = self.attn(image_features) 
            
            # Applying attention scores
            weighted_features = image_features * scores.view(-1, 1, 1)

        else:
            scores = None
            weighted_features = image_features

        h_mu, h_log_variance, decoded_features = self.vae(weighted_features)
        return scores, h_mu, h_log_variance, decoded_features

if __name__ == '__main__':
    pass
