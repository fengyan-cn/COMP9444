import torch
import torch.nn as nn
import math

class MultiHeadAttention(nn.Module):
    def __init__(self, model_input, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_k = model_input // self.num_heads
        self.W_q = nn.Linear(model_input, model_input)
        self.W_k = nn.Linear(model_input, model_input)
        self.W_v = nn.Linear(model_input, model_input)
        self.fc = nn.Linear(model_input, model_input)

    def forward(self, x):
        batch_size = x.size(0)

        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)

        Q = Q.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        attn_weights = torch.nn.functional.softmax(attn_scores, dim=-1)
        attn_output = torch.matmul(attn_weights, V)

        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.d_k)
        output = self.fc(attn_output)

        return output
    
d_model = 512
num_heads = 8
x = torch.randn(32, 10, d_model)
MultiHeadAttention_layer = MultiHeadAttention(d_model, num_heads=num_heads)
output = MultiHeadAttention_layer(x)
print(output.shape)