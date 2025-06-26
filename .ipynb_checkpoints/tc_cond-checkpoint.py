# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from typing import Tuple, List, Optional




# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import math
# from einops import rearrange
# import random

# # from torch_cluster import kmeans

# class PositionalEncoding3D(nn.Module):
#     """Updated 3D Positional Encoding."""
#     def __init__(self, 
#                  num_embed=8192, 
#                  spatial_size=[32, 32, 32], 
#                  embed_dim=3968, 
#                  trainable=True, 
#                  pos_emb_type='embedding'):
#         super().__init__()
        
#         if isinstance(spatial_size, int):
#             spatial_size = [spatial_size, spatial_size, spatial_size]

#         self.spatial_size = spatial_size
#         self.num_embed = num_embed + 1
#         self.embed_dim = embed_dim
#         self.trainable = trainable
#         self.pos_emb_type = pos_emb_type

#         assert self.pos_emb_type in ['embedding', 'parameter']
        
#         # self.emb = nn.Conv3d(256, embed_dim, kernel_size=3, padding=1)
#         self.flatten = nn.Flatten(start_dim=2)
#         if self.pos_emb_type == 'embedding':
#             self.height_emb = nn.Embedding(self.spatial_size[0], embed_dim)
#             self.width_emb = nn.Embedding(self.spatial_size[1], embed_dim)
#             self.depth_emb = nn.Embedding(self.spatial_size[2], embed_dim)
#         else:
#             self.height_emb = nn.Parameter(torch.zeros(1, self.spatial_size[0], embed_dim))
#             self.width_emb = nn.Parameter(torch.zeros(1, self.spatial_size[1], embed_dim))
#             self.depth_emb = nn.Parameter(torch.zeros(1, self.spatial_size[2], embed_dim))
        
#         self._set_trainable()

#     def _set_trainable(self):
#         if not self.trainable:
#             for param in self.parameters():
#                 param.requires_grad = False

#     def forward(self, index, **kwargs):
#         # assert index.dim() == 2  # B x L
#         # index = torch.clamp(index, min=0)  # Ensure indices are valid
#         print("enc_out enc_out enc_out enc_out flatten emb", index.shape)
#         # emb = self.emb(index)
#         emb = self.flatten(index)  # Shape: (B, embed_dim, L)
#         emb = emb.contiguous()

#         print("enc_out enc_out enc_out enc_out flatten emb", emb.shape)
        
#         # Transpose latent embeddings to match the attention input format
#         emb = emb.permute(0, 2, 1) 
#         # print("self.spatial_size[0]", self.spatial_size[0])
#         # print("self.spatial_size[0]", self.spatial_size[1])
#         # print("self.spatial_size[0]", self.spatial_size[2])
#         if emb.shape[1] > 0:
#         # if False:
#             if self.pos_emb_type == 'embedding':
#                 height_emb = self.height_emb(torch.arange(self.spatial_size[0], device=index.device).view(1, self.spatial_size[0])).unsqueeze(2).unsqueeze(3)  # Shape: 1 x H x 1 x 1 x embed_dim
#                 width_emb = self.width_emb(torch.arange(self.spatial_size[1], device=index.device).view(1, self.spatial_size[1])).unsqueeze(1).unsqueeze(3)   # Shape: 1 x 1 x W x 1 x embed_dim
#                 depth_emb = self.depth_emb(torch.arange(self.spatial_size[2], device=index.device).view(1, self.spatial_size[2])).unsqueeze(1).unsqueeze(1)   # Shape: 1 x 1 x 1 x D x embed_dim
#             else:
#                 height_emb = self.height_emb.unsqueeze(2).unsqueeze(3)  # Shape: 1 x H x 1 x 1 x embed_dim
#                 width_emb = self.width_emb.unsqueeze(1).unsqueeze(3)    # Shape: 1 x 1 x W x 1 x embed_dim
#                 depth_emb = self.depth_emb.unsqueeze(1).unsqueeze(1)    # Shape: 1 x 1 x 1 x D x embed_dim

#             pos_emb = (height_emb + width_emb + depth_emb).view(1, self.spatial_size[0] * self.spatial_size[1] * self.spatial_size[2], -1) # 1 x H x W x D -> 1 x L xD
#             emb = emb + pos_emb[:, :emb.shape[1], :]
        
#         return emb



# # class Conv_MLP(nn.Module):
# #     def __init__(self, n_embd, mlp_hidden_times, act, resid_pdrop):
# #         super().__init__()
# #         self.conv1 = nn.Conv3d(in_channels=n_embd, out_channels=int(mlp_hidden_times * n_embd), kernel_size=3, stride=1, padding=1)
# #         self.act = act
# #         self.conv2 = nn.Conv3d(in_channels=int(mlp_hidden_times * n_embd), out_channels=n_embd, kernel_size=3, stride=1, padding=1)

# #         # self.conv3 = nn.Conv3d(in_channels=n_embd, out_channels=int(mlp_hidden_times * n_embd), kernel_size=3, stride=1, padding=1)
# #         # self.act = act
# #         # self.conv4 = nn.Conv3d(in_channels=int(mlp_hidden_times * n_embd), out_channels=n_embd, kernel_size=3, stride=1, padding=1)
    
# #         self.dropout = nn.Dropout(resid_pdrop)

# #     def forward(self, x):
# #         n =  x.size()[1]
# #         x = rearrange(x, 'b (h w d) c -> b c h w d', h=8, w=8, d=8)
# #         x = (self.conv2(self.act(self.conv1(x))))
# #         # x = self.conv4(self.act(self.conv3(x)))
# #         x = rearrange(x, 'b c h w d -> b (h w d) c')
# #         return self.dropout(x)


# class ConvBlock(nn.Module):
#     """A Conv Block with LayerNorm."""
#     def __init__(self, embed_dim, kernel_size=3, stride=1, padding=1):
#         super(ConvBlock, self).__init__()
#         self.conv1 = nn.Conv3d(in_channels=embed_dim, out_channels=int(4 * embed_dim), kernel_size=3, stride=1, padding=1)
#         self.act = GELU2()
#         self.conv2 = nn.Conv3d(in_channels=int(4 * embed_dim), out_channels=embed_dim, kernel_size=3, stride=1, padding=1)

#     def forward(self, x):
#         # b, c, d, h, w = x.shape
#         x = rearrange(x, 'b (h w d) c -> b c h w d', h=15, w=15, d=9)
#         x = (self.conv2(self.act(self.conv1(x))))
#         # x = self.conv4(self.act(self.conv3(x)))
#         x = rearrange(x, 'b c h w d -> b (h w d) c')
#         return x


# # class FullAttention(nn.Module):
# #     """Full Attention Module."""
# #     def __init__(self, n_embd, n_head, attn_pdrop=0.0, resid_pdrop=0.0, causal=True):
# #         super().__init__()
# #         assert n_embd % n_head == 0
# #         self.key = nn.Linear(n_embd, n_embd)
# #         self.query = nn.Linear(n_embd, n_embd)
# #         self.value = nn.Linear(n_embd, n_embd)
# #         self.attn_drop = nn.Dropout(attn_pdrop)
# #         self.resid_drop = nn.Dropout(resid_pdrop)
# #         self.proj = nn.Linear(n_embd, n_embd)
# #         self.n_head = n_head
# #         self.causal = causal

# #     def forward(self, x, lay=int, mask=None):
# #         B, T, C = x.size()
# #         k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
# #         q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
# #         v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

# #         att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
# #         # if lay>= 5:
# #         #     # print("att being revered")
# #         #     att = 1-att
# #         if mask is not None:
# #             att = att.masked_fill(mask == 0, float('-inf'))

# #         att = F.softmax(att, dim=-1)
# #         att = self.attn_drop(att)
# #         y = att @ v
# #         y = y.transpose(1, 2).contiguous().view(B, T, C)
# #         y = self.resid_drop(self.proj(y))
# #         return y, att






# class FullAttention(nn.Module):
#     """Full Attention Module with Local-to-Global Attention."""
#     def __init__(self, n_embd, n_head, attn_pdrop=0.0, resid_pdrop=0.0, causal=True, window_size=(3, 3, 3)):
#         """
#         Args:
#             n_embd (int): Embedding dimension.
#             n_head (int): Number of attention heads.
#             attn_pdrop (float): Dropout probability for attention weights.
#             resid_pdrop (float): Dropout probability for residual connections.
#             causal (bool): Whether to apply causal masking (e.g., for autoregressive tasks).
#             window_size (tuple): (H, W, D) for local attention window. None means full attention.
#         """
#         super().__init__()
#         assert n_embd % n_head == 0
#         self.key = nn.Linear(n_embd, n_embd)
#         self.query = nn.Linear(n_embd, n_embd)
#         self.value = nn.Linear(n_embd, n_embd)
#         self.attn_drop = nn.Dropout(attn_pdrop)
#         self.resid_drop = nn.Dropout(resid_pdrop)
#         self.proj = nn.Linear(n_embd, n_embd)
#         self.n_head = n_head
#         self.causal = causal
#         self.window_size = window_size

#     def forward(self, x, lay=int, mask=None):
#         """
#         Forward pass with local-to-global attention.
#         Args:
#             x (torch.Tensor): Input tensor of shape (B, T, C).
#             mask (torch.Tensor): Optional mask of shape (B, T).
#         Returns:
#             torch.Tensor: Output tensor of shape (B, T, C).
#         """
#         B, T, C = x.size()
#         if self.window_size is not None:
#             H, W, D = self.window_size
#             # Step 1: Apply vectorized local attention
#             local_output = self._local_attention(x, H, W, D)

#             # Step 2: Aggregate global context (global attention)
#             global_context = self._global_attention(x)

#             # Step 3: Combine local and global features
#             output = local_output + global_context
#             att = 0
#         else:
#             # Full attention if no window size is defined
#             output, _ = self._full_attention(x, mask)

#         return output, att

#     def _local_attention(self, x, H, W, D):
#         """Compute local attention within defined window using vectorization."""
#         B, T, C = x.size()
#         window_tokens = H * W * D
#         assert T % window_tokens == 0, "Sequence length must be divisible by window size."

#         # Reshape for windows: (B * num_windows, window_tokens, C)
#         num_windows = T // window_tokens
#         x_windows = x.view(B, num_windows, window_tokens, C).reshape(-1, window_tokens, C)

#         # Compute local attention for all windows simultaneously
#         y, _ = self._full_attention(x_windows, mask=None)

#         # Reshape back to original sequence shape (B, T, C)
#         y = y.view(B, num_windows * window_tokens, C)
#         return y

#     def _global_attention(self, x):
#         """Compute global attention by aggregating context across all tokens."""
#         B, T, C = x.size()
#         k = self.key(x).view(B, self.n_head, T, C // self.n_head)
#         q = self.query(x).view(B, self.n_head, T, C // self.n_head)
#         v = self.value(x).view(B, self.n_head, T, C // self.n_head)

#         # Global attention across all tokens
#         att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
#         att = F.softmax(att, dim=-1)
#         att = self.attn_drop(att)
#         global_context = (att @ v).view(B, T, C)
#         return self.resid_drop(self.proj(global_context))

#     def _full_attention(self, x, mask):
#         """Standard full attention."""
#         B, T, C = x.size()
#         k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
#         q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
#         v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

#         att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
#         if mask is not None:
#             att = att.masked_fill(mask == 0, float('-inf'))
#         att = F.softmax(att, dim=-1)
#         att = self.attn_drop(att)
#         y = att @ v
#         y = y.transpose(1, 2).contiguous().view(B, T, C)
#         y = self.resid_drop(self.proj(y))
#         return y, att



# class FullAttentiondec(nn.Module):
#     """Full Attention Module with Local-to-Global Attention."""
#     def __init__(self, n_embd, n_head, attn_pdrop=0.0, resid_pdrop=0.0, causal=True, window_size=(5, 5, 3)):
#         """
#         Args:
#             n_embd (int): Embedding dimension.
#             n_head (int): Number of attention heads.
#             attn_pdrop (float): Dropout probability for attention weights.
#             resid_pdrop (float): Dropout probability for residual connections.
#             causal (bool): Whether to apply causal masking (e.g., for autoregressive tasks).
#             window_size (tuple): (H, W, D) for local attention window. None means full attention.
#         """
#         super().__init__()
#         assert n_embd % n_head == 0
#         self.key = nn.Linear(n_embd, n_embd)
#         self.query = nn.Linear(n_embd, n_embd)
#         self.value = nn.Linear(n_embd, n_embd)
#         self.attn_drop = nn.Dropout(attn_pdrop)
#         self.resid_drop = nn.Dropout(resid_pdrop)
#         self.proj = nn.Linear(n_embd, n_embd)
#         self.n_head = n_head
#         self.causal = causal
#         self.window_size = window_size

#     def forward(self, x, lay=int, mask=None):
#         """
#         Forward pass with local-to-global attention.
#         Args:
#             x (torch.Tensor): Input tensor of shape (B, T, C).
#             mask (torch.Tensor): Optional mask of shape (B, T).
#         Returns:
#             torch.Tensor: Output tensor of shape (B, T, C).
#         """
#         B, T, C = x.size()
#         if self.window_size is not None:
#             H, W, D = self.window_size
#             # Step 1: Apply vectorized local attention
#             local_output = self._local_attention(x, H, W, D)

#             # Step 2: Aggregate global context (global attention)
#             # global_context = self._global_attention(x)

#             # # Step 3: Combine local and global features
#             # output = local_output + global_context
#             output = local_output
#             att = 0
#         else:
#             # Full attention if no window size is defined
#             output, _ = self._full_attention(x, mask)

#         return output, att

#     def _local_attention(self, x, H, W, D):
#         """Compute local attention within defined window using vectorization."""
#         B, T, C = x.size()
#         window_tokens = H * W * D
#         assert T % window_tokens == 0, "Sequence length must be divisible by window size."

#         # Reshape for windows: (B * num_windows, window_tokens, C)
#         num_windows = T // window_tokens
#         x_windows = x.view(B, num_windows, window_tokens, C).reshape(-1, window_tokens, C)

#         # Compute local attention for all windows simultaneously
#         y, _ = self._full_attention(x_windows, mask=None)

#         # Reshape back to original sequence shape (B, T, C)
#         y = y.view(B, num_windows * window_tokens, C)
#         return y

#     def _global_attention(self, x):
#         """Compute global attention by aggregating context across all tokens."""
#         B, T, C = x.size()
#         k = self.key(x).view(B, self.n_head, T, C // self.n_head)
#         q = self.query(x).view(B, self.n_head, T, C // self.n_head)
#         v = self.value(x).view(B, self.n_head, T, C // self.n_head)

#         # Global attention across all tokens
#         att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
#         att = F.softmax(att, dim=-1)
#         att = self.attn_drop(att)
#         global_context = (att @ v).view(B, T, C)
#         return self.resid_drop(self.proj(global_context))

#     def _full_attention(self, x, mask):
#         """Standard full attention."""
#         B, T, C = x.size()
#         k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
#         q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
#         v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

#         att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
#         if mask is not None:
#             att = att.masked_fill(mask == 0, float('-inf'))
#         att = F.softmax(att, dim=-1)
#         att = self.attn_drop(att)
#         y = att @ v
#         y = y.transpose(1, 2).contiguous().view(B, T, C)
#         y = self.resid_drop(self.proj(y))
#         return y, att



# # class FullAttention(nn.Module):
# #     """Full Attention Module with Local-to-Global Attention."""
# #     def __init__(self, n_embd, n_head, attn_pdrop=0.0, resid_pdrop=0.0, causal=True, window_size=(3, 3, 3)):
# #         """
# #         Args:
# #             n_embd (int): Embedding dimension.
# #             n_head (int): Number of attention heads.
# #             attn_pdrop (float): Dropout probability for attention weights.
# #             resid_pdrop (float): Dropout probability for residual connections.
# #             causal (bool): Whether to apply causal masking (e.g., for autoregressive tasks).
# #             window_size (tuple): (H, W, D) for local attention window. None means full attention.
# #         """
# #         super().__init__()
# #         assert n_embd % n_head == 0
# #         self.key = nn.Linear(n_embd, n_embd)
# #         self.query = nn.Linear(n_embd, n_embd)
# #         self.value = nn.Linear(n_embd, n_embd)
# #         self.attn_drop = nn.Dropout(attn_pdrop)
# #         self.resid_drop = nn.Dropout(resid_pdrop)
# #         self.proj = nn.Linear(n_embd, n_embd)
# #         self.n_head = n_head
# #         self.causal = causal
# #         self.window_size = window_size

# #     def forward(self, x, lay=int, mask=None):
# #         """
# #         Forward pass with local-to-global attention.
# #         Args:
# #             x (torch.Tensor): Input tensor of shape (B, T, C).
# #             mask (torch.Tensor): Optional mask of shape (B, T).
# #         Returns:
# #             torch.Tensor: Output tensor of shape (B, T, C).
# #         """
# #         B, T, C = x.size()
# #         if self.window_size is not None:
# #             H, W, D = self.window_size
# #             # Step 1: Apply local attention
# #             local_output = self._local_attention(x, H, W, D)

# #             # Step 2: Aggregate global context (global attention)
# #             global_context = self._global_attention(x)

# #             # Step 3: Combine local and global features
# #             output = local_output + global_context
# #             att = 0
# #         else:
# #             # Full attention if no window size is defined
# #             output, _ = self._full_attention(x, mask)

# #         return output, att

# #     def _local_attention(self, x, H, W, D):
# #         """Compute local attention within defined window."""
# #         B, T, C = x.size()
# #         window_tokens = H * W * D
# #         assert T % window_tokens == 0, "Sequence length must be divisible by window size."

# #         # Reshape for windows: (B, num_windows, window_tokens, C)
# #         num_windows = T // window_tokens
# #         x_windows = x.view(B, num_windows, window_tokens, C)

# #         # Compute local attention for each window
# #         local_outputs = []
# #         for i in range(num_windows):
# #             window = x_windows[:, i, :, :]
# #             local_output, _ = self._full_attention(window, mask=None)
# #             local_outputs.append(local_output)

# #         # Concatenate windows back to the original shape
# #         local_output = torch.cat(local_outputs, dim=1)
# #         return local_output

# #     def _global_attention(self, x):
# #         """Compute global attention by aggregating context across all tokens."""
# #         B, T, C = x.size()
# #         k = self.key(x).view(B, self.n_head, T, C // self.n_head)
# #         q = self.query(x).view(B, self.n_head, T, C // self.n_head)
# #         v = self.value(x).view(B, self.n_head, T, C // self.n_head)

# #         # Global attention across all tokens
# #         att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
# #         att = F.softmax(att, dim=-1)
# #         att = self.attn_drop(att)
# #         global_context = (att @ v).view(B, T, C)
# #         return self.resid_drop(self.proj(global_context))

# #     def _full_attention(self, x, mask):
# #         """Standard full attention."""
# #         B, T, C = x.size()
# #         k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
# #         q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
# #         v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

# #         att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
# #         if mask is not None:
# #             att = att.masked_fill(mask == 0, float('-inf'))
# #         att = F.softmax(att, dim=-1)
# #         att = self.attn_drop(att)
# #         y = att @ v
# #         y = y.transpose(1, 2).contiguous().view(B, T, C)
# #         y = self.resid_drop(self.proj(y))
# #         return y, att

# class AttentionScalingWithHeads(nn.Module):
#     def __init__(self, embed_dim, num_heads=8):
#         super(AttentionScalingWithHeads, self).__init__()
#         self.embed_dim = embed_dim
#         self.num_heads = num_heads
#         self.head_dim = embed_dim // num_heads

#         # Linear layers for query projection
#         self.query_proj = nn.Linear(embed_dim, embed_dim)
#         self.softmax = nn.Softmax(dim=2)  # Normalize across sequence length

#     def forward(self, x):
#         batch_size, seq_len, embed_dim = x.size()

#         # Project to multi-head space
#         query = self.query_proj(x)  # (batch_size, seq_len, embed_dim)
#         query = query.view(batch_size, seq_len, self.num_heads, self.head_dim)  # (batch_size, seq_len, num_heads, head_dim)

#         # Compute attention scores
#         attention_scores = torch.mean(query, dim=-1)  # Average over head_dim
#         attention_scores = self.softmax(attention_scores)  # (batch_size, seq_len, num_heads)

#         # Reshape to apply scaling
#         attention_scores = attention_scores.view(batch_size, seq_len, self.num_heads, 1)
#         scaled_x = x.view(batch_size, seq_len, self.num_heads, self.head_dim) * attention_scores

#         # Flatten back to original embedding dimension
#         scaled_x = scaled_x.view(batch_size, seq_len, embed_dim)
#         return scaled_x




# class SEBlock(nn.Module):
#     def __init__(self, embed_dim, reduction_ratio=16):
#         super(SEBlock, self).__init__()
#         self.fc1 = nn.Linear(embed_dim, embed_dim // reduction_ratio)
#         self.fc2 = nn.Linear(embed_dim // reduction_ratio, embed_dim)
#         self.sigmoid = nn.Sigmoid()

#     def forward(self, x):
        
#         b, seq_len, c = x.size()  # batch_size, sequence_length, channels
        
#         # Global average pooling along the sequence length dimension (seq_len)
#         se = torch.mean(x, dim=1)  # Pooling across the sequence dimension
#         # print("se after mean is shape is", se.shape)
        
#         se = self.fc1(se)
#         se = torch.relu(se)
#         se = self.fc2(se)
#         se = self.sigmoid(se)  # Channel-wise attention
#         se = torch.unsqueeze(se, dim=1)
#         # print("se after unseq is shape is", se.shape)
#         # Scale the input tensor by the attention weights
#         return x * se  # Apply the mask across all sequence tokens



# class TransformerBlock(nn.Module):
#     """A Transformer Block with Full Attention and MLP."""
#     def __init__(self, embed_dim, num_heads, mlp_ratio=4, dropout=0.0, mlp_type = None):
#         super(TransformerBlock, self).__init__()
#         self.attn = FullAttention(embed_dim, num_heads, attn_pdrop=dropout, resid_pdrop=dropout)
#         if mlp_type == 'conv_mlp':
#             self.mlp = ConvBlock(embed_dim)
#         else:
#             self.mlp = nn.Sequential(
#             nn.Linear(embed_dim, mlp_ratio * embed_dim),
#             GELU2(),
#             nn.Linear(mlp_ratio * embed_dim, embed_dim),
#             nn.Dropout(dropout)
#         )
#         self.norm1 = nn.LayerNorm(embed_dim)
#         self.norm2 = nn.LayerNorm(embed_dim)
#         self.se_block = SEBlock(embed_dim, 16)
#         self.attention_scaling = AttentionScalingWithHeads(embed_dim, 8)

#     def forward(self, x, lay=int, mask=None):
#         attn_out, att = self.attn(x, lay, mask)
#         x = x + attn_out
#         x = self.norm1(x)

#         mlp_out = self.mlp(x)
#         x = x + mlp_out
#         x = self.norm2(x)
#         x = self.attention_scaling(x)
#         x = self.se_block(x)
#         return x, att

# class TransformerBlockdec(nn.Module):
#     """A Transformer Block with Full Attention and MLP."""
#     def __init__(self, embed_dim, num_heads, mlp_ratio=4, dropout=0.0, mlp_type = None):
#         super(TransformerBlockdec, self).__init__()
#         self.attn = FullAttentiondec(embed_dim, num_heads, attn_pdrop=dropout, resid_pdrop=dropout)
#         if mlp_type == 'conv_mlp':
#             self.mlp = ConvBlock(embed_dim)
#         else:
#             self.mlp = nn.Sequential(
#             nn.Linear(embed_dim, mlp_ratio * embed_dim),
#             GELU2(),
#             nn.Linear(mlp_ratio * embed_dim, embed_dim),
#             nn.Dropout(dropout)
#         )
#         self.norm1 = nn.LayerNorm(embed_dim)
#         self.norm2 = nn.LayerNorm(embed_dim)
#         self.se_block = SEBlock(embed_dim, 16)
#         self.attention_scaling = AttentionScalingWithHeads(embed_dim, 8)

#     def forward(self, x, lay=int, mask=None):
#         attn_out, att = self.attn(x, lay, mask)
#         x = x + attn_out
#         x = self.norm1(x)

#         mlp_out = self.mlp(x)
#         x = x + mlp_out
#         x = self.norm2(x)
#         x = self.attention_scaling(x)
#         x = self.se_block(x)
#         return x, att
# # class TransformerModel(nn.Module):
# #     """Transformer Model with Full Attention, Conv Blocks, and Positional Encoding."""
# #     def __init__(self, input_shape, embed_dim, num_layers, num_heads):
# #         super(TransformerModel, self).__init__()
# #         h, w, d = input_shape
#         # self.positional_encoding = PositionalEncoding3D(embed_dim=embed_dim, 
#         #                                                 spatial_size=[h, w, d])
#         # self.layers = nn.ModuleList([
#         #     nn.Sequential(
#         #         TransformerBlock(embed_dim, num_heads),
#         #         # ConvBlock(embed_dim)
#         #     )
#         #     for _ in range(num_layers)
#         # ])

# #     def forward(self, x):
# #         b, c, h, w, d = x.shape
# #         # x = rearrange(x, 'b c h w d -> b (h w d) c')
# #         x = self.positional_encoding(x)
# #         # x = rearrange(x, 'b c h w d -> b (h w d) c')
# #         for layer in self.layers:
# #             x = layer(x)

# #         x = rearrange(x, 'b (h w d) c -> b c h w d', h=h, w=w, d=d)
# #         return x

# class ConvBlock3D(nn.Module):
#     """Convolution Block with Conv3d, BatchNorm, ReLU, and Dropout"""
#     def __init__(self, in_channels, out_channels, dropout_prob):
#         super(ConvBlock3D, self).__init__()
#         self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)
#         self.relu = nn.ReLU(inplace=True)
#         self.batch_norm = nn.BatchNorm3d(out_channels)
#         self.dropout = nn.Dropout3d(p=dropout_prob)
    
#     def forward(self, x):
#         x = self.conv(x)
#         x = self.relu(x)
#         x = self.batch_norm(x)
#         x = self.dropout(x)
#         return x

# class TransformerModel(nn.Module):
#     """Transformer Model with Full Attention, Uncertainty Estimation, and Soft Masking."""
#     def __init__(self, input_shape, embed_dim, num_layers, num_heads):
#         super(TransformerModel, self).__init__()
#         h, w, d = input_shape
#         self.num_layers = num_layers
#         self.embed_dim = embed_dim
#         self.positional_encoding = PositionalEncoding3D(embed_dim=embed_dim, 
#                                                         spatial_size=[h, w, d])
#         self.layers = nn.ModuleList([
#             TransformerBlock(embed_dim, num_heads)  # Remove nn.Sequential wrapper
#             for _ in range(num_layers)
#         ])
#         self.uncertainty_layers = nn.ModuleList([nn.Linear(embed_dim, 1) for _ in range(num_layers)])
#         # self.emb = ConvBlock3D(512, 256, 0.2) 

#     def calculate_uncertainty(self, x, layer_idx):
#         """
#         Calculate token-level uncertainty for a specific layer using MC Dropout or learned variance.
#         """
#         variance = self.uncertainty_layers[layer_idx](x).sigmoid()  # Values in [0, 1]
#         return variance

#     def forward(self, x):
#         b, c, h, w, d = x.shape
#         x = self.positional_encoding(x)
#         # gt_x = rearrange(gt_x, 'b c h w d -> b (h w d) c')
#         # masked_gt_list = []
#         # masked_out_list = []
#         for i, layer in enumerate(self.layers):
#             x, att = layer(x,i)

#             # print("at shape is", att.shape)
            
#             if i >= 5 and i < (self.num_layers - 1):  # After the 6th layer, estimate uncertainty and mask
                
#                 uncertainty = self.calculate_uncertainty(x, i)
#                 # print("Uncertainty shape is", uncertainty.shape)
                
#                 # Calculate certainty mask
#                 certainty_mask = 1 - uncertainty  # Certainty is the complement of uncertainty
#                 x = x * certainty_mask  # Apply soft masking to the model's output
                
#                 # Apply the same mask to the ground truth
#                 # masked_gt = gt_x * certainty_mask  # Mask the ground truth similarly
#                 # masked_gt_list.append(masked_gt)  # Store masked GT for this layer
#                 # masked_out_list.append(x)
            
#         x = rearrange(x, 'b (h w d) c -> b c h w d', h=h, w=w, d=d)
#         # x = self.emb(x)
#         # gt_x_up = rearrange(gt_x_up, 'b (h w d) c -> b c h w d', h=h, w=w, d=d)
#         return x


# class TransformerModeldec(nn.Module):
#     """Transformer Model with Full Attention, Uncertainty Estimation, and Soft Masking."""
#     def __init__(self, input_shape, embed_dim, num_layers, num_heads):
#         super(TransformerModeldec, self).__init__()
#         h, w, d = input_shape
#         self.num_layers = num_layers
#         self.embed_dim = embed_dim
#         self.positional_encoding = PositionalEncoding3D(embed_dim=embed_dim, 
#                                                         spatial_size=[h, w, d])
#         self.layers = nn.ModuleList([
#             TransformerBlockdec(embed_dim, num_heads)  # Remove nn.Sequential wrapper
#             for _ in range(num_layers)
#         ])
#         # self.uncertainty_layers = nn.ModuleList([nn.Linear(embed_dim, 1) for _ in range(num_layers)])
#         # self.emb = ConvBlock3D(512, 256, 0.2) 

#     # def calculate_uncertainty(self, x, layer_idx):
#     #     """
#     #     Calculate token-level uncertainty for a specific layer using MC Dropout or learned variance.
#     #     """
#     #     variance = self.uncertainty_layers[layer_idx](x).sigmoid()  # Values in [0, 1]
#     #     return variance

#     def forward(self, x):
#         b, c, h, w, d = x.shape
#         x = self.positional_encoding(x)
#         # gt_x = rearrange(gt_x, 'b c h w d -> b (h w d) c')
#         # masked_gt_list = []
#         # masked_out_list = []
#         for i, layer in enumerate(self.layers):
#             x, att = layer(x,i)

#             # print("at shape is", att.shape)
            
#             # if i >= 5 and i < (self.num_layers - 1):  # After the 6th layer, estimate uncertainty and mask
                
#             #     uncertainty = self.calculate_uncertainty(x, i)
#             #     # print("Uncertainty shape is", uncertainty.shape)
                
#             #     # Calculate certainty mask
#             #     certainty_mask = 1 - uncertainty  # Certainty is the complement of uncertainty
#             #     x = x * certainty_mask  # Apply soft masking to the model's output
                
#                 # Apply the same mask to the ground truth
#                 # masked_gt = gt_x * certainty_mask  # Mask the ground truth similarly
#                 # masked_gt_list.append(masked_gt)  # Store masked GT for this layer
#                 # masked_out_list.append(x)
            
#         x = rearrange(x, 'b (h w d) c -> b c h w d', h=h, w=w, d=d)
#         # x = self.emb(x)
#         # gt_x_up = rearrange(gt_x_up, 'b (h w d) c -> b c h w d', h=h, w=w, d=d)
#         return x











# mse_loss = nn.MSELoss()

# class EMAQuantizer(nn.Module):
#     """
#     Vector Quantization module using Exponential Moving Average (EMA) to learn the codebook parameters based on  Neural
#     Discrete Representation Learning by Oord et al. (https://arxiv.org/abs/1711.00937) and the official implementation
#     that can be found at https://github.com/deepmind/sonnet/blob/v2/sonnet/src/nets/vqvae.py#L148 and commit
#     58d9a2746493717a7c9252938da7efa6006f3739.

#     This module is not compatible with TorchScript while working in a Distributed Data Parallelism Module. This is due
#     to lack of TorchScript support for torch.distributed module as per https://github.com/pytorch/pytorch/issues/41353
#     on 22/10/2022. If you want to TorchScript your model, please turn set `ddp_sync` to False.

#     Args:
#         spatial_dims :  number of spatial spatial_dims.
#         num_embeddings: number of atomic elements in the codebook.
#         embedding_dim: number of channels of the input and atomic elements.
#         commitment_cost: scaling factor of the MSE loss between input and its quantized version. Defaults to 0.25.
#         decay: EMA decay. Defaults to 0.99.
#         epsilon: epsilon value. Defaults to 1e-5.
#         embedding_init: initialization method for the codebook. Defaults to "normal".
#         ddp_sync: whether to synchronize the codebook across processes. Defaults to True.
#     """

#     def __init__(
#         self,
#         spatial_dims: int,
#         num_embeddings: int,
#         embedding_dim: int,
#         commitment_cost: float = 0.25,
#         decay: float = 0.99,
#         epsilon: float = 1e-5,
#         embedding_init: str = "normal",
#         ddp_sync: bool = True,
#         pretrained_embedding: Optional[torch.Tensor] = None,  # Add this parameter
#     ):
#         super().__init__()
#         self.spatial_dims: int = spatial_dims
#         self.embedding_dim: int = embedding_dim
#         self.num_embeddings: int = num_embeddings
    
#         assert self.spatial_dims in [2, 3], ValueError(
#             f"EMAQuantizer only supports 4D and 5D tensor inputs but received spatial dims {spatial_dims}."
#         )
    
#         # Initialize embedding
#         self.embedding: torch.nn.Embedding = torch.nn.Embedding(self.num_embeddings, self.embedding_dim)
        
#         # Load pretrained embedding if provided
#         if pretrained_embedding is not None:
#             if pretrained_embedding.shape != (self.num_embeddings, self.embedding_dim):
#                 raise ValueError(
#                     f"Pretrained embedding must have shape ({self.num_embeddings}, {self.embedding_dim}), "
#                     f"but got {pretrained_embedding.shape}."
#                 )
#             self.embedding.weight.data.copy_(pretrained_embedding)
#         elif embedding_init == "kaiming_uniform":
#             torch.nn.init.kaiming_uniform_(self.embedding.weight.data, mode="fan_in", nonlinearity="linear")
#         # Otherwise, use default initialization (normal)
    
#         self.embedding.weight.requires_grad = False
    
#         self.commitment_cost: float = commitment_cost
    
#         self.register_buffer("ema_cluster_size", torch.zeros(self.num_embeddings))
#         self.register_buffer("ema_w", self.embedding.weight.data.clone())
    
#         self.decay: float = decay
#         self.epsilon: float = epsilon
    
#         self.ddp_sync: bool = ddp_sync
    
#         # Precalculating required permutation shapes
#         self.flatten_permutation: Sequence[int] = [0] + list(range(2, self.spatial_dims + 2)) + [1]
#         self.quantization_permutation: Sequence[int] = [0, self.spatial_dims + 1] + list(
#             range(1, self.spatial_dims + 1)
#         )

#     def quantize(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
#         """
#         Given an input it projects it to the quantized space and returns additional tensors needed for EMA loss.

#         Args:
#             inputs: Encoding space tensors

#         Returns:
#             torch.Tensor: Flatten version of the input of shape [B*D*H*W, C].
#             torch.Tensor: One-hot representation of the quantization indices of shape [B*D*H*W, self.num_embeddings].
#             torch.Tensor: Quantization indices of shape [B,D,H,W,1]

#         """
#         encoding_indices_view = list(inputs.shape)
#         del encoding_indices_view[1]

#         with torch.cuda.amp.autocast(enabled=False):
#             inputs = inputs.float()

#             # Converting to channel last format
#             flat_input = inputs.permute(self.flatten_permutation).contiguous().view(-1, self.embedding_dim)

#             # Calculate Euclidean distances
#             distances = (
#                 (flat_input**2).sum(dim=1, keepdim=True)
#                 + (self.embedding.weight.t() ** 2).sum(dim=0, keepdim=True)
#                 - 2 * torch.mm(flat_input, self.embedding.weight.t())
#             )

#             # Mapping distances to indexes
#             encoding_indices = torch.max(-distances, dim=1)[1]
#             print("encoding_indices shape issssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssss", encoding_indices)
#             encodings = torch.nn.functional.one_hot(encoding_indices, self.num_embeddings).float()

#             # Quantize and reshape
#             encoding_indices = encoding_indices.view(encoding_indices_view)

#         return flat_input, encodings, encoding_indices

#     def embed(self, embedding_indices: torch.Tensor) -> torch.Tensor:
#         """
#         Given encoding indices of shape [B,D,H,W,1] embeds them in the quantized space
#         [B, D, H, W, self.embedding_dim] and reshapes them to [B, self.embedding_dim, D, H, W] to be fed to the
#         decoder.

#         Args:
#             embedding_indices: Tensor in channel last format which holds indices referencing atomic
#                 elements from self.embedding

#         Returns:
#             torch.Tensor: Quantize space representation of encoding_indices in channel first format.
#         """
#         with torch.cuda.amp.autocast(enabled=False):
#             return self.embedding(embedding_indices).permute(self.quantization_permutation).contiguous()

#     @torch.jit.unused
#     def distributed_synchronization(self, encodings_sum: torch.Tensor, dw: torch.Tensor) -> None:
#         """
#         TorchScript does not support torch.distributed.all_reduce. This function is a bypassing trick based on the
#         example: https://pytorch.org/docs/stable/generated/torch.jit.unused.html#torch.jit.unused

#         Args:
#             encodings_sum: The summation of one hot representation of what encoding was used for each
#                 position.
#             dw: The multiplication of the one hot representation of what encoding was used for each
#                 position with the flattened input.

#         Returns:
#             None
#         """
#         if self.ddp_sync and torch.distributed.is_initialized():
#             torch.distributed.all_reduce(tensor=encodings_sum, op=torch.distributed.ReduceOp.SUM)
#             torch.distributed.all_reduce(tensor=dw, op=torch.distributed.ReduceOp.SUM)
#         else:
#             pass

#     def forward(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
#         flat_input, encodings, encoding_indices = self.quantize(inputs)
#         quantized = self.embed(encoding_indices)

#         # Use EMA to update the embedding vectors
#         if self.training:
#             print("EMA Training Started")
#             with torch.no_grad():
#                 encodings_sum = encodings.sum(0)
#                 dw = torch.mm(encodings.t(), flat_input)

#                 if self.ddp_sync:
#                     self.distributed_synchronization(encodings_sum, dw)

#                 self.ema_cluster_size.data.mul_(self.decay).add_(torch.mul(encodings_sum, 1 - self.decay))

#                 # Laplace smoothing of the cluster size
#                 n = self.ema_cluster_size.sum()
#                 weights = (self.ema_cluster_size + self.epsilon) / (n + self.num_embeddings * self.epsilon) * n
#                 self.ema_w.data.mul_(self.decay).add_(torch.mul(dw, 1 - self.decay))
#                 self.embedding.weight.data.copy_(self.ema_w / weights.unsqueeze(1))
#         else:
#             encodings_sum=torch.zeros(256)

#         # print("self.embedding.weight.data", (self.embedding.weight.data).shape)
#         print("quantized ema shape is", quantized.shape)
#         print("inputs ema shape is", inputs.shape)
#         # Encoding Loss
        
#         loss = self.commitment_cost * mse_loss(quantized.detach(), inputs)
#         loss = loss

#         # Straight Through Estimator
#         quantized = inputs + (quantized - inputs).detach()

#         return quantized, loss, encoding_indices, encodings_sum, self.embedding.weight.data


# class VectorQuantizer(torch.nn.Module):
#     """
#     Vector Quantization wrapper that is needed as a workaround for the AMP to isolate the non fp16 compatible parts of
#     the quantization in their own class.

#     Args:
#         quantizer (torch.nn.Module):  Quantizer module that needs to return its quantized representation, loss and index
#             based quantized representation. Defaults to None
#     """

#     def __init__(self, quantizer: torch.nn.Module = None):
#         super().__init__()

#         self.quantizer: torch.nn.Module = quantizer

#         self.perplexity: torch.Tensor = torch.rand(1)

#     def forward(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
#         quantized, loss, encoding_indices, encodings_sum, embedding = self.quantizer(inputs)

#         # Perplexity calculations
#         avg_probs = (
#             torch.histc(encoding_indices.float(), bins=self.quantizer.num_embeddings, max=self.quantizer.num_embeddings)
#             .float()
#             .div(encoding_indices.numel())
#         )

#         self.perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
#         print("self.perplexity", self.perplexity)
#         # loss += 0.5 * self.perplexity

#         return loss, quantized, encodings_sum, embedding

#     def embed(self, embedding_indices: torch.Tensor) -> torch.Tensor:
#         return self.quantizer.embed(embedding_indices=embedding_indices)

#     def quantize(self, encodings: torch.Tensor) -> torch.Tensor:
#         quantized, loss, encoding_indices, encodings_sum, embedding = self.quantizer(encodings)

#         return encoding_indices


# # class EMAQuantizer_indice(nn.Module):
# #     """
# #     Vector Quantization module using Exponential Moving Average (EMA) to learn the codebook parameters based on  Neural
# #     Discrete Representation Learning by Oord et al. (https://arxiv.org/abs/1711.00937) and the official implementation
# #     that can be found at https://github.com/deepmind/sonnet/blob/v2/sonnet/src/nets/vqvae.py#L148 and commit
# #     58d9a2746493717a7c9252938da7efa6006f3739.

# #     This module is not compatible with TorchScript while working in a Distributed Data Parallelism Module. This is due
# #     to lack of TorchScript support for torch.distributed module as per https://github.com/pytorch/pytorch/issues/41353
# #     on 22/10/2022. If you want to TorchScript your model, please turn set `ddp_sync` to False.

# #     Args:
# #         spatial_dims :  number of spatial spatial_dims.
# #         num_embeddings: number of atomic elements in the codebook.
# #         embedding_dim: number of channels of the input and atomic elements.
# #         commitment_cost: scaling factor of the MSE loss between input and its quantized version. Defaults to 0.25.
# #         decay: EMA decay. Defaults to 0.99.
# #         epsilon: epsilon value. Defaults to 1e-5.
# #         embedding_init: initialization method for the codebook. Defaults to "normal".
# #         ddp_sync: whether to synchronize the codebook across processes. Defaults to True.
# #     """

# #     def __init__(
# #         self,
# #         spatial_dims: int,
# #         num_embeddings: int,
# #         embedding_dim: int,
# #         commitment_cost: float = 0.25,
# #         decay: float = 0.99,
# #         epsilon: float = 1e-5,
# #         embedding_init: str = "normal",
# #         ddp_sync: bool = True,
# #         pretrained_embedding: Optional[torch.Tensor] = None,  # Add this parameter
# #     ):
# #         super().__init__()
# #         self.spatial_dims: int = spatial_dims
# #         self.embedding_dim: int = embedding_dim
# #         self.num_embeddings: int = num_embeddings
    
# #         assert self.spatial_dims in [2, 3], ValueError(
# #             f"EMAQuantizer only supports 4D and 5D tensor inputs but received spatial dims {spatial_dims}."
# #         )
    
# #         # Initialize embedding
# #         self.embedding: torch.nn.Embedding = torch.nn.Embedding(self.num_embeddings, self.embedding_dim)
        
# #         # Load pretrained embedding if provided
# #         if pretrained_embedding is not None:
# #             if pretrained_embedding.shape != (self.num_embeddings, self.embedding_dim):
# #                 raise ValueError(
# #                     f"Pretrained embedding must have shape ({self.num_embeddings}, {self.embedding_dim}), "
# #                     f"but got {pretrained_embedding.shape}."
# #                 )
# #             self.embedding.weight.data.copy_(pretrained_embedding)
# #         elif embedding_init == "kaiming_uniform":
# #             torch.nn.init.kaiming_uniform_(self.embedding.weight.data, mode="fan_in", nonlinearity="linear")
# #         # Otherwise, use default initialization (normal)
    
# #         self.embedding.weight.requires_grad = False
    
# #         self.commitment_cost: float = commitment_cost
    
# #         self.register_buffer("ema_cluster_size", torch.zeros(self.num_embeddings))
# #         self.register_buffer("ema_w", self.embedding.weight.data.clone())
    
# #         self.decay: float = decay
# #         self.epsilon: float = epsilon
    
# #         self.ddp_sync: bool = ddp_sync
    
# #         # Precalculating required permutation shapes
# #         self.flatten_permutation: Sequence[int] = [0] + list(range(2, self.spatial_dims + 2)) + [1]
# #         self.quantization_permutation: Sequence[int] = [0, self.spatial_dims + 1] + list(
# #             range(1, self.spatial_dims + 1)
# #         )

# #     def quantize(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
# #         """
# #         Given an input it projects it to the quantized space and returns additional tensors needed for EMA loss.

# #         Args:
# #             inputs: Encoding space tensors

# #         Returns:
# #             torch.Tensor: Flatten version of the input of shape [B*D*H*W, C].
# #             torch.Tensor: One-hot representation of the quantization indices of shape [B*D*H*W, self.num_embeddings].
# #             torch.Tensor: Quantization indices of shape [B,D,H,W,1]

# #         """
# #         encoding_indices_view = list(inputs.shape)
# #         del encoding_indices_view[1]

# #         with torch.cuda.amp.autocast(enabled=False):
# #             inputs = inputs.float()

# #             # Converting to channel last format
# #             flat_input = inputs.permute(self.flatten_permutation).contiguous().view(-1, self.embedding_dim)

# #             # Calculate Euclidean distances
# #             distances = (
# #                 (flat_input**2).sum(dim=1, keepdim=True)
# #                 + (self.embedding.weight.t() ** 2).sum(dim=0, keepdim=True)
# #                 - 2 * torch.mm(flat_input, self.embedding.weight.t())
# #             )

# #             # Mapping distances to indexes
# #             encoding_indices = torch.max(-distances, dim=1)[1]
# #             print("encoding_indices shape issssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssss", encoding_indices)
# #             encodings = torch.nn.functional.one_hot(encoding_indices, self.num_embeddings).float()

# #             # Quantize and reshape
# #             encoding_indices = encoding_indices.view(encoding_indices_view)

# #         return flat_input, encodings, encoding_indices

# #     def embed(self, embedding_indices: torch.Tensor) -> torch.Tensor:
# #         """
# #         Given encoding indices of shape [B,D,H,W,1] embeds them in the quantized space
# #         [B, D, H, W, self.embedding_dim] and reshapes them to [B, self.embedding_dim, D, H, W] to be fed to the
# #         decoder.

# #         Args:
# #             embedding_indices: Tensor in channel last format which holds indices referencing atomic
# #                 elements from self.embedding

# #         Returns:
# #             torch.Tensor: Quantize space representation of encoding_indices in channel first format.
# #         """
# #         with torch.cuda.amp.autocast(enabled=False):
# #             return self.embedding(embedding_indices).permute(self.quantization_permutation).contiguous()

# #     @torch.jit.unused
# #     def distributed_synchronization(self, encodings_sum: torch.Tensor, dw: torch.Tensor) -> None:
# #         """
# #         TorchScript does not support torch.distributed.all_reduce. This function is a bypassing trick based on the
# #         example: https://pytorch.org/docs/stable/generated/torch.jit.unused.html#torch.jit.unused

# #         Args:
# #             encodings_sum: The summation of one hot representation of what encoding was used for each
# #                 position.
# #             dw: The multiplication of the one hot representation of what encoding was used for each
# #                 position with the flattened input.

# #         Returns:
# #             None
# #         """
# #         if self.ddp_sync and torch.distributed.is_initialized():
# #             torch.distributed.all_reduce(tensor=encodings_sum, op=torch.distributed.ReduceOp.SUM)
# #             torch.distributed.all_reduce(tensor=dw, op=torch.distributed.ReduceOp.SUM)
# #         else:
# #             pass

# #     def forward(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
# #         flat_input, encodings, encoding_indices = self.quantize(inputs)
# #         quantized = self.embed(encoding_indices)

# #         # Use EMA to update the embedding vectors
# #         if self.training:
# #             print("EMA Training Started")
# #             with torch.no_grad():
# #                 encodings_sum = encodings.sum(0)
# #                 dw = torch.mm(encodings.t(), flat_input)

# #                 if self.ddp_sync:
# #                     self.distributed_synchronization(encodings_sum, dw)

# #                 self.ema_cluster_size.data.mul_(self.decay).add_(torch.mul(encodings_sum, 1 - self.decay))

# #                 # Laplace smoothing of the cluster size
# #                 n = self.ema_cluster_size.sum()
# #                 weights = (self.ema_cluster_size + self.epsilon) / (n + self.num_embeddings * self.epsilon) * n
# #                 self.ema_w.data.mul_(self.decay).add_(torch.mul(dw, 1 - self.decay))
# #                 self.embedding.weight.data.copy_(self.ema_w / weights.unsqueeze(1))
# #         else:
# #             encodings_sum=torch.zeros(256)

# #         # print("self.embedding.weight.data", (self.embedding.weight.data).shape)
# #         print("quantized ema shape is", quantized.shape)
# #         print("inputs ema shape is", inputs.shape)
# #         # Encoding Loss
        
# #         loss = self.commitment_cost * mse_loss(quantized.detach(), inputs)
# #         loss = loss

# #         # Straight Through Estimator
# #         encoding_indices1=encoding_indices.unsqueeze(dim=1)
# #         additional_channels = torch.zeros(encoding_indices1.size(0), 63, *encoding_indices1.size()[2:], device=encoding_indices1.device) 
# #         extended_tensor=torch.cat([encoding_indices1, additional_channels], dim=1) 
# #         encoding_indices1 = inputs + (extended_tensor - inputs).detach()
# #         quantized = inputs + (quantized - inputs).detach()

# #         return quantized, loss, encoding_indices, encodings_sum, self.embedding.weight.data, encoding_indices1


# # class VectorQuantizer_indice(torch.nn.Module):
# #     """
# #     Vector Quantization wrapper that is needed as a workaround for the AMP to isolate the non fp16 compatible parts of
# #     the quantization in their own class.

# #     Args:
# #         quantizer (torch.nn.Module):  Quantizer module that needs to return its quantized representation, loss and index
# #             based quantized representation. Defaults to None
# #     """

# #     def __init__(self, quantizer: torch.nn.Module = None):
# #         super().__init__()

# #         self.quantizer: torch.nn.Module = quantizer

# #         self.perplexity: torch.Tensor = torch.rand(1)

# #     def forward(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
# #         quantized, loss, encoding_indices, encodings_sum, embedding, encoding_indices1 = self.quantizer(inputs)

# #         # Perplexity calculations
# #         avg_probs = (
# #             torch.histc(encoding_indices.float(), bins=self.quantizer.num_embeddings, max=self.quantizer.num_embeddings)
# #             .float()
# #             .div(encoding_indices.numel())
# #         )

# #         self.perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

# #         return loss, quantized, encodings_sum, embedding, encoding_indices1

# #     def embed(self, embedding_indices: torch.Tensor) -> torch.Tensor:
# #         return self.quantizer.embed(embedding_indices=embedding_indices)

# #     def quantize(self, encodings: torch.Tensor) -> torch.Tensor:
# #         quantized, loss, encoding_indices, encodings_sum, embedding = self.quantizer(encodings)

# #         return encoding_indices











# class ConvBlock3D_wo_p(nn.Module):
#     """Convolution Block with Conv3d, BatchNorm, ReLU, and Dropout"""
#     def __init__(self, in_channels, out_channels, dropout_prob):
#         super(ConvBlock3D_wo_p, self).__init__()
#         self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)
#         self.relu = nn.ReLU(inplace=True)
#         self.batch_norm = nn.BatchNorm3d(out_channels)
#         self.dropout = nn.Dropout3d(p=dropout_prob)
    
#     def forward(self, x):
#         x = self.conv(x)
#         x = self.relu(x)
#         x = self.batch_norm(x)
#         x = self.dropout(x)
#         return x






# class ConvBlock3D_won_p(nn.Module):
#     """Convolution Block with Conv3d, BatchNorm, ReLU, and Dropout"""
#     def __init__(self, in_channels, out_channels, dropout_prob):
#         super(ConvBlock3D_won_p, self).__init__()
#         self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)
#         self.relu = nn.ReLU(inplace=True)
#         self.batch_norm = nn.BatchNorm3d(out_channels)
#         self.dropout = nn.Dropout3d(p=dropout_prob)
    
#     def forward(self, x):
#         x = self.conv(x)
#         x = self.relu(x)
#         # x = self.batch_norm(x)
#         # x = self.dropout(x)
#         return x


# class Encoder3D(nn.Module):
#     """Encoder consisting of multiple convolution blocks with increasing feature maps"""
#     def __init__(self, in_channels, dropout_prob=0.5):
#         super(Encoder3D, self).__init__()
        
#         self.encoder1 = ConvBlock3D(in_channels, 8, dropout_prob)
#         self.encoder2 = ConvBlock3D(8, 16, dropout_prob)
#         self.encoder3 = ConvBlock3D(16, 32, dropout_prob)
#         self.encoder3_cat = ConvBlock3D(160, 32, dropout_prob)
#         self.encoder4 = ConvBlock3D(32, 64, dropout_prob)
#         self.encoder5 = ConvBlock3D(64, 128, dropout_prob)
#         # self.res5 = ResidualBlock(128)
#         self.encoder6 = ConvBlock3D(128, 256, dropout_prob)
#         # self.encoder7 = ConvBlock3D(256, 512, dropout_prob)
#         # self.encoder8 = ConvBlock3D(512, 1024, dropout_prob)
#         # self.encoder9 = ConvBlock3D(1024, 512, dropout_prob)
#         self.pool = nn.MaxPool3d(2)
#         self.condition_prob = 0.7
#     def forward(self, x, autoencoder_latent, is_train=False):
#         # x1 = self.encoder1(x)
#         # print(f"Encoder1 output shape: {x1.shape}")
#         # x2 = self.encoder2(self.pool(x1))
#         # print(f"Encoder2 output shape: {x2.shape}")
#         # x3 = self.encoder3(self.pool(x2))
#         # print(f"Encoder3 output shape: {x3.shape}")
#         # x4 = self.encoder4((x3))
#         # # print(f"Encoder4 output shape: {x4.shape}")
#         x1 = self.encoder1(x)
#         print(f"Encoder1 output shape: {x1.shape}")
#         # x1 = self.res1(x1)
#         x2 = self.encoder2(self.pool(x1))
#         print(f"Encoder2 output shape: {x2.shape}")
#         # x2 = self.res2(x2)
#         x3 = self.encoder3(self.pool(x2))
#         print(f"Encoder3 output shape: {x3.shape}") 
#         if is_train and random.random() >= self.condition_prob:
#             print("Including segmentation mask latent during training.")
#             x3 = self.encoder3_cat(torch.cat((autoencoder_latent, x3), dim=1))
#         else:
#             print("Skipping segmentation mask latent.")
        
#         print(f"Encoder3 (after possible concatenation) output shape: {x3.shape}")
#         # x3 = self.encoder3_cat(torch.cat((autoencoder_latent.float(), x3), dim=1))
#         # x3 = self.res3(x3)
#         x4 = self.encoder4(self.pool(x3))
#         print(f"Encoder4 output shape: {x4.shape}")
#         # x4 = self.res4(x4)
#         x5 = self.encoder5(self.pool(x4))
#         print(f"Encoder5 output shape: {x5.shape}")
#         x6 = self.encoder6((x5))
#         print(f"Encoder6 output shape: {x6.shape}")
#         # x7 = self.encoder7((x6))
#         # print(f"Encoder6 output shape: {x7.shape}")
#         # x8 = self.encoder8((x7))
#         # print(f"Encoder6 output shape: {x8.shape}")
#         # x8 = self.encoder9((x8))
#         # print(f"Encoder6 output shape: {x8.shape}")
#         # padding = (0, 4, 0, 1, 0, 1)  # (left, right, top, bottom, front, back)
#         # x6 = F.pad(x6, padding, mode='constant', value=0)
        
#         # x5 = self.res5(x5)
#         # x6 = self.encoder6(x5)
#         return x1, x2, x3, x4, x5, x6
        
# class BottleneckBlock(nn.Module):
#     """Bottleneck block with 128 to 128 features"""
#     def __init__(self, in_channels, dropout_prob=0.5):
#         super(BottleneckBlock, self).__init__()
#         self.bottleneck = ConvBlock3D_won_p(in_channels, in_channels, dropout_prob)
        
#     def forward(self, x):
#         x = self.bottleneck(x)
#         print(f"Bottleneck output shape: {x.shape}")
#         return x

# # class Indice3D(nn.Module):
# #     """Encoder consisting of multiple convolution blocks with increasing feature maps"""
# #     def __init__(self, in_channels, dropout_prob=0.5):
# #         super(Indice3D, self).__init__()
        
# #         self.encoder1 = ConvBlock3D(in_channels, 64, dropout_prob)
# #         self.encoder2 = ConvBlock3D(64, 64, dropout_prob)
# #         self.encoder3 = ConvBlock3D(64, 64, dropout_prob)
# #         self.encoder4 = ConvBlock3D(64, 64, dropout_prob)
# #         self.conv1 = nn.Conv3d(64, 64, kernel_size=3, padding=1)
# #         self.conv2 = nn.Conv3d(64, 64, kernel_size=3, padding=1)
# #         # self.conv3 = nn.Conv3d(1, 1, kernel_size=1) 
        
        
# #     def forward(self, x):
# #         x1 = self.encoder1(x)
# #         print(f"indice1 output shape: {x1.shape}")
# #         x2 = self.encoder2(x1)
# #         print(f"indice2 output shape: {x2.shape}")
# #         x3 = self.encoder3(x2)
# #         print(f"indice3 output shape: {x3.shape}")
# #         x4 = self.encoder4(x3)
# #         print(f"indice4 output shape: {x4.shape}")
# #         x5 = self.conv1(x4)
# #         x5 = F.relu(x5)
# #         x6 = self.conv2(x5)
# #         # x6 = F.relu(x6)
# #         # x7 = self.conv3(x6)
# #         # x7 = F.relu(x7)
# #         return x6

# class GELU2(nn.Module):
#     def __init__(self):
#         super().__init__()
#     def forward(self, x):
#         return x * F.sigmoid(1.702 * x)


# class SEBlock3D(nn.Module):
#     def __init__(self, in_channels, reduction=16):
#         super(SEBlock3D, self).__init__()
#         self.global_avg_pool = nn.AdaptiveAvgPool3d(1)  # Output shape: (batch, channels, 1, 1, 1)
#         self.fc1 = nn.Conv3d(in_channels, in_channels // reduction, kernel_size=1)
#         self.fc2 = nn.Conv3d(in_channels // reduction, in_channels, kernel_size=1)
#         self.sigmoid = nn.Sigmoid()

#     def forward(self, x):
#         x_squeezed = self.global_avg_pool(x)  # Global pooling
#         x_fc = torch.relu(self.fc1(x_squeezed))  # Fully connected layer 1
#         x_fc = self.sigmoid(self.fc2(x_fc))  # Fully connected layer 2 with sigmoid
#         return x * x_fc  # Scale input by channel attention weights

# class SpatialAttention3D(nn.Module):
#     def __init__(self, in_channels):
#         super(SpatialAttention3D, self).__init__()
#         self.conv1 = nn.Conv3d(in_channels, 1, kernel_size=1)  # Single channel output
#         self.sigmoid = nn.Sigmoid()

#     def forward(self, x):
#         # avg_pool = torch.mean(x, dim=1, keepdim=True)  # Pool across channels
#         spatial_attention = self.sigmoid(self.conv1(x))  # Spatial attention map
#         return x * spatial_attention  # Scale input by spatial attention map


# class CombinedAttention3D(nn.Module):
#     def __init__(self, in_channels, reduction=16):
#         super(CombinedAttention3D, self).__init__()
#         self.channel_attention = SEBlock3D(in_channels, reduction)
#         self.spatial_attention = SpatialAttention3D(in_channels)

#     def forward(self, x):
#         x = self.channel_attention(x)  # Channel-wise attention
#         x = self.spatial_attention(x)  # Spatial attention
#         return x

# class Decoder3D(nn.Module):
#     """Decoder with skip connections and upsampling"""
#     def __init__(self, dropout_prob=0.5):
#         super(Decoder3D, self).__init__()
#         self.upsample_0 = self.upsample_block1(256, dropout_prob)
#         self.transmodel = TransformerModeldec(input_shape=[30, 30, 18], embed_dim=256, num_layers=3, num_heads=8)
#         # self.transmodel2 = TransformerModeldec(input_shape=[30, 30, 18], embed_dim=256, num_layers=2, num_heads=8)
#         # self.attention = CombinedAttention3D(128, reduction=4)  # Add attention block
#         # self.conv3d = ConvBlock3D(192, 128, dropout_prob)
#         self.conv_0 = ConvBlock3D_wo_p(192, 64, dropout_prob)
#         self.upsample_1 = self.upsample_block(256, dropout_prob)
#         self.upsample_2 = self.upsample_block1(64, dropout_prob)
#         self.conv_1 = nn.Conv3d(96, 128, kernel_size=3, padding=1)
#         # self.conv_1 = nn.Conv3d(256, 128, kernel_size=3, padding=1)
#         # self.conv_2 = nn.Conv3d(128, 256, kernel_size=3)
#         # self.conv_3 = nn.Conv3d(256, 128, kernel_size=3, padding=1)
#         # self.conv_4 = nn.Conv3d(256, 128, kernel_size=3, padding=1)
#         # self.alpha = nn.Parameter(torch.tensor(0.5))  # Initial weight for low confidence
#         # self.beta = nn.Parameter(torch.tensor(0.5))   # Initial weight for high confidence
#         # self.upsample0 = self.upsample_block(128, dropout_prob)
#         # self.conv0 = nn.Conv3d(128, 64, kernel_size=3, padding=1)
        
#         # self.upsample1 = self.upsample_block(64, dropout_prob)
#         # self.conv1 = nn.Conv3d(64, 32, kernel_size=3, padding=1)
#         # self.upsample2 = self.upsample_block(32, dropout_prob)
#         # self.conv2 = nn.Conv3d(32, 16, kernel_size=3, padding=1)
#         # self.upsample3 = self.upsample_block(16, dropout_prob)
#         # self.conv3 = nn.Conv3d(16, 8, kernel_size=3, padding=1)
        
#         # self.final_conv = nn.Conv3d(8, 1, kernel_size=1)  # Assuming segmentation output is single channel
    
#     def upsample_block(self, in_channels, dropout_prob):
#         """Create an upsampling block with Conv3d, ReLU, BatchNorm, and Dropout"""
#         layers = [
#             nn.Upsample(scale_factor=1, mode='nearest'),
#             nn.Conv3d(in_channels, in_channels // 2, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.BatchNorm3d(in_channels // 2),
#             nn.Dropout3d(p=dropout_prob),
#         ]
#         return nn.Sequential(*layers)
#     def upsample_block1(self, in_channels, dropout_prob):
#         """Create an upsampling block with Conv3d, ReLU, BatchNorm, and Dropout"""
#         layers = [
#             nn.Upsample(scale_factor=2, mode='nearest'),
#             nn.Conv3d(in_channels, in_channels, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.BatchNorm3d(in_channels),
#             nn.Dropout3d(p=dropout_prob),
#         ]
#         return nn.Sequential(*layers)

#     def forward(self, x, x4, x3):
#         print(f"Decoder input (x): {x.shape}")
#         x_6 = self.upsample_0(x)
#         x_6 = self.transmodel(x_6)
#         x_6 = self.upsample_1(x_6)
#         # x_6 = self.conv_0(x_6)
#         # x_6 = x_6[:, :, 0:15, 0:15, 0:9]
#         # print("x_6 shape is", x_6.shape)
        
#         padding = (0, 1, 0, 0, 0, 0)  # (left, right, top, bottom, front, back)
#         x_6 = F.pad(x_6, padding, mode='constant', value=0)
        
#         # x_6 = self.conv_0(x_6)
#         # print("x_6_pad shape is", x_6.shape)
#         x_6 = self.conv_0(torch.cat((x_6, x4), dim=1))  # Concatenate with encoder3
#         x_6 = self.upsample_2(x_6)
#         x_6 = self.conv_1(torch.cat((x_6, x3), dim=1))  # Concatenate with encoder3
#         # x_6 = torch.softmax(x_6, dim=1)
#         # padding = (0, 1, 1, 1, 1, 1)  # (left, right, top, bottom, front, back)
#         # x_6 = F.pad(x_6, padding, mode='constant', value=0)
#         # x_6 = self.transmodel(x_6)
#         # padding = (0, 1, 0, 0, 0, 0)  # (left, right, top, bottom, front, back)
#         # x_6 = F.pad(x_6, padding, mode='constant', value=0)
#         # x_6 = self.conv_1(x_6)



#         # probs = F.softmax(x_6, dim=1)  # Compute softmax probabilities
#         # max_probs, _ = probs.max(dim=1)  # Get max probabilities per voxel
        
#         # # Learnable scaling factors
        
#         # # Soft scaling
#         # low_conf_scale = (1 - max_probs).unsqueeze(1) * self.alpha
#         # high_conf_scale = max_probs.unsqueeze(1) * self.beta
        
#         # # Apply scaling
#         # low_conf_features = x_6 * low_conf_scale
#         # high_conf_features = x_6 * high_conf_scale
#         # low_conf_features = self.conv_2(low_conf_features)
#         # padding = (0, 1, 1, 1, 1, 1)  # (left, right, top, bottom, front, back)
#         # low_conf_features = F.pad(low_conf_features, padding, mode='constant', value=0)
#         # low_conf_features = self.transmodel2(low_conf_features)
#         # padding = (0, 1, 0, 0, 0, 0)  # (left, right, top, bottom, front, back)
#         # low_conf_features = F.pad(low_conf_features, padding, mode='constant', value=0)
#         # low_conf_features = self.conv_3(low_conf_features)

#         # final_logits = torch.cat((low_conf_features, high_conf_features), dim=1)
#         # final_logits = self.conv_4(final_logits)
#         # x_6 = self.attention(x_6)
#         # x_6 = self.conv_0(x_6)
#         # padding = (0, 1, 1, 1, 1, 1)  # (left, right, top, bottom, front, back)
#         # x_6 = F.pad(x_6, padding, mode='constant', value=0)
#         # print("x_6_pad shape is", x_6.shape)
#         # print("x_6 shape is after concatenation", x_6.shape)
#         # x_6 = self.upsample0(x_6)  # First decoder layer
#         # print(f"Upsample1 output shape: {x_6.shape}")

#         # padding = (0, 1, 0, 0, 0, 0)  # (left, right, top, bottom, front, back)
#         # x_6 = F.pad(x_6, padding, mode='constant', value=0)
#         # x_7 = self.conv0(torch.cat([x_6, x4], dim=1))  # Concatenate with encoder3
#         # print(f"Conv1 output shape after concatenation: {x_7.shape}")
#         # x6 = self.upsample1(x_7)  # First decoder layer
#         # print(f"Upsample1 output shape: {x6.shape}")
#         # x7 = self.conv1(torch.cat([x6, x3], dim=1))  # Concatenate with encoder3
#         # print(f"Conv1 output shape after concatenation: {x7.shape}")
#         # x7 = self.upsample2(x7)
#         # print(f"Upsample2 output shape: {x7.shape}")
#         # padding = (0, 1, 0, 0, 0, 0)  # (left, right, top, bottom, front, back)
#         # x7 = F.pad(x7, padding, mode='constant', value=0)
#         # x8 = self.conv2(torch.cat([x7, x2], dim=1))  # Concatenate with encoder2
#         # print(f"Conv2 output shape after concatenation: {x8.shape}")
#         # x8 = self.upsample3(x8)
#         # print(f"Upsample3 output shape: {x8.shape}")
#         # padding = (0, 1, 0, 0, 0, 0)  # (left, right, top, bottom, front, back)
#         # x8 = F.pad(x8, padding, mode='constant', value=0)
#         # x9 = self.conv3(torch.cat([x8, x1], dim=1))  # Concatenate with encoder1
#         # print(f"Conv3 output shape after concatenation: {x9.shape}")
#         # out = self.final_conv(x9)  # Final output
#         # print(f"Final output shape: {out.shape}")
#         return x_6



# class LatentSpaceMaskReducer(nn.Module):
#     def __init__(self, input_channels, reduction_factor=1):
#         super(LatentSpaceMaskReducer, self).__init__()
#         self.reduction_factor = reduction_factor
#         # 1x1 convolution to reduce channels to 1
#         self.channel_reduction = nn.Conv3d(input_channels, 1, kernel_size=1)
#         # A small network to learn the mask
#         self.mask_network = nn.Sequential(
#             nn.Conv3d(1, 1, kernel_size=3, padding=1),
#             nn.Sigmoid()  # Output is a mask between 0 and 1
#         )

#     def generate_mask(self, reduced_tensor):
#         # Generate a mask based on the reduced tensor
#         mask = self.mask_network(reduced_tensor)
#         return mask

#     def reduce_latent(self, latent_tensor):
#         B, C, H, W, D = latent_tensor.shape
        
#         # Ensure tensor is on the correct device
        
#         # Reduce channels from C to 1 using a 1x1 convolution
#         reduced_tensor = self.channel_reduction(latent_tensor)  # Shape: (B, 1, D, H, W)

#         # Generate a mask that identifies the background and foreground
#         mask = self.generate_mask(reduced_tensor)  # Shape: (B, 1, D, H, W)

#         # Mask the original latent tensor
#         masked_tensor = latent_tensor * mask  # Element-wise multiplication

#         return masked_tensor




# # class EdgeRefinement3D(torch.nn.Module):
# #     def __init__(self):
# #         super(EdgeRefinement3D, self).__init__()
# #         self.conv = nn.Conv3d(128, 1, kernel_size=3, padding=1)

# #     def sobel_3d(self, input_tensor):
# #         """
# #         Apply a Sobel filter in 3D to compute gradients in the x, y, and z directions.
# #         """
# #         # Sobel kernel for the x direction
# #         sobel_x = torch.tensor(
# #             [[[[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
# #                [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
# #                [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]]]],
# #             dtype=torch.float32, device=input_tensor.device
# #         )

# #         # Sobel kernel for the y direction
# #         sobel_y = torch.tensor(
# #             [[[[[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
# #                [[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
# #                [[-1, -2, -1], [0, 0, 0], [1, 2, 1]]]]],
# #             dtype=torch.float32, device=input_tensor.device
# #         )

# #         # Sobel kernel for the z direction
# #         sobel_z = torch.tensor(
# #             [[[[[-1, -1, -1], [-1, -1, -1], [-1, -1, -1]],
# #                [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
# #                [[1, 1, 1], [1, 1, 1], [1, 1, 1]]]]],
# #             dtype=torch.float32, device=input_tensor.device
# #         )

# #         # Ensure input_tensor is 5D: (batch, channel, depth, height, width)
# #         if input_tensor.ndim != 5:
# #             raise ValueError("Expected input_tensor to be 5D (batch, channel, depth, height, width), got shape: {}".format(input_tensor.shape))

# #         # Apply Sobel filters in each direction
# #         print("input_tensor", input_tensor.shape)
# #         # F.conv3d(input_tensor, kernel=3, padding=1)
# #         grad_x = F.conv3d(input_tensor, sobel_x, stride=1, padding=1)
# #         grad_y = F.conv3d(input_tensor, sobel_y, stride=1, padding=1)
# #         grad_z = F.conv3d(input_tensor, sobel_z, stride=1, padding=1)

# #         # Combine gradients
# #         gradient_magnitude = torch.sqrt(grad_x**2 + grad_y**2 + grad_z**2)

# #         return gradient_magnitude

# #     def forward(self, input_tensor):
# #         """Forward pass for Edge Refinement."""
# #         x = self.conv(input_tensor)
# #         edge_map = self.sobel_3d(x)
# #         input_tensor_up = edge_map * input_tensor
# #         return input_tensor_up

# # class EdgeRefinement3D_lap(torch.nn.Module):
# #     def __init__(self):
# #         super(EdgeRefinement3D_lap, self).__init__()
# #         self.conv = nn.Conv3d(128, 1, kernel_size=3, padding=1)

# #     def laplacian_3d(self, input_tensor):
# #         """
# #         Apply a Laplacian filter in 3D to compute second-order derivatives.
# #         """
# #         # Laplacian kernel
# #         laplacian_kernel = torch.tensor(
# #             [[[[[0, 1, 0], [1, -6, 1], [0, 1, 0]],
# #                [[1, -6, 1], [-6, 36, -6], [1, -6, 1]],
# #                [[0, 1, 0], [1, -6, 1], [0, 1, 0]]]]],
# #             dtype=torch.float32, device=input_tensor.device
# #         )

# #         # Ensure input_tensor is 5D: (batch, channel, depth, height, width)
# #         if input_tensor.ndim != 5:
# #             raise ValueError("Expected input_tensor to be 5D (batch, channel, depth, height, width), got shape: {}".format(input_tensor.shape))

# #         # Apply the Laplacian filter
# #         laplacian_output = F.conv3d(input_tensor, laplacian_kernel, stride=1, padding=1)

# #         return laplacian_output

# #     def forward(self, input_tensor):
# #         """Forward pass for Edge Refinement."""
# #         x = self.conv(input_tensor)
# #         laplacian_map = self.laplacian_3d(x)
# #         input_tensor_up = laplacian_map * input_tensor
# #         return input_tensor_up



# # import torch
# # import torch.nn as nn
# # import torch.nn.functional as F

# class EdgeRefinement3D(torch.nn.Module):
#     def __init__(self):
#         super(EdgeRefinement3D, self).__init__()
#         self.conv = nn.Conv3d(128, 1, kernel_size=3, padding=1)

#     def gaussian_filter_3d(self, input_tensor, kernel_size=3, sigma=1.0):
#         """
#         Apply a Gaussian filter in 3D for smoothing.
#         """
#         # Create a 3D Gaussian kernel
#         kernel = self.create_gaussian_kernel(kernel_size, sigma, input_tensor.device)

#         # Ensure input_tensor is 5D: (batch, channel, depth, height, width)
#         if input_tensor.ndim != 5:
#             raise ValueError("Expected input_tensor to be 5D (batch, channel, depth, height, width), got shape: {}".format(input_tensor.shape))

#         # Apply the Gaussian filter using convolution
#         gaussian_output = F.conv3d(input_tensor, kernel, stride=1, padding=kernel_size//2)
#         return gaussian_output

#     def create_gaussian_kernel(self, kernel_size, sigma, device):
#         """
#         Create a 3D Gaussian kernel.
#         """
#         # Create a 1D Gaussian kernel
#         kernel_1d = torch.linspace(-(kernel_size // 2), kernel_size // 2, kernel_size, device=device)
#         kernel_1d = torch.exp(-0.5 * (kernel_1d / sigma) ** 2)

#         # Normalize the kernel
#         kernel_1d = kernel_1d / kernel_1d.sum()

#         # Create a 3D Gaussian kernel by taking the outer product of the 1D kernels
#         kernel_3d = kernel_1d.view(1, 1, kernel_size, 1, 1) * kernel_1d.view(1, 1, 1, kernel_size, 1) * kernel_1d.view(1, 1, 1, 1, kernel_size)
#         kernel_3d = kernel_3d.expand(1, 1, kernel_size, kernel_size, kernel_size)  # Expand to 3D kernel

#         # Ensure the kernel has the correct shape for convolution
#         return kernel_3d

#     def sobel_3d(self, input_tensor):
#         """
#         Apply a Sobel filter in 3D to compute gradients in the x, y, and z directions.
#         """
#         # Sobel kernel for the x direction
#         sobel_x = torch.tensor(
#             [[[[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
#                [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
#                [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]]]],
#             dtype=torch.float32, device=input_tensor.device
#         )

#         # Sobel kernel for the y direction
#         sobel_y = torch.tensor(
#             [[[[[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
#                [[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
#                [[-1, -2, -1], [0, 0, 0], [1, 2, 1]]]]],
#             dtype=torch.float32, device=input_tensor.device
#         )

#         # Sobel kernel for the z direction
#         sobel_z = torch.tensor(
#             [[[[[-1, -1, -1], [-1, -1, -1], [-1, -1, -1]],
#                [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
#                [[1, 1, 1], [1, 1, 1], [1, 1, 1]]]]],
#             dtype=torch.float32, device=input_tensor.device
#         )

#         # Ensure input_tensor is 5D: (batch, channel, depth, height, width)
#         if input_tensor.ndim != 5:
#             raise ValueError("Expected input_tensor to be 5D (batch, channel, depth, height, width), got shape: {}".format(input_tensor.shape))

#         # Apply Sobel filters in each direction
#         grad_x = F.conv3d(input_tensor, sobel_x, stride=1, padding=1)
#         grad_y = F.conv3d(input_tensor, sobel_y, stride=1, padding=1)
#         grad_z = F.conv3d(input_tensor, sobel_z, stride=1, padding=1)

#         # Combine gradients
#         gradient_magnitude = torch.sqrt(grad_x**2 + grad_y**2 + grad_z**2)

#         return gradient_magnitude

#     def forward(self, input_tensor):
#         """Forward pass for Edge Refinement."""
#         # Apply the Gaussian filter to the input tensor first
#         x = self.conv(input_tensor)
#         smoothed_input = self.gaussian_filter_3d(x)

#         # Apply Sobel filter to the smoothed input
#         edge_map = self.sobel_3d(smoothed_input)

#         # Refine the input tensor using the edge map
#         input_tensor_up = edge_map * input_tensor
#         return input_tensor_up







# # import torch
# # import torch.nn as nn
# # import torch.nn.functional as F

# class EdgeRefinement3D_lap(torch.nn.Module):
#     def __init__(self):
#         super(EdgeRefinement3D_lap, self).__init__()
#         self.conv = nn.Conv3d(128, 1, kernel_size=3, padding=1)

#     def gaussian_3d(self, input_tensor, sigma=1.0):
#         """
#         Apply a 3D Gaussian filter to smooth the image.
#         """
#         # Define a 3D Gaussian kernel using the Gaussian function.
#         kernel_size = 3  # Size of the kernel (3x3x3 for simplicity)
#         kernel = self.create_gaussian_kernel(kernel_size, sigma)
#         kernel = kernel.to(input_tensor.device)

#         # Apply the Gaussian filter (convolution).
#         smoothed_input = F.conv3d(input_tensor, kernel, stride=1, padding=1)
#         return smoothed_input

#     def create_gaussian_kernel(self, kernel_size, sigma):
#         """
#         Create a 3D Gaussian kernel.
#         """
#         # Create a 1D Gaussian kernel.
#         ax = torch.arange(-(kernel_size // 2), kernel_size // 2 + 1, dtype=torch.float32)
#         xx, yy, zz = torch.meshgrid(ax, ax, ax)
#         kernel = torch.exp(-(xx**2 + yy**2 + zz**2) / (2 * sigma**2))

#         # Normalize the kernel to ensure the sum is 1.
#         kernel = kernel / kernel.sum()

#         # Reshape the kernel into a 4D tensor for convolution (out_channels=1, in_channels=1, depth, height, width)
#         kernel = kernel.unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, depth, height, width)
#         return kernel

#     def laplacian_3d(self, input_tensor):
#         """
#         Apply a Laplacian filter in 3D to compute second-order derivatives.
#         """
#         # Laplacian kernel
#         laplacian_kernel = torch.tensor(
#             [[[[[0, 1, 0], [1, -6, 1], [0, 1, 0]],
#                [[1, -6, 1], [-6, 36, -6], [1, -6, 1]],
#                [[0, 1, 0], [1, -6, 1], [0, 1, 0]]]]],
#             dtype=torch.float32, device=input_tensor.device
#         )

#         # Ensure input_tensor is 5D: (batch, channel, depth, height, width)
#         if input_tensor.ndim != 5:
#             raise ValueError("Expected input_tensor to be 5D (batch, channel, depth, height, width), got shape: {}".format(input_tensor.shape))

#         # Apply the Laplacian filter
#         laplacian_output = F.conv3d(input_tensor, laplacian_kernel, stride=1, padding=1)
#         return laplacian_output

#     def forward(self, input_tensor):
#         """Forward pass for Edge Refinement."""
#         # Apply a 3D Gaussian filter for smoothing
#         x = self.conv(input_tensor)
#         smoothed_tensor = self.gaussian_3d(x)

#         # Apply Laplacian filter to the smoothed tensor
#         laplacian_map = self.laplacian_3d(smoothed_tensor)

#         # Refine the input tensor using the Laplacian map
#         input_tensor_up = laplacian_map * input_tensor
#         return input_tensor_up

# class SPADEGenerator(nn.Module):
#     def __init__(self, latent_channels, bottleneck_channels):
#         super().__init__()
#         self.conv_mean = nn.Conv3d(latent_channels, bottleneck_channels, kernel_size=3, padding=1)
#         self.conv_std = nn.Conv3d(latent_channels, bottleneck_channels, kernel_size=3, padding=1)

#     def forward(self, autoencoder_latent, x_bottt):
#         autoencoder_latent_mean = autoencoder_latent.mean(dim=(2, 3, 4), keepdim=True)
#         autoencoder_latent_std = autoencoder_latent.std(dim=(2, 3, 4), keepdim=True)
#         mean = self.conv_mean(x_bottt)  # Compute mean
#         std = self.conv_std(x_bottt)   # Compute std
#         mean = mean.mean(dim=(2, 3, 4), keepdim=True)
#         std = std.std(dim=(2, 3, 4), keepdim=True)
        
#         return mean, std, autoencoder_latent_mean, autoencoder_latent_std

# class SPADELayer(nn.Module):
#     def __init__(self, bottleneck_channels):
#         super().__init__()
#         self.epsilon = 1e-5  # To prevent division by zero

#     def forward(self, x_bottt, mean, std):
#         # Compute the batch mean and std of x_bottt
#         batch_mean = x_bottt.mean(dim=(2, 3, 4), keepdim=True)
#         batch_std = x_bottt.std(dim=(2, 3, 4), keepdim=True)

#         # Normalize x_bottt using its batch stats
#         normalized = (x_bottt - batch_mean) / (batch_std + self.epsilon)

#         # Apply spatial conditioning
#         output = std * normalized + mean
#         return output




# class UNet3DResidual(nn.Module):
#     def __init__(self, autoencoder_quantizer0, autoencoder_decoder, autoencoder_segmentataion, in_channels: int, out_channels: int, dropout_prob: float()):
#         super(UNet3DResidual, self).__init__()

#         self.dropout_prob = dropout_prob  # Dropout probability

#         # Initialize Encoder, Bottleneck, and Decoder as separate modules
#         self.encoder = Encoder3D(in_channels, dropout_prob)
#         # self.indice3D = Indice3D(64, dropout_prob)
#         self.bottleneck = BottleneckBlock(128, dropout_prob)
#         self.decoder = Decoder3D(dropout_prob)
#         self.transmodel = TransformerModel(input_shape=[15, 15, 9], embed_dim=256, num_layers=7, num_heads=8)
        # self.spade_generator = SPADEGenerator(128, 128)
        # self.spade_layer = SPADELayer(128)
#         # self.EdgeRefinement3D = EdgeRefinement3D()
#         # self.EdgeRefinement3D_lap = EdgeRefinement3D_lap()
#         # self.conv = nn.Conv3d(256, 128, kernel_size=3, padding=1)
#         # self.LatentSpaceMaskReducer = LatentSpaceMaskReducer(256)
#         # # self.pretrained_embedding = pretrained_embedding
#         self.quantizer0 = VectorQuantizer(
#             quantizer=EMAQuantizer(
#                 spatial_dims=3,
#                 num_embeddings=128,
#                 embedding_dim=128,
#                 commitment_cost=0.25,
#                 decay=0.99,
#                 epsilon=1e-5,
#                 embedding_init='uniform',
#                 ddp_sync=False,
#                 pretrained_embedding=None,
#             )
#         )
#         # self.quantizer0 = autoencoder_quantizer0
#         # self.TransformerModeldec_mlm = TransformerModeldec_mlm((30, 30, 18), 128, 5, 8)
#         # self.conv_mlm = nn.Conv3d(128, 128, kernel_size=3)
#         # self.LatentSpaceReducer = LatentSpaceReducer(4, 2)
#         # self.conv1 = nn.Conv3d(128, 64, kernel_size=3, padding=1)
#         # # # self.conv2 = nn.Conv3d(64, 32, kernel_size=3, padding=1)
#         # # # self.conv3 = nn.Conv3d(32, 64, kernel_size=3, padding=1)
#         # self.conv4 = nn.Conv3d(128, 32, kernel_size=3, padding=1)
#         # self.quantizer1 = VectorQuantizer_indice(
#         #     quantizer=EMAQuantizer_indice(
#         #         spatial_dims=3,
#         #         num_embeddings=1024,
#         #         embedding_dim=32,
#         #         commitment_cost=0.25,
#         #         decay=0.99,
#         #         epsilon=1e-5,
#         #         embedding_init='uniform',
#         #         ddp_sync=False,
#         #         pretrained_embedding=None,
#         #     )
#         # )
#         # self.model_pt_decoder = autoencoder_decoder
#         # self.model_pt_seg = autoencoder_segmentataion
        
#     def forward(self, x, autoencoder_latent, is_train=False):
#         # Encoder path
#         x1, x2, x3, x4, x5, x6 = self.encoder(x, autoencoder_latent, is_train=is_train)
#         # x6 = self.LatentSpaceMaskReducer.reduce_latent(x6)
#         # x6 = self.LatentSpaceReducer(x6)
#         x8=self.transmodel(x6)
#         x8 = self.decoder(x8, x4, x3)
#         softmax_output = torch.softmax(x8, dim=1)  # logits shape: (batch, classes, depth, height, width)
#         max_prob, max_index = torch.max(softmax_output, dim=1)
#         print("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", torch.sum(((max_prob))>0.3))
#         # print("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", torch.sum(((max_prob))>0.35))
#         # print("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", torch.sum(((max_prob))>0.4))
#         # print("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", torch.sum(((max_prob))>0.45))
#         # print("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", torch.sum(((max_prob))>0.5))
#         # entropy_map = -torch.sum(softmax_output * torch.log(softmax_output + 1e-10), dim=1)  # Avoid log(0) with small epsilon
#         # threshold = entropy_map.mean() + entropy_map.std()  # Example threshold
#         # high_entropy_mask = entropy_map > threshold
#         # print("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", torch.sum((high_entropy_mask)))
#         # print("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", ((high_entropy_mask.shape)))
#         # coords = torch.nonzero(high_entropy_mask, as_tuple=False)  # Get coordinates of high-entropy voxels
#         # min_coords = coords.min(dim=0).values  # Minimum corner of bounding box
#         # max_coords = coords.max(dim=0).values  # Maximum corner of bounding box
#         # print("min_coords", min_coords)
#         # print("max_coords", max_coords)
#         # flat_output = torch.argmax(softmax_output, dim=1).flatten()
#         # print("flat_output", flat_output)




#         # x8_sob = self.EdgeRefinement3D(x8)

        
#         # x8_lap = self.EdgeRefinement3D_lap(x8)

#         # x_cat = self.conv(torch.cat((x8_sob, x8_lap), dim=1))
#         x_bottt = (self.bottleneck(x8)).float()

        # mean, std, autoencoder_latent_mean, autoencoder_latent_std = self.spade_generator(autoencoder_latent, x_bottt)

        # # Apply SPADE
        # x_bottt = self.spade_layer(x_bottt, mean, std)

        
#         quantized_loss, quantized, encodings_sum, embedding = self.quantizer0(x_bottt)

#         # reconstruction = self.model_pt_decoder(quantized)
#         # reconstruction = self.model_pt_seg(reconstruction)
        
#         # b, c, h, w, d = x.shape
#         # softmax_out = torch.softmax(x8, dim=1)

#         # # Step 2: Create a confidence mask (probability > 0.5)
#         # confidence_mask = (softmax_out > 0.5)  # Shape: (4, 128, 30, 30, 19)
        
#         # # Step 3: Get the most confident predictions (argmax along channel dimension)
#         # argmax_indices = torch.argmax(softmax_out, dim=1)  # Shape: (4, 30, 30, 19)
        
#         # # Step 4: Mask the argmax indices based on confidence
#         # # Reduce the confidence mask to match the spatial dimensions
#         # confidence_reduced = confidence_mask.any(dim=1)  # Shape: (4, 30, 30, 19)
        
#         # # Apply the confidence mask to the argmax indices
#         # masked_indices = torch.where(confidence_reduced, argmax_indices, torch.tensor(-1))  # -1 indicates masked
#         # masked_indices = self.conv_mlm(masked_indices.unsqueeze(dim=1))
#         # padding = (0, 1, 1, 1, 1, 1)  # (left, right, top, bottom, front, back)
#         # masked_indices = F.pad(masked_indices, padding, mode='constant', value=0)
#         # masked_indices = masked_indices.squeeze(dim=1)
#         # masked_indices = self.TransformerModeldec_mlm(masked_indices)
#         # reconstruction = self.model_pt_decoder(quantized)
#         # segmentataion = self.model_pt_seg(reconstruction)
#         # x9 = self.conv4(x8)
#         # loss, quantized, encodings_sum, embedding
#         # Bottleneck
#         # x_bot = self.bottleneck(x6)

#         # x_bot = self.conv1(x_bot)
#         # x_bot = self.conv2(x_bot)

#         # indice = self.indice3D(x5)
#         # q_loss1, quantized1, encodings_sum1, embedding1, encoding_indices1 = self.quantizer1(indice)
        
#         # q_loss, z_quantized0, encodings_sum, embedding = self.quantizer0(x_bot)

#         # # z_quantized0_post = self.conv3(z_quantized0)
#         # z_quantized0 = self.conv4(z_quantized0)

#         # # Decoder path with skip connections
#         # out = self.decoder(z_quantized0, x6, x5, x4, x3, x2, x1)

#         return x8, x_bottt, quantized, quantized_loss, mean, std, autoencoder_latent_mean, autoencoder_latent_std
        
#         # return x8, quantized, segmentataion, quantized_loss









































import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List, Optional




import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from einops import rearrange
import random

# from torch_cluster import kmeans

class PositionalEncoding3D(nn.Module):
    """Updated 3D Positional Encoding."""
    def __init__(self, 
                 num_embed=8192, 
                 spatial_size=[32, 32, 32], 
                 embed_dim=3968, 
                 trainable=True, 
                 pos_emb_type='embedding'):
        super().__init__()
        
        if isinstance(spatial_size, int):
            spatial_size = [spatial_size, spatial_size, spatial_size]

        self.spatial_size = spatial_size
        self.num_embed = num_embed + 1
        self.embed_dim = embed_dim
        self.trainable = trainable
        self.pos_emb_type = pos_emb_type

        assert self.pos_emb_type in ['embedding', 'parameter']
        
        # self.emb = nn.Conv3d(256, embed_dim, kernel_size=3, padding=1)
        self.flatten = nn.Flatten(start_dim=2)
        if self.pos_emb_type == 'embedding':
            self.height_emb = nn.Embedding(self.spatial_size[0], embed_dim)
            self.width_emb = nn.Embedding(self.spatial_size[1], embed_dim)
            self.depth_emb = nn.Embedding(self.spatial_size[2], embed_dim)
        else:
            self.height_emb = nn.Parameter(torch.zeros(1, self.spatial_size[0], embed_dim))
            self.width_emb = nn.Parameter(torch.zeros(1, self.spatial_size[1], embed_dim))
            self.depth_emb = nn.Parameter(torch.zeros(1, self.spatial_size[2], embed_dim))
        
        self._set_trainable()

    def _set_trainable(self):
        if not self.trainable:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, index, **kwargs):
        # assert index.dim() == 2  # B x L
        # index = torch.clamp(index, min=0)  # Ensure indices are valid
        print("enc_out enc_out enc_out enc_out flatten emb", index.shape)
        # emb = self.emb(index)
        emb = self.flatten(index)  # Shape: (B, embed_dim, L)
        emb = emb.contiguous()

        print("enc_out enc_out enc_out enc_out flatten emb", emb.shape)
        
        # Transpose latent embeddings to match the attention input format
        emb = emb.permute(0, 2, 1) 
        # print("self.spatial_size[0]", self.spatial_size[0])
        # print("self.spatial_size[0]", self.spatial_size[1])
        # print("self.spatial_size[0]", self.spatial_size[2])
        if emb.shape[1] > 0:
        # if False:
            if self.pos_emb_type == 'embedding':
                height_emb = self.height_emb(torch.arange(self.spatial_size[0], device=index.device).view(1, self.spatial_size[0])).unsqueeze(2).unsqueeze(3)  # Shape: 1 x H x 1 x 1 x embed_dim
                width_emb = self.width_emb(torch.arange(self.spatial_size[1], device=index.device).view(1, self.spatial_size[1])).unsqueeze(1).unsqueeze(3)   # Shape: 1 x 1 x W x 1 x embed_dim
                depth_emb = self.depth_emb(torch.arange(self.spatial_size[2], device=index.device).view(1, self.spatial_size[2])).unsqueeze(1).unsqueeze(1)   # Shape: 1 x 1 x 1 x D x embed_dim
            else:
                height_emb = self.height_emb.unsqueeze(2).unsqueeze(3)  # Shape: 1 x H x 1 x 1 x embed_dim
                width_emb = self.width_emb.unsqueeze(1).unsqueeze(3)    # Shape: 1 x 1 x W x 1 x embed_dim
                depth_emb = self.depth_emb.unsqueeze(1).unsqueeze(1)    # Shape: 1 x 1 x 1 x D x embed_dim

            pos_emb = (height_emb + width_emb + depth_emb).view(1, self.spatial_size[0] * self.spatial_size[1] * self.spatial_size[2], -1) # 1 x H x W x D -> 1 x L xD
            emb = emb + pos_emb[:, :emb.shape[1], :]
        
        return emb



# class Conv_MLP(nn.Module):
#     def __init__(self, n_embd, mlp_hidden_times, act, resid_pdrop):
#         super().__init__()
#         self.conv1 = nn.Conv3d(in_channels=n_embd, out_channels=int(mlp_hidden_times * n_embd), kernel_size=3, stride=1, padding=1)
#         self.act = act
#         self.conv2 = nn.Conv3d(in_channels=int(mlp_hidden_times * n_embd), out_channels=n_embd, kernel_size=3, stride=1, padding=1)

#         # self.conv3 = nn.Conv3d(in_channels=n_embd, out_channels=int(mlp_hidden_times * n_embd), kernel_size=3, stride=1, padding=1)
#         # self.act = act
#         # self.conv4 = nn.Conv3d(in_channels=int(mlp_hidden_times * n_embd), out_channels=n_embd, kernel_size=3, stride=1, padding=1)
    
#         self.dropout = nn.Dropout(resid_pdrop)

#     def forward(self, x):
#         n =  x.size()[1]
#         x = rearrange(x, 'b (h w d) c -> b c h w d', h=8, w=8, d=8)
#         x = (self.conv2(self.act(self.conv1(x))))
#         # x = self.conv4(self.act(self.conv3(x)))
#         x = rearrange(x, 'b c h w d -> b (h w d) c')
#         return self.dropout(x)


class ConvBlock(nn.Module):
    """A Conv Block with LayerNorm."""
    def __init__(self, embed_dim, kernel_size=3, stride=1, padding=1):
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Conv3d(in_channels=embed_dim, out_channels=int(4 * embed_dim), kernel_size=3, stride=1, padding=1)
        self.act = GELU2()
        self.conv2 = nn.Conv3d(in_channels=int(4 * embed_dim), out_channels=embed_dim, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        # b, c, d, h, w = x.shape
        x = rearrange(x, 'b (h w d) c -> b c h w d', h=15, w=15, d=9)
        x = (self.conv2(self.act(self.conv1(x))))
        # x = self.conv4(self.act(self.conv3(x)))
        x = rearrange(x, 'b c h w d -> b (h w d) c')
        return x


class FullAttention(nn.Module):
    """Full Attention Module."""
    def __init__(self, n_embd, n_head, attn_pdrop=0.0, resid_pdrop=0.0, causal=True):
        super().__init__()
        assert n_embd % n_head == 0
        self.key = nn.Linear(n_embd, n_embd)
        self.query = nn.Linear(n_embd, n_embd)
        self.value = nn.Linear(n_embd, n_embd)
        self.attn_drop = nn.Dropout(attn_pdrop)
        self.resid_drop = nn.Dropout(resid_pdrop)
        self.proj = nn.Linear(n_embd, n_embd)
        self.n_head = n_head
        self.causal = causal

    def forward(self, x, lay=int, mask=None):
        B, T, C = x.size()
        k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        # if lay>= 5:
        #     # print("att being revered")
        #     att = 1-att
        if mask is not None:
            att = att.masked_fill(mask == 0, float('-inf'))

        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_drop(self.proj(y))
        return y, att






# class FullAttention(nn.Module):
#     """Full Attention Module with Local-to-Global Attention."""
#     def __init__(self, n_embd, n_head, attn_pdrop=0.0, resid_pdrop=0.0, causal=True, window_size=(3, 3, 3)):
#         """
#         Args:
#             n_embd (int): Embedding dimension.
#             n_head (int): Number of attention heads.
#             attn_pdrop (float): Dropout probability for attention weights.
#             resid_pdrop (float): Dropout probability for residual connections.
#             causal (bool): Whether to apply causal masking (e.g., for autoregressive tasks).
#             window_size (tuple): (H, W, D) for local attention window. None means full attention.
#         """
#         super().__init__()
#         assert n_embd % n_head == 0
#         self.key = nn.Linear(n_embd, n_embd)
#         self.query = nn.Linear(n_embd, n_embd)
#         self.value = nn.Linear(n_embd, n_embd)
#         self.attn_drop = nn.Dropout(attn_pdrop)
#         self.resid_drop = nn.Dropout(resid_pdrop)
#         self.proj = nn.Linear(n_embd, n_embd)
#         self.n_head = n_head
#         self.causal = causal
#         self.window_size = window_size

#     def forward(self, x, lay=int, mask=None):
#         """
#         Forward pass with local-to-global attention.
#         Args:
#             x (torch.Tensor): Input tensor of shape (B, T, C).
#             mask (torch.Tensor): Optional mask of shape (B, T).
#         Returns:
#             torch.Tensor: Output tensor of shape (B, T, C).
#         """
#         B, T, C = x.size()
#         if self.window_size is not None:
#             H, W, D = self.window_size
#             # Step 1: Apply vectorized local attention
#             local_output = self._local_attention(x, H, W, D)

#             # Step 2: Aggregate global context (global attention)
#             global_context = self._global_attention(x)

#             # Step 3: Combine local and global features
#             output = local_output + global_context
#             att = 0
#         else:
#             # Full attention if no window size is defined
#             output, _ = self._full_attention(x, mask)

#         return output, att

#     def _local_attention(self, x, H, W, D):
#         """Compute local attention within defined window using vectorization."""
#         B, T, C = x.size()
#         window_tokens = H * W * D
#         assert T % window_tokens == 0, "Sequence length must be divisible by window size."

#         # Reshape for windows: (B * num_windows, window_tokens, C)
#         num_windows = T // window_tokens
#         x_windows = x.view(B, num_windows, window_tokens, C).reshape(-1, window_tokens, C)

#         # Compute local attention for all windows simultaneously
#         y, _ = self._full_attention(x_windows, mask=None)

#         # Reshape back to original sequence shape (B, T, C)
#         y = y.view(B, num_windows * window_tokens, C)
#         return y

#     def _global_attention(self, x):
#         """Compute global attention by aggregating context across all tokens."""
#         B, T, C = x.size()
#         k = self.key(x).view(B, self.n_head, T, C // self.n_head)
#         q = self.query(x).view(B, self.n_head, T, C // self.n_head)
#         v = self.value(x).view(B, self.n_head, T, C // self.n_head)

#         # Global attention across all tokens
#         att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
#         att = F.softmax(att, dim=-1)
#         att = self.attn_drop(att)
#         global_context = (att @ v).view(B, T, C)
#         return self.resid_drop(self.proj(global_context))

#     def _full_attention(self, x, mask):
#         """Standard full attention."""
#         B, T, C = x.size()
#         k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
#         q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
#         v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

#         att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
#         if mask is not None:
#             att = att.masked_fill(mask == 0, float('-inf'))
#         att = F.softmax(att, dim=-1)
#         att = self.attn_drop(att)
#         y = att @ v
#         y = y.transpose(1, 2).contiguous().view(B, T, C)
#         y = self.resid_drop(self.proj(y))
#         return y, att



class FullAttentiondec(nn.Module):
    """Full Attention Module with Local-to-Global Attention."""
    def __init__(self, n_embd, n_head, attn_pdrop=0.0, resid_pdrop=0.0, causal=True, window_size=(5, 5, 3)):
        """
        Args:
            n_embd (int): Embedding dimension.
            n_head (int): Number of attention heads.
            attn_pdrop (float): Dropout probability for attention weights.
            resid_pdrop (float): Dropout probability for residual connections.
            causal (bool): Whether to apply causal masking (e.g., for autoregressive tasks).
            window_size (tuple): (H, W, D) for local attention window. None means full attention.
        """
        super().__init__()
        assert n_embd % n_head == 0
        self.key = nn.Linear(n_embd, n_embd)
        self.query = nn.Linear(n_embd, n_embd)
        self.value = nn.Linear(n_embd, n_embd)
        self.attn_drop = nn.Dropout(attn_pdrop)
        self.resid_drop = nn.Dropout(resid_pdrop)
        self.proj = nn.Linear(n_embd, n_embd)
        self.n_head = n_head
        self.causal = causal
        self.window_size = window_size

    def forward(self, x, lay=int, mask=None):
        """
        Forward pass with local-to-global attention.
        Args:
            x (torch.Tensor): Input tensor of shape (B, T, C).
            mask (torch.Tensor): Optional mask of shape (B, T).
        Returns:
            torch.Tensor: Output tensor of shape (B, T, C).
        """
        B, T, C = x.size()
        if self.window_size is not None:
            H, W, D = self.window_size
            # Step 1: Apply vectorized local attention
            local_output = self._local_attention(x, H, W, D)

            # Step 2: Aggregate global context (global attention)
            # global_context = self._global_attention(x)

            # # Step 3: Combine local and global features
            # output = local_output + global_context
            output = local_output
            att = 0
        else:
            # Full attention if no window size is defined
            output, _ = self._full_attention(x, mask)

        return output, att

    def _local_attention(self, x, H, W, D):
        """Compute local attention within defined window using vectorization."""
        B, T, C = x.size()
        window_tokens = H * W * D
        assert T % window_tokens == 0, "Sequence length must be divisible by window size."

        # Reshape for windows: (B * num_windows, window_tokens, C)
        num_windows = T // window_tokens
        x_windows = x.view(B, num_windows, window_tokens, C).reshape(-1, window_tokens, C)

        # Compute local attention for all windows simultaneously
        y, _ = self._full_attention(x_windows, mask=None)

        # Reshape back to original sequence shape (B, T, C)
        y = y.view(B, num_windows * window_tokens, C)
        return y

    def _global_attention(self, x):
        """Compute global attention by aggregating context across all tokens."""
        B, T, C = x.size()
        k = self.key(x).view(B, self.n_head, T, C // self.n_head)
        q = self.query(x).view(B, self.n_head, T, C // self.n_head)
        v = self.value(x).view(B, self.n_head, T, C // self.n_head)

        # Global attention across all tokens
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        global_context = (att @ v).view(B, T, C)
        return self.resid_drop(self.proj(global_context))

    def _full_attention(self, x, mask):
        """Standard full attention."""
        B, T, C = x.size()
        k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        if mask is not None:
            att = att.masked_fill(mask == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_drop(self.proj(y))
        return y, att



# class FullAttention(nn.Module):
#     """Full Attention Module with Local-to-Global Attention."""
#     def __init__(self, n_embd, n_head, attn_pdrop=0.0, resid_pdrop=0.0, causal=True, window_size=(3, 3, 3)):
#         """
#         Args:
#             n_embd (int): Embedding dimension.
#             n_head (int): Number of attention heads.
#             attn_pdrop (float): Dropout probability for attention weights.
#             resid_pdrop (float): Dropout probability for residual connections.
#             causal (bool): Whether to apply causal masking (e.g., for autoregressive tasks).
#             window_size (tuple): (H, W, D) for local attention window. None means full attention.
#         """
#         super().__init__()
#         assert n_embd % n_head == 0
#         self.key = nn.Linear(n_embd, n_embd)
#         self.query = nn.Linear(n_embd, n_embd)
#         self.value = nn.Linear(n_embd, n_embd)
#         self.attn_drop = nn.Dropout(attn_pdrop)
#         self.resid_drop = nn.Dropout(resid_pdrop)
#         self.proj = nn.Linear(n_embd, n_embd)
#         self.n_head = n_head
#         self.causal = causal
#         self.window_size = window_size

#     def forward(self, x, lay=int, mask=None):
#         """
#         Forward pass with local-to-global attention.
#         Args:
#             x (torch.Tensor): Input tensor of shape (B, T, C).
#             mask (torch.Tensor): Optional mask of shape (B, T).
#         Returns:
#             torch.Tensor: Output tensor of shape (B, T, C).
#         """
#         B, T, C = x.size()
#         if self.window_size is not None:
#             H, W, D = self.window_size
#             # Step 1: Apply local attention
#             local_output = self._local_attention(x, H, W, D)

#             # Step 2: Aggregate global context (global attention)
#             global_context = self._global_attention(x)

#             # Step 3: Combine local and global features
#             output = local_output + global_context
#             att = 0
#         else:
#             # Full attention if no window size is defined
#             output, _ = self._full_attention(x, mask)

#         return output, att

#     def _local_attention(self, x, H, W, D):
#         """Compute local attention within defined window."""
#         B, T, C = x.size()
#         window_tokens = H * W * D
#         assert T % window_tokens == 0, "Sequence length must be divisible by window size."

#         # Reshape for windows: (B, num_windows, window_tokens, C)
#         num_windows = T // window_tokens
#         x_windows = x.view(B, num_windows, window_tokens, C)

#         # Compute local attention for each window
#         local_outputs = []
#         for i in range(num_windows):
#             window = x_windows[:, i, :, :]
#             local_output, _ = self._full_attention(window, mask=None)
#             local_outputs.append(local_output)

#         # Concatenate windows back to the original shape
#         local_output = torch.cat(local_outputs, dim=1)
#         return local_output

#     def _global_attention(self, x):
#         """Compute global attention by aggregating context across all tokens."""
#         B, T, C = x.size()
#         k = self.key(x).view(B, self.n_head, T, C // self.n_head)
#         q = self.query(x).view(B, self.n_head, T, C // self.n_head)
#         v = self.value(x).view(B, self.n_head, T, C // self.n_head)

#         # Global attention across all tokens
#         att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
#         att = F.softmax(att, dim=-1)
#         att = self.attn_drop(att)
#         global_context = (att @ v).view(B, T, C)
#         return self.resid_drop(self.proj(global_context))

#     def _full_attention(self, x, mask):
#         """Standard full attention."""
#         B, T, C = x.size()
#         k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
#         q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
#         v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

#         att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
#         if mask is not None:
#             att = att.masked_fill(mask == 0, float('-inf'))
#         att = F.softmax(att, dim=-1)
#         att = self.attn_drop(att)
#         y = att @ v
#         y = y.transpose(1, 2).contiguous().view(B, T, C)
#         y = self.resid_drop(self.proj(y))
#         return y, att

class AttentionScalingWithHeads(nn.Module):
    def __init__(self, embed_dim, num_heads=8):
        super(AttentionScalingWithHeads, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # Linear layers for query projection
        self.query_proj = nn.Linear(embed_dim, embed_dim)
        self.softmax = nn.Softmax(dim=2)  # Normalize across sequence length

    def forward(self, x):
        batch_size, seq_len, embed_dim = x.size()

        # Project to multi-head space
        query = self.query_proj(x)  # (batch_size, seq_len, embed_dim)
        query = query.view(batch_size, seq_len, self.num_heads, self.head_dim)  # (batch_size, seq_len, num_heads, head_dim)

        # Compute attention scores
        attention_scores = torch.mean(query, dim=-1)  # Average over head_dim
        attention_scores = self.softmax(attention_scores)  # (batch_size, seq_len, num_heads)

        # Reshape to apply scaling
        attention_scores = attention_scores.view(batch_size, seq_len, self.num_heads, 1)
        scaled_x = x.view(batch_size, seq_len, self.num_heads, self.head_dim) * attention_scores

        # Flatten back to original embedding dimension
        scaled_x = scaled_x.view(batch_size, seq_len, embed_dim)
        return scaled_x




class SEBlock(nn.Module):
    def __init__(self, embed_dim, reduction_ratio=16):
        super(SEBlock, self).__init__()
        self.fc1 = nn.Linear(embed_dim, embed_dim // reduction_ratio)
        self.fc2 = nn.Linear(embed_dim // reduction_ratio, embed_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        
        b, seq_len, c = x.size()  # batch_size, sequence_length, channels
        
        # Global average pooling along the sequence length dimension (seq_len)
        se = torch.mean(x, dim=1)  # Pooling across the sequence dimension
        # print("se after mean is shape is", se.shape)
        
        se = self.fc1(se)
        se = torch.relu(se)
        se = self.fc2(se)
        se = self.sigmoid(se)  # Channel-wise attention
        se = torch.unsqueeze(se, dim=1)
        # print("se after unseq is shape is", se.shape)
        # Scale the input tensor by the attention weights
        return x * se  # Apply the mask across all sequence tokens



class TransformerBlock(nn.Module):
    """A Transformer Block with Full Attention and MLP."""
    def __init__(self, embed_dim, num_heads, mlp_ratio=4, dropout=0.0, mlp_type = None):
        super(TransformerBlock, self).__init__()
        self.attn = FullAttention(embed_dim, num_heads, attn_pdrop=dropout, resid_pdrop=dropout)
        if mlp_type == 'conv_mlp':
            self.mlp = ConvBlock(embed_dim)
        else:
            self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_ratio * embed_dim),
            GELU2(),
            nn.Linear(mlp_ratio * embed_dim, embed_dim),
            nn.Dropout(dropout)
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.se_block = SEBlock(embed_dim, 16)
        self.attention_scaling = AttentionScalingWithHeads(embed_dim, 8)

    def forward(self, x, lay=int, mask=None):
        attn_out, att = self.attn(x, lay, mask)
        x = x + attn_out
        x = self.norm1(x)

        mlp_out = self.mlp(x)
        x = x + mlp_out
        x = self.norm2(x)
        x = self.attention_scaling(x)
        x = self.se_block(x)
        return x, att

class TransformerBlockdec(nn.Module):
    """A Transformer Block with Full Attention and MLP."""
    def __init__(self, embed_dim, num_heads, mlp_ratio=4, dropout=0.0, mlp_type = None):
        super(TransformerBlockdec, self).__init__()
        self.attn = FullAttentiondec(embed_dim, num_heads, attn_pdrop=dropout, resid_pdrop=dropout)
        if mlp_type == 'conv_mlp':
            self.mlp = ConvBlock(embed_dim)
        else:
            self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_ratio * embed_dim),
            GELU2(),
            nn.Linear(mlp_ratio * embed_dim, embed_dim),
            nn.Dropout(dropout)
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.se_block = SEBlock(embed_dim, 16)
        self.attention_scaling = AttentionScalingWithHeads(embed_dim, 8)

    def forward(self, x, lay=int, mask=None):
        attn_out, att = self.attn(x, lay, mask)
        x = x + attn_out
        x = self.norm1(x)

        mlp_out = self.mlp(x)
        x = x + mlp_out
        x = self.norm2(x)
        x = self.attention_scaling(x)
        x = self.se_block(x)
        return x, att
# class TransformerModel(nn.Module):
#     """Transformer Model with Full Attention, Conv Blocks, and Positional Encoding."""
#     def __init__(self, input_shape, embed_dim, num_layers, num_heads):
#         super(TransformerModel, self).__init__()
#         h, w, d = input_shape
        # self.positional_encoding = PositionalEncoding3D(embed_dim=embed_dim, 
        #                                                 spatial_size=[h, w, d])
        # self.layers = nn.ModuleList([
        #     nn.Sequential(
        #         TransformerBlock(embed_dim, num_heads),
        #         # ConvBlock(embed_dim)
        #     )
        #     for _ in range(num_layers)
        # ])

#     def forward(self, x):
#         b, c, h, w, d = x.shape
#         # x = rearrange(x, 'b c h w d -> b (h w d) c')
#         x = self.positional_encoding(x)
#         # x = rearrange(x, 'b c h w d -> b (h w d) c')
#         for layer in self.layers:
#             x = layer(x)

#         x = rearrange(x, 'b (h w d) c -> b c h w d', h=h, w=w, d=d)
#         return x

class ConvBlock3D(nn.Module):
    """Convolution Block with Conv3d, BatchNorm, ReLU, and Dropout"""
    def __init__(self, in_channels, out_channels, dropout_prob):
        super(ConvBlock3D, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.batch_norm = nn.BatchNorm3d(out_channels)
        self.dropout = nn.Dropout3d(p=dropout_prob)
    
    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        x = self.batch_norm(x)
        x = self.dropout(x)
        return x

class TransformerModel(nn.Module):
    """Transformer Model with Full Attention, Uncertainty Estimation, and Soft Masking."""
    def __init__(self, input_shape, embed_dim, num_layers, num_heads):
        super(TransformerModel, self).__init__()
        h, w, d = input_shape
        self.num_layers = num_layers
        self.embed_dim = embed_dim
        self.positional_encoding = PositionalEncoding3D(embed_dim=embed_dim, 
                                                        spatial_size=[h, w, d])
        self.layers = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads)  # Remove nn.Sequential wrapper
            for _ in range(num_layers)
        ])
        self.uncertainty_layers = nn.ModuleList([nn.Linear(embed_dim, 1) for _ in range(num_layers)])
        # self.emb = ConvBlock3D(512, 256, 0.2) 

    def calculate_uncertainty(self, x, layer_idx):
        """
        Calculate token-level uncertainty for a specific layer using MC Dropout or learned variance.
        """
        variance = self.uncertainty_layers[layer_idx](x).sigmoid()  # Values in [0, 1]
        return variance

    def forward(self, x):
        b, c, h, w, d = x.shape
        x = self.positional_encoding(x)
        # gt_x = rearrange(gt_x, 'b c h w d -> b (h w d) c')
        # masked_gt_list = []
        # masked_out_list = []
        for i, layer in enumerate(self.layers):
            x, att = layer(x,i)

            # print("at shape is", att.shape)
            
            if i >= 5 and i < (self.num_layers - 1):  # After the 6th layer, estimate uncertainty and mask
                
                uncertainty = self.calculate_uncertainty(x, i)
                # print("Uncertainty shape is", uncertainty.shape)
                
                # Calculate certainty mask
                certainty_mask = 1 - uncertainty  # Certainty is the complement of uncertainty
                x = x * certainty_mask  # Apply soft masking to the model's output
                
                # Apply the same mask to the ground truth
                # masked_gt = gt_x * certainty_mask  # Mask the ground truth similarly
                # masked_gt_list.append(masked_gt)  # Store masked GT for this layer
                # masked_out_list.append(x)
            
        x = rearrange(x, 'b (h w d) c -> b c h w d', h=h, w=w, d=d)
        # x = self.emb(x)
        # gt_x_up = rearrange(gt_x_up, 'b (h w d) c -> b c h w d', h=h, w=w, d=d)
        return x


class TransformerModeldec(nn.Module):
    """Transformer Model with Full Attention, Uncertainty Estimation, and Soft Masking."""
    def __init__(self, input_shape, embed_dim, num_layers, num_heads):
        super(TransformerModeldec, self).__init__()
        h, w, d = input_shape
        self.num_layers = num_layers
        self.embed_dim = embed_dim
        self.positional_encoding = PositionalEncoding3D(embed_dim=embed_dim, 
                                                        spatial_size=[h, w, d])
        self.layers = nn.ModuleList([
            TransformerBlockdec(embed_dim, num_heads)  # Remove nn.Sequential wrapper
            for _ in range(num_layers)
        ])
        # self.uncertainty_layers = nn.ModuleList([nn.Linear(embed_dim, 1) for _ in range(num_layers)])
        # self.emb = ConvBlock3D(512, 256, 0.2) 

    # def calculate_uncertainty(self, x, layer_idx):
    #     """
    #     Calculate token-level uncertainty for a specific layer using MC Dropout or learned variance.
    #     """
    #     variance = self.uncertainty_layers[layer_idx](x).sigmoid()  # Values in [0, 1]
    #     return variance

    def forward(self, x):
        b, c, h, w, d = x.shape
        x = self.positional_encoding(x)
        # gt_x = rearrange(gt_x, 'b c h w d -> b (h w d) c')
        # masked_gt_list = []
        # masked_out_list = []
        for i, layer in enumerate(self.layers):
            x, att = layer(x,i)

            # print("at shape is", att.shape)
            
            # if i >= 5 and i < (self.num_layers - 1):  # After the 6th layer, estimate uncertainty and mask
                
            #     uncertainty = self.calculate_uncertainty(x, i)
            #     # print("Uncertainty shape is", uncertainty.shape)
                
            #     # Calculate certainty mask
            #     certainty_mask = 1 - uncertainty  # Certainty is the complement of uncertainty
            #     x = x * certainty_mask  # Apply soft masking to the model's output
                
                # Apply the same mask to the ground truth
                # masked_gt = gt_x * certainty_mask  # Mask the ground truth similarly
                # masked_gt_list.append(masked_gt)  # Store masked GT for this layer
                # masked_out_list.append(x)
            
        x = rearrange(x, 'b (h w d) c -> b c h w d', h=h, w=w, d=d)
        # x = self.emb(x)
        # gt_x_up = rearrange(gt_x_up, 'b (h w d) c -> b c h w d', h=h, w=w, d=d)
        return x











mse_loss = nn.MSELoss()

class EMAQuantizer(nn.Module):
    """
    Vector Quantization module using Exponential Moving Average (EMA) to learn the codebook parameters based on  Neural
    Discrete Representation Learning by Oord et al. (https://arxiv.org/abs/1711.00937) and the official implementation
    that can be found at https://github.com/deepmind/sonnet/blob/v2/sonnet/src/nets/vqvae.py#L148 and commit
    58d9a2746493717a7c9252938da7efa6006f3739.

    This module is not compatible with TorchScript while working in a Distributed Data Parallelism Module. This is due
    to lack of TorchScript support for torch.distributed module as per https://github.com/pytorch/pytorch/issues/41353
    on 22/10/2022. If you want to TorchScript your model, please turn set `ddp_sync` to False.

    Args:
        spatial_dims :  number of spatial spatial_dims.
        num_embeddings: number of atomic elements in the codebook.
        embedding_dim: number of channels of the input and atomic elements.
        commitment_cost: scaling factor of the MSE loss between input and its quantized version. Defaults to 0.25.
        decay: EMA decay. Defaults to 0.99.
        epsilon: epsilon value. Defaults to 1e-5.
        embedding_init: initialization method for the codebook. Defaults to "normal".
        ddp_sync: whether to synchronize the codebook across processes. Defaults to True.
    """

    def __init__(
        self,
        spatial_dims: int,
        num_embeddings: int,
        embedding_dim: int,
        commitment_cost: float = 0.25,
        decay: float = 0.99,
        epsilon: float = 1e-5,
        embedding_init: str = "normal",
        ddp_sync: bool = True,
        pretrained_embedding: Optional[torch.Tensor] = None,  # Add this parameter
    ):
        super().__init__()
        self.spatial_dims: int = spatial_dims
        self.embedding_dim: int = embedding_dim
        self.num_embeddings: int = num_embeddings
    
        assert self.spatial_dims in [2, 3], ValueError(
            f"EMAQuantizer only supports 4D and 5D tensor inputs but received spatial dims {spatial_dims}."
        )
    
        # Initialize embedding
        self.embedding: torch.nn.Embedding = torch.nn.Embedding(self.num_embeddings, self.embedding_dim)
        
        # Load pretrained embedding if provided
        if pretrained_embedding is not None:
            if pretrained_embedding.shape != (self.num_embeddings, self.embedding_dim):
                raise ValueError(
                    f"Pretrained embedding must have shape ({self.num_embeddings}, {self.embedding_dim}), "
                    f"but got {pretrained_embedding.shape}."
                )
            self.embedding.weight.data.copy_(pretrained_embedding)
        elif embedding_init == "kaiming_uniform":
            torch.nn.init.kaiming_uniform_(self.embedding.weight.data, mode="fan_in", nonlinearity="linear")
        # Otherwise, use default initialization (normal)
    
        self.embedding.weight.requires_grad = False
    
        self.commitment_cost: float = commitment_cost
    
        self.register_buffer("ema_cluster_size", torch.zeros(self.num_embeddings))
        self.register_buffer("ema_w", self.embedding.weight.data.clone())
    
        self.decay: float = decay
        self.epsilon: float = epsilon
    
        self.ddp_sync: bool = ddp_sync
    
        # Precalculating required permutation shapes
        self.flatten_permutation: Sequence[int] = [0] + list(range(2, self.spatial_dims + 2)) + [1]
        self.quantization_permutation: Sequence[int] = [0, self.spatial_dims + 1] + list(
            range(1, self.spatial_dims + 1)
        )

    def quantize(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Given an input it projects it to the quantized space and returns additional tensors needed for EMA loss.

        Args:
            inputs: Encoding space tensors

        Returns:
            torch.Tensor: Flatten version of the input of shape [B*D*H*W, C].
            torch.Tensor: One-hot representation of the quantization indices of shape [B*D*H*W, self.num_embeddings].
            torch.Tensor: Quantization indices of shape [B,D,H,W,1]

        """
        encoding_indices_view = list(inputs.shape)
        del encoding_indices_view[1]

        with torch.cuda.amp.autocast(enabled=False):
            inputs = inputs.float()

            # Converting to channel last format
            flat_input = inputs.permute(self.flatten_permutation).contiguous().view(-1, self.embedding_dim)

            # Calculate Euclidean distances
            distances = (
                (flat_input**2).sum(dim=1, keepdim=True)
                + (self.embedding.weight.t() ** 2).sum(dim=0, keepdim=True)
                - 2 * torch.mm(flat_input, self.embedding.weight.t())
            )

            # Mapping distances to indexes
            encoding_indices = torch.max(-distances, dim=1)[1]
            print("encoding_indices shape issssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssss", encoding_indices)
            encodings = torch.nn.functional.one_hot(encoding_indices, self.num_embeddings).float()

            # Quantize and reshape
            encoding_indices = encoding_indices.view(encoding_indices_view)

        return flat_input, encodings, encoding_indices

    def embed(self, embedding_indices: torch.Tensor) -> torch.Tensor:
        """
        Given encoding indices of shape [B,D,H,W,1] embeds them in the quantized space
        [B, D, H, W, self.embedding_dim] and reshapes them to [B, self.embedding_dim, D, H, W] to be fed to the
        decoder.

        Args:
            embedding_indices: Tensor in channel last format which holds indices referencing atomic
                elements from self.embedding

        Returns:
            torch.Tensor: Quantize space representation of encoding_indices in channel first format.
        """
        with torch.cuda.amp.autocast(enabled=False):
            return self.embedding(embedding_indices).permute(self.quantization_permutation).contiguous()

    @torch.jit.unused
    def distributed_synchronization(self, encodings_sum: torch.Tensor, dw: torch.Tensor) -> None:
        """
        TorchScript does not support torch.distributed.all_reduce. This function is a bypassing trick based on the
        example: https://pytorch.org/docs/stable/generated/torch.jit.unused.html#torch.jit.unused

        Args:
            encodings_sum: The summation of one hot representation of what encoding was used for each
                position.
            dw: The multiplication of the one hot representation of what encoding was used for each
                position with the flattened input.

        Returns:
            None
        """
        if self.ddp_sync and torch.distributed.is_initialized():
            torch.distributed.all_reduce(tensor=encodings_sum, op=torch.distributed.ReduceOp.SUM)
            torch.distributed.all_reduce(tensor=dw, op=torch.distributed.ReduceOp.SUM)
        else:
            pass

    def forward(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        flat_input, encodings, encoding_indices = self.quantize(inputs)
        quantized = self.embed(encoding_indices)

        # Use EMA to update the embedding vectors
        if self.training:
            print("EMA Training Started")
            with torch.no_grad():
                encodings_sum = encodings.sum(0)
                dw = torch.mm(encodings.t(), flat_input)

                if self.ddp_sync:
                    self.distributed_synchronization(encodings_sum, dw)

                self.ema_cluster_size.data.mul_(self.decay).add_(torch.mul(encodings_sum, 1 - self.decay))

                # Laplace smoothing of the cluster size
                n = self.ema_cluster_size.sum()
                weights = (self.ema_cluster_size + self.epsilon) / (n + self.num_embeddings * self.epsilon) * n
                self.ema_w.data.mul_(self.decay).add_(torch.mul(dw, 1 - self.decay))
                self.embedding.weight.data.copy_(self.ema_w / weights.unsqueeze(1))
        else:
            encodings_sum=torch.zeros(256)

        # print("self.embedding.weight.data", (self.embedding.weight.data).shape)
        print("quantized ema shape is", quantized.shape)
        print("inputs ema shape is", inputs.shape)
        # Encoding Loss
        
        loss = self.commitment_cost * mse_loss(quantized.detach(), inputs)
        loss = loss

        # Straight Through Estimator
        quantized = inputs + (quantized - inputs).detach()

        return quantized, loss, encoding_indices, encodings_sum, self.embedding.weight.data


class VectorQuantizer(torch.nn.Module):
    """
    Vector Quantization wrapper that is needed as a workaround for the AMP to isolate the non fp16 compatible parts of
    the quantization in their own class.

    Args:
        quantizer (torch.nn.Module):  Quantizer module that needs to return its quantized representation, loss and index
            based quantized representation. Defaults to None
    """

    def __init__(self, quantizer: torch.nn.Module = None):
        super().__init__()

        self.quantizer: torch.nn.Module = quantizer

        self.perplexity: torch.Tensor = torch.rand(1)

    def forward(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        quantized, loss, encoding_indices, encodings_sum, embedding = self.quantizer(inputs)

        # Perplexity calculations
        avg_probs = (
            torch.histc(encoding_indices.float(), bins=self.quantizer.num_embeddings, max=self.quantizer.num_embeddings)
            .float()
            .div(encoding_indices.numel())
        )

        self.perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        print("self.perplexity", self.perplexity)
        # loss += 0.5 * self.perplexity

        return loss, quantized, encodings_sum, embedding

    def embed(self, embedding_indices: torch.Tensor) -> torch.Tensor:
        return self.quantizer.embed(embedding_indices=embedding_indices)

    def quantize(self, encodings: torch.Tensor) -> torch.Tensor:
        quantized, loss, encoding_indices, encodings_sum, embedding = self.quantizer(encodings)

        return encoding_indices


# class EMAQuantizer_indice(nn.Module):
#     """
#     Vector Quantization module using Exponential Moving Average (EMA) to learn the codebook parameters based on  Neural
#     Discrete Representation Learning by Oord et al. (https://arxiv.org/abs/1711.00937) and the official implementation
#     that can be found at https://github.com/deepmind/sonnet/blob/v2/sonnet/src/nets/vqvae.py#L148 and commit
#     58d9a2746493717a7c9252938da7efa6006f3739.

#     This module is not compatible with TorchScript while working in a Distributed Data Parallelism Module. This is due
#     to lack of TorchScript support for torch.distributed module as per https://github.com/pytorch/pytorch/issues/41353
#     on 22/10/2022. If you want to TorchScript your model, please turn set `ddp_sync` to False.

#     Args:
#         spatial_dims :  number of spatial spatial_dims.
#         num_embeddings: number of atomic elements in the codebook.
#         embedding_dim: number of channels of the input and atomic elements.
#         commitment_cost: scaling factor of the MSE loss between input and its quantized version. Defaults to 0.25.
#         decay: EMA decay. Defaults to 0.99.
#         epsilon: epsilon value. Defaults to 1e-5.
#         embedding_init: initialization method for the codebook. Defaults to "normal".
#         ddp_sync: whether to synchronize the codebook across processes. Defaults to True.
#     """

#     def __init__(
#         self,
#         spatial_dims: int,
#         num_embeddings: int,
#         embedding_dim: int,
#         commitment_cost: float = 0.25,
#         decay: float = 0.99,
#         epsilon: float = 1e-5,
#         embedding_init: str = "normal",
#         ddp_sync: bool = True,
#         pretrained_embedding: Optional[torch.Tensor] = None,  # Add this parameter
#     ):
#         super().__init__()
#         self.spatial_dims: int = spatial_dims
#         self.embedding_dim: int = embedding_dim
#         self.num_embeddings: int = num_embeddings
    
#         assert self.spatial_dims in [2, 3], ValueError(
#             f"EMAQuantizer only supports 4D and 5D tensor inputs but received spatial dims {spatial_dims}."
#         )
    
#         # Initialize embedding
#         self.embedding: torch.nn.Embedding = torch.nn.Embedding(self.num_embeddings, self.embedding_dim)
        
#         # Load pretrained embedding if provided
#         if pretrained_embedding is not None:
#             if pretrained_embedding.shape != (self.num_embeddings, self.embedding_dim):
#                 raise ValueError(
#                     f"Pretrained embedding must have shape ({self.num_embeddings}, {self.embedding_dim}), "
#                     f"but got {pretrained_embedding.shape}."
#                 )
#             self.embedding.weight.data.copy_(pretrained_embedding)
#         elif embedding_init == "kaiming_uniform":
#             torch.nn.init.kaiming_uniform_(self.embedding.weight.data, mode="fan_in", nonlinearity="linear")
#         # Otherwise, use default initialization (normal)
    
#         self.embedding.weight.requires_grad = False
    
#         self.commitment_cost: float = commitment_cost
    
#         self.register_buffer("ema_cluster_size", torch.zeros(self.num_embeddings))
#         self.register_buffer("ema_w", self.embedding.weight.data.clone())
    
#         self.decay: float = decay
#         self.epsilon: float = epsilon
    
#         self.ddp_sync: bool = ddp_sync
    
#         # Precalculating required permutation shapes
#         self.flatten_permutation: Sequence[int] = [0] + list(range(2, self.spatial_dims + 2)) + [1]
#         self.quantization_permutation: Sequence[int] = [0, self.spatial_dims + 1] + list(
#             range(1, self.spatial_dims + 1)
#         )

#     def quantize(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
#         """
#         Given an input it projects it to the quantized space and returns additional tensors needed for EMA loss.

#         Args:
#             inputs: Encoding space tensors

#         Returns:
#             torch.Tensor: Flatten version of the input of shape [B*D*H*W, C].
#             torch.Tensor: One-hot representation of the quantization indices of shape [B*D*H*W, self.num_embeddings].
#             torch.Tensor: Quantization indices of shape [B,D,H,W,1]

#         """
#         encoding_indices_view = list(inputs.shape)
#         del encoding_indices_view[1]

#         with torch.cuda.amp.autocast(enabled=False):
#             inputs = inputs.float()

#             # Converting to channel last format
#             flat_input = inputs.permute(self.flatten_permutation).contiguous().view(-1, self.embedding_dim)

#             # Calculate Euclidean distances
#             distances = (
#                 (flat_input**2).sum(dim=1, keepdim=True)
#                 + (self.embedding.weight.t() ** 2).sum(dim=0, keepdim=True)
#                 - 2 * torch.mm(flat_input, self.embedding.weight.t())
#             )

#             # Mapping distances to indexes
#             encoding_indices = torch.max(-distances, dim=1)[1]
#             print("encoding_indices shape issssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssss", encoding_indices)
#             encodings = torch.nn.functional.one_hot(encoding_indices, self.num_embeddings).float()

#             # Quantize and reshape
#             encoding_indices = encoding_indices.view(encoding_indices_view)

#         return flat_input, encodings, encoding_indices

#     def embed(self, embedding_indices: torch.Tensor) -> torch.Tensor:
#         """
#         Given encoding indices of shape [B,D,H,W,1] embeds them in the quantized space
#         [B, D, H, W, self.embedding_dim] and reshapes them to [B, self.embedding_dim, D, H, W] to be fed to the
#         decoder.

#         Args:
#             embedding_indices: Tensor in channel last format which holds indices referencing atomic
#                 elements from self.embedding

#         Returns:
#             torch.Tensor: Quantize space representation of encoding_indices in channel first format.
#         """
#         with torch.cuda.amp.autocast(enabled=False):
#             return self.embedding(embedding_indices).permute(self.quantization_permutation).contiguous()

#     @torch.jit.unused
#     def distributed_synchronization(self, encodings_sum: torch.Tensor, dw: torch.Tensor) -> None:
#         """
#         TorchScript does not support torch.distributed.all_reduce. This function is a bypassing trick based on the
#         example: https://pytorch.org/docs/stable/generated/torch.jit.unused.html#torch.jit.unused

#         Args:
#             encodings_sum: The summation of one hot representation of what encoding was used for each
#                 position.
#             dw: The multiplication of the one hot representation of what encoding was used for each
#                 position with the flattened input.

#         Returns:
#             None
#         """
#         if self.ddp_sync and torch.distributed.is_initialized():
#             torch.distributed.all_reduce(tensor=encodings_sum, op=torch.distributed.ReduceOp.SUM)
#             torch.distributed.all_reduce(tensor=dw, op=torch.distributed.ReduceOp.SUM)
#         else:
#             pass

#     def forward(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
#         flat_input, encodings, encoding_indices = self.quantize(inputs)
#         quantized = self.embed(encoding_indices)

#         # Use EMA to update the embedding vectors
#         if self.training:
#             print("EMA Training Started")
#             with torch.no_grad():
#                 encodings_sum = encodings.sum(0)
#                 dw = torch.mm(encodings.t(), flat_input)

#                 if self.ddp_sync:
#                     self.distributed_synchronization(encodings_sum, dw)

#                 self.ema_cluster_size.data.mul_(self.decay).add_(torch.mul(encodings_sum, 1 - self.decay))

#                 # Laplace smoothing of the cluster size
#                 n = self.ema_cluster_size.sum()
#                 weights = (self.ema_cluster_size + self.epsilon) / (n + self.num_embeddings * self.epsilon) * n
#                 self.ema_w.data.mul_(self.decay).add_(torch.mul(dw, 1 - self.decay))
#                 self.embedding.weight.data.copy_(self.ema_w / weights.unsqueeze(1))
#         else:
#             encodings_sum=torch.zeros(256)

#         # print("self.embedding.weight.data", (self.embedding.weight.data).shape)
#         print("quantized ema shape is", quantized.shape)
#         print("inputs ema shape is", inputs.shape)
#         # Encoding Loss
        
#         loss = self.commitment_cost * mse_loss(quantized.detach(), inputs)
#         loss = loss

#         # Straight Through Estimator
#         encoding_indices1=encoding_indices.unsqueeze(dim=1)
#         additional_channels = torch.zeros(encoding_indices1.size(0), 63, *encoding_indices1.size()[2:], device=encoding_indices1.device) 
#         extended_tensor=torch.cat([encoding_indices1, additional_channels], dim=1) 
#         encoding_indices1 = inputs + (extended_tensor - inputs).detach()
#         quantized = inputs + (quantized - inputs).detach()

#         return quantized, loss, encoding_indices, encodings_sum, self.embedding.weight.data, encoding_indices1


# class VectorQuantizer_indice(torch.nn.Module):
#     """
#     Vector Quantization wrapper that is needed as a workaround for the AMP to isolate the non fp16 compatible parts of
#     the quantization in their own class.

#     Args:
#         quantizer (torch.nn.Module):  Quantizer module that needs to return its quantized representation, loss and index
#             based quantized representation. Defaults to None
#     """

#     def __init__(self, quantizer: torch.nn.Module = None):
#         super().__init__()

#         self.quantizer: torch.nn.Module = quantizer

#         self.perplexity: torch.Tensor = torch.rand(1)

#     def forward(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
#         quantized, loss, encoding_indices, encodings_sum, embedding, encoding_indices1 = self.quantizer(inputs)

#         # Perplexity calculations
#         avg_probs = (
#             torch.histc(encoding_indices.float(), bins=self.quantizer.num_embeddings, max=self.quantizer.num_embeddings)
#             .float()
#             .div(encoding_indices.numel())
#         )

#         self.perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

#         return loss, quantized, encodings_sum, embedding, encoding_indices1

#     def embed(self, embedding_indices: torch.Tensor) -> torch.Tensor:
#         return self.quantizer.embed(embedding_indices=embedding_indices)

#     def quantize(self, encodings: torch.Tensor) -> torch.Tensor:
#         quantized, loss, encoding_indices, encodings_sum, embedding = self.quantizer(encodings)

#         return encoding_indices











class ConvBlock3D_wo_p(nn.Module):
    """Convolution Block with Conv3d, BatchNorm, ReLU, and Dropout"""
    def __init__(self, in_channels, out_channels, dropout_prob):
        super(ConvBlock3D_wo_p, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.batch_norm = nn.BatchNorm3d(out_channels)
        self.dropout = nn.Dropout3d(p=dropout_prob)
    
    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        x = self.batch_norm(x)
        x = self.dropout(x)
        return x






class ConvBlock3D_won_p(nn.Module):
    """Convolution Block with Conv3d, BatchNorm, ReLU, and Dropout"""
    def __init__(self, in_channels, out_channels, dropout_prob):
        super(ConvBlock3D_won_p, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.batch_norm = nn.BatchNorm3d(out_channels)
        self.dropout = nn.Dropout3d(p=dropout_prob)
    
    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        # x = self.batch_norm(x)
        # x = self.dropout(x)
        return x

# class TransformerModeldec_mlm(nn.Module):
#     def __init__(self, input_channels=32, embed_size=(32, 60, 60, 38), max_register_size=128):
#         super(TransformerModeldec_mlm, self).__init__()
        
#         # 5-layer 3D Conv network
#         self.conv_layers = nn.Sequential(
#             nn.Conv3d(32, 32, kernel_size=3, stride=1, padding=1),
#             nn.ReLU(),
#             nn.Conv3d(32, 32, kernel_size=3, stride=1, padding=1),
#             nn.ReLU()
#         )
        
#         # Register buffer for storing embeddings
#         self.register_buffer('embedding', torch.zeros((max_register_size, *embed_size)))
#         self.register_counter = 0
#         self.max_register_size = max_register_size

#     def find_closest_embedding(self, normalized):
#         # distances = torch.norm(self.gamma_embedding - normalized, dim=[1, 2, 3, 4])
#         flat_input = normalized.view(normalized.shape[0], -1)  # Flatten spatial dimensions
#         flat_embeddings = self.embedding.view(self.max_register_size, -1)  # Flatten stored embeddings
    
#         distances = (
#             (flat_input**2).sum(dim=1, keepdim=True)  # ||x||^2
#             + (flat_embeddings**2).sum(dim=1, keepdim=False).unsqueeze(0)  # ||y||^2
#             - 2 * torch.mm(flat_input, flat_embeddings.t())  # -2 * x^T y
#         )
    
#         return torch.argmin(distances, dim=1)


#     def forward(self, pred_latent, gt_latent, is_train=False):
#         if is_train:
#             avg_embedding = gt_latent.mean(dim=(0), keepdim=True)
#             if self.register_counter < self.max_register_size:
#                 min_idx = self.register_counter
#                 self.register_counter += 1                
#             else:
#                 print("Using custom distance calculation")
#                 min_idx = self.find_closest_embedding(pred_latent)
#                 print("min_idx", min_idx)
#             self.embedding[min_idx].data.copy_(0.9 * self.embedding[min_idx].to(pred_latent.device) + 0.1 * avg_embedding.squeeze(0))
#             output = self.embedding[min_idx].to(pred_latent.device)
#             print("output after register buffer is", output.shape)
#             # x = torch.cat([output, pred_latent], dim=1)
#             x = output + pred_latent
#             output = self.conv_layers(x)   
        
#         else:
#             min_idx = self.find_closest_embedding(pred_latent)
#             output = self.embedding[min_idx].to(pred_latent.device)
#             # output = output.expand_as(pred_latent)
#             x = output + pred_latent
#             output = self.conv_layers(x)    
        
#         return output





class Encoder3D(nn.Module):
    """Encoder consisting of multiple convolution blocks with increasing feature maps"""
    def __init__(self, in_channels, dropout_prob=0.5):
        super(Encoder3D, self).__init__()
        
        self.encoder1 = ConvBlock3D(in_channels, 8, dropout_prob)
        self.encoder2 = ConvBlock3D(8, 16, dropout_prob)
        self.encoder3 = ConvBlock3D(16, 32, dropout_prob)
        # self.encoder3_cat = TransformerModeldec_mlm()
        self.encoder4 = ConvBlock3D(32, 64, dropout_prob)
        self.encoder5 = ConvBlock3D(64, 128, dropout_prob)
        # self.res5 = ResidualBlock(128)
        self.encoder6 = ConvBlock3D(128, 256, dropout_prob)
        # self.register_buffer('running_mean', torch.zeros(1, 32, 60, 60, 38))  # Match output dimensions
        # self.register_buffer('running_var', torch.ones(1, 32, 60, 60, 38))
        # self.encoder7 = ConvBlock3D(256, 512, dropout_prob)
        # self.encoder8 = ConvBlock3D(512, 1024, dropout_prob)
        # self.encoder9 = ConvBlock3D(1024, 512, dropout_prob)
        self.pool = nn.MaxPool3d(2)
        
    def forward(self, x, autoenc, is_train=False):
        # x1 = self.encoder1(x)
        # print(f"Encoder1 output shape: {x1.shape}")
        # x2 = self.encoder2(self.pool(x1))
        # print(f"Encoder2 output shape: {x2.shape}")
        # x3 = self.encoder3(self.pool(x2))
        # print(f"Encoder3 output shape: {x3.shape}")
        # x4 = self.encoder4((x3))
        # # print(f"Encoder4 output shape: {x4.shape}")
        x1 = self.encoder1(x)
        print(f"Encoder1 output shape: {x1.shape}")
        # x1 = self.res1(x1)
        x2 = self.encoder2(self.pool(x1))
        print(f"Encoder2 output shape: {x2.shape}")
        # x2 = self.res2(x2)
        x3 = self.encoder3(self.pool(x2))
        print(f"Encoder3 output shape: {x3.shape}")  
        # x3 = self.encoder3_cat(x3, autoenc, is_train=is_train)
        # if is_train and random.random() >=0.6:
        #     x3 = self.encoder3_cat(torch.cat((x3, autoenc), dim=1))
        #     self.running_mean.data.copy_(0.9 * self.running_mean + 0.1 * x3.mean(dim=(0), keepdim=True))
        #     self.running_var.data.copy_(0.9 * self.running_var + 0.1 * x3.mean(dim=(0), keepdim=True))

        # else:
        #     gamma = self.running_mean
        #     beta = self.running_var
        #     x3 = x3 * (1 + gamma) + beta
        # x3 = self.res3(x3)
        x4 = self.encoder4(self.pool(x3))
        print(f"Encoder4 output shape: {x4.shape}")
        # x4 = self.res4(x4)
        x5 = self.encoder5(self.pool(x4))
        print(f"Encoder5 output shape: {x5.shape}")
        x6 = self.encoder6((x5))
        print(f"Encoder6 output shape: {x6.shape}")
        # x7 = self.encoder7((x6))
        # print(f"Encoder7 output shape: {x7.shape}")
        # x8 = self.encoder8((x7))
        # print(f"Encoder6 output shape: {x8.shape}")
        # x8 = self.encoder9((x8))
        # print(f"Encoder6 output shape: {x8.shape}")
        # padding = (0, 4, 0, 1, 0, 1)  # (left, right, top, bottom, front, back)
        # x6 = F.pad(x6, padding, mode='constant', value=0)
        
        # x5 = self.res5(x5)
        # x6 = self.encoder6(x5)
        return x1, x2, x3, x4, x5, x6
        
class BottleneckBlock(nn.Module):
    """Bottleneck block with 128 to 128 features"""
    def __init__(self, in_channels, dropout_prob=0.5):
        super(BottleneckBlock, self).__init__()
        self.bottleneck = ConvBlock3D_won_p(in_channels, in_channels, dropout_prob)
        
    def forward(self, x):
        x = self.bottleneck(x)
        print(f"Bottleneck output shape: {x.shape}")
        return x

# class Indice3D(nn.Module):
#     """Encoder consisting of multiple convolution blocks with increasing feature maps"""
#     def __init__(self, in_channels, dropout_prob=0.5):
#         super(Indice3D, self).__init__()
        
#         self.encoder1 = ConvBlock3D(in_channels, 64, dropout_prob)
#         self.encoder2 = ConvBlock3D(64, 64, dropout_prob)
#         self.encoder3 = ConvBlock3D(64, 64, dropout_prob)
#         self.encoder4 = ConvBlock3D(64, 64, dropout_prob)
#         self.conv1 = nn.Conv3d(64, 64, kernel_size=3, padding=1)
#         self.conv2 = nn.Conv3d(64, 64, kernel_size=3, padding=1)
#         # self.conv3 = nn.Conv3d(1, 1, kernel_size=1) 
        
        
#     def forward(self, x):
#         x1 = self.encoder1(x)
#         print(f"indice1 output shape: {x1.shape}")
#         x2 = self.encoder2(x1)
#         print(f"indice2 output shape: {x2.shape}")
#         x3 = self.encoder3(x2)
#         print(f"indice3 output shape: {x3.shape}")
#         x4 = self.encoder4(x3)
#         print(f"indice4 output shape: {x4.shape}")
#         x5 = self.conv1(x4)
#         x5 = F.relu(x5)
#         x6 = self.conv2(x5)
#         # x6 = F.relu(x6)
#         # x7 = self.conv3(x6)
#         # x7 = F.relu(x7)
#         return x6

class GELU2(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return x * F.sigmoid(1.702 * x)


class SEBlock3D(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(SEBlock3D, self).__init__()
        self.global_avg_pool = nn.AdaptiveAvgPool3d(1)  # Output shape: (batch, channels, 1, 1, 1)
        self.fc1 = nn.Conv3d(in_channels, in_channels // reduction, kernel_size=1)
        self.fc2 = nn.Conv3d(in_channels // reduction, in_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x_squeezed = self.global_avg_pool(x)  # Global pooling
        x_fc = torch.relu(self.fc1(x_squeezed))  # Fully connected layer 1
        x_fc = self.sigmoid(self.fc2(x_fc))  # Fully connected layer 2 with sigmoid
        return x * x_fc  # Scale input by channel attention weights

class SpatialAttention3D(nn.Module):
    def __init__(self, in_channels):
        super(SpatialAttention3D, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, 1, kernel_size=1)  # Single channel output
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # avg_pool = torch.mean(x, dim=1, keepdim=True)  # Pool across channels
        spatial_attention = self.sigmoid(self.conv1(x))  # Spatial attention map
        return x * spatial_attention  # Scale input by spatial attention map


class CombinedAttention3D(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(CombinedAttention3D, self).__init__()
        self.channel_attention = SEBlock3D(in_channels, reduction)
        self.spatial_attention = SpatialAttention3D(in_channels)

    def forward(self, x):
        x = self.channel_attention(x)  # Channel-wise attention
        x = self.spatial_attention(x)  # Spatial attention
        return x

class Decoder3D(nn.Module):
    """Decoder with skip connections and upsampling"""
    def __init__(self, dropout_prob=0.5):
        super(Decoder3D, self).__init__()
        self.upsample_0 = self.upsample_block1(256, dropout_prob)
        # self.transmodel = TransformerModeldec(input_shape=[30, 30, 18], embed_dim=256, num_layers=3, num_heads=8)
        # self.transmodel2 = TransformerModeldec(input_shape=[30, 30, 18], embed_dim=256, num_layers=2, num_heads=8)
        # self.attention = CombinedAttention3D(128, reduction=4)  # Add attention block
        # self.conv3d = ConvBlock3D(192, 128, dropout_prob)
        self.conv_0 = ConvBlock3D_wo_p(128, 128, dropout_prob)
        self.upsample_1 = self.upsample_block(128, dropout_prob)
        self.upsample_2 = self.upsample_block2(128, dropout_prob)
        self.conv_1 = nn.Conv3d(160, 512, kernel_size=3, padding=1)
        # self.conv_1 = nn.Conv3d(256, 128, kernel_size=3, padding=1)
        # self.conv_2 = nn.Conv3d(128, 256, kernel_size=3)
        # self.conv_3 = nn.Conv3d(256, 128, kernel_size=3, padding=1)
        # self.conv_4 = nn.Conv3d(256, 128, kernel_size=3, padding=1)
        # self.alpha = nn.Parameter(torch.tensor(0.5))  # Initial weight for low confidence
        # self.beta = nn.Parameter(torch.tensor(0.5))   # Initial weight for high confidence
        # self.upsample0 = self.upsample_block(128, dropout_prob)
        # self.conv0 = nn.Conv3d(128, 64, kernel_size=3, padding=1)
        
        # self.upsample1 = self.upsample_block(64, dropout_prob)
        # self.conv1 = nn.Conv3d(64, 32, kernel_size=3, padding=1)
        # self.upsample2 = self.upsample_block(32, dropout_prob)
        # self.conv2 = nn.Conv3d(32, 16, kernel_size=3, padding=1)
        # self.upsample3 = self.upsample_block(16, dropout_prob)
        # self.conv3 = nn.Conv3d(16, 8, kernel_size=3, padding=1)
        
        # self.final_conv = nn.Conv3d(8, 1, kernel_size=1)  # Assuming segmentation output is single channel
    
    def upsample_block(self, in_channels, dropout_prob):
        
        """Create an upsampling block with Conv3d, ReLU, BatchNorm, and Dropout"""
        layers = [
            nn.Upsample(scale_factor=1, mode='nearest'),
            nn.Conv3d(in_channels, in_channels // 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm3d(in_channels // 2),
            nn.Dropout3d(p=dropout_prob),
        ]
        return nn.Sequential(*layers)

    def upsample_block2(self, in_channels, dropout_prob):
        
        """Create an upsampling block with Conv3d, ReLU, BatchNorm, and Dropout"""
        layers = [
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv3d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm3d(in_channels),
            nn.Dropout3d(p=dropout_prob),
        ]
        return nn.Sequential(*layers)


    def upsample_block1(self, in_channels, dropout_prob):
        """Create an upsampling block with Conv3d, ReLU, BatchNorm, and Dropout"""
        layers = [
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv3d(in_channels, in_channels//2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm3d(in_channels//2),
            nn.Dropout3d(p=dropout_prob),
        ]
        return nn.Sequential(*layers)

    def forward(self, x, x4, x3):
        print(f"Decoder input (x): {x.shape}")
        x_6 = self.upsample_0(x)
        print(f"Decoder input up0 (x): {x_6.shape}")
        # x_6 = self.transmodel(x_6)
        x_6 = self.upsample_1(x_6)
        print(f"Decoder input up1 (x): {x_6.shape}")
        # x_6 = self.conv_0(x_6)
        # x_6 = x_6[:, :, 0:15, 0:15, 0:9]
        # print("x_6 shape is", x_6.shape)
        
        padding = (0, 1, 0, 0, 0, 0)  # (left, right, top, bottom, front, back)
        x_6 = F.pad(x_6, padding, mode='constant', value=0)
        
        # x_6 = self.conv_0(x_6)
        # print("x_6_pad shape is", x_6.shape)
        x_6 = self.conv_0(torch.cat((x_6, x4), dim=1))  # Concatenate with encoder3
        print(f"Decoder input up0 cat (x): {x_6.shape}")
        x_6 = self.upsample_2(x_6)
        print(f"Decoder input up2 (x): {x_6.shape}")
        x_6 = self.conv_1(torch.cat((x_6, x3), dim=1))  # Concatenate with encoder3
        # x_6 = torch.softmax(x_6, dim=1)
        # padding = (0, 1, 1, 1, 1, 1)  # (left, right, top, bottom, front, back)
        # x_6 = F.pad(x_6, padding, mode='constant', value=0)
        # x_6 = self.transmodel(x_6)
        # padding = (0, 1, 0, 0, 0, 0)  # (left, right, top, bottom, front, back)
        # x_6 = F.pad(x_6, padding, mode='constant', value=0)
        # x_6 = self.conv_1(x_6)



        # probs = F.softmax(x_6, dim=1)  # Compute softmax probabilities
        # max_probs, _ = probs.max(dim=1)  # Get max probabilities per voxel
        
        # # Learnable scaling factors
        
        # # Soft scaling
        # low_conf_scale = (1 - max_probs).unsqueeze(1) * self.alpha
        # high_conf_scale = max_probs.unsqueeze(1) * self.beta
        
        # # Apply scaling
        # low_conf_features = x_6 * low_conf_scale
        # high_conf_features = x_6 * high_conf_scale
        # low_conf_features = self.conv_2(low_conf_features)
        # padding = (0, 1, 1, 1, 1, 1)  # (left, right, top, bottom, front, back)
        # low_conf_features = F.pad(low_conf_features, padding, mode='constant', value=0)
        # low_conf_features = self.transmodel2(low_conf_features)
        # padding = (0, 1, 0, 0, 0, 0)  # (left, right, top, bottom, front, back)
        # low_conf_features = F.pad(low_conf_features, padding, mode='constant', value=0)
        # low_conf_features = self.conv_3(low_conf_features)

        # final_logits = torch.cat((low_conf_features, high_conf_features), dim=1)
        # final_logits = self.conv_4(final_logits)
        # x_6 = self.attention(x_6)
        # x_6 = self.conv_0(x_6)
        # padding = (0, 1, 1, 1, 1, 1)  # (left, right, top, bottom, front, back)
        # x_6 = F.pad(x_6, padding, mode='constant', value=0)
        # print("x_6_pad shape is", x_6.shape)
        # print("x_6 shape is after concatenation", x_6.shape)
        # x_6 = self.upsample0(x_6)  # First decoder layer
        # print(f"Upsample1 output shape: {x_6.shape}")

        # padding = (0, 1, 0, 0, 0, 0)  # (left, right, top, bottom, front, back)
        # x_6 = F.pad(x_6, padding, mode='constant', value=0)
        # x_7 = self.conv0(torch.cat([x_6, x4], dim=1))  # Concatenate with encoder3
        # print(f"Conv1 output shape after concatenation: {x_7.shape}")
        # x6 = self.upsample1(x_7)  # First decoder layer
        # print(f"Upsample1 output shape: {x6.shape}")
        # x7 = self.conv1(torch.cat([x6, x3], dim=1))  # Concatenate with encoder3
        # print(f"Conv1 output shape after concatenation: {x7.shape}")
        # x7 = self.upsample2(x7)
        # print(f"Upsample2 output shape: {x7.shape}")
        # padding = (0, 1, 0, 0, 0, 0)  # (left, right, top, bottom, front, back)
        # x7 = F.pad(x7, padding, mode='constant', value=0)
        # x8 = self.conv2(torch.cat([x7, x2], dim=1))  # Concatenate with encoder2
        # print(f"Conv2 output shape after concatenation: {x8.shape}")
        # x8 = self.upsample3(x8)
        # print(f"Upsample3 output shape: {x8.shape}")
        # padding = (0, 1, 0, 0, 0, 0)  # (left, right, top, bottom, front, back)
        # x8 = F.pad(x8, padding, mode='constant', value=0)
        # x9 = self.conv3(torch.cat([x8, x1], dim=1))  # Concatenate with encoder1
        # print(f"Conv3 output shape after concatenation: {x9.shape}")
        # out = self.final_conv(x9)  # Final output
        # print(f"Final output shape: {out.shape}")
        return x_6



class LatentSpaceMaskReducer(nn.Module):
    def __init__(self, input_channels, reduction_factor=1):
        super(LatentSpaceMaskReducer, self).__init__()
        self.reduction_factor = reduction_factor
        # 1x1 convolution to reduce channels to 1
        self.channel_reduction = nn.Conv3d(input_channels, 1, kernel_size=1)
        # A small network to learn the mask
        self.mask_network = nn.Sequential(
            nn.Conv3d(1, 1, kernel_size=3, padding=1),
            nn.Sigmoid()  # Output is a mask between 0 and 1
        )

    def generate_mask(self, reduced_tensor):
        # Generate a mask based on the reduced tensor
        mask = self.mask_network(reduced_tensor)
        return mask

    def reduce_latent(self, latent_tensor):
        B, C, H, W, D = latent_tensor.shape
        
        # Ensure tensor is on the correct device
        
        # Reduce channels from C to 1 using a 1x1 convolution
        reduced_tensor = self.channel_reduction(latent_tensor)  # Shape: (B, 1, D, H, W)

        # Generate a mask that identifies the background and foreground
        mask = self.generate_mask(reduced_tensor)  # Shape: (B, 1, D, H, W)

        # Mask the original latent tensor
        masked_tensor = latent_tensor * mask  # Element-wise multiplication

        return masked_tensor




# class EdgeRefinement3D(torch.nn.Module):
#     def __init__(self):
#         super(EdgeRefinement3D, self).__init__()
#         self.conv = nn.Conv3d(128, 1, kernel_size=3, padding=1)

#     def sobel_3d(self, input_tensor):
#         """
#         Apply a Sobel filter in 3D to compute gradients in the x, y, and z directions.
#         """
#         # Sobel kernel for the x direction
#         sobel_x = torch.tensor(
#             [[[[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
#                [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
#                [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]]]],
#             dtype=torch.float32, device=input_tensor.device
#         )

#         # Sobel kernel for the y direction
#         sobel_y = torch.tensor(
#             [[[[[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
#                [[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
#                [[-1, -2, -1], [0, 0, 0], [1, 2, 1]]]]],
#             dtype=torch.float32, device=input_tensor.device
#         )

#         # Sobel kernel for the z direction
#         sobel_z = torch.tensor(
#             [[[[[-1, -1, -1], [-1, -1, -1], [-1, -1, -1]],
#                [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
#                [[1, 1, 1], [1, 1, 1], [1, 1, 1]]]]],
#             dtype=torch.float32, device=input_tensor.device
#         )

#         # Ensure input_tensor is 5D: (batch, channel, depth, height, width)
#         if input_tensor.ndim != 5:
#             raise ValueError("Expected input_tensor to be 5D (batch, channel, depth, height, width), got shape: {}".format(input_tensor.shape))

#         # Apply Sobel filters in each direction
#         print("input_tensor", input_tensor.shape)
#         # F.conv3d(input_tensor, kernel=3, padding=1)
#         grad_x = F.conv3d(input_tensor, sobel_x, stride=1, padding=1)
#         grad_y = F.conv3d(input_tensor, sobel_y, stride=1, padding=1)
#         grad_z = F.conv3d(input_tensor, sobel_z, stride=1, padding=1)

#         # Combine gradients
#         gradient_magnitude = torch.sqrt(grad_x**2 + grad_y**2 + grad_z**2)

#         return gradient_magnitude

#     def forward(self, input_tensor):
#         """Forward pass for Edge Refinement."""
#         x = self.conv(input_tensor)
#         edge_map = self.sobel_3d(x)
#         input_tensor_up = edge_map * input_tensor
#         return input_tensor_up

# class EdgeRefinement3D_lap(torch.nn.Module):
#     def __init__(self):
#         super(EdgeRefinement3D_lap, self).__init__()
#         self.conv = nn.Conv3d(128, 1, kernel_size=3, padding=1)

#     def laplacian_3d(self, input_tensor):
#         """
#         Apply a Laplacian filter in 3D to compute second-order derivatives.
#         """
#         # Laplacian kernel
#         laplacian_kernel = torch.tensor(
#             [[[[[0, 1, 0], [1, -6, 1], [0, 1, 0]],
#                [[1, -6, 1], [-6, 36, -6], [1, -6, 1]],
#                [[0, 1, 0], [1, -6, 1], [0, 1, 0]]]]],
#             dtype=torch.float32, device=input_tensor.device
#         )

#         # Ensure input_tensor is 5D: (batch, channel, depth, height, width)
#         if input_tensor.ndim != 5:
#             raise ValueError("Expected input_tensor to be 5D (batch, channel, depth, height, width), got shape: {}".format(input_tensor.shape))

#         # Apply the Laplacian filter
#         laplacian_output = F.conv3d(input_tensor, laplacian_kernel, stride=1, padding=1)

#         return laplacian_output

#     def forward(self, input_tensor):
#         """Forward pass for Edge Refinement."""
#         x = self.conv(input_tensor)
#         laplacian_map = self.laplacian_3d(x)
#         input_tensor_up = laplacian_map * input_tensor
#         return input_tensor_up



# import torch
# import torch.nn as nn
# import torch.nn.functional as F

class EdgeRefinement3D(torch.nn.Module):
    def __init__(self):
        super(EdgeRefinement3D, self).__init__()
        self.conv = nn.Conv3d(128, 1, kernel_size=3, padding=1)

    def gaussian_filter_3d(self, input_tensor, kernel_size=3, sigma=1.0):
        """
        Apply a Gaussian filter in 3D for smoothing.
        """
        # Create a 3D Gaussian kernel
        kernel = self.create_gaussian_kernel(kernel_size, sigma, input_tensor.device)

        # Ensure input_tensor is 5D: (batch, channel, depth, height, width)
        if input_tensor.ndim != 5:
            raise ValueError("Expected input_tensor to be 5D (batch, channel, depth, height, width), got shape: {}".format(input_tensor.shape))

        # Apply the Gaussian filter using convolution
        gaussian_output = F.conv3d(input_tensor, kernel, stride=1, padding=kernel_size//2)
        return gaussian_output

    def create_gaussian_kernel(self, kernel_size, sigma, device):
        """
        Create a 3D Gaussian kernel.
        """
        # Create a 1D Gaussian kernel
        kernel_1d = torch.linspace(-(kernel_size // 2), kernel_size // 2, kernel_size, device=device)
        kernel_1d = torch.exp(-0.5 * (kernel_1d / sigma) ** 2)

        # Normalize the kernel
        kernel_1d = kernel_1d / kernel_1d.sum()

        # Create a 3D Gaussian kernel by taking the outer product of the 1D kernels
        kernel_3d = kernel_1d.view(1, 1, kernel_size, 1, 1) * kernel_1d.view(1, 1, 1, kernel_size, 1) * kernel_1d.view(1, 1, 1, 1, kernel_size)
        kernel_3d = kernel_3d.expand(1, 1, kernel_size, kernel_size, kernel_size)  # Expand to 3D kernel

        # Ensure the kernel has the correct shape for convolution
        return kernel_3d

    def sobel_3d(self, input_tensor):
        """
        Apply a Sobel filter in 3D to compute gradients in the x, y, and z directions.
        """
        # Sobel kernel for the x direction
        sobel_x = torch.tensor(
            [[[[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
               [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
               [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]]]],
            dtype=torch.float32, device=input_tensor.device
        )

        # Sobel kernel for the y direction
        sobel_y = torch.tensor(
            [[[[[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
               [[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
               [[-1, -2, -1], [0, 0, 0], [1, 2, 1]]]]],
            dtype=torch.float32, device=input_tensor.device
        )

        # Sobel kernel for the z direction
        sobel_z = torch.tensor(
            [[[[[-1, -1, -1], [-1, -1, -1], [-1, -1, -1]],
               [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
               [[1, 1, 1], [1, 1, 1], [1, 1, 1]]]]],
            dtype=torch.float32, device=input_tensor.device
        )

        # Ensure input_tensor is 5D: (batch, channel, depth, height, width)
        if input_tensor.ndim != 5:
            raise ValueError("Expected input_tensor to be 5D (batch, channel, depth, height, width), got shape: {}".format(input_tensor.shape))

        # Apply Sobel filters in each direction
        grad_x = F.conv3d(input_tensor, sobel_x, stride=1, padding=1)
        grad_y = F.conv3d(input_tensor, sobel_y, stride=1, padding=1)
        grad_z = F.conv3d(input_tensor, sobel_z, stride=1, padding=1)

        # Combine gradients
        gradient_magnitude = torch.sqrt(grad_x**2 + grad_y**2 + grad_z**2)

        return gradient_magnitude

    def forward(self, input_tensor):
        """Forward pass for Edge Refinement."""
        # Apply the Gaussian filter to the input tensor first
        x = self.conv(input_tensor)
        smoothed_input = self.gaussian_filter_3d(x)

        # Apply Sobel filter to the smoothed input
        edge_map = self.sobel_3d(smoothed_input)

        # Refine the input tensor using the edge map
        input_tensor_up = edge_map * input_tensor
        return input_tensor_up







# import torch
# import torch.nn as nn
# import torch.nn.functional as F

class EdgeRefinement3D_lap(torch.nn.Module):
    def __init__(self):
        super(EdgeRefinement3D_lap, self).__init__()
        self.conv = nn.Conv3d(128, 1, kernel_size=3, padding=1)

    def gaussian_3d(self, input_tensor, sigma=1.0):
        """
        Apply a 3D Gaussian filter to smooth the image.
        """
        # Define a 3D Gaussian kernel using the Gaussian function.
        kernel_size = 3  # Size of the kernel (3x3x3 for simplicity)
        kernel = self.create_gaussian_kernel(kernel_size, sigma)
        kernel = kernel.to(input_tensor.device)

        # Apply the Gaussian filter (convolution).
        smoothed_input = F.conv3d(input_tensor, kernel, stride=1, padding=1)
        return smoothed_input

    def create_gaussian_kernel(self, kernel_size, sigma):
        """
        Create a 3D Gaussian kernel.
        """
        # Create a 1D Gaussian kernel.
        ax = torch.arange(-(kernel_size // 2), kernel_size // 2 + 1, dtype=torch.float32)
        xx, yy, zz = torch.meshgrid(ax, ax, ax)
        kernel = torch.exp(-(xx**2 + yy**2 + zz**2) / (2 * sigma**2))

        # Normalize the kernel to ensure the sum is 1.
        kernel = kernel / kernel.sum()

        # Reshape the kernel into a 4D tensor for convolution (out_channels=1, in_channels=1, depth, height, width)
        kernel = kernel.unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, depth, height, width)
        return kernel

    def laplacian_3d(self, input_tensor):
        """
        Apply a Laplacian filter in 3D to compute second-order derivatives.
        """
        # Laplacian kernel
        laplacian_kernel = torch.tensor(
            [[[[[0, 1, 0], [1, -6, 1], [0, 1, 0]],
               [[1, -6, 1], [-6, 36, -6], [1, -6, 1]],
               [[0, 1, 0], [1, -6, 1], [0, 1, 0]]]]],
            dtype=torch.float32, device=input_tensor.device
        )

        # Ensure input_tensor is 5D: (batch, channel, depth, height, width)
        if input_tensor.ndim != 5:
            raise ValueError("Expected input_tensor to be 5D (batch, channel, depth, height, width), got shape: {}".format(input_tensor.shape))

        # Apply the Laplacian filter
        laplacian_output = F.conv3d(input_tensor, laplacian_kernel, stride=1, padding=1)
        return laplacian_output

    def forward(self, input_tensor):
        """Forward pass for Edge Refinement."""
        # Apply a 3D Gaussian filter for smoothing
        x = self.conv(input_tensor)
        smoothed_tensor = self.gaussian_3d(x)

        # Apply Laplacian filter to the smoothed tensor
        laplacian_map = self.laplacian_3d(smoothed_tensor)

        # Refine the input tensor using the Laplacian map
        input_tensor_up = laplacian_map * input_tensor
        return input_tensor_up


class SPADEGenerator(nn.Module):
    def __init__(self, latent_channels, bottleneck_channels):
        super().__init__()
        self.conv_mean = nn.Conv3d(latent_channels, bottleneck_channels, kernel_size=3, padding=1)
        self.conv_std = nn.Conv3d(latent_channels, bottleneck_channels, kernel_size=3, padding=1)

    def forward(self, autoencoder_latent, x_bottt):
        autoencoder_latent_mean = autoencoder_latent.mean(dim=(2, 3, 4), keepdim=True)
        autoencoder_latent_std = autoencoder_latent.std(dim=(2, 3, 4), keepdim=True)
        mean = self.conv_mean(x_bottt)  # Compute mean
        std = self.conv_std(x_bottt)   # Compute std
        mean = mean.mean(dim=(2, 3, 4), keepdim=True)
        std = std.std(dim=(2, 3, 4), keepdim=True)
        
        return mean, std, autoencoder_latent_mean, autoencoder_latent_std

class SPADELayer(nn.Module):
    def __init__(self, bottleneck_channels):
        super().__init__()
        self.epsilon = 1e-5  # To prevent division by zero

    def forward(self, x_bottt, mean, std):
        # Compute the batch mean and std of x_bottt
        batch_mean = x_bottt.mean(dim=(2, 3, 4), keepdim=True)
        batch_std = x_bottt.std(dim=(2, 3, 4), keepdim=True)

        # Normalize x_bottt using its batch stats
        normalized = (x_bottt - batch_mean) / (batch_std + self.epsilon)

        # Apply spatial conditioning
        output = std * normalized + mean
        return output


class Learnable3DDilation(nn.Module):
    def __init__(self, kernel_size=3):
        super(Learnable3DDilation, self).__init__()
        self.kernel = nn.Parameter(torch.ones(1, 1, kernel_size, kernel_size, kernel_size))  

    def forward(self, x):
        return F.max_pool3d(x, kernel_size=self.kernel.shape[-1], stride=1, padding=1)

class Learnable3DErosion(nn.Module):
    def __init__(self, kernel_size=3):
        super(Learnable3DErosion, self).__init__()
        self.kernel = nn.Parameter(torch.ones(1, 1, kernel_size, kernel_size, kernel_size))  

    def forward(self, x):
        x_inv = -x  
        eroded_inv = F.max_pool3d(x_inv, kernel_size=self.kernel.shape[-1], stride=1, padding=1)
        return -eroded_inv

class HalfMorphologicalGradient3D(nn.Module):
    def __init__(self, kernel_size=3, mode="external"):
        """
        mode = "external" → Outer boundary (Dilation - Original)
        mode = "internal" → Inner boundary (Original - Erosion)
        """
        super(HalfMorphologicalGradient3D, self).__init__()
        self.dilation = Learnable3DDilation(kernel_size)
        self.erosion = Learnable3DErosion(kernel_size)
        self.mode = mode

    def forward(self, x):
        if self.mode == "external":
            return self.dilation(x) # Outer boundary
        elif self.mode == "internal":
            return x - self.erosion(x)  # Inner boundary
        else:
            raise ValueError("Mode must be 'external' or 'internal'")

class SegmentationModel(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, num_classes: int) -> None:
        super(SegmentationModel, self).__init__()
        self.conv1 = nn.Conv3d(4, 2, kernel_size=3, padding=1)
        self.conv2 = nn.Conv3d(2, 2, kernel_size=1)  # Output channels equal to num_classes
        # self.conv4_op = nn.Conv3d(4, 4, kernel_size=1)  # Output channels equal to num_classes
        # self.conv5_op = nn.Conv3d(4, 4, kernel_size=1)  # Output channels equal to num_classes
        # self.conv6_op = nn.Conv3d(4, 4, kernel_size=1)  # Output channels equal to num_classes
    def forward(self, x: torch.Tensor) -> torch.Tensor:
    
        print("X shape  before is", x.shape)
        # if x.shape[4] <= 100:
        #     x = F.pad(x, (1, 0, 0, 1, 1, 0), mode='constant', value=0)
        # else:
        #     x = F.pad(x, (1, 0, 0, 1, 1, 0), mode='constant', value=0)
        x = self.conv1(x)
        segmentation_mask1 = self.conv2(x)
        # print("X shape after is", segmentation_mask1.shape)
        
        
        # # Compute softmax probabilities over classes
        output_probabilities = F.softmax(segmentation_mask1, dim=1)
        return output_probabilities


class MultiScaleChannelManipulationNetwork(nn.Module):
    def __init__(self, in_channels=512, out_channels=256):
        super(MultiScaleChannelManipulationNetwork, self).__init__()

        # Increasing channels (32 -> 64 -> 128)
        self.conv1_1 = self.multi_scale_block(in_channels, 256)
        self.conv1_2 = self.multi_scale_block(256, 256)
        
        # Maintain 128 channels
        # self.conv2 = self.multi_scale_block(256, 512)

        # Decreasing channels (128 -> 64 -> 32)
        # self.conv3_1 = self.multi_scale_block(128, 64)
        # self.conv3_2 = self.multi_scale_block(64, 32)

        # Final 1x1 convolution to refine output
        self.final_conv = nn.Conv3d(768, 512, kernel_size=1)
        # self.final_activation = nn.ReLU()  # Apply ReLU to final output

    def multi_scale_block(self, in_channels, out_channels):
        """Multi-scale block with kernel sizes 1x1, 3x3, and 5x5."""
        return nn.Sequential(
            MultiScaleConv3D(in_channels, out_channels),
            nn.ReLU()  # Apply ReLU after multi-scale convolution
        )

    def forward(self, latent):
        # Pass through each stage
        x = self.conv1_1(latent)  # 32 -> 64
        x = self.conv1_2(x)  # 64 -> 128
        # x = self.conv2(x)    # Maintain 128 channels
        # x = self.conv3_1(x)  # 128 -> 64
        # x = self.conv3_2(x)  # 64 -> 32
        x_cat = torch.cat((latent, x), dim=1)
        x = self.final_conv(x_cat)  # 32 -> output_channels
        # x = self.final_activation(x)  # Final ReLU activation
        return x


class MultiScaleConv3D(nn.Module):
    """Applies 3D convolutions with different kernel sizes (1x1, 3x3, 5x5) and concatenates the outputs."""
    def __init__(self, in_channels, out_channels):
        super(MultiScaleConv3D, self).__init__()

        # 3 parallel convolutions with different kernel sizes
        self.conv1x1 = nn.Conv3d(in_channels, out_channels, kernel_size=1)
        self.conv3x3 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv5x5 = nn.Conv3d(in_channels, out_channels, kernel_size=5, padding=2)

        # 1x1 convolution to fix channel count
        self.channel_reduce = nn.Conv3d(out_channels*3, out_channels, kernel_size=1)

        # Activation for each convolution
        self.relu = nn.ReLU()

    def forward(self, x):
        x1 = self.relu(self.conv1x1(x))
        x2 = self.relu(self.conv3x3(x))
        x3 = self.relu(self.conv5x5(x))

        # Concatenate the multi-scale outputs along the channel dimension
        x_cat = torch.cat((x1, x2, x3), dim=1)

        # Reduce the number of channels back to out_channels
        x_out = self.channel_reduce(x_cat)

        return x_out




class TC_stack(nn.Module):
    def __init__(self, autoencoder_quantizer0,  autoencoder_conv3, autoencoder_conv4, autoencoder_decoder, autoencoder_segmentataion, in_channels: int, out_channels: int, dropout_prob: float()):
        super(TC_stack, self).__init__()

        self.dropout_prob = dropout_prob  # Dropout probability

        # Initialize Encoder, Bottleneck, and Decoder as separate modules
        self.encoder = Encoder3D(in_channels, dropout_prob)
        # self.encoder1 = Encoder3D(1, dropout_prob)
        # self.encoder2 = Encoder3D(1, dropout_prob)
        # self.encoder3 = Encoder3D(1, dropout_prob)
        # self.indice3D = Indice3D(64, dropout_prob)
        self.bottleneck = BottleneckBlock(128, dropout_prob)
        self.decoder = Decoder3D(dropout_prob)
        # self.EdgeRefinement3D = EdgeRefinement3D()
        # self.EdgeRefinement3D_lap = EdgeRefinement3D_lap()
        # self.conv = nn.Conv3d(256, 128, kernel_size=3, padding=1)
        # self.LatentSpaceMaskReducer = LatentSpaceMaskReducer(256)
        # # self.pretrained_embedding = pretrained_embedding
        # self.quantizer0 = VectorQuantizer(
        #     quantizer=EMAQuantizer(
        #         spatial_dims=3,
        #         num_embeddings=512,
        #         embedding_dim=512,
        #         commitment_cost=0.25,
        #         decay=0.99,
        #         epsilon=1e-5,
        #         embedding_init='uniform',
        #         ddp_sync=False,
        #         pretrained_embedding=None,
        #     )
        # )
        self.transmodel = TransformerModel(input_shape=[15, 15, 9], embed_dim=256, num_layers=8, num_heads=8)
        # self.HalfMorphologicalGradient3D = HalfMorphologicalGradient3D()
        # self.conv_gdt = nn.Conv3d(8, 4, kernel_size=3, padding=1)
        # self.conv_spade = nn.Conv3d(512, 32, kernel_size=3, padding=1)
        # self.conv_spade2 = nn.Conv3d(32, 512, kernel_size=3, padding=1)
        # self.spade_generator = SPADEGenerator(32, 32)
        # self.spade_layer = SPADELayer(32)
        self.quantizer1 = autoencoder_quantizer0
        # self.MultiScaleChannelManipulationNetwork = MultiScaleChannelManipulationNetwork()
        # self.TransformerModeldec_mlm = TransformerModeldec_mlm((30, 30, 18), 128, 5, 8)
        # self.conv_mlm = nn.Conv3d(128, 128, kernel_size=3)
        # self.LatentSpaceReducer = LatentSpaceReducer(4, 2)
        # self.conv_hg = nn.Conv3d(8, 4, kernel_size=3, padding=1)
        # self.conv1 = nn.Conv3d(1024, 256, kernel_size=3, padding=1)
        # self.conv4 = nn.Conv3d(256, 64, kernel_size=3, padding=1)
        # self.conv3 = nn.Conv3d(512, 32, kernel_size=3, padding=1)
        # # # self.conv2 = nn.Conv3d(64, 32, kernel_size=3, padding=1)
        # # # self.conv3 = nn.Conv3d(32, 64, kernel_size=3, padding=1)
        # self.conv4 = nn.Conv3d(512, 32, kernel_size=3, padding=1)
        # self.quantizer1 = VectorQuantizer(
        #     quantizer=EMAQuantizer(
        #         spatial_dims=3,
        #         num_embeddings=512,
        #         embedding_dim=32,
        #         commitment_cost=0.25,
        #         decay=0.99,
        #         epsilon=1e-5,
        #         embedding_init='uniform',
        #         ddp_sync=False,
        #         pretrained_embedding=None,
        #     )
        # )
        self.conv3 = autoencoder_conv3
        self.conv4 = autoencoder_conv4
        self.model_pt_decoder = autoencoder_decoder
        self.model_pt_seg = autoencoder_segmentataion
        # self.SegmentationModel1 = autoencoder_segmentataion
        # self.SegmentationModel2 = SegmentationModel(4, 2, 2)
    def forward(self, images, autoencdoer_latent, is_train=False):
        # Encoder path
        # images_gdt = self.HalfMorphologicalGradient3D(images)
        # padding = (1, 1, 1, 1, 1, 1)  # (left, right, top, bottom, front, back)
        # images_gdt = F.pad(images_gdt, padding, mode='constant', value=0)
        # images = self.conv_gdt(torch.cat((images, images_gdt), dim=1))
        x1, x2, x3, x4, x5, x6 = self.encoder(images, autoencdoer_latent, is_train=is_train)
        # x1t2, x2t2, x3t2, x4t2, x5t2, x6t2 = self.encoder1(t2)
        # x1t2f, x2t2f, x3t2f, x4t2f, x5t2f, x6t2f = self.encoder2(t2f)
        # x1t1c, x2t1c, x3t1c, x4t1c, x5t1c, x6t1c = self.encoder3(t1c)
        # print("shape is", (torch.cat((x6t1, x6t2, x6t2f, x6t1c), dim=1)).shape)
        # x6 = self.conv1(torch.cat((x6t1, x6t2, x6t2f, x6t1c), dim=1))
        # x4 = self.conv4(torch.cat((x4t1, x4t2, x4t2f, x4t1c), dim=1))
        # x3 = self.conv3(torch.cat((x3t1, x3t2, x3t2f, x3t1c), dim=1))
        # x6 = self.LatentSpaceMaskReducer.reduce_latent(x6)
        # x6 = self.LatentSpaceReducer(x6)
        x8=self.transmodel(x6)
        x8 = self.decoder(x8, x4, x3)
        # x8 = self.MultiScaleChannelManipulationNetwork(x8)
        # x8_spd = self.conv_spade(x8)
        # mean, std, autoencoder_latent_mean, autoencoder_latent_std = self.spade_generator(autoencoder_latent, x8)

        # Apply SPADE
        # x8_spd = self.spade_layer(x8, mean, std)
        # x8 = self.conv_spade2(x8_spd)
        # x_bottt = self.conv4(x8)
        # print("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", torch.sum(((max_prob))>0.35))
        # print("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", torch.sum(((max_prob))>0.4))
        # print("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", torch.sum(((max_prob))>0.45))
        # print("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", torch.sum(((max_prob))>0.5))
        # entropy_map = -torch.sum(softmax_output * torch.log(softmax_output + 1e-10), dim=1)  # Avoid log(0) with small epsilon
        # threshold = entropy_map.mean() + entropy_map.std()  # Example threshold
        # high_entropy_mask = entropy_map > threshold
        # print("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", torch.sum((high_entropy_mask)))
        # print("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", ((high_entropy_mask.shape)))
        # coords = torch.nonzero(high_entropy_mask, as_tuple=False)  # Get coordinates of high-entropy voxels
        # min_coords = coords.min(dim=0).values  # Minimum corner of bounding box
        # max_coords = coords.max(dim=0).values  # Maximum corner of bounding box
        # print("min_coords", min_coords)
        # print("max_coords", max_coords)
        # flat_output = torch.argmax(softmax_output, dim=1).flatten()
        # print("flat_output", flat_output)

        x8 = torch.argmax(x8, dim=1)
        embeddingsss = self.quantizer1.embed(x8)
        # embeddingsss_quant = self.MultiScaleChannelManipulationNetwork(embeddingsss)
        embeddingsss=self.conv3(embeddingsss)
        embeddingsss=self.conv4(embeddingsss)

        # # x8_sob = self.EdgeRefinement3D(x8)

        
        # # x8_lap = self.EdgeRefinement3D_lap(x8)

        # # x_cat = self.conv(torch.cat((x8_sob, x8_lap), dim=1))
        # # x_bottt = (self.bottleneck(x8)).float()
        # # quantized_loss, quantized, encodings_sum, embedding = self.quantizer1(x8_spd)

        embeddingsss = self.model_pt_decoder(embeddingsss)
        
        embeddingsss1 = self.model_pt_seg(embeddingsss)

        # print("embeddingsss1 shape is", embeddingsss1.shape)
        

        # embeddingsss2 = self.SegmentationModel2(embeddingsss)
        # embeddingsss = torch.cat((images_gdt, embeddingsss), dim=1)
        # embeddingsss = self.conv_hg(embeddingsss) 
        # embeddingsss_pt = self.model_pt_seg(embeddingsss)
        # embeddingsss = self.HalfMorphologicalGradient3D(embeddingsss)
        
        
        # b, c, h, w, d = x.shape
        # softmax_out = torch.softmax(x8, dim=1)

        # # Step 2: Create a confidence mask (probability > 0.5)
        # confidence_mask = (softmax_out > 0.5)  # Shape: (4, 128, 30, 30, 19)
        
        # # Step 3: Get the most confident predictions (argmax along channel dimension)
        # argmax_indices = torch.argmax(softmax_out, dim=1)  # Shape: (4, 30, 30, 19)
        
        # # Step 4: Mask the argmax indices based on confidence
        # # Reduce the confidence mask to match the spatial dimensions
        # confidence_reduced = confidence_mask.any(dim=1)  # Shape: (4, 30, 30, 19)
        
        # # Apply the confidence mask to the argmax indices
        # masked_indices = torch.where(confidence_reduced, argmax_indices, torch.tensor(-1))  # -1 indicates masked
        # masked_indices = self.conv_mlm(masked_indices.unsqueeze(dim=1))
        # padding = (0, 1, 1, 1, 1, 1)  # (left, right, top, bottom, front, back)
        # masked_indices = F.pad(masked_indices, padding, mode='constant', value=0)
        # masked_indices = masked_indices.squeeze(dim=1)
        # masked_indices = self.TransformerModeldec_mlm(masked_indices)
        # reconstruction = self.model_pt_decoder(quantized)
        # segmentataion = self.model_pt_seg(reconstruction)
        # x9 = self.conv4(x8)
        # loss, quantized, encodings_sum, embedding
        # Bottleneck
        # x_bot = self.bottleneck(x6)

        # x_bot = self.conv1(x_bot)
        # x_bot = self.conv2(x_bot)

        # indice = self.indice3D(x5)
        # q_loss1, quantized1, encodings_sum1, embedding1, encoding_indices1 = self.quantizer1(indice)
        
        # q_loss, z_quantized0, encodings_sum, embedding = self.quantizer0(x_bot)

        # # z_quantized0_post = self.conv3(z_quantized0)
        # z_quantized0 = self.conv4(z_quantized0)

        # # Decoder path with skip connections
        # out = self.decoder(z_quantized0, x6, x5, x4, x3, x2, x1)
        quantized = 0.0
        quantized_loss = 0.0

        return embeddingsss1
        
        # return x8, quantized, segmentataion, quantized_loss
