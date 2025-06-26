import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List
from einops import rearrange
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
    ):
        super().__init__()
        self.spatial_dims: int = spatial_dims
        self.embedding_dim: int = embedding_dim
        self.num_embeddings: int = num_embeddings

        assert self.spatial_dims in [2, 3], ValueError(
            f"EMAQuantizer only supports 4D and 5D tensor inputs but received spatial dims {spatial_dims}."
        )

        self.embedding: torch.nn.Embedding = torch.nn.Embedding(self.num_embeddings, self.embedding_dim)
        if embedding_init == "normal":
            # Initialization is passed since the default one is normal inside the nn.Embedding
            pass
        elif embedding_init == "kaiming_uniform":
            torch.nn.init.kaiming_uniform_(self.embedding.weight.data, mode="fan_in", nonlinearity="linear")
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
            encodings = torch.nn.functional.one_hot(encoding_indices, self.num_embeddings).float()
            encoding_probabilities = torch.softmax(-distances / 1.0, dim=1)
            entropy_loss = -torch.sum(encoding_probabilities * torch.log(encoding_probabilities + 1e-8)) / encoding_probabilities.size(0)
            entropy_loss = 0.01 * entropy_loss 
            # Quantize and reshape
            encoding_indices = encoding_indices.view(encoding_indices_view)

        return flat_input, encodings, encoding_indices, entropy_loss

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
        flat_input, encodings, encoding_indices, entropy_loss = self.quantize(inputs)
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
        # loss += entropy_loss

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

        # self.perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        self.perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        print("self.perplexity", self.perplexity)
        # loss += 0.01 * self.perplexity

        return loss, quantized, encodings_sum, embedding, encoding_indices

    def embed(self, embedding_indices: torch.Tensor) -> torch.Tensor:
        return self.quantizer.embed(embedding_indices)

    def quantize(self, encodings: torch.Tensor) -> torch.Tensor:
        _, _, encoding_indices, _, _ = self.quantizer(encodings)

        return encoding_indices




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





class ConvBlock3D_won(nn.Module):
    """Convolution Block with Conv3d, BatchNorm, ReLU, and Dropout"""
    def __init__(self, in_channels, out_channels, dropout_prob):
        super(ConvBlock3D_won, self).__init__()
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

class AttentionScalingWithHeads(nn.Module):
    def __init__(self, embed_dim, num_heads=8):
        super(AttentionScalingWithHeads, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # Linear layers for query projection
        self.query_proj = nn.Linear(embed_dim // 2, embed_dim)
        self.query_proj_up = nn.Linear(embed_dim, embed_dim)
        self.query_proj_up_cat = nn.Linear(embed_dim * 2, embed_dim)
        self.softmax = nn.Softmax(dim=2)  # Normalize across sequence length
        self.flatten = nn.Flatten(start_dim=2)
    def forward(self, x, x_up):
        b, c, h, w, d = x_up.shape
        x = self.flatten(x)
        x_up = self.flatten(x_up)
        x_up = x_up.permute(0, 2, 1) 
        print(f"x_up output shape: {x_up.shape}")
        query_up = self.query_proj_up(x_up)  # (batch_size, seq_len, embed_dim)
        print(f"query_up output shape: {query_up.shape}")
        query_up = query_up.permute(0, 2, 1) 
        batch_size_up, embed_dim_up, seq_len_up = query_up.size()
        
        

        # Project to multi-head space
        x = x.permute(0, 2, 1) 
        print("size of query ", x.shape)
        query = self.query_proj(x)  # (batch_size, seq_len, embed_dim)
        query = query.permute(0, 2, 1)
        print("size of query ", query.shape)
        batch_size, embed_dim_pre, seq_len_pre = query.size()
        query = query.view(batch_size, embed_dim_pre, seq_len_pre)  
       
        attention_scores = self.softmax(attention_scores, dim=-1)  # (batch_size, seq_len, num_heads)

       
        print("size of liner ", query_up.shape)
        print("size of liner ", attention_scores.shape)
        scaled_x = attention_scores * query_up
        
        print("size of scaled_x ", scaled_x.shape)
        scaled_x = rearrange(scaled_x, 'b (h w d) c -> b c h w d', h=h, w=w, d=d)
        return scaled_x

class Encoder3D(nn.Module):
    """Encoder consisting of multiple convolution blocks with increasing feature maps"""
    def __init__(self, in_channels, dropout_prob=0.5):
        super(Encoder3D, self).__init__()
        
        self.encoder1 = ConvBlock3D(in_channels, 8, dropout_prob)
        # self.res1 = ResidualBlock(8)
        self.encoder2 = ConvBlock3D(8, 16, dropout_prob)
        # self.res2 = ResidualBlock(16)
        self.encoder3 = ConvBlock3D(16, 32, dropout_prob)
        self.encoder4 = ConvBlock3D(32, 64, dropout_prob)
        self.encoder5 = ConvBlock3D(64, 128, dropout_prob)
        self.encoder6 = ConvBlock3D(128, 128, dropout_prob)
        
        self.pool = nn.MaxPool3d(2)
        
    def forward(self, x):
        x1 = self.encoder1(x)
        print(f"Encoder1 output shape: {x1.shape}")
        # x1 = self.res1(x1)
        x2 = self.encoder2(self.pool(x1))
        print(f"Encoder2 output shape: {x2.shape}")
        # x2 = self.res2(x2)
        x3 = self.encoder3(self.pool(x2))
        print(f"Encoder3 output shape: {x3.shape}")
        # x3 = self.res3(x3)
        x4 = self.encoder4((x3))
        # x4 = self.attention_scaling4(x3, x4)
        print(f"Encoder4 output shape: {x4.shape}")
        # x4 = self.res4(x4)
        # x4 = self.encoder5_pre(x4)
        # padding = (0, 1, 1, 1, 1, 1)  # (left, right, top, bottom, front, back)
        # x4 = F.pad(x4, padding, mode='constant', value=0)
        print(f"Encoder4 output shape: {x4.shape}")
        x5 = self.encoder5((x4))
        print(f"Encoder5 output shape: {x5.shape}")
        # x5 = self.attention_scaling5(x4, x5)
        print(f"Encoder5 output shape: {x5.shape}")
        # x5 = self.res5(x5)
        x5 = self.encoder6(x5)
        return x5


class BottleneckBlock(nn.Module):
    """Bottleneck block with 128 to 128 features"""
    def __init__(self, in_channels, dropout_prob=0.3):
        super(BottleneckBlock, self).__init__()
        self.bottleneck = ConvBlock3D(in_channels, in_channels, dropout_prob)
        
    def forward(self, x):
        x = self.bottleneck(x)
        print(f"Bottleneck output shape: {x.shape}")
        return x


class Decoder3D(nn.Module):
    """Decoder with skip connections and upsampling"""
    def __init__(self, dropout_prob=0.5):
        super(Decoder3D, self).__init__()
        self.res1 = ConvBlock3D(128, 128, dropout_prob)
        
        self.upsample1 = self.upsample_block1(128, dropout_prob)
        
        self.upsample2 = self.upsample_block1(64, dropout_prob)
        
        self.upsample3 = self.upsample_block(32, dropout_prob)
        
        self.upsample4 = self.upsample_block(16, dropout_prob)
        
        
        self.final_conv = nn.Conv3d(8, 4, kernel_size=3, padding=1)  # Assuming segmentation output is single channel
        
    
    def upsample_block(self, in_channels, dropout_prob):
        """Create an upsampling block with Conv3d, ReLU, BatchNorm, and Dropout"""
        layers = [
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv3d(in_channels, in_channels // 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm3d(in_channels // 2),
            nn.Dropout3d(p=dropout_prob),
        ]
        return nn.Sequential(*layers)
    def upsample_block1(self, in_channels, dropout_prob):
        """Create an upsampling block with Conv3d, ReLU, BatchNorm, and Dropout"""
        layers = [
            nn.Upsample(scale_factor=1, mode='nearest'),
            nn.Conv3d(in_channels, in_channels // 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm3d(in_channels // 2),
            nn.Dropout3d(p=dropout_prob),
        ]
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.res1(x)
        print(f"Decoder input (x): {x.shape}")
        x6 = self.upsample1(x)  # First decoder layer
        print(f"Upsample1 output shape: {x6.shape}")
       
        x7 = self.upsample2(x6)
        print(f"Upsample2 output shape: {x7.shape}")
        
       
        x8 = self.upsample3(x7)
        print(f"Upsample3 output shape: {x8.shape}")
        padding = (0, 1, 0, 0, 0, 0)  # (left, right, top, bottom, front, back)
        x8 = F.pad(x8, padding, mode='constant', value=0)
       

        x9 = self.upsample4(x8)
        print(f"Upsample3 output shape: {x9.shape}")
        padding = (0, 1, 0, 0, 0, 0)  # (left, right, top, bottom, front, back)
        x9 = F.pad(x9, padding, mode='constant', value=0)
        print(f"Upsample3 output shape: {x9.shape}")
       
        out = self.final_conv(x9)  # Final output
        print(f"Final output shape: {out.shape}")
        return out

class SegmentationModel(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, num_classes: int) -> None:
        super(SegmentationModel, self).__init__()
        self.conv1 = nn.Conv3d(4, 2, kernel_size=3, padding=1)
        self.conv2 = nn.Conv3d(2, 2, kernel_size=1)  # Output channels equal to num_classes
       
    def forward(self, x: torch.Tensor) -> torch.Tensor:
    
        print("X shape  before is", x.shape)
        
        x = self.conv1(x)
        segmentation_mask1 = self.conv2(x)
        
        output_probabilities = F.softmax(segmentation_mask1, dim=1)
        return output_probabilities





class VQVAE_seq_WT(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dropout_prob: float()):
        super(VQVAE_seq_WT, self).__init__()

        self.dropout_prob = dropout_prob  # Dropout probability

        # Initialize Encoder, Bottleneck, and Decoder as separate modules
        self.encoder = Encoder3D(in_channels, dropout_prob)
        self.bottleneck = BottleneckBlock(128, dropout_prob)
        self.decoder = Decoder3D(dropout_prob)
        self.segmentation=SegmentationModel(4, 4, 4)
        self.quantizer0 = VectorQuantizer(
            quantizer=EMAQuantizer(
                spatial_dims=3,
                num_embeddings=512,
                embedding_dim=32,
                commitment_cost=0.25,
                decay=0.99,
                epsilon=1e-5,
                embedding_init='uniform',
                ddp_sync=False,
            )
        )
        self.quantizer1 = VectorQuantizer(
            quantizer=EMAQuantizer(
                spatial_dims=3,
                num_embeddings=512,
                embedding_dim=32,
                commitment_cost=0.25,
                decay=0.99,
                epsilon=1e-5,
                embedding_init='uniform',
                ddp_sync=False,
            )
        )
        self.quantizer2 = VectorQuantizer(
            quantizer=EMAQuantizer(
                spatial_dims=3,
                num_embeddings=512,
                embedding_dim=32,
                commitment_cost=0.25,
                decay=0.99,
                epsilon=1e-5,
                embedding_init='uniform',
                ddp_sync=False,
            )
        )
        self.conv1 = nn.Conv3d(128, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv3d(64, 32, kernel_size=3, padding=1)
        
        self.conv3 = nn.Conv3d(32, 64, kernel_size=3, padding=1)
        self.conv4 = nn.Conv3d(64, 128, kernel_size=3, padding=1)
    def forward(self, x):
        # Encoder path
        x4 = self.encoder(x)
        
        # Bottleneck
        x5 = self.bottleneck(x4)
        x5 = self.conv1(x5)
        x5 = self.conv2(x5)
        # x5 = self.conv3(x5)
        quantization_loss0, z_quantized0, encodings_sum0, embedding0, encoding_indices = self.quantizer0(x5)

        # z_quantized0_post = self.conv4(z_quantized0)
        z_quantized0_post = self.conv3(z_quantized0)
        z_quantized0_post = self.conv4(z_quantized0_post)

        # Decoder path with skip connections
        reconstruction = self.decoder(z_quantized0_post)
        segmentation_mask = self.segmentation(reconstruction)
        
       
        
        
        print("segmentation_mask", reconstruction.shape)

        
        total_quantization_loss = torch.mean(quantization_loss0)
        print("total_quantization_loss2222222222222222", (total_quantization_loss))
# #        

        return z_quantized0, segmentation_mask, total_quantization_loss, encodings_sum0, embedding0
