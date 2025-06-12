# SCUNet-NGswin/models/scunet_ngswin.py

# -*- coding: utf-8 -*-
import math
import torch
import torch.nn as nn
import numpy as np
from thop import profile
from einops import rearrange 
from einops.layers.torch import Rearrange, Reduce
from timm.models.layers import trunc_normal_, DropPath

# importing NGSwin model
# from models.NGSwin.my_model.ngswin_model.ngswin import NGswin
from models.NGSwin.my_model.ngswin_model.nstb import NSTB


class WMSA(nn.Module):
    """ Self-attention module in Swin Transformer
    """

    def __init__(self, input_dim, output_dim, head_dim, window_size, type):
        super(WMSA, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.head_dim = head_dim 
        self.scale = self.head_dim ** -0.5
        self.n_heads = input_dim//head_dim
        self.window_size = window_size
        self.type=type
        self.embedding_layer = nn.Linear(self.input_dim, 3*self.input_dim, bias=True)

        # TODO recover
        # self.relative_position_params = nn.Parameter(torch.zeros(self.n_heads, 2 * window_size - 1, 2 * window_size -1))
        self.relative_position_params = nn.Parameter(torch.zeros((2 * window_size - 1)*(2 * window_size -1), self.n_heads))

        self.linear = nn.Linear(self.input_dim, self.output_dim)

        trunc_normal_(self.relative_position_params, std=.02)
        self.relative_position_params = torch.nn.Parameter(self.relative_position_params.view(2*window_size-1, 2*window_size-1, self.n_heads).transpose(1,2).transpose(0,1))

    def generate_mask(self, h, w, p, shift):
        """ generating the mask of SW-MSA
        Args:
            shift: shift parameters in CyclicShift.
        Returns:
            attn_mask: should be (1 1 w p p),
        """
        # supporting sqaure.
        attn_mask = torch.zeros(h, w, p, p, p, p, dtype=torch.bool, device=self.relative_position_params.device)
        if self.type == 'W':
            return attn_mask

        s = p - shift
        attn_mask[-1, :, :s, :, s:, :] = True
        attn_mask[-1, :, s:, :, :s, :] = True
        attn_mask[:, -1, :, :s, :, s:] = True
        attn_mask[:, -1, :, s:, :, :s] = True
        attn_mask = rearrange(attn_mask, 'w1 w2 p1 p2 p3 p4 -> 1 1 (w1 w2) (p1 p2) (p3 p4)')
        return attn_mask

    def forward(self, x):
        """ Forward pass of Window Multi-head Self-attention module.
        Args:
            x: input tensor with shape of [b h w c];
            attn_mask: attention mask, fill -inf where the value is True; 
        Returns:
            output: tensor shape [b h w c]
        """
        if self.type!='W': x = torch.roll(x, shifts=(-(self.window_size//2), -(self.window_size//2)), dims=(1,2))
        x = rearrange(x, 'b (w1 p1) (w2 p2) c -> b w1 w2 p1 p2 c', p1=self.window_size, p2=self.window_size)
        h_windows = x.size(1)
        w_windows = x.size(2)
        # sqaure validation
        # assert h_windows == w_windows

        x = rearrange(x, 'b w1 w2 p1 p2 c -> b (w1 w2) (p1 p2) c', p1=self.window_size, p2=self.window_size)
        qkv = self.embedding_layer(x)
        q, k, v = rearrange(qkv, 'b nw np (threeh c) -> threeh b nw np c', c=self.head_dim).chunk(3, dim=0)
        sim = torch.einsum('hbwpc,hbwqc->hbwpq', q, k) * self.scale
        # Adding learnable relative embedding
        sim = sim + rearrange(self.relative_embedding(), 'h p q -> h 1 1 p q')
        # Using Attn Mask to distinguish different subwindows.
        if self.type != 'W':
            attn_mask = self.generate_mask(h_windows, w_windows, self.window_size, shift=self.window_size//2)
            sim = sim.masked_fill_(attn_mask, float("-inf"))

        probs = nn.functional.softmax(sim, dim=-1)
        output = torch.einsum('hbwij,hbwjc->hbwic', probs, v)
        output = rearrange(output, 'h b w p c -> b w p (h c)')
        output = self.linear(output)
        output = rearrange(output, 'b (w1 w2) (p1 p2) c -> b (w1 p1) (w2 p2) c', w1=h_windows, p1=self.window_size)

        if self.type!='W': output = torch.roll(output, shifts=(self.window_size//2, self.window_size//2), dims=(1,2))
        return output

    def relative_embedding(self):
        cord = torch.tensor(np.array([[i, j] for i in range(self.window_size) for j in range(self.window_size)]))
        relation = cord[:, None, :] - cord[None, :, :] + self.window_size -1
        # negative is allowed
        return self.relative_position_params[:, relation[:,:,0].long(), relation[:,:,1].long()]


class Block(nn.Module):
    def __init__(self, input_dim, output_dim, head_dim, window_size, drop_path, type='W', input_resolution=None):
        """ SwinTransformer Block
        """
        super(Block, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        assert type in ['W', 'SW']
        self.type = type
        if input_resolution <= window_size:
            self.type = 'W'

        print("Block Initial Type: {}, drop_path_rate:{:.6f}".format(self.type, drop_path))
        self.ln1 = nn.LayerNorm(input_dim)
        self.msa = WMSA(input_dim, input_dim, head_dim, window_size, self.type)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.ln2 = nn.LayerNorm(input_dim)
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 4 * input_dim),
            nn.GELU(),
            nn.Linear(4 * input_dim, output_dim),
        )

    def forward(self, x):
        x = x + self.drop_path(self.msa(self.ln1(x)))
        x = x + self.drop_path(self.mlp(self.ln2(x)))
        return x

# original ConvTransBlock
class ConvTransBlock(nn.Module):
    def __init__(self, conv_dim, trans_dim, head_dim, window_size, drop_path, type='W', input_resolution=None):
        """ SwinTransformer and Conv Block
        """
        super(ConvTransBlock, self).__init__()
        self.conv_dim = conv_dim
        self.trans_dim = trans_dim
        self.head_dim = head_dim
        self.window_size = window_size
        self.drop_path = drop_path
        self.type = type
        self.input_resolution = input_resolution

        assert self.type in ['W', 'SW']
        if self.input_resolution <= self.window_size:
            self.type = 'W'

        self.trans_block = Block(self.trans_dim, self.trans_dim, self.head_dim, self.window_size, self.drop_path, self.type, self.input_resolution)
        self.conv1_1 = nn.Conv2d(self.conv_dim+self.trans_dim, self.conv_dim+self.trans_dim, 1, 1, 0, bias=True)
        self.conv1_2 = nn.Conv2d(self.conv_dim+self.trans_dim, self.conv_dim+self.trans_dim, 1, 1, 0, bias=True)

        self.conv_block = nn.Sequential(
                nn.Conv2d(self.conv_dim, self.conv_dim, 3, 1, 1, bias=False),
                nn.ReLU(True),
                nn.Conv2d(self.conv_dim, self.conv_dim, 3, 1, 1, bias=False)
                )


    def forward(self, x):
       
        # 1) Split input into convolutional and transformer components
        conv_x, trans_x = torch.split(self.conv1_1(x), (self.conv_dim, self.trans_dim), dim=1)

        # 2) Process convolutional component (Conv Block)
        conv_x = self.conv_block(conv_x) + conv_x

        # 3) Process transformer component (Swin Transformer Block)
        trans_x = Rearrange('b c h w -> b h w c')(trans_x)  # Rearrange for transformer processing
        trans_x = self.trans_block(trans_x)                # Apply Swin Transformer block
        trans_x = Rearrange('b h w c -> b c h w')(trans_x)  # Rearrange back to original format

        # 4) Combine convolutional and transformer components
        res = self.conv1_2(torch.cat((conv_x, trans_x), dim=1))

        # 5) Add residual connection
        x = x + res

        return x


class ConvNSTBBlock(nn.Module):
    """
    ConvNSTBBlock:
      - Splits input channels into a convolutional branch and an NSTB branch.
      - Conv branch: two 3×3 convs + ReLU + residual.
      - NSTB branch: flatten → NSTB (N-Gram Swin) → reshape.
      - Fuse: concatenate both outputs, apply 1×1 conv, then add a residual from x.
    """

    def __init__(
        self,
        conv_dim: int,
        trans_dim: int,
        head_dim: int,
        window_size: int,
        drop_path: float,
        block_type: str = 'W',   # either 'W' or 'SW'
        input_resolution: int = 256
    ):
        super(ConvNSTBBlock, self).__init__()
        self.conv_dim = conv_dim
        self.trans_dim = trans_dim
        self.window_size = window_size
        self.drop_path = drop_path
        self.type = block_type
        self.input_resolution = input_resolution

        # If resolution ≤ window_size, force windowed (no shift)
        if self.input_resolution <= self.window_size:
            self.type = 'W'

        # --- 1) First 1×1 conv: mix all incoming channels, then we will split ---
        self.conv1_1 = nn.Conv2d(
            in_channels=self.conv_dim + self.trans_dim,
            out_channels=self.conv_dim + self.trans_dim,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True
        )

        # --- 2) Convolutional branch: two 3×3 convs + ReLU, with a residual ---
        self.conv_block = nn.Sequential(
            nn.Conv2d(self.conv_dim, self.conv_dim, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(True),
            nn.Conv2d(self.conv_dim, self.conv_dim, kernel_size=3, stride=1, padding=1, bias=False)
        )

        # --- 3) NSTB branch: the N-Gram Swin Transformer block ---
        # We pass in (dim=trans_dim) and compute how many patches fit in each direction.
        self.nstb_block = NSTB(
            dim=self.trans_dim,
            training_num_patches=(
                self.input_resolution // self.window_size,
                self.input_resolution // self.window_size
            ),
            ngram=2,                              # you can tune this (e.g. 2 or 3)
            num_heads=self.trans_dim // head_dim, # total attention heads
            window_size=self.window_size,
            shift_size=self.window_size // 2,
            mlp_ratio=2.0,
            qkv_bias=True,
            drop=0.0,
            attn_drop=0.0,
            drop_path=self.drop_path,
            act_layer=nn.GELU,
            norm_layer=nn.LayerNorm
        )

        # --- 4) Second 1×1 conv: fuse concatenated conv-out + nstb-out back into (conv_dim+trans_dim) channels ---
        self.conv1_2 = nn.Conv2d(
            in_channels=self.conv_dim + self.trans_dim,
            out_channels=self.conv_dim + self.trans_dim,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: shape (B, conv_dim+trans_dim, H, W)
        Returns: (B, conv_dim+trans_dim, H, W), after fusing conv & NSTB branches + residual
        """

        # --- A) Mix & split ---
        mixed = self.conv1_1(x)
        # Now split into two chunks: conv_x ∈ (B, conv_dim, H, W) and trans_x ∈ (B, trans_dim, H, W)
        conv_x, trans_x = torch.split(mixed, [self.conv_dim, self.trans_dim], dim=1)

        # --- B) Convolutional branch ---
        # 1) Two 3×3 convs + ReLU
        conv_intermediate = self.conv_block(conv_x)            # → (B, conv_dim, H, W)
        # 2) Residual add inside conv branch
        conv_out = conv_x + conv_intermediate                   # → (B, conv_dim, H, W)

        # --- C) NSTB branch ---
        B, C_t, H, W = trans_x.shape
        # 1) Flatten spatial dims into a sequence of length H*W
        nstb_in = rearrange(trans_x, 'b c h w -> b (h w) c')   # → (B, H*W, trans_dim)
        # 2) Run through NSTB: returns (inp, out, num_patches). We only need the “out” sequence.
        _, nstb_out_flat, _ = self.nstb_block(nstb_in, (H, W))
        # 3) Reshape back to spatial feature map
        nstb_out = rearrange(nstb_out_flat, 'b (h w) c -> b c h w', h=H, w=W)
        #    → (B, trans_dim, H, W)

        # --- D) Fuse conv_out + nstb_out ---
        fused = torch.cat((conv_out, nstb_out), dim=1)          # → (B, conv_dim+trans_dim, H, W)
        merged = self.conv1_2(fused)                            # → (B, conv_dim+trans_dim, H, W)

        # --- E) Final residual from original x ---
        out = x + merged                                         # → (B, conv_dim+trans_dim, H, W)
        return out
class TransNSTBBlock(nn.Module):
    """
    TransNSTBBlock:
      - Splits input channels via a 1×1 conv, keeps only the transformer slice.
      - Runs a Swin Transformer block on one copy of that slice.
      - Runs an NSTB (N-Gram Swin) block on another copy of that slice.
      - Concatenates Swin-out and NSTB-out, fuses via 1×1 conv, then adds original input as a residual.
    """

    def __init__(
        self,
        conv_dim: int,
        trans_dim: int,
        head_dim: int,
        window_size: int,
        drop_path: float,
        block_type: str = 'W',    # either 'W' or 'SW'
        input_resolution: int = 256
    ):
        super(TransNSTBBlock, self).__init__()

        self.conv_dim = conv_dim
        self.trans_dim = trans_dim
        self.head_dim = head_dim
        self.window_size = window_size
        self.drop_path = drop_path
        self.type = block_type
        self.input_resolution = input_resolution

        assert self.type in ['W', 'SW']
        # If the feature map is smaller than one window, force “W” (no shift)
        if self.input_resolution <= self.window_size:
            self.type = 'W'

        # --- 1) First 1×1 conv: mix all incoming channels (conv_dim + trans_dim) → (conv_dim + trans_dim) ---
        self.conv1_1 = nn.Conv2d(
            in_channels=self.conv_dim + self.trans_dim,
            out_channels=self.conv_dim + self.trans_dim,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True
        )

        # --- 2) Swin Transformer block path (on trans_dim channels) ---
        #    We create one Swin block that expects input shape (B, H, W, trans_dim)
        self.trans_block = Block(
            self.trans_dim,        # input_dim = output_dim = trans_dim
            self.trans_dim,
            self.head_dim,
            self.window_size,
            self.drop_path,
            self.type,
            self.input_resolution
        )

        # --- 3) NSTB (N-Gram Swin Transformer) block path ---
        self.nstb_block = NSTB(
            dim=self.trans_dim,
            training_num_patches=(
                self.input_resolution // self.window_size,
                self.input_resolution // self.window_size
            ),
            ngram=2,                              # you can adjust n-gram size as needed
            num_heads=self.trans_dim // self.head_dim,
            window_size=self.window_size,
            shift_size=self.window_size // 2,
            mlp_ratio=2.0,
            qkv_bias=True,
            drop=0.0,
            attn_drop=0.0,
            drop_path=self.drop_path,
            act_layer=nn.GELU,
            norm_layer=nn.LayerNorm
        )

        # --- 4) Second 1×1 conv: fuse concatenated Swin-out + NSTB-out → (conv_dim + trans_dim) channels ---
        #    Note: We concatenate two trans_dim outputs, so conv1_2’s input channels = 2*trans_dim,
        #    and we output back to (conv_dim + trans_dim).
        self.conv1_2 = nn.Conv2d(
            in_channels=2 * self.trans_dim,
            out_channels=self.conv_dim + self.trans_dim,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, conv_dim + trans_dim, H, W)
        Returns: (B, conv_dim + trans_dim, H, W)
        """

        # --- A) Mix channels via 1×1 conv, then split ---
        mixed = self.conv1_1(x)
        # We only care about the “trans_dim” slice. The conv_dim slice is discarded.
        _, trans_x = torch.split(mixed, [self.conv_dim, self.trans_dim], dim=1)
        # Keep a copy of raw trans_x for NSTB
        raw_trans = trans_x  # ∈ (B, trans_dim, H, W)

        # --- B) Swin Transformer branch on trans_x ---
        # 1) Rearrange (B, C, H, W) → (B, H, W, C)
        trans_x_permuted = rearrange(trans_x, 'b c h w -> b h w c')
        # 2) Apply Swin block
        swin_out_permuted = self.trans_block(trans_x_permuted)  # ∈ (B, H, W, trans_dim)
        # 3) Rearrange back to (B, C, H, W)
        swin_out = rearrange(swin_out_permuted, 'b h w c -> b c h w')  # ∈ (B, trans_dim, H, W)

        # --- C) NSTB branch on raw_trans ---
        B, C_t, H, W = raw_trans.shape
        # 1) Flatten spatial dims: (B, C_t, H, W) → (B, H*W, C_t)
        nstb_in = rearrange(raw_trans, 'b c h w -> b (h w) c')
        # 2) Apply NSTB: returns (inp, out_flat, num_patches). We take out_flat.
        _, nstb_out_flat, _ = self.nstb_block(nstb_in, (H, W))
        # 3) Reshape (B, H*W, C_t) → (B, C_t, H, W)
        nstb_out = rearrange(nstb_out_flat, 'b (h w) c -> b c h w', h=H, w=W)  # ∈ (B, trans_dim, H, W)

        # --- D) Fuse Swin‐out and NSTB‐out, then add a residual from x ---
        # 1) Concatenate along channel axis: (B, 2*trans_dim, H, W)
        fused_trans = torch.cat((swin_out, nstb_out), dim=1)
        # 2) 1×1 conv to mix them + project back to (conv_dim + trans_dim) channels
        merged = self.conv1_2(fused_trans)  # ∈ (B, conv_dim + trans_dim, H, W)
        # 3) Residual add: add original input x
        out = x + merged                    # ∈ (B, conv_dim + trans_dim, H, W)

        return out

class ConvTransNSTBBlock(nn.Module):
    """
    ConvTransNSTBBlock:
      - Splits input into three parallel branches: Conv, Swin-Transformer, and NSTB.
      - Conv branch: two 3×3 convs + ReLU + residual.
      - Swin branch: layer-norm → windowed self-attention → MLP → residual (inside Swin Block).
      - NSTB branch: flatten → N-Gram Swin Transformer → reshape → no internal conv.
      - Fuse: concatenate all three outputs, apply 1×1 conv to mix, then add original input as a residual.
    """

    def __init__(
        self,
        conv_dim: int,
        trans_dim: int,
        head_dim: int,
        window_size: int,
        drop_path: float,
        block_type: str = 'W',    # 'W' for window, 'SW' for shifted window
        input_resolution: int = 256
    ):
        super(ConvTransNSTBBlock, self).__init__()

        self.conv_dim = conv_dim
        self.trans_dim = trans_dim
        self.head_dim = head_dim
        self.window_size = window_size
        self.drop_path = drop_path
        self.type = block_type
        self.input_resolution = input_resolution

        assert self.type in ['W', 'SW']
        if self.input_resolution <= self.window_size:
            self.type = 'W'

        # 1) First 1×1 conv: mix all incoming channels (conv_dim + trans_dim) → (conv_dim + trans_dim)
        self.conv1_1 = nn.Conv2d(
            in_channels=self.conv_dim + self.trans_dim,
            out_channels=self.conv_dim + self.trans_dim,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True
        )

        # 2) Convolutional branch: two 3×3 convs + ReLU, with a residual
        self.conv_block = nn.Sequential(
            nn.Conv2d(self.conv_dim, self.conv_dim, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(True),
            nn.Conv2d(self.conv_dim, self.conv_dim, kernel_size=3, stride=1, padding=1, bias=False)
        )

        # 3) Swin Transformer block (on trans_dim channels)
        self.trans_block = Block(
            input_dim=self.trans_dim,
            output_dim=self.trans_dim,
            head_dim=self.head_dim,
            window_size=self.window_size,
            drop_path=self.drop_path,
            type=self.type,
            input_resolution=self.input_resolution
        )

        # 4) NSTB block (N-Gram Swin Transformer)      
        self.nstb_block = NSTB(
            dim=self.trans_dim,
            training_num_patches=(
                self.input_resolution // self.window_size,
                self.input_resolution // self.window_size
            ),
            ngram=2,                              # adjust n-gram size as needed
            num_heads=self.trans_dim // self.head_dim,
            window_size=self.window_size,
            shift_size=self.window_size // 2,
            mlp_ratio=2.0,
            qkv_bias=True,
            drop=0.0,
            attn_drop=0.0,
            drop_path=self.drop_path,
            act_layer=nn.GELU,
            norm_layer=nn.LayerNorm
        )

        # 5) Second 1×1 conv: fuse concatenated outputs (conv_out, swin_out, nstb_out) → (conv_dim + trans_dim)
        #    concatenated channel count = conv_dim + trans_dim + trans_dim = conv_dim + 2*trans_dim
        self.conv1_2 = nn.Conv2d(
            in_channels=self.conv_dim + 2 * self.trans_dim,
            out_channels=self.conv_dim + self.trans_dim,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, conv_dim + trans_dim, H, W)
        returns: (B, conv_dim + trans_dim, H, W)
        """

        # --- A) Mix & split into conv_x and trans_x ---
        mixed = self.conv1_1(x)
        conv_x, trans_x = torch.split(mixed, [self.conv_dim, self.trans_dim], dim=1)
        raw_trans = trans_x  # save for NSTB branch

        # --- B) Convolutional branch ---
        # 1) two 3×3 convs + ReLU
        conv_intermediate = self.conv_block(conv_x)               # (B, conv_dim, H, W)
        # 2) add residual from conv_x
        conv_out = conv_x + conv_intermediate                      # (B, conv_dim, H, W)

        # --- C) Swin Transformer branch ---
        # 1) rearrange (B, C, H, W) → (B, H, W, C)
        trans_x_permuted = rearrange(trans_x, 'b c h w -> b h w c')
        # 2) apply Swin block
        swin_out_permuted = self.trans_block(trans_x_permuted)    # (B, H, W, trans_dim)
        # 3) rearrange back to (B, C, H, W)
        swin_out = rearrange(swin_out_permuted, 'b h w c -> b c h w')  # (B, trans_dim, H, W)

        # --- D) NSTB branch ---
        B, C_t, H, W = raw_trans.shape
        # 1) flatten spatial dims: (B, trans_dim, H, W) → (B, H*W, trans_dim)
        nstb_in = rearrange(raw_trans, 'b c h w -> b (h w) c')
        # 2) run NSTB: returns (inp, out_flat, num_patches)
        _, nstb_out_flat, _ = self.nstb_block(nstb_in, (H, W))
        # 3) reshape back: (B, H*W, trans_dim) → (B, trans_dim, H, W)
        nstb_out = rearrange(nstb_out_flat, 'b (h w) c -> b c h w', h=H, w=W)

        # --- E) Fuse conv_out, swin_out, nstb_out ---
        fused = torch.cat((conv_out, swin_out, nstb_out), dim=1)   # (B, conv_dim + 2*trans_dim, H, W)
        merged = self.conv1_2(fused)                               # (B, conv_dim + trans_dim, H, W)

        # --- F) Final residual ---
        out = x + merged                                            # (B, conv_dim + trans_dim, H, W)
        return out


class SCUNet(nn.Module):

    def __init__(self, in_nc=1, config=[2,2,2,2,2,2,2], dim=64, drop_path_rate=0.0, input_resolution=256):
        super(SCUNet, self).__init__()
        self.config = config
        self.dim = dim
        self.head_dim = 32
        self.window_size = 8

        # drop path rate for each layer
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(config))]

        self.m_head = [nn.Conv2d(in_nc, dim, 3, 1, 1, bias=False)]

        begin = 0
        self.m_down1 = [ConvTransBlock(dim//2, dim//2, self.head_dim, self.window_size, dpr[i+begin], 'W' if not i%2 else 'SW', input_resolution) 
                      for i in range(config[0])] + \
                      [nn.Conv2d(dim, 2*dim, 2, 2, 0, bias=False)]

        begin += config[0]
        self.m_down2 = [ConvTransBlock(dim, dim, self.head_dim, self.window_size, dpr[i+begin], 'W' if not i%2 else 'SW', input_resolution//2)
                      for i in range(config[1])] + \
                      [nn.Conv2d(2*dim, 4*dim, 2, 2, 0, bias=False)]

        begin += config[1]
        self.m_down3 = [ConvTransBlock(2*dim, 2*dim, self.head_dim, self.window_size, dpr[i+begin], 'W' if not i%2 else 'SW',input_resolution//4)
                      for i in range(config[2])] + \
                      [nn.Conv2d(4*dim, 8*dim, 2, 2, 0, bias=False)]

        begin += config[2]
        self.m_body = [ConvTransBlock(4*dim, 4*dim, self.head_dim, self.window_size, dpr[i+begin], 'W' if not i%2 else 'SW', input_resolution//8)
                    for i in range(config[3])]

        begin += config[3]
        self.m_up3 = [nn.ConvTranspose2d(8*dim, 4*dim, 2, 2, 0, bias=False),] + \
                      [ConvTransBlock(2*dim, 2*dim, self.head_dim, self.window_size, dpr[i+begin], 'W' if not i%2 else 'SW',input_resolution//4)
                      for i in range(config[4])]
                      
        begin += config[4]
        self.m_up2 = [nn.ConvTranspose2d(4*dim, 2*dim, 2, 2, 0, bias=False),] + \
                      [ConvTransBlock(dim, dim, self.head_dim, self.window_size, dpr[i+begin], 'W' if not i%2 else 'SW', input_resolution//2)
                      for i in range(config[5])]
                      
        begin += config[5]
        self.m_up1 = [nn.ConvTranspose2d(2*dim, dim, 2, 2, 0, bias=False),] + \
                    [ConvTransBlock(dim//2, dim//2, self.head_dim, self.window_size, dpr[i+begin], 'W' if not i%2 else 'SW', input_resolution) 
                      for i in range(config[6])]

        self.m_tail = [nn.Conv2d(dim, in_nc, 3, 1, 1, bias=False)]

        self.m_head = nn.Sequential(*self.m_head)
        self.m_down1 = nn.Sequential(*self.m_down1)
        self.m_down2 = nn.Sequential(*self.m_down2)
        self.m_down3 = nn.Sequential(*self.m_down3)
        self.m_body = nn.Sequential(*self.m_body)
        self.m_up3 = nn.Sequential(*self.m_up3)
        self.m_up2 = nn.Sequential(*self.m_up2)
        self.m_up1 = nn.Sequential(*self.m_up1)
        self.m_tail = nn.Sequential(*self.m_tail)  
        #self.apply(self._init_weights)

    def forward(self, x0):

        h, w = x0.size()[-2:]
        paddingBottom = int(np.ceil(h/64)*64-h)
        paddingRight = int(np.ceil(w/64)*64-w)
        x0 = nn.ReplicationPad2d((0, paddingRight, 0, paddingBottom))(x0)

        x1 = self.m_head(x0)
        x2 = self.m_down1(x1)
        x3 = self.m_down2(x2)
        x4 = self.m_down3(x3)
        x = self.m_body(x4)
        x = self.m_up3(x+x4)
        x = self.m_up2(x+x3)
        x = self.m_up1(x+x2)
        x = self.m_tail(x+x1)

        x = x[..., :h, :w]
        
        return x


    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)



if __name__ == '__main__':

    # torch.cuda.empty_cache()
    net = SCUNet()

    x = torch.randn((2, 1, 64, 128))
    x = net(x)
    print(x.shape)
