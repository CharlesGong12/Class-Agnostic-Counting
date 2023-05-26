import time
from functools import partial
import math
import random

import numpy as np
import open_clip

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.utils

from timm.models.vision_transformer import PatchEmbed, Block
from models_crossvit import CrossAttentionBlock

from util.pos_embed import get_2d_sincos_pos_embed

class SupervisedMAE(nn.Module):
    def __init__(self, img_size=384, patch_size=16, in_chans=3,
                 embed_dim=512, depth=24, num_heads=16,
                 decoder_embed_dim=512, decoder_depth=2, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False):
        super().__init__()

        self.decoder_embed = nn.Linear(512, decoder_embed_dim, bias=True)
        # print(self.decoder_embed)
        self.shot_token = nn.Parameter(torch.zeros(512))

        # Exemplar encoder with CNN
        self.clip_enc, _, _ = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')

        self.decoder_blocks = nn.ModuleList([
            CrossAttentionBlock(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(decoder_depth)])

        self.decoder_norm = norm_layer(decoder_embed_dim)
        # Density map regresssion module
        self.decode_head0 = nn.Sequential(
            nn.Conv2d(decoder_embed_dim, 256, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(8, 256),
            nn.ReLU(inplace=True)
        )
        self.decode_head1 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(8, 256),
            nn.ReLU(inplace=True)
        )
        self.decode_head2 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(8, 256),
            nn.ReLU(inplace=True)
        )
        self.decode_head3 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(8, 256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 1, kernel_size=1, stride=1)
        )  
    
        # --------------------------------------------------------------------------

        self.norm_pix_loss = norm_pix_loss

        self.initialize_weights()

    def initialize_weights(self):
        torch.nn.init.normal_(self.shot_token, std=.02)
        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_decoder(self, x, word_vectors, shot_num=3):
        #print(x.shape, self.decoder_embed)
        x = self.decoder_embed(x)
        #print(x.shape)

        if shot_num > 0:
            y = self.clip_enc.encode_text(word_vectors.squeeze(1)).unsqueeze(1)
            # print("Encoded", y.shape)
        else:
            y = self.shot_token.repeat(word_vectors.shape[1],1).unsqueeze(0).to(x.device)
            y = y.transpose(0,1) # y [3,N,C]->[N,3,C]
        # print("Reshaped", x.shape, y.shape)
        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x, y)
            #print(x.shape)
        x = self.decoder_norm(x)
        
        # Density map regression
        n, hw, c = x.shape
        h = w = int(math.sqrt(hw))
        x = x.transpose(1, 2).reshape(n, c, h, w)
        #print(x.shape)
        x = F.interpolate(
                        self.decode_head0(x), size=x.shape[-1]*2, mode='bilinear', align_corners=False)
        #print(x.shape)
        x = F.interpolate(
                        self.decode_head1(x), size=x.shape[-1]*2, mode='bilinear', align_corners=False)
        #print(x.shape)
        x = F.interpolate(
                        self.decode_head2(x), size=x.shape[-1]*2, mode='bilinear', align_corners=False)
        #print(x.shape)
        x = F.interpolate(
                        self.decode_head3(x), size=x.shape[-1]*2, mode='bilinear', align_corners=False)
        #print(x.shape)
        x=F.interpolate(x, size=(224,224), mode='bilinear', align_corners=False)
        x = x.squeeze(-3)
        #print(x.shape)
        return x

    def forward(self, imgs, word_vectors, shot_num):
        #print(imgs.shape, word_vectors.shape, shot_num)
        with torch.no_grad():
            latent = self.clip_enc.encode_image(imgs).unsqueeze(1)  # [N, 1, 384]
        #print(latent.shape)
        pred = self.forward_decoder(latent, word_vectors, shot_num)  # [N, 384, 384]
        #print(pred.shape)
        return pred


def mae_vit_base_patch16_dec512d8b(**kwargs):
    model = SupervisedMAE(
        patch_size=16, embed_dim=768, depth=12, num_heads=12,
        decoder_embed_dim=512, decoder_depth=2, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def mae_vit_large_patch16_dec512d8b(**kwargs):
    model = SupervisedMAE(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16,
        decoder_embed_dim=512, decoder_depth=2, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def mae_vit_huge_patch14_dec512d8b(**kwargs):
    model = SupervisedMAE(
        patch_size=14, embed_dim=1280, depth=32, num_heads=16,
        decoder_embed_dim=512, decoder_depth=2, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def mae_vit_base_patch16_fim4(**kwargs):
    model = SupervisedMAE(
        patch_size=16, embed_dim=768, depth=12, num_heads=12,
        decoder_embed_dim=512, decoder_depth=4, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def mae_vit_base_patch16_fim6(**kwargs):
    model = SupervisedMAE(
        patch_size=16, embed_dim=768, depth=12, num_heads=12,
        decoder_embed_dim=512, decoder_depth=6, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


# set recommended archs
mae_vit_base_patch16 = mae_vit_base_patch16_dec512d8b  
mae_vit_base4_patch16 = mae_vit_base_patch16_fim4 # decoder: 4 blocks
mae_vit_base6_patch16 = mae_vit_base_patch16_fim6 # decoder: 6 blocks
mae_vit_large_patch16 = mae_vit_large_patch16_dec512d8b  
mae_vit_huge_patch14 = mae_vit_huge_patch14_dec512d8b  
