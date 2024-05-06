"""
两模态的注意力权重相互指导，相互增强
"""

import math
import logging
from functools import partial
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.models.layers import to_2tuple

from lib.models.layers.patch_embed import PatchEmbed
from .utils import combine_tokens, recover_tokens
from .vit import VisionTransformer
from lib.models.caformer.block import Block,CMABlock

from lib.utils.misc import NestedTensor

from .position_encoding import build_position_encoding

_logger = logging.getLogger(__name__)


class VisionTransformer_CMA(VisionTransformer):
    """ Vision Transformer with candidate elimination (CE) module

    A PyTorch impl of : `An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale`
        - https://arxiv.org/abs/2010.11929

    Includes distillation token & head support for `DeiT: Data-efficient Image Transformers`
        - https://arxiv.org/abs/2012.12877
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=True, representation_size=None, distilled=False,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., embed_layer=PatchEmbed, norm_layer=None,
                 act_layer=None, weight_init='', 
                 caia_loc=None):
        """
        Args:
            img_size (int, tuple): input image size
            patch_size (int, tuple): patch size
            in_chans (int): number of input channels
            num_classes (int): number of classes for classification head
            embed_dim (int): embedding dimension
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            representation_size (Optional[int]): enable and set representation layer (pre-logits) to this value if set
            distilled (bool): model includes a distillation token and head as in DeiT models
            drop_rate (float): dropout rate
            attn_drop_rate (float): attention dropout rate
            drop_path_rate (float): stochastic depth rate
            embed_layer (nn.Module): patch embedding layer
            norm_layer: (nn.Module): normalization layer
            weight_init: (str): weight init scheme
        """
        # super().__init__()
        super().__init__()
        if isinstance(img_size, tuple):
            self.img_size = img_size
        else:
            self.img_size = to_2tuple(img_size)
        self.patch_size = patch_size
        self.in_chans = in_chans

        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.num_tokens = 2 if distilled else 1
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU

        self.patch_embed = embed_layer(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.dist_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if distilled else None
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        blocks = []
        feat_size=256
        for i in range(depth):

            if caia_loc[i]:
                blocks.append(
                    CMABlock(
                        dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop_rate,
                        attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, act_layer=act_layer,
                        layer_num=i, feat_size=feat_size+64
                        )
                    )
            else:
                blocks.append(
                    Block(
                        dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop_rate,
                        attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, act_layer=act_layer,
                        layer_num=i
                        )
                    )

        self.blocks = nn.Sequential(*blocks)
        self.norm = norm_layer(embed_dim)

        # 不能学习的位置编码
        self.match_dim = 64
        self.pos_emb = build_position_encoding(self.match_dim)

        self.init_weights(weight_init)

    def forward_features(self, z_rgb, z_tir, x_rgb, x_tir, mask_z=None, mask_x=None):

        B, H, W = x_rgb.shape[0], x_rgb.shape[2], x_rgb.shape[3]


        x_rgb = self.patch_embed(x_rgb)
        z_rgb = self.patch_embed(z_rgb)
        x_tir = self.patch_embed(x_tir)
        z_tir = self.patch_embed(z_tir)


        # 创建位置编码【和ostrack的不同，这个是固定的】
        N = math.ceil(x_rgb.shape[1]**0.5)      # 向上取整
        mask = torch.zeros([B,N,N], dtype=torch.bool).cuda()
        pos_emb = self.pos_emb(NestedTensor(x_rgb, mask))   # B,C,H,W
        pos_emb = pos_emb.flatten(2).transpose(-1,-2)  # BxHWxC
        
        N = math.ceil(z_rgb.shape[1]**0.5)
        mask = torch.zeros([B,N,N], dtype=torch.bool).cuda()
        pos_emb_z = self.pos_emb(NestedTensor(z_rgb, mask))   # B,C,H,W
        pos_emb_z = pos_emb_z.flatten(2).transpose(-1,-2)  # BxHWxC


        # attention mask handling
        # B, H, W
        if mask_z is not None and mask_x is not None:
            # 对最后一个维度进行下采样
            mask_z = F.interpolate(mask_z[None].float(), scale_factor=1. / self.patch_size).to(torch.bool)[0]
            mask_z = mask_z.flatten(1).unsqueeze(-1)

            mask_x = F.interpolate(mask_x[None].float(), scale_factor=1. / self.patch_size).to(torch.bool)[0]
            mask_x = mask_x.flatten(1).unsqueeze(-1)

            mask_x = combine_tokens(mask_z, mask_x, mode=self.cat_mode)
            mask_x = mask_x.squeeze(-1)

        z_rgb += self.pos_embed_z
        x_rgb += self.pos_embed_x
        z_tir += self.pos_embed_z
        x_tir += self.pos_embed_x

        x_rgb = combine_tokens(z_rgb, x_rgb, mode=self.cat_mode)
        x_tir = combine_tokens(z_tir, x_tir, mode=self.cat_mode)


        x_rgb = self.pos_drop(x_rgb)
        x_tir = self.pos_drop(x_tir)
        
        lens_x = self.pos_embed_x.shape[1]
        

        ################# main stream
        for i, blk in enumerate(self.blocks):
            x_rgb, x_tir, attn_rgb, attn_tir = \
                blk(x_rgb, x_tir, mask_x, pos_emb, pos_emb_z)
            


        x_rgb = self.norm(x_rgb)
        x_tir = self.norm(x_tir)

        aux_dict_rgb = {
            "attn": attn_rgb,
        }

        aux_dict_tir = {
            "attn": attn_tir,
        }

        return x_rgb, x_tir, aux_dict_rgb, aux_dict_tir


    def forward(self, z_rgb,z_tir, x_rgb,x_tir, ce_template_mask=None, ce_keep_rate=None, return_last_attn=False):

        return self.forward_features(z_rgb=z_rgb, z_tir=z_tir, x_rgb=x_rgb, x_tir=x_tir)




def _create_vision_transformer(pretrained=False, **kwargs):
    model = VisionTransformer_CMA(**kwargs)

    if pretrained:
        if 'npz' in pretrained:
            model.load_pretrained(pretrained, prefix='')
        else:
            checkpoint = torch.load(pretrained, map_location="cpu")
            
            try:
                missing_keys, unexpected_keys = model.load_state_dict(checkpoint["net"], strict=False)
            except:
                try:
                    missing_keys, unexpected_keys = model.load_state_dict(checkpoint, strict=False)
                except:
                    missing_keys, unexpected_keys = model.load_state_dict(checkpoint["model"], strict=False)
            print('Load pretrained model from: ' + pretrained)

    return model


def vit_base_patch16_224_cma(pretrained=False, **kwargs):
    """ ViT-Base model (ViT-B/16) from original paper (https://arxiv.org/abs/2010.11929).
    """
    model_kwargs = dict(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model = _create_vision_transformer(pretrained=pretrained, **model_kwargs)
    return model


def vit_large_patch16_224_ce(pretrained=False, **kwargs):
    """ ViT-Large model (ViT-L/16) from original paper (https://arxiv.org/abs/2010.11929).
    """
    model_kwargs = dict(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16, **kwargs)
    model = _create_vision_transformer(pretrained=pretrained, **model_kwargs)
    return model
