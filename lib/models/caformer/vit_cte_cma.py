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
from lib.models.caformer.block import CTEBlock,CMA_CTEBlock

from lib.utils.misc import NestedTensor

from .position_encoding import build_position_encoding

_logger = logging.getLogger(__name__)


class VisionTransformer_CTE_CMA(VisionTransformer):
    """ Vision Transformer with candidate elimination (CE) module

    A PyTorch impl of : `An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale`
        - https://arxiv.org/abs/2010.11929

    Includes distillation token & head support for `DeiT: Data-efficient Image Transformers`
        - https://arxiv.org/abs/2012.12877
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=True, representation_size=None, distilled=False,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., embed_layer=PatchEmbed, norm_layer=None,
                 act_layer=None, weight_init='', cte_loc=None, cte_keep_ratio=None,
                 cma_loc=None, cma_area=["ST"]):
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
        super().__init__()
        self.cma_loc = cma_loc
        self.cma_area = cma_area
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
        ce_index = 0
        self.cte_loc = cte_loc
        feat_size=256
        mark = False
        for i in range(depth):
            cte_keep_ratio_i = 1.0
            if mark:
                try:
                    feat_size = int(math.ceil(feat_size*cte_keep_ratio[ce_index]))
                except:
                    feat_size = int(math.ceil(feat_size*cte_keep_ratio[-1]))
                mark=False
            if cte_loc is not None and i in cte_loc:
                cte_keep_ratio_i = cte_keep_ratio[ce_index]
                mark=True
                ce_index += 1

            if cma_loc[i]:
                blocks.append(
                    CMA_CTEBlock(
                        dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop_rate,
                        attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, act_layer=act_layer,
                        keep_ratio_search=cte_keep_ratio_i, layer_num=i, feat_size=feat_size+64, area=cma_area
                        )
                    )
            else:
                blocks.append(
                    CTEBlock(
                        dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop_rate,
                        attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, act_layer=act_layer,
                        keep_ratio_search=cte_keep_ratio_i, layer_num=i
                        )
                    )

        self.blocks = nn.Sequential(*blocks)
        self.norm = norm_layer(embed_dim)

        # 不能学习的位置编码
        self.match_dim = 64
        self.pos_emb = build_position_encoding(self.match_dim)

        self.init_weights(weight_init)

    def forward_features(self, z_rgb, z_tir, x_rgb, x_tir, mask_z=None, mask_x=None,
                         cte_template_mask=None, cte_keep_rate=None,
                         return_last_attn=False, weight=0.5
                         ):
        
        global_index_s_layers = []

        B, H, W = x_rgb.shape[0], x_rgb.shape[2], x_rgb.shape[3]


        x_rgb = self.patch_embed(x_rgb)
        z_rgb = self.patch_embed(z_rgb)
        x_tir = self.patch_embed(x_tir)
        z_tir = self.patch_embed(z_tir)


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
        

        lens_z = self.pos_embed_z.shape[1]
        lens_x = self.pos_embed_x.shape[1]

        ################# 
        global_index_t = torch.linspace(0, lens_z - 1, lens_z).to(x_rgb.device).to(torch.int64)
        global_index_t = global_index_t.repeat(B, 1)

        global_index_s = torch.linspace(0, lens_x - 1, lens_x).to(x_rgb.device).to(torch.int64)
        global_index_s = global_index_s.repeat(B, 1)
        removed_indexes_s = []
        

        pos_emb_input = pos_emb.clone()
        ################# main stream
        attn_rgb_li = []; attn_tir_li = []
        for i, blk in enumerate(self.blocks):
            x_rgb, x_tir, global_index_t, global_index_s, removed_index_s, attn_rgb, attn_tir = \
                blk(x_rgb, x_tir, global_index_t, global_index_s, \
                        mask_x, cte_template_mask, cte_keep_rate, pos_emb_input, pos_emb_z, weight)
            attn_rgb_li.append(attn_rgb); attn_tir_li.append(attn_tir)
            
            if self.cte_loc is not None and i in self.cte_loc:
                pos_emb_input = pos_emb.gather(1, global_index_s.unsqueeze(-1).repeat(1,1,self.match_dim)).clone()
                removed_indexes_s.append(removed_index_s)
                if not self.training:
                    global_index_s_layers.append(global_index_s)


        x_rgb = self.norm(x_rgb)
        x_tir = self.norm(x_tir)

        lens_x_new = global_index_s.shape[1]
        lens_z_new = global_index_t.shape[1]
        z_rgb = x_rgb[:, :lens_z_new]
        x_rgb = x_rgb[:, lens_z_new:]
        z_tir = x_tir[:, :lens_z_new]
        x_tir = x_tir[:, lens_z_new:]


        # 将shape还原
        if removed_indexes_s and removed_indexes_s[0] is not None:
            removed_indexes_cat = torch.cat(removed_indexes_s, dim=1)

            pruned_lens_x = lens_x - lens_x_new
            pad_x = torch.zeros([B, pruned_lens_x, x_rgb.shape[2]], device=x_rgb.device)
            x_rgb = torch.cat([x_rgb, pad_x], dim=1)
            index_all = torch.cat([global_index_s, removed_indexes_cat], dim=1)
            # recover original token order
            C = x_rgb.shape[-1]
            # x = x.gather(1, index_all.unsqueeze(-1).expand(B, -1, C).argsort(1))
            x_rgb = torch.zeros_like(x_rgb).scatter_(dim=1, index=index_all.unsqueeze(-1).expand(B, -1, C).to(torch.int64), src=x_rgb)

        x_rgb = recover_tokens(x_rgb, lens_z_new, lens_x, mode=self.cat_mode)

        # re-concatenate with the template, which may be further used by other modules
        x_rgb = torch.cat([z_rgb, x_rgb], dim=1)

        aux_dict_rgb = {
            "attn": attn_rgb_li,
            "removed_indexes_s": removed_indexes_s,  # used for visualization
            "global_index_s": global_index_s_layers,
            "cma_loc": self.cma_loc,
            "cma_area": self.cma_area
        }


        # 将shape还原
        if removed_indexes_s and removed_indexes_s[0] is not None:
            removed_indexes_cat = torch.cat(removed_indexes_s, dim=1)

            pruned_lens_x = lens_x - lens_x_new
            pad_x = torch.zeros([B, pruned_lens_x, x_tir.shape[2]], device=x_tir.device)
            x_tir = torch.cat([x_tir, pad_x], dim=1)
            index_all = torch.cat([global_index_s, removed_indexes_cat], dim=1)
            # recover original token order
            C = x_tir.shape[-1]
            # x = x.gather(1, index_all.unsqueeze(-1).expand(B, -1, C).argsort(1))
            x_tir = torch.zeros_like(x_tir).scatter_(dim=1, index=index_all.unsqueeze(-1).expand(B, -1, C).to(torch.int64), src=x_tir)

        x_tir = recover_tokens(x_tir, lens_z_new, lens_x, mode=self.cat_mode)

        # re-concatenate with the template, which may be further used by other modules
        x_tir = torch.cat([z_tir, x_tir], dim=1)

        aux_dict_tir = {
            "attn": attn_tir_li,
            "removed_indexes_s": removed_indexes_s,  # used for visualization
            "global_index_s": global_index_s_layers,
            "cma_loc": self.cma_loc,
            "cma_area": self.cma_area
        }

        return x_rgb, x_tir, aux_dict_rgb, aux_dict_tir


    def forward(self, z_rgb,z_tir, x_rgb,x_tir, cte_template_mask=None, cte_keep_rate=None, return_last_attn=False, weight=0.5):

        return self.forward_features(z_rgb=z_rgb, z_tir=z_tir,
                                     x_rgb=x_rgb, x_tir=x_tir,
                                     cte_template_mask=cte_template_mask, cte_keep_rate=cte_keep_rate, weight=weight)




def _create_vision_transformer(pretrained=False, **kwargs):
    model = VisionTransformer_CTE_CMA(**kwargs)

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


def vit_base_patch16_224_cte_cma(pretrained=False, **kwargs):
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
