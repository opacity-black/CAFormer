import torch.nn as nn
from timm.models.layers import Mlp, DropPath, trunc_normal_, lecun_normal_
from lib.models.layers.attn_blocks import candidate_elimination
from .attention import Attention, CrossModulatedAttention
# from torch.nn.functional import *
import torch
import torch.nn.functional as F



class CTEBlock(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, keep_ratio_search=1.0,
                layer_num=None):
        super().__init__()

        self.layer_num = layer_num
        self.area = None
        
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)

        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)

        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.keep_ratio_search = keep_ratio_search


    def keep_token(self, x, attn, global_index_template=None, global_index_search=None, \
                   cte_template_mask=None, keep_ratio_search=None):
        
        lens_t = global_index_template.shape[1]
        removed_index_search = None
        if self.keep_ratio_search < 1 and (keep_ratio_search is None or keep_ratio_search < 1):
            keep_ratio_search = self.keep_ratio_search if keep_ratio_search is None else keep_ratio_search
            x, global_index_search, removed_index_search = \
                candidate_elimination(attn, x, lens_t, keep_ratio_search, global_index_search, cte_template_mask)

        return x, global_index_template, global_index_search, removed_index_search


    def forward(self, x_rgb, x_tir, global_index_t, global_index_s, mask, 
                cte_template_mask, keep_ratio_search, *args, **args_dict):
        
        x_rgb_attn, corrmap_rgb = self.attn(self.norm1(x_rgb), mask=mask, return_attention=True)
        x_tir_attn, corrmap_tir = self.attn(self.norm1(x_tir), mask=mask, return_attention=True)

        x_rgb = x_rgb + self.drop_path(x_rgb_attn)
        x_tir = x_tir + self.drop_path(x_tir_attn)

        attn_all = corrmap_rgb+corrmap_tir
        x_rgb, _, _, _ = \
            self.keep_token(x_rgb, attn_all, global_index_t, global_index_s, cte_template_mask, keep_ratio_search)
        x_tir, global_index_t, global_index_s, removed_index_s = \
            self.keep_token(x_tir, attn_all, global_index_t, global_index_s, cte_template_mask, keep_ratio_search)

        x_rgb = x_rgb + self.drop_path(self.mlp(self.norm2(x_rgb)))
        x_tir = x_tir + self.drop_path(self.mlp(self.norm2(x_tir)))

        return x_rgb, x_tir, global_index_t, global_index_s, removed_index_s, corrmap_rgb, corrmap_tir






class CMA_CTEBlock(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, keep_ratio_search=1.0,
                layer_num=None,  match_dim=64, feat_size=320, area=["ST"]):
        super().__init__()

        self.layer_num = layer_num
        self.area = area
        
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        # CMA module
        if 'S' in self.area:   # area 3
            self.inner_attn_s = CrossModulatedAttention(dropout=drop, match_dim=match_dim, feat_size=feat_size, v_dim=feat_size, area="S")
        if 'ST' in self.area:   # area 3
            self.inner_attn_st = CrossModulatedAttention(dropout=drop, match_dim=match_dim, feat_size=feat_size, v_dim=64, area="ST")
        if 'TT' in self.area:   # area 1
            self.inner_attn_tt = CrossModulatedAttention(dropout=drop, match_dim=match_dim, feat_size=feat_size, v_dim=64, area="TT")
        if 'SS' in self.area:   # area 4
            self.inner_attn_ss = CrossModulatedAttention(dropout=drop, match_dim=match_dim, feat_size=feat_size, v_dim=feat_size-64, area="SS")
        if 'TS' in self.area:   # area 2
            self.inner_attn_ts = CrossModulatedAttention(dropout=drop, match_dim=match_dim, feat_size=feat_size, v_dim=feat_size-64, area="TS")

        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)

        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.keep_ratio_search = keep_ratio_search
        
        self.attn_map = {'RGB':[0,0,0], 'TIR':[0,0,0]}


    def keep_token(self, x, attn, global_index_template=None, global_index_search=None, \
                   cte_template_mask=None, keep_ratio_search=None):
        
        lens_t = global_index_template.shape[1]
        removed_index_search = None
        if self.keep_ratio_search < 1 and (keep_ratio_search is None or keep_ratio_search < 1):
            keep_ratio_search = self.keep_ratio_search if keep_ratio_search is None else keep_ratio_search
            x, global_index_search, removed_index_search = \
                candidate_elimination(attn, x, lens_t, keep_ratio_search, global_index_search, cte_template_mask)

        return x, global_index_template, global_index_search, removed_index_search


    def areaof(self, x, area):
        if area=='t':
            return x[..., :64, :]
        if area=='s':
            return x[..., 64:, :]
        if area=="st":
            return x[..., 64:, :64]
        if area=="ss":
            return x[..., 64:, 64:]
        if area=="ts":
            return x[..., :64, 64:]
        if area=="tt":
            return x[..., :64, :64]


    def forward(self, x_rgb, x_tir, global_index_t, global_index_s, mask, 
                cte_template_mask, keep_ratio_search, pos_emb, pos_emb_z, weight=0.5):
        

        x_rgb_norm = self.norm1(x_rgb)
        corrmap_rgb = self.attn.get_corrmap(x_rgb_norm, x_rgb_norm, mask=mask)
        x_tir_norm = self.norm1(x_tir)
        corrmap_tir = self.attn.get_corrmap(x_tir_norm, x_tir_norm, mask=mask)

        # if not self.training:         # For visualization
            # self.attn_map['RGB'][0] = corrmap_rgb.clone().detach().softmax(-1).sum(1).squeeze()
            # self.attn_map['TIR'][0] = corrmap_tir.clone().detach().softmax(-1).sum(1).squeeze()

        if "ST" in self.area:       # this is selected
            corrmap_rgb_st_f, corrmap_tir_st_f = self.inner_attn_st(
                self.areaof(corrmap_rgb, 's'), self.areaof(corrmap_tir, 's'), pos_emb)
            corrmap_rgb_st, corrmap_tir_st = self.areaof(corrmap_rgb, 'st'), self.areaof(corrmap_tir, 'st')
            corrmap_rgb_st += corrmap_rgb_st_f
            corrmap_tir_st += corrmap_tir_st_f
        if "SS" in self.area:
            corrmap_rgb_ss_f, corrmap_tir_ss_f = self.inner_attn_ss(
                self.areaof(corrmap_rgb, 's'), self.areaof(corrmap_tir, 's'), pos_emb)
            corrmap_rgb_ss, corrmap_tir_ss = self.areaof(corrmap_rgb, 'ss'), self.areaof(corrmap_tir, 'ss')
            corrmap_rgb_ss += corrmap_rgb_ss_f
            corrmap_tir_ss += corrmap_tir_ss_f
        if "TS" in self.area:
            corrmap_rgb_ts_f, corrmap_tir_ts_f = self.inner_attn_ts(
                self.areaof(corrmap_rgb, 't'), self.areaof(corrmap_tir, 't'), pos_emb_z)
            corrmap_rgb_ts, corrmap_tir_ts = self.areaof(corrmap_rgb, 'ts'), self.areaof(corrmap_tir, 'ts')
            corrmap_rgb_ts += corrmap_rgb_ts_f
            corrmap_tir_ts += corrmap_tir_ts_f
        if "TT" in self.area:
            corrmap_rgb_tt_f, corrmap_tir_tt_f = self.inner_attn_tt(
                self.areaof(corrmap_rgb, 't'), self.areaof(corrmap_tir, 't'), pos_emb_z)
            corrmap_rgb_tt, corrmap_tir_tt = self.areaof(corrmap_rgb, 'tt'), self.areaof(corrmap_tir, 'tt')
            corrmap_rgb_tt += corrmap_rgb_tt_f
            corrmap_tir_tt += corrmap_tir_tt_f
        if "S" in self.area:
            corrmap_rgb_s_f, corrmap_tir_s_f = self.inner_attn_s(
                self.areaof(corrmap_rgb, 's'), self.areaof(corrmap_tir, 's'), pos_emb)
            corrmap_rgb_s, corrmap_tir_s = self.areaof(corrmap_rgb, 's'), self.areaof(corrmap_tir, 's')
            corrmap_rgb_s += corrmap_rgb_s_f
            corrmap_tir_s += corrmap_tir_s_f

        
        # if not self.training:         # 
        #     self.attn_map['RGB'][1] = corrmap_rgb.clone().detach().softmax(-1)
        #     self.attn_map['TIR'][1] = corrmap_tir.clone().detach().softmax(-1)

        # self.attn_st = [self.layer_num, attn_rgb[:,:,64:,:64].clone(), attn_tir[:,:,64:,:64].clone()]

        corrmap_rgb = corrmap_rgb.softmax(dim=-1)
        corrmap_tir = corrmap_tir.softmax(dim=-1)

        #if not self.training:         # 
        #    self.attn_map['RGB'][2] = corrmap_rgb.sum(1).squeeze().clone().detach()
        #    self.attn_map['TIR'][2] = corrmap_tir.sum(1).squeeze().clone().detach()

        x_rgb_attn = self.attn.get_attn_x(corrmap_rgb, x_rgb_norm, need_softmax=False)
        x_tir_attn = self.attn.get_attn_x(corrmap_tir, x_tir_norm, need_softmax=False)

        x_rgb = x_rgb + self.drop_path(x_rgb_attn)
        x_tir = x_tir + self.drop_path(x_tir_attn)
        
        attn_all = corrmap_rgb*weight+corrmap_tir*(1-weight)
        x_rgb, _,_,_ = \
            self.keep_token(x_rgb, attn_all, global_index_t, global_index_s, cte_template_mask, keep_ratio_search)
        x_tir, global_index_t, global_index_s, removed_index_s = \
            self.keep_token(x_tir, attn_all, global_index_t, global_index_s, cte_template_mask, keep_ratio_search)

        x_rgb = x_rgb + self.drop_path(self.mlp(self.norm2(x_rgb)))
        x_tir = x_tir + self.drop_path(self.mlp(self.norm2(x_tir)))

        return x_rgb, x_tir, global_index_t, global_index_s, removed_index_s, corrmap_rgb, corrmap_tir






class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                layer_num=None):
        super().__init__()

        self.layer_num = layer_num
        self.area = None
        
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)

        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)

        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)


    def forward(self, x_rgb, x_tir, mask, pos_emb, pos_emb_z):
        
        x_rgb_attn, corrmap_rgb = self.attn(self.norm1(x_rgb), mask=mask, return_attention=True)
        x_tir_attn, corrmap_tir = self.attn(self.norm1(x_tir), mask=mask, return_attention=True)

        # 残差连接
        x_rgb = x_rgb + self.drop_path(x_rgb_attn)
        x_tir = x_tir + self.drop_path(x_tir_attn)

        # 通道注意力
        x_rgb = x_rgb + self.drop_path(self.mlp(self.norm2(x_rgb)))
        x_tir = x_tir + self.drop_path(self.mlp(self.norm2(x_tir)))

        return x_rgb, x_tir, corrmap_rgb, corrmap_tir






class CMABlock(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                layer_num=None,  match_dim=64, feat_size=320, area=["st"]):
        super().__init__()

        self.layer_num = layer_num
        self.area = area
        
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        # CMA module
        if 'st' in self.area:
            self.inner_attn_st = CrossModulatedAttention(dropout=drop, match_dim=match_dim, feat_size=feat_size, v_dim=64)
        if 'ss' in self.area:
            self.inner_attn_ss = CrossModulatedAttention(dropout=drop, match_dim=match_dim, feat_size=feat_size, v_dim=feat_size-64)

        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)

        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)


    def areaof(self, x, area):
        if area=='t':
            return x[..., :64, :]
        if area=='s':
            return x[..., 64:, :]
        if area=="st":
            return x[..., 64:, :64]
        if area=="ss":
            return x[..., 64:, 64:]



    def forward(self, x_rgb, x_tir, mask, pos_emb, pos_emb_z):
        

        x_rgb_norm = self.norm1(x_rgb)
        corrmap_rgb = self.attn.get_corrmap(x_rgb_norm, x_rgb_norm, mask=mask)
        x_tir_norm = self.norm1(x_tir)
        corrmap_tir = self.attn.get_corrmap(x_tir_norm, x_tir_norm, mask=mask)

        # if not self.training:         
        #     self.attn_map['RGB'][0] = corrmap_rgb.clone().detach().softmax(-1)
        #     self.attn_map['TIR'][0] = corrmap_tir.clone().detach().softmax(-1)

        if "st" in self.area:
            corrmap_rgb_st_f, corrmap_tir_st_f = self.inner_attn_st(
                self.areaof(corrmap_rgb, 's'), self.areaof(corrmap_tir, 's'), pos_emb)
            corrmap_rgb_st, corrmap_tir_st = self.areaof(corrmap_rgb, 'st'), self.areaof(corrmap_tir, 'st')
            corrmap_rgb_st += corrmap_rgb_st_f
            corrmap_tir_st += corrmap_tir_st_f

        
        # if not self.training:         
        #     self.attn_map['RGB'][1] = attn_rgb_f.clone().detach().softmax(-1)
        #     self.attn_map['TIR'][1] = attn_tir_f.clone().detach().softmax(-1)

        corrmap_rgb = corrmap_rgb.softmax(dim=-1)
        corrmap_tir = corrmap_tir.softmax(dim=-1)

        # if not self.training:         
        #     self.attn_map['RGB'][2] = corrmap_rgb.clone().detach()
        #     self.attn_map['TIR'][2] = corrmap_tir.clone().detach()

        x_rgb_attn = self.attn.get_attn_x(corrmap_rgb, x_rgb_norm, need_softmax=False)
        x_tir_attn = self.attn.get_attn_x(corrmap_tir, x_tir_norm, need_softmax=False)

        x_rgb = x_rgb + self.drop_path(x_rgb_attn)
        x_tir = x_tir + self.drop_path(x_tir_attn)
        
        x_rgb = x_rgb + self.drop_path(self.mlp(self.norm2(x_rgb)))
        x_tir = x_tir + self.drop_path(self.mlp(self.norm2(x_tir)))

        return x_rgb, x_tir, corrmap_rgb, corrmap_tir


