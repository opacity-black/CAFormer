import torch.nn as nn
import torch
from torch.nn.init import xavier_normal_, constant_, xavier_uniform_



class InnerAttention(nn.Module):
    def __init__(self, dim, nhead, v_dim=None, identical_connection=True):
        """
        将注意力权重作为特征
        """
        super().__init__()
        v_dim = dim if v_dim==None else v_dim
        self.q_linear = nn.Linear(dim, dim)
        self.k_linear = nn.Linear(dim, dim)
        self.out_linear = nn.Linear(v_dim, v_dim)
        self.num_heads = nhead
        head_dim = dim // nhead
        self.scale = head_dim ** -0.5
        self.id_connect = identical_connection
        self.init_params()

    def init_params(self):
        for n, m in self.named_modules():
            if 'linear' in n:
                xavier_uniform_(m.weight)
                constant_(m.bias, 0.)

    def forward(self, query, key, value, return_attn=False):
        B,N1,C = query.shape
        B,N2,C = key.shape
        B,N2,vC = value.shape
        query = self.q_linear(query).reshape(B, N1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        key = self.k_linear(key).reshape(B, N2, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        value = value.reshape(B, N2, self.num_heads, vC // self.num_heads).permute(0, 2, 1, 3)

        attn = query @ key.transpose(-1,-2) * self.scale
        attn = attn.softmax(dim=-1)
        value = (attn @ value).transpose(1, 2).reshape(B, -1, vC)

        if self.id_connect:
            value = self.out_linear(value) + value
        else:
            value = self.out_linear(value)

        if return_attn:
            return value, attn
        else:
            return value




class Attention(nn.Module):
    def __init__(self, dim, num_heads=12, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.q_linear = nn.Linear(dim, dim, bias=qkv_bias)
        self.k_linear = nn.Linear(dim, dim, bias=qkv_bias)
        self.v_linear = nn.Linear(dim, dim, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)


    def get_corrmap(self, q, k, mask):
        """
        输出原始的QK权重，未softmax
        """
        # q: B, N, C
        B, N1, C = q.shape
        B, N2, C = k.shape
        q = self.q_linear(q).reshape(B, N1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3) # B,hn,N,C/hn
        k = self.k_linear(k).reshape(B, N2, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale

        if mask is not None:
            attn = attn.masked_fill(mask.unsqueeze(1).unsqueeze(2), float('-inf'),)
        return attn



    def get_attn_x(self, attn, v, return_attention=False, need_softmax=True):
        """
        根据attn权重，组织value
        """
        if need_softmax:        
            attn = attn.softmax(dim=-1)
            
        B,N,C = v.shape
        v = self.v_linear(v).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, -1, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        if return_attention:
            return x, attn
        else:
            return x


    def general_forward(self, query, key, value, mask, return_attention=False):
        # q: B, N, C
        B, N1, C = query.shape
        B, N2, C = key.shape
        q = self.q_linear(query).reshape(B, N1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3) # B,hn,N,C/hn
        k = self.k_linear(key).reshape(B, N2, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = self.v_linear(value).reshape(B, N2, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale

        if mask is not None:
            attn = attn.masked_fill(mask.unsqueeze(1).unsqueeze(2), float('-inf'),)

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N1, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        if return_attention:
            return x, attn
        else:
            return x
        

    def forward(self, q, k=None, v=None, mask=None, return_attention=False):
        k = q if k==None else k
        v = q if v==None else v
        return self.general_forward(q,k,v,mask, return_attention)



class CrossModulatedAttention(nn.Module):
    
    def __init__(self, dropout, match_dim, feat_size, v_dim, area:str="ST"):
        super(CrossModulatedAttention, self).__init__()
        assert area in ["ST", "TS", "TT", "SS", "S"]
        self.match_dim = match_dim
        self.feat_size = feat_size
        self.v_dim = v_dim
        self.corr_proj = nn.Linear(self.feat_size, self.match_dim)
        # self.corr_attn = InnerAttention(self.match_dim, 1, dropout=dropout, vdim=self.v_dim, batch_first=True)
        self.corr_attn = InnerAttention(self.match_dim, 1, v_dim=self.v_dim)
        self.feat_norm0 = nn.LayerNorm(self.feat_size)
        self.feat_norm1 = nn.LayerNorm(self.match_dim)
        # self.feat_norm2 = nn.LayerNorm(self.v_dim)
        self.dropout = nn.Dropout(dropout)
        self.init_params()
        self.area = area


    def init_params(self):
        for n, m in self.named_modules():
            if 'proj' in n:
                xavier_uniform_(m.weight)
                constant_(m.bias, 0.)


    def forward(self, corr_map_rgb, corr_map_tir, pos_emb):

        N=64

        B,hn,N1,N2 = corr_map_rgb.shape
        corr_map_rgb = corr_map_rgb.reshape(B*hn, N1, N2)
        corr_map_tir = corr_map_tir.reshape(B*hn, N1, N2)

        pos_emb = pos_emb.repeat(hn,1,1,1).transpose(0,1).reshape(B*hn, N1, self.match_dim)   # 每个batch_size的pos_emb不一样

        q = k = torch.cat([self.feat_norm1(self.corr_proj(self.feat_norm0(corr_map_rgb.clone()))) + pos_emb, 
                           self.feat_norm1(self.corr_proj(self.feat_norm0(corr_map_tir.clone()))) + pos_emb], dim=-2)

        if self.area in ["TT", "ST"]:
            corr_map = self.dropout( self.corr_attn(q, k, value = \
                                torch.cat([corr_map_rgb[..., :N], corr_map_tir[..., :N]], dim=-2), return_attn=True)[0])
            attn_rgb = corr_map[:,:N1,:].reshape(B,hn,N1,N)
            attn_tir = corr_map[:,N1:,:].reshape(B,hn,N1,N)
            
        elif self.area in ["SS", "TS"]:
            corr_map = self.dropout( self.corr_attn(q, k, value = \
                                torch.cat([corr_map_rgb[..., N:], corr_map_tir[..., N:]], dim=-2), return_attn=True)[0])
            attn_rgb = corr_map[:,:N1,:].reshape(B,hn,N1,N2-64)
            attn_tir = corr_map[:,N1:,:].reshape(B,hn,N1,N2-64)

        elif self.area in ["S"]:
            corr_map = self.dropout( self.corr_attn(q, k, value = \
                                torch.cat([corr_map_rgb, corr_map_tir], dim=-2), return_attn=True)[0])
            attn_rgb = corr_map[:,:N1,:].reshape(B,hn,N1,N2)
            attn_tir = corr_map[:,N1:,:].reshape(B,hn,N1,N2)

        return attn_rgb, attn_tir
