"""
这里放注意力融合的不同方式
"""

import torch
import torch.nn as nn
from ..caiatrack.position_encoding import build_position_encoding
from torch.nn.init import xavier_uniform_, constant_



class add_attn(nn.Module):
    """
    主要是用来验证的
    """
    def __init__(self):
        super().__init__()
        # self.scale_rgb = nn.Parameter(torch.tensor(0.))
        # self.scale_tir = nn.Parameter(torch.tensor(1.))

    def forward(self, attn_rgb, attn_tir, *args):
        # attn_rgb/attn_tir: B,hn,N,N

        # 固定权重
        attn_rgb_f = attn_tir
        attn_tir_f = attn_rgb

        # 自己学习权重
        # attn_rgb_f = attn_tir.clone()*self.scale_rgb
        # attn_tir_f = attn_rgb.clone()*self.scale_tir

        # 只改变纵向分布，不变横向分布，约束性不如直接相加
        # attn_rgb_f = 0
        # attn_tir_f = attn_rgb.mean(-1).unsqueeze(-1).repeat(1,1,1,64)
        # 只改变横向分布，不变纵向分布，约束性较差
        # N=attn_rgb.shape[2]
        # attn_rgb_f = 0
        # attn_tir_f = attn_rgb.mean(-2).unsqueeze(-2).repeat(1,1,N,1)

        return attn_rgb_f, attn_tir_f
    


class cross_attn_fusion(nn.Module):
    def __init__(self, dim, nhead=1):
        """
        针对ST区域的交互
        """
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.q_linear = nn.Linear(dim, dim)
        self.k_linear = nn.Linear(dim, dim)
        self.out_linear = nn.Linear(dim,dim)
        self.num_heads = nhead
        head_dim = dim // nhead
        self.scale = head_dim ** -0.5
        self.init_params()
        self.attn = [0,0]

    def init_params(self):
        for n, m in self.named_modules():
            if 'linear' in n:
                xavier_uniform_(m.weight)
                constant_(m.bias, 0.)


    def crossattn(self, query, key, value, return_attn=False):
        B,N1,C = query.shape
        B,N2,C = key.shape
        query = self.q_linear(query).reshape(B, N1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        key = self.k_linear(key).reshape(B, N2, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        value = value.reshape(B, N2, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        attn = query @ key.transpose(-1,-2) * self.scale
        attn = attn.softmax(dim=-1)
        value = (attn @ value).transpose(1, 2).reshape(B, -1, C)

        if not self.training:
            self.attn.append(attn)
            del self.attn[0]

        value = self.out_linear(value)

        if return_attn:
            return value, attn
        else:
            return value


    def forward(self, attn_rgb, attn_tir, pos_emb):
        # attn_rgb/attn_tir: B,hn,N,C
        # pos_emb: B,HW,C

        B,hn,N1,N2 = attn_rgb.shape
        attn_rgb = attn_rgb.reshape(B*hn, N1, N2)
        attn_tir = attn_tir.reshape(B*hn, N1, N2)
        pos_emb = pos_emb.repeat(hn,1,1,1).transpose(0,1).reshape(B*hn, N1, N2)   # 每个batch_size的pos_emb不一样

        # layer norm and add pos_emb
        attn_rgb_norm = self.norm1(attn_rgb.clone()) + pos_emb
        attn_tir_norm = self.norm1(attn_tir.clone()) + pos_emb

        # cross attention
        attn_rgb = self.crossattn(query = attn_rgb_norm, 
                                  key = torch.cat([attn_tir_norm, attn_rgb_norm], dim=-2), 
                                  value = torch.cat([attn_tir, attn_rgb], dim=-2))
        
        attn_tir = self.crossattn(query = attn_tir_norm, 
                                  key = torch.cat([attn_tir_norm, attn_rgb_norm], dim=-2), 
                                  value = torch.cat([attn_tir, attn_rgb], dim=-2))

        attn_rgb = attn_rgb.reshape(B,hn,N1,N2)
        attn_tir = attn_tir.reshape(B,hn,N1,N2)

        return attn_rgb, attn_tir
    



class cross_attn_fusion_(nn.Module):
    def __init__(self, dim, nhead=1, enable_modal_embed=False):
        """
        针对ST区域的交互
        """
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.q_linear = nn.Linear(dim, dim)
        self.k_linear = nn.Linear(dim, dim)
        self.out_linear = nn.Linear(dim,dim)
        self.inner_attn = nn.Identity()
        self.num_heads = nhead
        head_dim = dim // nhead
        self.scale = head_dim ** -0.5
        self.init_params()
        self.attn = [0,0]
        # 2023年5月22日，添加了模态嵌入
        if enable_modal_embed:
            self.modal_embed_rgb = nn.Parameter(torch.zeros(dim))
            self.modal_embed_tir = nn.Parameter(torch.zeros(dim))
        else:
            self.modal_embed_rgb = 0
            self.modal_embed_tir = 0

    def init_params(self):
        for n, m in self.named_modules():
            if 'linear' in n:
                xavier_uniform_(m.weight)
                constant_(m.bias, 0.)


    def crossattn(self, query, key, value, return_attn=False):
        B,N1,C = query.shape
        B,N2,C = key.shape
        query = self.q_linear(query).reshape(B, N1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        key = self.k_linear(key).reshape(B, N2, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        value = value.reshape(B, N2, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        attn = query @ key.transpose(-1,-2) * self.scale
        attn = attn.softmax(dim=-1)
        value = (attn @ value).transpose(1, 2).reshape(B, -1, C)

        # if not self.training:
        #     self.attn.append(attn)
        #     del self.attn[0]
        self.inner_attn([None, attn])

        value = self.out_linear(value)

        if return_attn:
            return value, attn
        else:
            return value


    def forward(self, attn_rgb, attn_tir, pos_emb):
        # attn_rgb/attn_tir: B,hn,N,C
        # pos_emb: B,HW,C

        B,hn,N1,N2 = attn_rgb.shape
        attn_rgb = attn_rgb.reshape(B*hn, N1, N2)
        attn_tir = attn_tir.reshape(B*hn, N1, N2)
        pos_emb = pos_emb.repeat(hn,1,1,1).transpose(0,1).reshape(B*hn, N1, N2)   # 每个batch_size的pos_emb不一样

        # layer norm and add pos_emb
        attn_rgb_norm = self.norm1(attn_rgb.clone()) + pos_emb + self.modal_embed_rgb  # 这里必须加clone，否则反向传播时会报错
        attn_tir_norm = self.norm1(attn_tir.clone()) + pos_emb + self.modal_embed_tir

        # cross attention
        attn_rgb = self.crossattn(query = attn_rgb_norm, 
                                  key = torch.cat([attn_rgb_norm, attn_tir_norm], dim=-2), 
                                  value = torch.cat([attn_rgb, attn_tir], dim=-2))
        
        attn_tir = self.crossattn(query = attn_tir_norm, 
                                  key = torch.cat([attn_rgb_norm, attn_tir_norm], dim=-2), 
                                  value = torch.cat([attn_rgb, attn_tir], dim=-2))

        attn_rgb = attn_rgb.reshape(B,hn,N1,N2)
        attn_tir = attn_tir.reshape(B,hn,N1,N2)

        return attn_rgb, attn_tir
    



class CAiA_v0(nn.Module):
    """
    针对ST区域的交互，调整其横向分布
    """
    def __init__(self, dim, nhead=1, enable_modal_embed=False):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.q_linear = nn.Linear(dim, dim)
        self.k_linear = nn.Linear(dim, dim)
        self.out_linear = nn.Linear(dim,dim)
        self.inner_attn = nn.Identity()
        self.num_heads = nhead
        head_dim = dim // nhead
        self.scale = head_dim ** -0.5
        self.init_params()
        self.attn = [0,0]
        # 2023年5月22日，添加了模态嵌入
        if enable_modal_embed:
            self.modal_embed_rgb = nn.Parameter(torch.zeros(dim))
            self.modal_embed_tir = nn.Parameter(torch.zeros(dim))
        else:
            self.modal_embed_rgb = 0
            self.modal_embed_tir = 0

    def init_params(self):
        for n, m in self.named_modules():
            if 'linear' in n:
                xavier_uniform_(m.weight)
                constant_(m.bias, 0.)


    def crossattn(self, query, key, value, return_attn=False):
        B,N1,C = query.shape
        B,N2,C = key.shape
        query = self.q_linear(query).reshape(B, N1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        key = self.k_linear(key).reshape(B, N2, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        value = value.reshape(B, N2, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        attn = query @ key.transpose(-1,-2) * self.scale
        attn = attn.softmax(dim=-1)
        value = (attn @ value).transpose(1, 2).reshape(B, -1, C)

        # if not self.training:
        #     self.attn.append(attn)
        #     del self.attn[0]
        self.inner_attn([None, attn])

        value = self.out_linear(value)

        if return_attn:
            return value, attn
        else:
            return value


    def forward(self, attn_rgb, attn_tir, pos_emb):
        # 输入的是softmax之后的注意力权重
        # attn_rgb/attn_tir: B,hn,N,C
        # pos_emb: B,HW,C

        B,hn,N1,N2 = attn_rgb.shape
        attn_rgb = attn_rgb.reshape(B*hn, N1, N2)
        attn_tir = attn_tir.reshape(B*hn, N1, N2)
        pos_emb = pos_emb.repeat(hn,1,1,1).transpose(0,1).reshape(B*hn, N1, N2)   # 每个batch_size的pos_emb不一样

        # layer norm and add pos_emb
        attn_rgb_norm = self.norm1(attn_rgb.clone()) + pos_emb + self.modal_embed_rgb  # 这里必须加clone，否则反向传播时会报错
        attn_tir_norm = self.norm1(attn_tir.clone()) + pos_emb + self.modal_embed_tir

        # cross attention
        attn_rgb = self.crossattn(query = attn_rgb_norm, 
                                  key = torch.cat([attn_rgb_norm, attn_tir_norm], dim=-2), 
                                  value = torch.cat([attn_rgb, attn_tir], dim=-2))
        
        attn_tir = self.crossattn(query = attn_tir_norm, 
                                  key = torch.cat([attn_rgb_norm, attn_tir_norm], dim=-2), 
                                  value = torch.cat([attn_rgb, attn_tir], dim=-2))

        attn_rgb = attn_rgb.reshape(B,hn,N1,N2)
        attn_tir = attn_tir.reshape(B,hn,N1,N2)

        return attn_rgb, attn_tir
    
    

class MLP_two(nn.Module):
    """
    针对ST区域的交互，调整其相对大小，每个模态不一样
    """
    def __init__(self, dim=256):
        # dim: All search region token num
        super().__init__()
        # 行注意力
        self.col_attn = nn.Sequential(
            nn.LayerNorm(dim*2),
            nn.Linear(dim*2, dim),
            nn.ReLU(),
            nn.Linear(dim, dim*2),
            nn.Sigmoid(),
        )
        self.dim = dim
        self.init_params()

    def init_params(self):
        for n, m in self.named_modules():
            if isinstance(m, nn.Linear):         # v2, v2_3, v3
                xavier_uniform_(m.weight)
                constant_(m.bias, 0.)


    def forward(self, attn_rgb_weight, attn_tir_weight, global_index_s):
        # 输入的是softmax之后的注意力权重的ST强度
        # 输出是调整之后的强度
        # attn_rgb_weight/attn_tir_weight: B,hn,H
        # global_index_s: B,N

        B,hn,N1 = attn_rgb_weight.shape
        idx = global_index_s.unsqueeze(-2).repeat(1,hn,1)
        # 生成行向量权重
        vex_col_rgb = torch.zeros([B,hn,self.dim]).cuda()
        vex_col_rgb.scatter_(dim=-1, index=idx, src=attn_rgb_weight.clone())
        vex_col_tir = torch.zeros([B,hn,self.dim]).cuda()
        vex_col_tir.scatter_(dim=-1, index=idx, src=attn_tir_weight.clone())
        vex_col = torch.cat([vex_col_rgb, vex_col_tir], -1)
        vex_col = self.col_attn(vex_col)    # B,hn,512

        vex_col_rgb = vex_col[:,:,:256].gather(dim=-1, index=idx).reshape(B,hn,N1)
        vex_col_tir = vex_col[:,:,256:].gather(dim=-1, index=idx).reshape(B,hn,N1)
        
        # if self.training:
        #     self.weight = vex_row.mean() + vex_col.mean()/(global_index_s.shape[1]/256)
        
        return vex_col_rgb, vex_col_tir



class MLP_one(nn.Module):
    """
    针对ST区域的交互，调整其相对大小，两个模态共享强度
    """
    def __init__(self, dim=256):
        # dim: All search region token num
        super().__init__()
        # 行注意力
        self.col_attn = nn.Sequential(
            nn.LayerNorm(dim*2),
            nn.Linear(dim*2, dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
            nn.Sigmoid(),
        )
        self.dim = dim
        self.init_params()

    def init_params(self):
        for n, m in self.named_modules():
            if isinstance(m, nn.Linear):
                xavier_uniform_(m.weight)
                constant_(m.bias, 0.)


    def forward(self, attn_rgb_weight, attn_tir_weight, global_index_s):
        # 输入的是softmax之后的注意力权重的ST强度
        # 输出是调整之后的强度
        # attn_rgb_weight/attn_tir_weight: B,hn,H
        # global_index_s: B,N

        B,hn,N1 = attn_rgb_weight.shape
        idx = global_index_s.unsqueeze(-2).repeat(1,hn,1)
        # 生成行向量权重
        vex_col_rgb = torch.zeros([B,hn,self.dim]).cuda()
        vex_col_rgb.scatter_(dim=-1, index=idx, src=attn_rgb_weight.clone())
        vex_col_tir = torch.zeros([B,hn,self.dim]).cuda()
        vex_col_tir.scatter_(dim=-1, index=idx, src=attn_tir_weight.clone())
        vex_col = torch.cat([vex_col_rgb, vex_col_tir], -1)
        vex_col = self.col_attn(vex_col)    # B,hn,256

        vex_col = vex_col.gather(dim=-1, index=idx).reshape(B,hn,N1)
        
        # if self.training:
        #     self.weight = vex_row.mean() + vex_col.mean()/(global_index_s.shape[1]/256)
        
        return vex_col




class InnerAttention(nn.Module):
    def __init__(self, dim, nhead, v_dim=None, identical_connection=False, ratio=1):
        """
        将注意力权重作为特征
        """
        super().__init__()
        v_dim = dim if v_dim==None else v_dim
        self.q_linear = nn.Linear(dim, dim*ratio)
        self.k_linear = nn.Linear(dim, dim*ratio)
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
            value = self.out_linear(value)+value
        else:
            value = self.out_linear(value)

        if return_attn:
            return value, attn
        else:
            return value




class ContrAttnAdd(nn.Module):
    def __init__(self, dim, nhead=1):
        """
        哪个贡献值高，就加哪个的注意力权重
        """
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.q_linear = nn.Linear(dim, dim)
        self.k_linear = nn.Linear(dim, dim)
        self.out_linear = nn.Linear(dim,dim)
        self.num_heads = nhead
        head_dim = dim // nhead
        self.scale = head_dim ** -0.5
        self.init_params()
        self.attn = [0,0]

    def init_params(self):
        for n, m in self.named_modules():
            if 'linear' in n:
                xavier_uniform_(m.weight)
                constant_(m.bias, 0.)


    def crossattn(self, query, key, value, return_attn=False):
        B,N1,C = query.shape
        B,N2,C = key.shape
        query = self.q_linear(query).reshape(B, N1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        key = self.k_linear(key).reshape(B, N2, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        value = value.reshape(B, N2, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        attn = query @ key.transpose(-1,-2) * self.scale
        attn = attn.softmax(dim=-1)
        value = (attn @ value).transpose(1, 2).reshape(B, -1, C)

        if not self.training:
            self.attn.append(attn)
            del self.attn[0]

        value = self.out_linear(value)

        if return_attn:
            return value, attn
        else:
            return value


    def forward(self, attn_rgb, attn_tir, pos_emb):
        # attn_rgb/attn_tir: B,hn,N,C
        # pos_emb: B,HW,C

        B,hn,N1,N2 = attn_rgb.shape
        attn_rgb_copy = attn_rgb.reshape(B*hn, N1, N2)
        attn_tir_copy = attn_tir.reshape(B*hn, N1, N2)
        pos_emb = pos_emb.repeat(hn,1,1,1).transpose(0,1).reshape(B*hn, N1, N2)   # 每个batch_size的pos_emb不一样

        # layer norm and add pos_emb
        attn_rgb_norm = self.norm1(attn_rgb_copy) + pos_emb  # 这里必须加clone，否则反向传播时会报错
        attn_tir_norm = self.norm1(attn_tir_copy) + pos_emb

        # cross attention
        self.crossattn(query = attn_rgb_norm, 
                        key = torch.cat([attn_tir_norm, attn_rgb_norm], dim=-2), 
                        value = torch.cat([attn_tir, attn_rgb], dim=-2))
        
        self.crossattn(query = attn_tir_norm, 
                        key = torch.cat([attn_tir_norm, attn_rgb_norm], dim=-2), 
                        value = torch.cat([attn_tir, attn_rgb], dim=-2))
        
        attn_sum = sum(self.attn)
        length = attn_sum.shape[-1]//2
        contr_t = attn_sum[..., :length].sum()/length   # TIR贡献
        contr_r = attn_sum[..., length:].sum()/length   # RGB贡献

        if contr_t>contr_r:
            return attn_tir, attn_tir
        else:
            return attn_rgb, attn_rgb
    


class CAiA_v1(nn.Module):
    def __init__(self, dim, nhead):
        """
        结构上是AiA的v1
        """
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.embed_linear = nn.Linear(dim,dim)
        self.q_linear = nn.Linear(dim, dim)
        self.k_linear = nn.Linear(dim, dim)
        self.out_linear = nn.Linear(dim,dim)
        self.num_heads = nhead
        head_dim = dim // nhead
        self.scale = head_dim ** -0.5
        self.init_params()
        self.attn = [0,0]

    def init_params(self):
        for n, m in self.named_modules():
            if 'linear' in n:
                xavier_uniform_(m.weight)
                constant_(m.bias, 0.)


    def crossattn(self, query, key, value, return_attn=False):
        B,N1,C = query.shape
        B,N2,C = key.shape
        query = self.q_linear(query).reshape(B, N1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        key = self.k_linear(key).reshape(B, N2, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        value = value.reshape(B, N2, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        attn = query @ key.transpose(-1,-2) * self.scale
        attn = attn.softmax(dim=-1)
        value = (attn @ value).transpose(1, 2).reshape(B, -1, C)

        if not self.training:
            self.attn.append(attn)
            del self.attn[0]

        value = self.out_linear(value) + value

        if return_attn:
            return value, attn
        else:
            return value


    def forward(self, attn_rgb, attn_tir, pos_emb):
        # attn_rgb/attn_tir: B,hn,N,C
        # pos_emb: B,HW,C

        B,hn,N1,N2 = attn_rgb.shape
        attn_rgb = attn_rgb.reshape(B*hn, N1, N2)
        attn_tir = attn_tir.reshape(B*hn, N1, N2)
        pos_emb = pos_emb.repeat(hn,1,1,1).transpose(0,1).reshape(B*hn, N1, N2)   # 每个batch_size的pos_emb不一样

        # layer norm and add pos_emb
        attn_rgb_norm = self.norm1(self.embed_linear(attn_rgb.clone())) + pos_emb  # 这里必须加clone，否则反向传播时会报错
        attn_tir_norm = self.norm1(self.embed_linear(attn_tir.clone())) + pos_emb
        attn_rgb_norm2 = self.norm2(attn_rgb.clone())
        attn_tir_norm2 = self.norm2(attn_tir.clone())

        # cross attention
        attn_rgb = self.crossattn(query = attn_rgb_norm, 
                                  key = torch.cat([attn_tir_norm, attn_rgb_norm], dim=-2), 
                                  value = torch.cat([attn_tir_norm2, attn_rgb_norm2], dim=-2))
        
        attn_tir = self.crossattn(query = attn_tir_norm, 
                                  key = torch.cat([attn_tir_norm, attn_rgb_norm], dim=-2), 
                                  value = torch.cat([attn_tir_norm2, attn_rgb_norm2], dim=-2))

        attn_rgb = attn_rgb.reshape(B,hn,N1,N2)
        attn_tir = attn_tir.reshape(B,hn,N1,N2)

        return attn_rgb, attn_tir
    


class CAiA_v3(nn.Module):
    def __init__(self, dim, inner_dim=None):
        """
        LayerNorm可能导致CAiA的退化，换成了BatchNorm
        相当于AiA的v3
        """
        inner_dim = dim if inner_dim==None else inner_dim
        super().__init__()
        # self.norm1 = nn.LayerNorm(inner_dim)
        self.norm1 = nn.BatchNorm2d(12)
        self.norm2 = nn.LayerNorm(dim)
        self.embed_linear = nn.Linear(dim, inner_dim)
        self.inner_attn = InnerAttention(inner_dim, nhead=1, v_dim=dim, identical_connection=False)
        self.v_linear = nn.Linear(dim, dim)
        self.init_params()

    def init_params(self):
        for n, m in self.named_modules():
            if 'linear' in n:
                xavier_uniform_(m.weight)
                constant_(m.bias, 0.)


    def forward(self, attn_rgb, attn_tir, pos_emb):
        # attn_rgb/attn_tir: B,hn,N,C
        # pos_emb: B,HW,C

        B,hn,N1,N2 = attn_rgb.shape
        attn_rgb = attn_rgb.reshape(B*hn, N1, N2)
        attn_tir = attn_tir.reshape(B*hn, N1, N2)

        # layer norm and add pos_emb
        pos_emb = pos_emb.repeat(hn,1,1,1).transpose(0,1).reshape(B*hn, N1, -1)   # 每个batch_size的pos_emb不一样,单个bs里的不同hn一样
        attn_rgb_norm = self.norm1(self.embed_linear(attn_rgb)) + pos_emb
        attn_tir_norm = self.norm1(self.embed_linear(attn_tir)) + pos_emb

        key = torch.cat([attn_rgb_norm, attn_tir_norm], dim=-2)
        value = self.v_linear(self.norm2(torch.cat([attn_rgb, attn_tir], dim=-2)))
        # cross attention
        attn_rgb,_ = self.inner_attn(query = attn_rgb_norm, key = key, value = value, return_attn=True)
        attn_tir,_ = self.inner_attn(query = attn_tir_norm, key = key, value = value, return_attn=True)

        attn_rgb = attn_rgb.reshape(B,hn,N1,N2)
        attn_tir = attn_tir.reshape(B,hn,N1,N2)

        return attn_rgb, attn_tir
    
    
class UniAttnention_v3(nn.Module):
    """
    
    """
    def __init__(self, dim, af_head=1, enable_modal_embed=False, inner_dim=None):
        super().__init__()
        inner_dim = dim if inner_dim==None else inner_dim
        self.norm1 = nn.LayerNorm(inner_dim)
        self.norm2 = nn.LayerNorm(dim)
        self.embed_linear = nn.Linear(dim, inner_dim)
        self.attn = InnerAttention(dim, nhead=af_head, v_dim=dim)
        self.v_linear = nn.Linear(dim, dim)
        # 2023年5月22日，添加了模态嵌入
        if enable_modal_embed:
            self.modal_embed_rgb = nn.Parameter(torch.zeros(dim))
            self.modal_embed_tir = nn.Parameter(torch.zeros(dim))
        else:
            self.modal_embed_rgb = 0
            self.modal_embed_tir = 0
        self.init_params()

    def init_params(self):
        for n, m in self.named_modules():
            if 'linear' in n:
                xavier_uniform_(m.weight)
                constant_(m.bias, 0.)


    def forward(self, attn_rgb, attn_tir, pos_emb):
        # attn_rgb/attn_tir: B,hn,N,C
        # pos_emb: B,HW,C

        B,hn,N1,N2 = attn_rgb.shape
        attn_rgb = attn_rgb.reshape(B*hn, N1, N2)
        attn_tir = attn_tir.reshape(B*hn, N1, N2)

        # layer norm and add pos_emb
        pos_emb = pos_emb.repeat(hn,1,1,1).transpose(0,1).reshape(B*hn, N1, N2)   # 每个batch_size的pos_emb不一样,单个bs里的不同hn一样
        attn_rgb_norm = self.norm1(self.embed_linear(attn_rgb)) + pos_emb+ self.modal_embed_rgb
        attn_tir_norm = self.norm1(self.embed_linear(attn_tir)) + pos_emb+ self.modal_embed_tir

        key = torch.cat([attn_rgb_norm, attn_tir_norm], dim=-2)
        value = self.v_linear(self.norm2(torch.cat([attn_rgb, attn_tir], dim=-2)))
        # cross attention
        attn_out,_ = self.attn(query = attn_rgb_norm+attn_tir_norm, key = key, value = value, return_attn=True)

        # Residual connection
        attn_out = attn_out + (attn_rgb + attn_tir)
        attn_out = attn_out.reshape(B,hn,N1,N2)
        return attn_out


# class UniAttnention_v4(nn.Module):
#     """
#     Decoder
#     """
#     def __init__(self, dim, af_head=1, enable_modal_embed=False):
#         super().__init__()
#         print("Now, you are using `UniAttnention_v4`")
#         self.norm1 = nn.LayerNorm(dim)
#         self.norm2 = nn.LayerNorm(dim)
#         self.embed_linear = nn.Linear(dim, dim)

#         self.cross_attn = InnerAttention(dim, nhead=af_head)
#         self.self_attn = InnerAttention(dim, nhead=af_head)

#         self.v_linear = nn.Linear(dim, dim)
#         # 2023年5月22日，添加了模态嵌入
#         if enable_modal_embed:
#             self.modal_embed_rgb = nn.Parameter(torch.zeros(dim))
#             self.modal_embed_tir = nn.Parameter(torch.zeros(dim))
#         else:
#             self.modal_embed_rgb = 0
#             self.modal_embed_tir = 0
#         self.init_params()
#         self.init_query = None

#     def init_params(self):
#         for n, m in self.named_modules():
#             if 'linear' in n:
#                 xavier_uniform_(m.weight)
#                 constant_(m.bias, 0.)


#     def forward(self, attn_rgb, attn_tir, pos_emb):
#         # attn_rgb/attn_tir: B,hn,N,C
#         # pos_emb: B,HW,C

#         B,hn,N1,N2 = attn_rgb.shape
#         attn_rgb = attn_rgb.reshape(B*hn, N1, N2)
#         attn_tir = attn_tir.reshape(B*hn, N1, N2)

#         # self.init_query = torch.ones([B*hn, N1, N2]).to(attn_rgb.device) if self.init_query==None else self.init_query

#         attn_query = attn_rgb+attn_tir
#         attn_query = self.self_attn(attn_query, attn_query, attn_query) + attn_query

#         # layer norm and add pos_emb
#         pos_emb = pos_emb.repeat(hn,1,1,1).transpose(0,1).reshape(B*hn, N1, N2)   # 每个batch_size的pos_emb不一样,单个bs里的不同hn一样
#         attn_rgb_norm = self.norm1(self.embed_linear(attn_rgb)) + pos_emb+ self.modal_embed_rgb
#         attn_tir_norm = self.norm1(self.embed_linear(attn_tir)) + pos_emb+ self.modal_embed_tir

#         key = torch.cat([attn_rgb_norm, attn_tir_norm], dim=-2)
#         value = self.v_linear(self.norm2(torch.cat([attn_rgb, attn_tir], dim=-2)))
#         # cross attention
#         attn_out,_ = self.cross_attn(query = attn_rgb_norm+attn_tir_norm, key = key, value = value, return_attn=True)

#         # Residual connection
#         # if not self.training:
#         #     # attn_out = attn_out*0.5 + (attn_rgb + attn_tir)*0.5
#         #     attn_out = attn_out + (attn_rgb + attn_tir)
#         attn_out = attn_out.reshape(B,hn,N1,N2)
#         return attn_out



# 效果一般
class SE_attn_fusion(nn.Module):
    def __init__(self, dim=12):
        """
        使用SENet的方式，分别做横向通道注意力和纵向通道注意力
        针对ST区域的交互
        dim是hn
        """
        super().__init__()
        # 行注意力
        self.col_attn = nn.Sequential(
            nn.LayerNorm(256*2),     # v2_1, v4
            nn.Linear(256*2, 256),
            nn.ReLU(),
            nn.Linear(256, 256*2),
            nn.Sigmoid(),         # v2, v2_1, v2_3
            # nn.Tanh()
        )
        # 列注意力
        # self.row_attn = nn.Sequential(
        #     nn.LayerNorm(64*2),     # v2_1, v4
        #     nn.Linear(64*2, 64),
        #     nn.ReLU(),
        #     nn.Linear(64, 64*2),
        #     nn.Sigmoid(),         # v2, v2_1
        #     # nn.Tanh()
        # )
        self.init_params()

    def init_params(self):
        for n, m in self.named_modules():
            if isinstance(m, nn.Linear):         # v2, v2_3, v3
                xavier_uniform_(m.weight)
                constant_(m.bias, 0.)
            # if isinstance(m, nn.Sequential):
            #     xavier_uniform_(m[0].weight)         # v2_1
            #     constant_(m[0].bias, 0.)
            #     constant_(m[2].weight, 0.)
            #     constant_(m[2].bias, 0.)


    def forward(self, attn_rgb, attn_tir, global_index_s):
        # attn_rgb/attn_tir: B,hn,N,C
        # global_index_s: B,N

        B,hn,N1,N2 = attn_rgb.shape
        idx = global_index_s.unsqueeze(-2).repeat(1,hn,1)
        # 生成行向量权重
        vex_col_rgb = torch.zeros([B,hn,256]).cuda()
        # vex_col_rgb.scatter_(dim=-1, index=idx, src=attn_rgb.detach().mean(-1))
        vex_col_rgb.scatter_(dim=-1, index=idx, src=attn_rgb.detach().max(-1).values)  # v3, v4
        vex_col_tir = torch.zeros([B,hn,256]).cuda()
        # vex_col_tir.scatter_(dim=-1, index=idx, src=attn_tir.detach().mean(-1))
        vex_col_tir.scatter_(dim=-1, index=idx, src=attn_tir.detach().max(-1).values)
        vex_col = torch.cat([vex_col_rgb, vex_col_tir], -1)
        vex_col = self.col_attn(vex_col)    # B,hn,512
        vex_col_rgb = vex_col[:,:,:256].gather(dim=-1, index=idx).reshape(B,hn,N1,1)
        vex_col_tir = vex_col[:,:,256:].gather(dim=-1, index=idx).reshape(B,hn,N1,1)
        attn_rgb_col = vex_col_rgb*attn_rgb.clone()
        attn_tir_col = vex_col_tir*attn_tir.clone()
        
        # 生成列向量权重
        # # vex_row = torch.cat([attn_rgb.detach().mean(-2), attn_tir.detach().mean(-2)], -1)
        # vex_row = torch.cat([attn_rgb.detach().max(-2).values, attn_tir.detach().max(-2).values], -1)
        # vex_row = self.row_attn(vex_row)    # B,hn,128
        # vex_row_rgb = vex_row[:,:,:64].reshape(B,hn,1,N2)
        # vex_row_tir = vex_row[:,:,64:].reshape(B,hn,1,N2)
        # attn_rgb_row = vex_row_rgb*attn_rgb.clone()
        # attn_tir_row = vex_row_tir*attn_tir.clone()

        # if self.training:
        #     self.weight = vex_row.mean() + vex_col.mean()/(global_index_s.shape[1]/256)
        
        # return attn_tir_col+attn_tir_row, attn_rgb_row+attn_rgb_col
        return attn_tir_col, attn_rgb_col





class AdaptiveUniAttn(nn.Module):
    def __init__(self, dim=64):
        super().__init__()
        self.inner_attn = nn.MultiheadAttention(embed_dim=dim, num_heads=2)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x_rgb, x_tir, pos_emd):
        B,hn,H,W = x_rgb.shape

        Q = torch.ones_like(x_rgb)
        self.inner_attn
