"""

"""
import math
import os
from typing import List

import torch
from torch import nn
from torch.nn.modules.transformer import _get_clones

from lib.models.layers.head import build_box_head
from .vit import vit_base_patch16_224
from .vit_cte_cma import vit_base_patch16_224_cte_cma
from .vit_cma import vit_base_patch16_224_cma
from lib.utils.box_ops import box_xyxy_to_cxcywh

import importlib


class CAFormer(nn.Module):

    def __init__(self, transformer, box_head, aux_loss=False, head_type="CORNER"):
        """ Initializes the model.
        Parameters:
            transformer: torch module of the transformer architecture.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
        """
        super().__init__()
        self.backbone = transformer
        self.box_head = box_head

        self.aux_loss = aux_loss
        self.head_type = head_type
        if head_type == "CORNER" or head_type == "CENTER":
            self.feat_sz_s = int(box_head.feat_sz)
            self.feat_len_s = int(box_head.feat_sz ** 2)

        if self.aux_loss:
            self.box_head = _get_clones(self.box_head, 6)


    def forward(self, template: list,
                search: list,
                cte_template_mask=None,
                cte_keep_rate=None,
                return_last_attn=False,
                ):
        # NOTE: x[0] is rgb
        assert len(template)==2 and len(search)==2
        x_rgb, x_tir, aux_dict_rgb, aux_dict_tir = self.backbone(z_rgb=template[0], z_tir=template[1],
                                                                x_rgb=search[0], x_tir=search[1],
                                    cte_template_mask=cte_template_mask,
                                    cte_keep_rate=cte_keep_rate,
                                    return_last_attn=return_last_attn, )

        aux_dict = {'aux_dict_rgb':aux_dict_rgb, 'aux_dict_tir':aux_dict_tir}
        x = torch.cat([x_rgb,x_tir],2)
        # Forward head
        feat_last = x
        if isinstance(x, list):
            feat_last = x[-1]
        out = self.forward_head(feat_last, None)

        out.update(aux_dict)
        out['backbone_feat'] = x
        return out

    def forward_head(self, cat_feature, gt_score_map=None, box_head=None):
        """
        cat_feature: output embeddings of the backbone, it can be (HW1+HW2, B, C) or (HW2, B, C)
        """
        if box_head==None:
            box_head=self.box_head
        enc_opt = cat_feature[:, -self.feat_len_s:]  # encoder output for the search region (B, HW, C)
        opt = (enc_opt.unsqueeze(-1)).permute((0, 3, 2, 1)).contiguous()
        bs, Nq, C, HW = opt.size()
        opt_feat = opt.view(-1, C, self.feat_sz_s, self.feat_sz_s)

        if self.head_type == "CORNER":
            # run the corner head
            pred_box, score_map = box_head(opt_feat, True)
            outputs_coord = box_xyxy_to_cxcywh(pred_box)
            outputs_coord_new = outputs_coord.view(bs, Nq, 4)
            out = {'pred_boxes': outputs_coord_new,
                   'score_map': score_map,
                   }
            return out

        elif self.head_type == "CENTER":
            # run the center head
            score_map_ctr, bbox, size_map, offset_map = box_head(opt_feat, gt_score_map)
            # outputs_coord = box_xyxy_to_cxcywh(bbox)
            outputs_coord = bbox
            outputs_coord_new = outputs_coord.view(bs, Nq, 4)
            out = {'pred_boxes': outputs_coord_new,
                   'score_map': score_map_ctr,
                   'size_map': size_map,
                   'offset_map': offset_map}
            return out
        else:
            raise NotImplementedError


def build_caformer(cfg, training=True):
    pretrained = False


    if cfg.MODEL.BACKBONE.TYPE == 'vit_base_patch16_224_cte_cma':  
        backbone = vit_base_patch16_224_cte_cma(pretrained, drop_path_rate=cfg.TRAIN.DROP_PATH_RATE,
                                           cte_loc=cfg.MODEL.BACKBONE.CTE_LOC,
                                           cte_keep_ratio=cfg.MODEL.BACKBONE.CTE_KEEP_RATIO,
                                           cma_loc=cfg.MODEL.BACKBONE.CMA_LOC,
                                           cma_area=cfg.MODEL.BACKBONE.CMA_AREA,
                                           )
             
        # for RGBT hidden_dim
        hidden_dim = backbone.embed_dim
        patch_start_index = 1

    elif cfg.MODEL.BACKBONE.TYPE == 'vit_base_patch16_224_cma':  
        backbone = vit_base_patch16_224_cma(pretrained, drop_path_rate=cfg.TRAIN.DROP_PATH_RATE,
                                           cma_loc=cfg.MODEL.BACKBONE.CMA_LOC,
                                           )
             
        # for RGBT hidden_dim
        hidden_dim = backbone.embed_dim
        patch_start_index = 1

    else:
        raise NotImplementedError

    backbone.finetune_track(cfg=cfg, patch_start_index=patch_start_index)

    box_head = build_box_head(cfg, hidden_dim*2)

    model = CAFormer(
        backbone,
        box_head,
        aux_loss=False,
        head_type=cfg.MODEL.HEAD.TYPE,
    )


    if 'CAFormer' in cfg.MODEL.PRETRAIN_FILE and training:
        checkpoint = torch.load(cfg.MODEL.PRETRAIN_FILE, map_location="cpu")
        missing_keys, unexpected_keys = model.load_state_dict(checkpoint, strict=False)
        print('Load pretrained model from: ' + cfg.MODEL.PRETRAIN_FILE)
        print('missing_keys, unexpected_keys',missing_keys, unexpected_keys)

    elif training:
        checkpoint = torch.load(cfg.MODEL.PRETRAIN_FILE, map_location="cpu")
        try:
            missing_keys, unexpected_keys = model.load_state_dict(checkpoint["net"], strict=False)
        except:
            try:
                missing_keys, unexpected_keys = model.load_state_dict(checkpoint, strict=False)
            except:
                missing_keys, unexpected_keys = model.load_state_dict(checkpoint["model"], strict=False)
        print('Load pretrained model from: ' + cfg.MODEL.PRETRAIN_FILE)
        print('missing_keys, unexpected_keys',missing_keys, unexpected_keys)
    return model


