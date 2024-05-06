from . import BaseActor
from lib.utils.misc import NestedTensor
from lib.utils.box_ops import box_cxcywh_to_xyxy, box_xywh_to_xyxy
import torch
from lib.utils.merge import merge_template_search
from ...utils.heapmap_utils import generate_heatmap
from ...utils.ce_utils import generate_mask_cond, adjust_keep_rate


class OSTrackActor(BaseActor):
    """ Actor for training OSTrack models """

    def __init__(self, net, objective, loss_weight, settings, cfg=None):
        super().__init__(net, objective)
        self.loss_weight = loss_weight
        self.settings = settings
        self.bs = self.settings.batchsize  # batch size
        self.cfg = cfg

    def __call__(self, data):
        """
        args:
            data - The input data, should contain the fields 'template', 'search', 'gt_bbox'.
            template_images: (N_t, batch, 3, H, W)
            search_images: (N_s, batch, 3, H, W)
        returns:
            loss    - the training loss
            status  -  dict containing detailed losses
        """
        # forward pass
        out_dict = self.forward_pass(data)

        # compute losses
        try:
            out_dict['rgb_result']
            loss, status = self.compute_losses_dgl(out_dict, data)
        except:
            loss, status = self.compute_losses(out_dict, data)

        return loss, status

    def forward_pass(self, data):
        # currently only support 1 template and 1 search region


        # print(data['template_images'].shape) # torch.Size([2, 32, 3, 128, 128])
        # print(data['search_images'].shape) # torch.Size([2, 32, 3, 128, 128])

        assert len(data['template_images']) == 2
        assert len(data['search_images']) == 2

        #exit()
        # template_list = []
        # for i in range(self.settings.num_template):
        #     template_img_i = data['template_images'][i].view(-1,
        #                                                      *data['template_images'].shape[2:])  # (batch, 3, 128, 128)
        #     # template_att_i = data['template_att'][i].view(-1, *data['template_att'].shape[2:])  # (batch, 128, 128)
        #     template_list.append(template_img_i) # [template_rgb, template_tir]


        template_list = []
        for i in range(len(data['template_images'])):
            template_img_i = data['template_images'][i].view(-1, *data['template_images'].shape[2:])  # (batch, 3, 320, 320)
            template_list.append(template_img_i)

        search_list = []
        for i in range(len(data['search_images'])):
            search_img_i = data['search_images'][i].view(-1, *data['search_images'].shape[2:])  # (batch, 3, 320, 320)
            search_list.append(search_img_i)

        # search_att = data['search_att'][0].view(-1, *data['search_att'].shape[2:])  # (batch, 320, 320)

        box_mask_z = None
        ce_keep_rate = None
        if self.cfg.MODEL.BACKBONE.CE_LOC:
            box_mask_z = generate_mask_cond(self.cfg, self.cfg.TRAIN.BATCH_SIZE, template_list[0].device,
                                            data['template_anno'][0])

            ce_start_epoch = self.cfg.TRAIN.CE_START_EPOCH
            ce_warm_epoch = self.cfg.TRAIN.CE_WARM_EPOCH
            ce_keep_rate = adjust_keep_rate(data['epoch'], warmup_epochs=ce_start_epoch,
                                                total_epochs=ce_start_epoch + ce_warm_epoch,
                                                ITERS_PER_EPOCH=1,
                                                base_keep_rate=self.cfg.MODEL.BACKBONE.CE_KEEP_RATIO[0])

        if len(template_list) == 1:
            template_list = template_list[0]

        out_dict = self.net(template=torch.stack(template_list),
                            search=torch.stack(search_list),
                            ce_template_mask=box_mask_z,
                            ce_keep_rate=ce_keep_rate,
                            return_last_attn=False)

        return out_dict

    def compute_losses(self, pred_dict, gt_dict, return_status=True):
        # gt gaussian map
        gt_bbox = gt_dict['search_anno'][-1]  # (Ns, batch, 4) (x1,y1,w,h) -> (batch, 4)
        gt_gaussian_maps = generate_heatmap(gt_dict['search_anno'], self.cfg.DATA.SEARCH.SIZE, self.cfg.MODEL.BACKBONE.STRIDE)
        gt_gaussian_maps = gt_gaussian_maps[-1].unsqueeze(1)    # B, 1, 16, 16

        # Get boxes
        pred_boxes = pred_dict['pred_boxes']
        if torch.isnan(pred_boxes).any():
            raise ValueError("Network outputs is NAN! Stop Training")
        num_queries = pred_boxes.size(1)
        pred_boxes_vec = box_cxcywh_to_xyxy(pred_boxes).view(-1, 4)  # (B,N,4) --> (BN,4) (x1,y1,x2,y2)
        gt_boxes_vec = box_xywh_to_xyxy(gt_bbox)[:, None, :].repeat((1, num_queries, 1)).view(-1, 4).clamp(min=0.0,
                                                                                                           max=1.0)  # (B,4) --> (B,1,4) --> (B,N,4)
        # compute giou and iou
        try:
            giou_loss, iou = self.objective['giou'](pred_boxes_vec, gt_boxes_vec)  # (BN,4) (BN,4)
        except:
            giou_loss, iou = torch.tensor(0.0).cuda(), torch.tensor(0.0).cuda()
        # compute l1 loss
        l1_loss = self.objective['l1'](pred_boxes_vec, gt_boxes_vec)  # (BN,4) (BN,4)
        # compute location loss
        if 'score_map' in pred_dict:
            location_loss = self.objective['focal'](pred_dict['score_map'], gt_gaussian_maps)
        else:
            location_loss = torch.tensor(0.0, device=l1_loss.device)
        
        # weighted sum
        loss = self.loss_weight['giou'] * giou_loss + self.loss_weight['l1'] * l1_loss + self.loss_weight['focal'] * location_loss
        if return_status:
            # status for log
            mean_iou = iou.detach().mean()
            status = {"Loss/total": loss.item(),
                      "Loss/giou": giou_loss.item(),
                      "Loss/l1": l1_loss.item(),
                      "Loss/location": location_loss.item(),
                      "IoU": mean_iou.item()}
            return loss, status
        else:
            return loss


    def compute_losses_dgl(self, pred_dict, gt_dict, return_status=True):
        """损失小的，提高他的贡献值"""

        def once_loss(loss_f, pred_rgb, pred_tir, gt):
            loss_rgb = loss_f(pred_rgb, gt)
            loss_tir = loss_f(pred_tir, gt)
            return loss_rgb, loss_tir


        # gt gaussian map
        gt_bbox = gt_dict['search_anno'][-1]  # (Ns, batch, 4) (x1,y1,w,h) -> (batch, 4)
        gt_gaussian_maps = generate_heatmap(gt_dict['search_anno'], self.cfg.DATA.SEARCH.SIZE, self.cfg.MODEL.BACKBONE.STRIDE)
        gt_gaussian_maps = gt_gaussian_maps[-1].unsqueeze(1)

        # Get boxes
        pred_boxes = pred_dict['pred_boxes']
        if torch.isnan(pred_boxes).any():
            raise ValueError("Network outputs is NAN! Stop Training")
        num_queries = pred_boxes.size(1)
        pred_boxes_vec = box_cxcywh_to_xyxy(pred_boxes).view(-1, 4)  # (B,N,4) --> (BN,4) (x1,y1,x2,y2)

        # 另外两个头
        pred_boxes_rgb = pred_dict['rgb_result']['pred_boxes']
        if torch.isnan(pred_boxes_rgb).any():
            raise ValueError("Network outputs is NAN! Stop Training")
        # pred_boxes_vec_rgb = box_cxcywh_to_xyxy(pred_boxes_rgb).view(-1, 4)  # (B,N,4) --> (BN,4) (x1,y1,x2,y2)
        
        pred_boxes_tir = pred_dict['tir_result']['pred_boxes']
        if torch.isnan(pred_boxes_tir).any():
            raise ValueError("Network outputs is NAN! Stop Training")
        # pred_boxes_vec_tir = box_cxcywh_to_xyxy(pred_boxes_tir).view(-1, 4)  # (B,N,4) --> (BN,4) (x1,y1,x2,y2)

        gt_boxes_vec = box_xywh_to_xyxy(gt_bbox)[:, None, :].repeat((1, num_queries, 1)).view(-1, 4).clamp(min=0.0,
                                                                                                           max=1.0)  # (B,4) --> (B,1,4) --> (B,N,4)
        ## fusion损失   
        # compute giou and iou
        giou_loss, iou = self.objective['giou'](pred_boxes_vec, gt_boxes_vec)  # (BN,4) (BN,4)输出1个值
        # compute l1 loss
        l1_loss = self.objective['l1'](pred_boxes_vec, gt_boxes_vec)  # (BN,4) (BN,4)
        # compute location loss
        location_loss = self.objective['focal'](pred_dict['score_map'], gt_gaussian_maps)
        # compute dynamic guide loss
        dgl_weight = 0.1        # frozen用的0.6
        dgl_loss = []
        modal_weights = torch.stack(self.modal_weight[1][0], dim=1)
        for pred_rgb, pred_tir, gt, modal_weight in zip(pred_dict['rgb_result']['score_map'], pred_dict['tir_result']['score_map'], 
                                                        gt_gaussian_maps, modal_weights):
            loss_rgb, loss_tir = once_loss(self.objective['focal'], pred_rgb, pred_tir, gt)
            # dgl_loss.append((modal_weight[0] if loss_rgb>loss_tir else modal_weight[1])*abs(loss_rgb-loss_tir))     # 第一版方案
            dgl_loss.append(max(loss_rgb,loss_tir))     # 第二版方案
        dgl_loss = sum(dgl_loss)/len(dgl_loss)

        # weighted sum
        loss = self.loss_weight['giou'] * giou_loss + self.loss_weight['l1'] * l1_loss + self.loss_weight['focal'] * location_loss

        if return_status:
            # status for log
            mean_iou = iou.detach().mean()
            status = {"Loss/total": loss.item(),
                      "Loss/giou": giou_loss.item(),
                      "Loss/l1": l1_loss.item(),
                      "Loss/location": location_loss.item(),
                      "IoU": mean_iou.item(), 
                      "dgl": dgl_loss.item(), }
            return loss+dgl_loss*dgl_weight, status
        else:
            return loss+dgl_loss*dgl_weight
