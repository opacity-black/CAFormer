from lib.models.caformer.block import CMABlock,CMA_CTEBlock
import torch


class MyQueue:
    def __init__(self, length:int) -> None:
        self.max_length = length
        self.data = []

    def put(self, val):
        self.data.append(val)
        if len(self.data)>self.max_length:
            del self.data[0]

    def get(self):
        return self.data
    
    def __getitem__(self, idx=-1):
        return self.data[idx]
    
    def get_max(self):
        m = max(self.data)
        reverse_idx = self.data.index(m) - len(self.data)
        return m, reverse_idx
    
    def clear(self):
        self.data = []


class ReadModalContribution:
    def __init__(self, net:CMABlock, length=300) -> None:
        self.contribute_rgb_once = []
        self.contribute_tir_once = []
        self.contribute = MyQueue(length)

        def get_innerattnmap(this, input, output):
            rgb, tir = self._weightByAttention(output[1])
            self.contribute_rgb_once.append(rgb)
            self.contribute_tir_once.append(tir)
        
        def get_last_innerattnmap(this, input, output):
            rgb, tir = self._weightByAttention(output[1])
            self.contribute_rgb_once.append(rgb)
            self.contribute_tir_once.append(tir)
            self.contribute.put([sum(self.contribute_rgb_once)/len(self.contribute_rgb_once), 
                                sum(self.contribute_tir_once)/len(self.contribute_tir_once)])
            self.contribute_rgb_once = []
            self.contribute_tir_once = []

        idxs = [i for i,blk in enumerate(net.backbone.blocks) if (isinstance(blk, CMA_CTEBlock) or isinstance(blk, CMABlock))]
        for i in idxs[:-1]:
            net.backbone.blocks[i].inner_attn_st.corr_attn.register_forward_hook(get_innerattnmap)
        net.backbone.blocks[idxs[-1]].inner_attn_st.corr_attn.register_forward_hook(get_last_innerattnmap)
    
    def _weightByAttention(self, attn_map, hn2B=12):
        # attn_map like this:  
        # | rgb_self  | rgb_cross |
        # -------------------------
        # | tir_cross | tir_self  |
        B, hn, N1, N2 = attn_map.shape  # hn=1
        attn_map = attn_map.reshape(-1, hn2B, N1, N2)
        N1, N2 = N1//2, N2//2
        x = attn_map.mean(1)    # B,N1,N2
        rgb = x[..., :N2].sum(-1).mean(-1)   # B,
        tir = x[..., N2:].sum(-1).mean(-1)
        # rgb = x[..., N2:, :N2].sum(-1).mean(-1)   # B,
        # tir = x[..., :N2, N2:].sum(-1).mean(-1)
        if B==hn2B:
            return rgb[0], tir[0]
        return rgb, tir


    def __getitem__(self, num): # time, modal, B
        # return torch.tensor(self.contribute.get()[-num:])   # no scale
        # return self.contribute.get()[-num:]   # no scale
        return torch.stack([(torch.stack(cell)*40).softmax(-1) for cell in self.contribute.get()[-num:]], dim=0)    # scale for CAiATrack_uce_st

