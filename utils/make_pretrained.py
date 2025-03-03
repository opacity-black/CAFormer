import torch

# Download SOT pretrained model from OSTrack, then run this program.
stateDict = torch.load("./pretrained/OSTrack_ep0300.pth.tar", map_location="cpu")
for k,v in list(stateDict.items()):
    # if 'box_head' in k:
    #     stateDict[k.replace('box_head', 'box_head_v')] = stateDict[k]
    #     stateDict[k.replace('box_head', 'box_head_i')] = stateDict[k]
    if k in ['box_head.conv1_ctr.0.weight','box_head.conv1_offset.0.weight','box_head.conv1_size.0.weight']:
        stateDict[k] = torch.cat([v,v],1)

    if 'qkv' in k:
        q_key = k.replace('qkv','q_linear')
        k_key = k.replace('qkv','k_linear')
        v_key = k.replace('qkv','v_linear')
        if 'weight' in k:
            stateDict[q_key] = v[:768, :]
            stateDict[k_key] = v[768:768*2, :]
            stateDict[v_key] = v[768*2:, :]
        elif 'bias' in k:
            stateDict[q_key] = v[:768]
            stateDict[k_key] = v[768:768*2]
            stateDict[v_key] = v[768*2:]
        del stateDict[k]
torch.save(stateDict, "./pretrained/CAFormer_SOTPretrained.pth.tar")