import torch
import os,sys


# 删除包里的梯度，节省空间
def delGrad(file):
    stateDict = torch.load(file, map_location="cpu")
    del stateDict['optimizer']
    torch.save(stateDict, file)


def qkv2q_k_v(file, new_file):
    # 将原来合在一起的qkv参数分开成q，k，v
    stateDict = torch.load(file, map_location="cpu")['net']
    stateDict_new = {}
    for k,v in list(stateDict.items()):
        if 'qkv' in k:
            print(f'transfer \"{k}\".')
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
        # else:
        #     stateDict_new[k] = v
    torch.save(stateDict, new_file)


def param_anl(file):
    # adapter参数分析
    stateDict = torch.load(file, map_location="cpu")['net']
    for k,v in list(stateDict.items()):
        if 'adapt' in k:
            print(f"{k}, \tmean={v.mean()}, \tstd={v.std()}\n")


if __name__=="__main__":
    delGrad(file='')