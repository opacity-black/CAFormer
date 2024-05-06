import torch
import argparse

# 删除包里的梯度，节省空间
def delGrad(file):
    print('reading...', file)
    stateDict = torch.load(file, map_location="cpu")
    try:
        del stateDict['optimizer']
    except:
        raise "Delete fail !!!"
    print('Overwriting...')
    torch.save(stateDict, file)
    print('Done')


if __name__=="__main__":
    
    parser = argparse.ArgumentParser(description='Delete the optimizer info in checkpoint file.')
    parser.add_argument('file_path', type=str, help='Path of Checkpoint File.', nargs='+') 
    args = parser.parse_args()

    for item in args.file_path:
        delGrad(item)