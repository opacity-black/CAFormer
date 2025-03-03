
class EnvironmentSettings:
    def __init__(self):

        self.lmdb_dir="/home/zhaojiacong/datasets/lmdb_dataset/"
        
        self.workspace_dir = './'    # Base directory for saving network checkpoints.
        self.tensorboard_dir = './tensorboard/'   # Directory for tensorboard files.
        self.wandb_dir = './wandb/'

        self.lasot_dir = '/home/liulei/Datasets/lasot/'
        self.got10k_dir = '/home/zhuyabin/dataset1/GOT/train/'
        self.trackingnet_dir = '/home/zhuyabin/dataset1/TrackingNet/'
        self.coco_dir = '/home/zhuyabin/dataset1/COCO2014/'
        self.lvis_dir = ''
        self.sbd_dir = ''
        self.imagenet_dir = ''
        self.imagenetdet_dir = ''
        self.ecssd_dir = ''
        self.hkuis_dir = ''
        self.msra10k_dir = ''
        self.davis_dir = ''
        self.youtubevos_dir = ''

        self.rgbt210_dir = '' # '/home/liulei/Datasets/RGBT210/'
        self.rgbt234_dir = '/media/data3/zhaogaotian/dataset/RGBT234/'
        self.gtot_dir = '/media/data3/zhaogaotian/dataset/GTOT/'

        self.lasher_dir = "/data1/Datasets/Tracking/LasHeR/"
        self.lasher_trainingset_dir = "/data1/Datasets/Tracking/LasHeR/"
        self.lasher_testingset_dir = "/data1/Datasets/Tracking/LasHeR/"
        self.UAV_RGBT_dir = "/home/zhuyabin/dataset1/VTUAV"
