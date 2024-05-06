from lib.test.evaluation.environment import EnvSettings

def local_env_settings():
    settings = EnvSettings()
    settings.lmdb_path="/home/zhaojiacong/datasets/lmdb_dataset/"
    
    settings.gtot_path = '/Datasets/GTOT/'
    settings.lasher_path = "/data1/Datasets/Tracking/LasHeR/"
    settings.lashertestingSet_path = '/media/data3/zhaogaotian/dataset/LasHeR/'
    settings.lasher_unaligned = "/home/zhaojiacong/datasets/LasHeR_Ualigned/"
    
    # settings.network_path = '/home/zhaojiacong/all_pretrained/tomp/author/'    # Where tracking networks are stored.  # 预训练权重的路径
    settings.network_path = '/home/zhaojiacong/all_pretrained/trained_by_me/'    # 预训练权重的路径
    settings.results_path = './tracking_result/'
    # settings.result_plot_path = '/home/zhaojiacong/all_result/lasher_test/'
    # settings.result_plot_path = '/home/zhaojiacong/all_result/rgbt234/'
    
    settings.rgbt210_path = ''
    settings.rgbt234_path = "/data1/Datasets/Tracking/RGBT234/"
    settings.vtuav_path = "/home/zhuyabin/dataset1/VTUAV"
    settings.gtot_dir = '/home/ps/GTOT/'
    settings.segmentation_path = '/data/liulei/pytracking/pytracking/segmentation_results/'

    settings.save_dir="./output/"
    settings.prj_dir = "./"


    return settings

