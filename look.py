#config文件测试版

import datetime
from mmengine import Config

net='niuniunet'
date = datetime.date.today()


#cfg = Config.fromfile('/home/niu/mmsegmentation/configs/pspnet/pspnet_r50-d8_4xb4-80k_ade20k-512x512.py')
#cfg = Config.fromfile('/home/niu/mmsegmentation/configs/ccnet/ccnet_r50-d8_4xb4-80k_ade20k-512x512.py')
#cfg = Config.fromfile('/home/niu/mmsegmentation/configs/danet/danet_r50-d8_4xb4-80k_ade20k-512x512.py')
#cfg = Config.fromfile('/home/niu/mmsegmentation/configs/unet/unet-s5-d16_fcn_4xb4-160k_cityscapes-512x1024.py')
#cfg = Config.fromfile('/home/niu/mmsegmentation/configs/dmnet/dmnet_r50-d8_4xb4-80k_ade20k-512x512.py')
#cfg = Config.fromfile('/home/niu/mmsegmentation/configs/hrnet/fcn_hr18_4xb4-80k_ade20k-512x512.py')
#cfg = Config.fromfile("/home/niu/mmsegmentation/configs/resnest/resnest_s101-d8_pspnet_4xb4-160k_ade20k-512x512.py")
dataset_cfg = Config.fromfile('./configs/_base_/datasets/WHUDataset_pipeline.py')


cfg.merge_from_dict(dataset_cfg)

NUM_CLASS = 2

cfg.crop_size = (512, 512)
cfg.model.data_preprocessor.size = cfg.crop_size

# 单卡训练时，需要把 SyncBN 改成 BN
cfg.norm_cfg = dict(type='BN', requires_grad=True)
cfg.model.backbone.norm_cfg = cfg.norm_cfg
cfg.model.decode_head.norm_cfg = cfg.norm_cfg
cfg.model.auxiliary_head.norm_cfg = cfg.norm_cfg

# 模型 decode/auxiliary 输出头，指定为类别个数
cfg.model.decode_head.num_classes = NUM_CLASS
cfg.model.auxiliary_head.num_classes = NUM_CLASS

# 训练 Batch Size
cfg.train_dataloader.batch_size = 8

# 结果保存目录
cfg.work_dir = './work_dirs/WHUDataset-' + net

# 模型保存与日志记录
cfg.train_cfg.max_iters = 60000 # 训练迭代次数
cfg.train_cfg.val_interval = 500 # 评估模型间隔
cfg.default_hooks.logger.interval = 100 # 日志记录间隔
cfg.default_hooks.checkpoint.interval = 1000 # 模型权重保存间隔
cfg.default_hooks.checkpoint.max_keep_ckpts = 1 # 最多保留几个模型权重
cfg.default_hooks.checkpoint.save_best = 'mIoU' # 保留指标最高的模型权重

# 随机数种子
cfg['randomness'] = dict(seed=0)

cfg.dump('whuconfigs/WHUDataset_'+net+'_'+str(date)+'.py')