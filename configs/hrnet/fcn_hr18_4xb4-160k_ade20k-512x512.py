_base_ = [
    '../_base_/models/fcn_hr18.py', '../_base_/datasets/ade20k.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_160k.py'
]
crop_size = (512, 512)
data_preprocessor = dict(size=crop_size)
model = dict(
    data_preprocessor=data_preprocessor, 
    decode_head=dict(num_classes=150),
    auxiliary_head= dict(
    type='FCNHead',
    in_channels=144,
    channels=64
    )
    
    )


#10月26日添加
#model['auxiliary_head'] = dict(
 #   type='FCNHead',
 #   in_channels=256,
  #  in_index=3,
  #  channels=64,
  #  num_convs=1,
 #   concat_input=False,
 #   dropout_ratio=-1,
  #  num_classes=150,
    
  #  loss_decode=dict(
  #      type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4)
#)

