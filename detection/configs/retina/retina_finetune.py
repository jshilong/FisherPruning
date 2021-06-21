_base_ = [
    '../_base_/models/retinanet_r50_fpn.py',
    '../_base_/datasets/coco_detection.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]
# optimizer
optimizer = dict(lr=0.01)
custom_hooks = [
    dict(type='FisherPruningHook',
         pruning=False,
         deploy_from='path to the pruned model')
]
#
model = dict(backbone=dict(frozen_stages=-1, ))
