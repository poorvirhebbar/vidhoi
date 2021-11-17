#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Video models."""

import torch
import torch.nn as nn
import torchvision

import slowfast.utils.weight_init_helper as init_helper
from slowfast.models.batchnorm_helper import get_norm

from . import head_helper, resnet_helper, stem_helper
from .build import MODEL_REGISTRY

# Number of blocks for different stages given the model depth.
_MODEL_STAGE_DEPTH = {50: (3, 4, 6, 3), 101: (3, 4, 23, 3)}

# Basis of temporal kernel sizes for each of the stage.
_TEMPORAL_KERNEL_BASIS = {
    "c2d": [
        [[1]],  # conv1 temporal kernel.
        [[1]],  # res2 temporal kernel.
        [[1]],  # res3 temporal kernel.
        [[1]],  # res4 temporal kernel.
        [[1]],  # res5 temporal kernel.
    ],
    "c2d_nopool": [
        [[1]],  # conv1 temporal kernel.
        [[1]],  # res2 temporal kernel.
        [[1]],  # res3 temporal kernel.
        [[1]],  # res4 temporal kernel.
        [[1]],  # res5 temporal kernel.
    ],
    "i3d": [
        [[5]],  # conv1 temporal kernel.
        [[3]],  # res2 temporal kernel.
        [[3, 1]],  # res3 temporal kernel.
        [[3, 1]],  # res4 temporal kernel.
        [[1, 3]],  # res5 temporal kernel.
    ],
    "i3d_nopool": [
        [[5]],  # conv1 temporal kernel.
        [[3]],  # res2 temporal kernel.
        [[3, 1]],  # res3 temporal kernel.
        [[3, 1]],  # res4 temporal kernel.
        [[1, 3]],  # res5 temporal kernel.
    ],
    "slow": [
        [[1]],  # conv1 temporal kernel.
        [[1]],  # res2 temporal kernel.
        [[1]],  # res3 temporal kernel.
        [[3]],  # res4 temporal kernel.
        [[3]],  # res5 temporal kernel.
    ],
    "slowfast": [
        [[1], [5]],  # conv1 temporal kernel for slow and fast pathway.
        [[1], [3]],  # res2 temporal kernel for slow and fast pathway.
        [[1], [3]],  # res3 temporal kernel for slow and fast pathway.
        [[3], [3]],  # res4 temporal kernel for slow and fast pathway.
        [[3], [3]],  # res5 temporal kernel for slow and fast pathway.
    ],
    "fast": [
        [[5]],  # conv1 temporal kernel 
        [[3]],  # res2 temporal kernel 
        [[3]],  # res3 temporal kernel 
        [[3]],  # res4 temporal kernel
        [[3]],  # res5 temporal kernel
    ],
}

_POOL1 = {
    "c2d": [[2, 1, 1]],
    "c2d_nopool": [[1, 1, 1]],
    "i3d": [[2, 1, 1]],
    "i3d_nopool": [[1, 1, 1]],
    "slow": [[1, 1, 1]],
    "slowfast": [[1, 1, 1], [1, 1, 1]],
    "fast": [[1, 1, 1]],
    "baseline": [[1, 1, 1]],
    "resnext101":[[1, 1, 1]],
    "resnet101":[[1, 1, 1]],
    "lighten":[[1, 1, 1]],
}


class FuseFastToSlow(nn.Module):
    """
    Fuses the information from the Fast pathway to the Slow pathway. Given the
    tensors from Slow pathway and Fast pathway, fuse information from Fast to
    Slow, then return the fused tensors from Slow and Fast pathway in order.
    """

    def __init__(
        self,
        dim_in,
        fusion_conv_channel_ratio,
        fusion_kernel,
        alpha,
        eps=1e-5,
        bn_mmt=0.1,
        inplace_relu=True,
        norm_module=nn.BatchNorm3d,
    ):
        """
        Args:
            dim_in (int): the channel dimension of the input.
            fusion_conv_channel_ratio (int): channel ratio for the convolution
                used to fuse from Fast pathway to Slow pathway.
            fusion_kernel (int): kernel size of the convolution used to fuse
                from Fast pathway to Slow pathway.
            alpha (int): the frame rate ratio between the Fast and Slow pathway.
            eps (float): epsilon for batch norm.
            bn_mmt (float): momentum for batch norm. Noted that BN momentum in
                PyTorch = 1 - BN momentum in Caffe2.
            inplace_relu (bool): if True, calculate the relu on the original
                input without allocating new memory.
            norm_module (nn.Module): nn.Module for the normalization layer. The
                default is nn.BatchNorm3d.
        """
        super(FuseFastToSlow, self).__init__()
        self.conv_f2s = nn.Conv3d(
            dim_in,
            dim_in * fusion_conv_channel_ratio,
            kernel_size=[fusion_kernel, 1, 1],
            stride=[alpha, 1, 1],
            padding=[fusion_kernel // 2, 0, 0],
            bias=False,
        )
        self.bn = norm_module(
            num_features=dim_in * fusion_conv_channel_ratio,
            eps=eps,
            momentum=bn_mmt,
        )
        self.relu = nn.ReLU(inplace_relu)

    def forward(self, x):
        x_s = x[0]
        x_f = x[1]
        fuse = self.conv_f2s(x_f)
        fuse = self.bn(fuse)
        fuse = self.relu(fuse)
        x_s_fuse = torch.cat([x_s, fuse], 1)
        return [x_s_fuse, x_f]


@MODEL_REGISTRY.register()
class Baseline(nn.Module):
    """
    Model builder for baseline network (without SlowFast).
    """

    def __init__(self, cfg):
        """
        The `__init__` method of any subclass should also contain these
            arguments.
        Args:
            cfg (CfgNode): model building configs, details are in the
                comments of the config file.
        """
        super(Baseline, self).__init__()
        self.norm_module = get_norm(cfg)
        self.enable_detection = cfg.DETECTION.ENABLE
        self.enable_detection_hoi = cfg.DETECTION.ENABLE_HOI
        self.num_pathways = 1
        self._construct_network(cfg)
        init_helper.init_weights_baseline(
            self, cfg.MODEL.FC_INIT_STD, cfg.RESNET.ZERO_INIT_FINAL_BN, cfg
        )

    def _construct_network(self, cfg):
        """
        Builds a Baseline model. The first pathway is the Slow pathway and the
            second pathway is the Fast pathway.
        Args:
            cfg (CfgNode): model building configs, details are in the
                comments of the config file.
        """
        assert cfg.MODEL.ARCH in _POOL1.keys()
        pool_size = _POOL1[cfg.MODEL.ARCH]
        assert len({len(pool_size), self.num_pathways}) == 1
        assert cfg.RESNET.DEPTH in _MODEL_STAGE_DEPTH.keys()

        (d2, d3, d4, d5) = _MODEL_STAGE_DEPTH[cfg.RESNET.DEPTH]

        num_groups = cfg.RESNET.NUM_GROUPS
        width_per_group = cfg.RESNET.WIDTH_PER_GROUP
        dim_inner = num_groups * width_per_group
        out_dim_ratio = (
            cfg.SLOWFAST.BETA_INV // cfg.SLOWFAST.FUSION_CONV_CHANNEL_RATIO
        )

        # initielize backbone resnet model
        resnet50 = torchvision.models.resnet50(pretrained=(False if cfg.RESNET.SCRATCH else True))
        self.backbone = torch.nn.Sequential(*list(resnet50.children())[:-2])
        if not cfg.RESNET.SCRATCH and cfg.BASELINE.FREEZE_BACKBONE:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # if cfg.DETECTION.ENABLE:
        #     if cfg.DETECTION.ENABLE_HOI:
        self.head = head_helper.ResNetPoolHead(
            dim_in=[width_per_group * 32],
            num_classes=cfg.MODEL.NUM_CLASSES,
            pool_size=[[cfg.DATA.NUM_FRAMES // pool_size[0][0], 1, 1,]],
            is_baseline=True
        )
        self.hoi_head = head_helper.HOIHead(cfg, 
            resolution=[cfg.DETECTION.ROI_XFORM_RESOLUTION] * 2, scale_factor=cfg.DETECTION.SPATIAL_SCALE_FACTOR, aligned=cfg.DETECTION.ALIGNED,)
     
    def forward(self, x, bboxes=None, obj_classes=None, obj_classes_lengths=None, action_labels=None, gt_obj_classes=None, gt_obj_classes_lengths=None, trajectories=None, human_poses=None, trajectory_boxes=None, skeleton_imgs=None, trajectory_box_masks=None):
        # x is a list of single path way features. E.g.,
        # (Pdb) x[0].shape
        # torch.Size([batch_size, 3, 1, 224, 224])
        x = x[0].squeeze(2)
        x = [self.backbone(x)]
        # (Pdb) x.shape
        # torch.Size([16, 2048, 7, 7])
        if self.enable_detection:
            if self.enable_detection_hoi:
                x = self.head(x)
                # (Pdb) x.shape
                # torch.Size([batch_size, 80])
                # bboxes = torch.Size([65, 5])
                x = self.hoi_head(x, bboxes, obj_classes, obj_classes_lengths, action_labels, gt_obj_classes, gt_obj_classes_lengths)
            else:
                x = self.head(x, bboxes)
        else:
            x = self.head(x)
        return x


@MODEL_REGISTRY.register()
class SlowFast(nn.Module):
    """
    SlowFast model builder for SlowFast network.

    Christoph Feichtenhofer, Haoqi Fan, Jitendra Malik, and Kaiming He.
    "SlowFast networks for video recognition."
    https://arxiv.org/pdf/1812.03982.pdf
    """

    def __init__(self, cfg):
        """
        The `__init__` method of any subclass should also contain these
            arguments.
        Args:
            cfg (CfgNode): model building configs, details are in the
                comments of the config file.
        """
        super(SlowFast, self).__init__()
        self.norm_module = get_norm(cfg)
        self.enable_detection = cfg.DETECTION.ENABLE
        self.enable_detection_hoi = cfg.DETECTION.ENABLE_HOI
        self.enable_toi_pooling = cfg.DETECTION.ENABLE_TOI_POOLING
        self.num_pathways = 2
        self._construct_network(cfg)
        init_helper.init_weights(
            self, cfg.MODEL.FC_INIT_STD, cfg.RESNET.ZERO_INIT_FINAL_BN, cfg
        )

    def _construct_network(self, cfg):
        """
        Builds a SlowFast model. The first pathway is the Slow pathway and the
            second pathway is the Fast pathway.
        Args:
            cfg (CfgNode): model building configs, details are in the
                comments of the config file.
        """
        assert cfg.MODEL.ARCH in _POOL1.keys()
        pool_size = _POOL1[cfg.MODEL.ARCH]
        assert len({len(pool_size), self.num_pathways}) == 1
        assert cfg.RESNET.DEPTH in _MODEL_STAGE_DEPTH.keys()

        (d2, d3, d4, d5) = _MODEL_STAGE_DEPTH[cfg.RESNET.DEPTH]

        num_groups = cfg.RESNET.NUM_GROUPS
        width_per_group = cfg.RESNET.WIDTH_PER_GROUP
        dim_inner = num_groups * width_per_group
        out_dim_ratio = (
            cfg.SLOWFAST.BETA_INV // cfg.SLOWFAST.FUSION_CONV_CHANNEL_RATIO
        )

        temp_kernel = _TEMPORAL_KERNEL_BASIS[cfg.MODEL.ARCH]

        self.s1 = stem_helper.VideoModelStem(
            dim_in=cfg.DATA.INPUT_CHANNEL_NUM,
            dim_out=[width_per_group, width_per_group // cfg.SLOWFAST.BETA_INV],
            kernel=[temp_kernel[0][0] + [7, 7], temp_kernel[0][1] + [7, 7]],
            stride=[[1, 2, 2]] * 2,
            padding=[
                [temp_kernel[0][0][0] // 2, 3, 3],
                [temp_kernel[0][1][0] // 2, 3, 3],
            ],
            norm_module=self.norm_module,
        )
        self.s1_fuse = FuseFastToSlow(
            width_per_group // cfg.SLOWFAST.BETA_INV,
            cfg.SLOWFAST.FUSION_CONV_CHANNEL_RATIO,
            cfg.SLOWFAST.FUSION_KERNEL_SZ,
            cfg.SLOWFAST.ALPHA,
            norm_module=self.norm_module,
        )

        self.s2 = resnet_helper.ResStage(
            dim_in=[
                width_per_group + width_per_group // out_dim_ratio,
                width_per_group // cfg.SLOWFAST.BETA_INV,
            ],
            dim_out=[
                width_per_group * 4,
                width_per_group * 4 // cfg.SLOWFAST.BETA_INV,
            ],
            dim_inner=[dim_inner, dim_inner // cfg.SLOWFAST.BETA_INV],
            temp_kernel_sizes=temp_kernel[1],
            stride=cfg.RESNET.SPATIAL_STRIDES[0],
            num_blocks=[d2] * 2,
            num_groups=[num_groups] * 2,
            num_block_temp_kernel=cfg.RESNET.NUM_BLOCK_TEMP_KERNEL[0],
            nonlocal_inds=cfg.NONLOCAL.LOCATION[0],
            nonlocal_group=cfg.NONLOCAL.GROUP[0],
            nonlocal_pool=cfg.NONLOCAL.POOL[0],
            instantiation=cfg.NONLOCAL.INSTANTIATION,
            trans_func_name=cfg.RESNET.TRANS_FUNC,
            dilation=cfg.RESNET.SPATIAL_DILATIONS[0],
            norm_module=self.norm_module,
        )
        self.s2_fuse = FuseFastToSlow(
            width_per_group * 4 // cfg.SLOWFAST.BETA_INV,
            cfg.SLOWFAST.FUSION_CONV_CHANNEL_RATIO,
            cfg.SLOWFAST.FUSION_KERNEL_SZ,
            cfg.SLOWFAST.ALPHA,
            norm_module=self.norm_module,
        )

        for pathway in range(self.num_pathways):
            pool = nn.MaxPool3d(
                kernel_size=pool_size[pathway],
                stride=pool_size[pathway],
                padding=[0, 0, 0],
            )
            self.add_module("pathway{}_pool".format(pathway), pool)

        self.s3 = resnet_helper.ResStage(
            dim_in=[
                width_per_group * 4 + width_per_group * 4 // out_dim_ratio,
                width_per_group * 4 // cfg.SLOWFAST.BETA_INV,
            ],
            dim_out=[
                width_per_group * 8,
                width_per_group * 8 // cfg.SLOWFAST.BETA_INV,
            ],
            dim_inner=[dim_inner * 2, dim_inner * 2 // cfg.SLOWFAST.BETA_INV],
            temp_kernel_sizes=temp_kernel[2],
            stride=cfg.RESNET.SPATIAL_STRIDES[1],
            num_blocks=[d3] * 2,
            num_groups=[num_groups] * 2,
            num_block_temp_kernel=cfg.RESNET.NUM_BLOCK_TEMP_KERNEL[1],
            nonlocal_inds=cfg.NONLOCAL.LOCATION[1],
            nonlocal_group=cfg.NONLOCAL.GROUP[1],
            nonlocal_pool=cfg.NONLOCAL.POOL[1],
            instantiation=cfg.NONLOCAL.INSTANTIATION,
            trans_func_name=cfg.RESNET.TRANS_FUNC,
            dilation=cfg.RESNET.SPATIAL_DILATIONS[1],
            norm_module=self.norm_module,
        )
        self.s3_fuse = FuseFastToSlow(
            width_per_group * 8 // cfg.SLOWFAST.BETA_INV,
            cfg.SLOWFAST.FUSION_CONV_CHANNEL_RATIO,
            cfg.SLOWFAST.FUSION_KERNEL_SZ,
            cfg.SLOWFAST.ALPHA,
            norm_module=self.norm_module,
        )

        self.s4 = resnet_helper.ResStage(
            dim_in=[
                width_per_group * 8 + width_per_group * 8 // out_dim_ratio,
                width_per_group * 8 // cfg.SLOWFAST.BETA_INV,
            ],
            dim_out=[
                width_per_group * 16,
                width_per_group * 16 // cfg.SLOWFAST.BETA_INV,
            ],
            dim_inner=[dim_inner * 4, dim_inner * 4 // cfg.SLOWFAST.BETA_INV],
            temp_kernel_sizes=temp_kernel[3],
            stride=cfg.RESNET.SPATIAL_STRIDES[2],
            num_blocks=[d4] * 2,
            num_groups=[num_groups] * 2,
            num_block_temp_kernel=cfg.RESNET.NUM_BLOCK_TEMP_KERNEL[2],
            nonlocal_inds=cfg.NONLOCAL.LOCATION[2],
            nonlocal_group=cfg.NONLOCAL.GROUP[2],
            nonlocal_pool=cfg.NONLOCAL.POOL[2],
            instantiation=cfg.NONLOCAL.INSTANTIATION,
            trans_func_name=cfg.RESNET.TRANS_FUNC,
            dilation=cfg.RESNET.SPATIAL_DILATIONS[2],
            norm_module=self.norm_module,
        )
        self.s4_fuse = FuseFastToSlow(
            width_per_group * 16 // cfg.SLOWFAST.BETA_INV,
            cfg.SLOWFAST.FUSION_CONV_CHANNEL_RATIO,
            cfg.SLOWFAST.FUSION_KERNEL_SZ,
            cfg.SLOWFAST.ALPHA,
            norm_module=self.norm_module,
        )

        self.s5 = resnet_helper.ResStage(
            dim_in=[
                width_per_group * 16 + width_per_group * 16 // out_dim_ratio,
                width_per_group * 16 // cfg.SLOWFAST.BETA_INV,
            ],
            dim_out=[
                width_per_group * 32,
                width_per_group * 32 // cfg.SLOWFAST.BETA_INV,
            ],
            dim_inner=[dim_inner * 8, dim_inner * 8 // cfg.SLOWFAST.BETA_INV],
            temp_kernel_sizes=temp_kernel[4],
            stride=cfg.RESNET.SPATIAL_STRIDES[3],
            num_blocks=[d5] * 2,
            num_groups=[num_groups] * 2,
            num_block_temp_kernel=cfg.RESNET.NUM_BLOCK_TEMP_KERNEL[3],
            nonlocal_inds=cfg.NONLOCAL.LOCATION[3],
            nonlocal_group=cfg.NONLOCAL.GROUP[3],
            nonlocal_pool=cfg.NONLOCAL.POOL[3],
            instantiation=cfg.NONLOCAL.INSTANTIATION,
            trans_func_name=cfg.RESNET.TRANS_FUNC,
            dilation=cfg.RESNET.SPATIAL_DILATIONS[3],
            norm_module=self.norm_module,
        )

        if cfg.SLOWFAST.FREEZE_BACKBONE:
            for param in self.s1.parameters():
                param.requires_grad = False
            for param in self.s2.parameters():
                param.requires_grad = False
            for param in self.s3.parameters():
                param.requires_grad = False
            for param in self.s4.parameters():
                param.requires_grad = False
            for param in self.s5.parameters():
                param.requires_grad = False
            for param in self.s1_fuse.parameters():
                param.requires_grad = False
            for param in self.s2_fuse.parameters():
                param.requires_grad = False
            for param in self.s3_fuse.parameters():
                param.requires_grad = False
            for param in self.s4_fuse.parameters():
                param.requires_grad = False

        if cfg.DETECTION.ENABLE:
            if cfg.DETECTION.ENABLE_HOI:
                if cfg.DETECTION.ENABLE_TOI_POOLING:
                    self.toi_head = head_helper.ResNetToIHead(
                        dim_in=[
                            width_per_group * 32,
                            width_per_group * 32 // cfg.SLOWFAST.BETA_INV,
                        ],
                        num_classes=cfg.MODEL.NUM_CLASSES,
                        pool_size=[None, None] # changed temporal pooling kernel size
                        if cfg.MULTIGRID.SHORT_CYCLE else 
                        [
                            [
                                13,
                                #cfg.DATA.NUM_FRAMES
                                #// cfg.SLOWFAST.ALPHA
                                #// pool_size[0][0],
                                1,
                                1,
                            ],
                            [13,#cfg.DATA.NUM_FRAMES // pool_size[1][0], 
                                1, 1],
                        ],
                        resolution=[[cfg.DETECTION.ROI_XFORM_RESOLUTION] * 2] * 2,
                        scale_factor=[cfg.DETECTION.SPATIAL_SCALE_FACTOR] * 2,
                        num_frames=cfg.DATA.NUM_FRAMES,
                        alpha=cfg.SLOWFAST.ALPHA,
                        aligned=cfg.DETECTION.ALIGNED,
                    )
                self.head = head_helper.ResNetPoolHead(
                    dim_in=[
                        width_per_group * 32,
                        width_per_group * 32 // cfg.SLOWFAST.BETA_INV,
                    ],
                    num_classes=cfg.MODEL.NUM_CLASSES,
                    pool_size=[None, None]
                    if cfg.MULTIGRID.SHORT_CYCLE else 
                    [
                        [
                            cfg.DATA.NUM_FRAMES
                            // cfg.SLOWFAST.ALPHA
                            // pool_size[0][0],
                            1,
                            1,
                        ],
                        [cfg.DATA.NUM_FRAMES // pool_size[1][0], 1, 1],
                    ],
                )
                self.hoi_head = head_helper.HOIHead(cfg, 
                    resolution=[cfg.DETECTION.ROI_XFORM_RESOLUTION] * 2, scale_factor=cfg.DETECTION.SPATIAL_SCALE_FACTOR, aligned=cfg.DETECTION.ALIGNED)
            else:
                self.head = head_helper.ResNetRoIHead(
                    dim_in=[
                        width_per_group * 32,
                        width_per_group * 32 // cfg.SLOWFAST.BETA_INV,
                    ],
                    num_classes=cfg.MODEL.NUM_CLASSES,
                    pool_size=[
                        [
                            cfg.DATA.NUM_FRAMES
                            // cfg.SLOWFAST.ALPHA
                            // pool_size[0][0],
                            1,
                            1,
                        ],
                        [cfg.DATA.NUM_FRAMES // pool_size[1][0], 1, 1],
                    ],
                    resolution=[[cfg.DETECTION.ROI_XFORM_RESOLUTION] * 2] * 2,
                    scale_factor=[cfg.DETECTION.SPATIAL_SCALE_FACTOR] * 2,
                    dropout_rate=cfg.MODEL.DROPOUT_RATE,
                    act_func=cfg.MODEL.HEAD_ACT,
                    aligned=cfg.DETECTION.ALIGNED,
                )
        else:
            self.head = head_helper.ResNetBasicHead(
                dim_in=[
                    width_per_group * 32,
                    width_per_group * 32 // cfg.SLOWFAST.BETA_INV,
                ],
                num_classes=cfg.MODEL.NUM_CLASSES,
                pool_size=[None, None]
                if cfg.MULTIGRID.SHORT_CYCLE
                else [
                    [
                        cfg.DATA.NUM_FRAMES
                        // cfg.SLOWFAST.ALPHA
                        // pool_size[0][0],
                        cfg.DATA.CROP_SIZE // 32 // pool_size[0][1],
                        cfg.DATA.CROP_SIZE // 32 // pool_size[0][2],
                    ],
                    [
                        cfg.DATA.NUM_FRAMES // pool_size[1][0],
                        cfg.DATA.CROP_SIZE // 32 // pool_size[1][1],
                        cfg.DATA.CROP_SIZE // 32 // pool_size[1][2],
                    ],
                ],  # None for AdaptiveAvgPool3d((1, 1, 1))
                dropout_rate=cfg.MODEL.DROPOUT_RATE,
                act_func=cfg.MODEL.HEAD_ACT,
            )

    def forward(self, x, bboxes=None, obj_classes=None, obj_classes_lengths=None, action_labels=None, gt_obj_classes=None, gt_obj_classes_lengths=None, trajectories=None, human_poses=None, trajectory_boxes=None, skeleton_imgs=None, trajectory_box_masks=None):
        # x is a list of two path way features. E.g.,
        # (Pdb) x[0].shape
        # torch.Size([batch_size, 3, 8, 224, 224])
        # (Pdb) x[1].shape
        # torch.Size([batch_size, 3, 32, 224, 224])
        with torch.no_grad():
            x = self.s1(x)
            x = self.s1_fuse(x)
            x = self.s2(x)
            x = self.s2_fuse(x)
            for pathway in range(self.num_pathways):
                pool = getattr(self, "pathway{}_pool".format(pathway))
                x[pathway] = pool(x[pathway])
            x = self.s3(x)
            x = self.s3_fuse(x)
            x = self.s4(x)
            x = self.s4_fuse(x)
            x = self.s5(x)
        # (Pdb) x[0].shape
        # torch.Size([batch_size, 2048, 8, 14, 14])
        # (Pdb) x[1].shape
        # torch.Size([batch_size, 256, 32, 14, 14])
        if self.enable_detection:
            if self.enable_detection_hoi:
                toi_pooled_features = self.toi_head(x, bboxes, trajectory_boxes) if self.enable_toi_pooling else None # need to go before x = self.head(x)
                #import pdb; pdb.set_trace()
                x = self.head(x) # head only does temporal pooling of features without roi pooling
                # len(x)
                # 16(batch_size)
                # x[0].shape
                # torch.Size([2304,14,14])
                x = self.hoi_head(x, bboxes, obj_classes, obj_classes_lengths, action_labels, gt_obj_classes, gt_obj_classes_lengths, trajectories, human_poses, toi_pooled_features, trajectory_boxes, skeleton_imgs, trajectory_box_masks)
            else:
                x = self.head(x, bboxes)
        else:
            x = self.head(x)
        
        return x


@MODEL_REGISTRY.register()
class ResNet(nn.Module):
    """
    ResNet model builder. It builds a ResNet like network backbone without
    lateral connection (C2D, I3D, Slow).

    Christoph Feichtenhofer, Haoqi Fan, Jitendra Malik, and Kaiming He.
    "SlowFast networks for video recognition."
    https://arxiv.org/pdf/1812.03982.pdf

    Xiaolong Wang, Ross Girshick, Abhinav Gupta, and Kaiming He.
    "Non-local neural networks."
    https://arxiv.org/pdf/1711.07971.pdf
    """

    def __init__(self, cfg):
        """
        The `__init__` method of any subclass should also contain these
            arguments.

        Args:
            cfg (CfgNode): model building configs, details are in the
                comments of the config file.
        """
        super(ResNet, self).__init__()
        self.norm_module = get_norm(cfg)
        self.enable_detection = cfg.DETECTION.ENABLE
        # self.enable_detection_hoi = cfg.DETECTION.ENABLE_HOI
        self.num_pathways = 1
        self._construct_network(cfg)
        init_helper.init_weights(
            self, cfg.MODEL.FC_INIT_STD, cfg.RESNET.ZERO_INIT_FINAL_BN
        )

    def _construct_network(self, cfg):
        """
        Builds a single pathway ResNet model.

        Args:
            cfg (CfgNode): model building configs, details are in the
                comments of the config file.
        """
        assert cfg.MODEL.ARCH in _POOL1.keys()
        pool_size = _POOL1[cfg.MODEL.ARCH]
        assert len({len(pool_size), self.num_pathways}) == 1
        assert cfg.RESNET.DEPTH in _MODEL_STAGE_DEPTH.keys()

        (d2, d3, d4, d5) = _MODEL_STAGE_DEPTH[cfg.RESNET.DEPTH]

        num_groups = cfg.RESNET.NUM_GROUPS
        width_per_group = cfg.RESNET.WIDTH_PER_GROUP
        dim_inner = num_groups * width_per_group

        temp_kernel = _TEMPORAL_KERNEL_BASIS[cfg.MODEL.ARCH]

        self.s1 = stem_helper.VideoModelStem(
            dim_in=cfg.DATA.INPUT_CHANNEL_NUM,
            dim_out=[width_per_group],
            kernel=[temp_kernel[0][0] + [7, 7]],
            stride=[[1, 2, 2]],
            padding=[[temp_kernel[0][0][0] // 2, 3, 3]],
            norm_module=self.norm_module,
        )

        self.s2 = resnet_helper.ResStage(
            dim_in=[width_per_group],
            dim_out=[width_per_group * 4],
            dim_inner=[dim_inner],
            temp_kernel_sizes=temp_kernel[1],
            stride=cfg.RESNET.SPATIAL_STRIDES[0],
            num_blocks=[d2],
            num_groups=[num_groups],
            num_block_temp_kernel=cfg.RESNET.NUM_BLOCK_TEMP_KERNEL[0],
            nonlocal_inds=cfg.NONLOCAL.LOCATION[0],
            nonlocal_group=cfg.NONLOCAL.GROUP[0],
            nonlocal_pool=cfg.NONLOCAL.POOL[0],
            instantiation=cfg.NONLOCAL.INSTANTIATION,
            trans_func_name=cfg.RESNET.TRANS_FUNC,
            stride_1x1=cfg.RESNET.STRIDE_1X1,
            inplace_relu=cfg.RESNET.INPLACE_RELU,
            dilation=cfg.RESNET.SPATIAL_DILATIONS[0],
            norm_module=self.norm_module,
        )

        for pathway in range(self.num_pathways):
            pool = nn.MaxPool3d(
                kernel_size=pool_size[pathway],
                stride=pool_size[pathway],
                padding=[0, 0, 0],
            )
            self.add_module("pathway{}_pool".format(pathway), pool)

        self.s3 = resnet_helper.ResStage(
            dim_in=[width_per_group * 4],
            dim_out=[width_per_group * 8],
            dim_inner=[dim_inner * 2],
            temp_kernel_sizes=temp_kernel[2],
            stride=cfg.RESNET.SPATIAL_STRIDES[1],
            num_blocks=[d3],
            num_groups=[num_groups],
            num_block_temp_kernel=cfg.RESNET.NUM_BLOCK_TEMP_KERNEL[1],
            nonlocal_inds=cfg.NONLOCAL.LOCATION[1],
            nonlocal_group=cfg.NONLOCAL.GROUP[1],
            nonlocal_pool=cfg.NONLOCAL.POOL[1],
            instantiation=cfg.NONLOCAL.INSTANTIATION,
            trans_func_name=cfg.RESNET.TRANS_FUNC,
            stride_1x1=cfg.RESNET.STRIDE_1X1,
            inplace_relu=cfg.RESNET.INPLACE_RELU,
            dilation=cfg.RESNET.SPATIAL_DILATIONS[1],
            norm_module=self.norm_module,
        )

        self.s4 = resnet_helper.ResStage(
            dim_in=[width_per_group * 8],
            dim_out=[width_per_group * 16],
            dim_inner=[dim_inner * 4],
            temp_kernel_sizes=temp_kernel[3],
            stride=cfg.RESNET.SPATIAL_STRIDES[2],
            num_blocks=[d4],
            num_groups=[num_groups],
            num_block_temp_kernel=cfg.RESNET.NUM_BLOCK_TEMP_KERNEL[2],
            nonlocal_inds=cfg.NONLOCAL.LOCATION[2],
            nonlocal_group=cfg.NONLOCAL.GROUP[2],
            nonlocal_pool=cfg.NONLOCAL.POOL[2],
            instantiation=cfg.NONLOCAL.INSTANTIATION,
            trans_func_name=cfg.RESNET.TRANS_FUNC,
            stride_1x1=cfg.RESNET.STRIDE_1X1,
            inplace_relu=cfg.RESNET.INPLACE_RELU,
            dilation=cfg.RESNET.SPATIAL_DILATIONS[2],
            norm_module=self.norm_module,
        )

        self.s5 = resnet_helper.ResStage(
            dim_in=[width_per_group * 16],
            dim_out=[width_per_group * 32],
            dim_inner=[dim_inner * 8],
            temp_kernel_sizes=temp_kernel[4],
            stride=cfg.RESNET.SPATIAL_STRIDES[3],
            num_blocks=[d5],
            num_groups=[num_groups],
            num_block_temp_kernel=cfg.RESNET.NUM_BLOCK_TEMP_KERNEL[3],
            nonlocal_inds=cfg.NONLOCAL.LOCATION[3],
            nonlocal_group=cfg.NONLOCAL.GROUP[3],
            nonlocal_pool=cfg.NONLOCAL.POOL[3],
            instantiation=cfg.NONLOCAL.INSTANTIATION,
            trans_func_name=cfg.RESNET.TRANS_FUNC,
            stride_1x1=cfg.RESNET.STRIDE_1X1,
            inplace_relu=cfg.RESNET.INPLACE_RELU,
            dilation=cfg.RESNET.SPATIAL_DILATIONS[3],
            norm_module=self.norm_module,
        )

        if self.enable_detection:
            self.head = head_helper.ResNetRoIHead(
                dim_in=[width_per_group * 32],
                num_classes=cfg.MODEL.NUM_CLASSES,
                pool_size=[None, None]
                    if cfg.MULTIGRID.SHORT_CYCLE else 
                    [[cfg.DATA.NUM_FRAMES // pool_size[0][0], 1, 1]],
                resolution=[[cfg.DETECTION.ROI_XFORM_RESOLUTION] * 2],
                scale_factor=[cfg.DETECTION.SPATIAL_SCALE_FACTOR],
                dropout_rate=cfg.MODEL.DROPOUT_RATE,
                act_func=cfg.MODEL.HEAD_ACT,
                aligned=cfg.DETECTION.ALIGNED,
            )
        else:
            self.head = head_helper.ResNetBasicHead(
                dim_in=[width_per_group * 32],
                num_classes=cfg.MODEL.NUM_CLASSES,
                pool_size=[None, None]
                if cfg.MULTIGRID.SHORT_CYCLE
                else [
                    [
                        cfg.DATA.NUM_FRAMES // pool_size[0][0],
                        cfg.DATA.CROP_SIZE // 32 // pool_size[0][1],
                        cfg.DATA.CROP_SIZE // 32 // pool_size[0][2],
                    ]
                ],  # None for AdaptiveAvgPool3d((1, 1, 1))
                dropout_rate=cfg.MODEL.DROPOUT_RATE,
                act_func=cfg.MODEL.HEAD_ACT,
            )

    def forward(self, x, bboxes=None):
        x = self.s1(x)
        x = self.s2(x)
        for pathway in range(self.num_pathways):
            pool = getattr(self, "pathway{}_pool".format(pathway))
            x[pathway] = pool(x[pathway])
        x = self.s3(x)
        x = self.s4(x)
        x = self.s5(x)
        if self.enable_detection:
            x = self.head(x, bboxes)
        else:
            x = self.head(x)
        return x


@MODEL_REGISTRY.register()
class LightenBackbone(nn.Module):
    """
    Feature generator for lighten
    """

    def __init__(self, cfg):
        """
        The `__init__` method of any subclass should also contain these
            arguments.
        Args:
            cfg (CfgNode): model building configs, details are in the
                comments of the config file.
        """
        super(LightenBackbone, self).__init__()
        self.norm_module = get_norm(cfg)
        self.enable_detection = cfg.DETECTION.ENABLE
        self.enable_detection_hoi = cfg.DETECTION.ENABLE_HOI
        self.enable_toi_pooling = cfg.DETECTION.ENABLE_TOI_POOLING
        self._construct_network(cfg)

    def _construct_network(self, cfg):
        """
        Builds a Feature Generator model.
        Args:
            cfg (CfgNode): model building configs, details are in the
                comments of the config file.
        """
        pool_size = _POOL1[cfg.MODEL.ARCH]
        width_per_group = cfg.RESNET.WIDTH_PER_GROUP
        
        self.in_feat_dim = 1024 # resnext output feature dimension
        self.in_hidden = 512
        self.out_feat_dim = 256 # final feature dimension

        self.resnext101 = torch.hub.load('pytorch/vision:v0.4.0','resnext101_32x8d',pretrained=True)
        self.resnext101 = nn.Sequential(*(list(self.resnext101.children())[:-3]))
        
        if cfg.DETECTION.ENABLE:
            if cfg.DETECTION.ENABLE_HOI:
                if cfg.DETECTION.ENABLE_TOI_POOLING:
                    self.toi_head = head_helper.ResNextRoIHead(
                        dim_in=[
                            width_per_group * 32
                        ],
                        num_classes=cfg.MODEL.NUM_CLASSES,
                        pool_size=pool_size, 
                        resolution=[[cfg.DETECTION.ROI_XFORM_RESOLUTION] * 2] * 2,
                        scale_factor=[cfg.DETECTION.SPATIAL_SCALE_FACTOR] * 2,
                        num_frames=cfg.DATA.NUM_FRAMES,
                        alpha=cfg.SLOWFAST.ALPHA,
                        aligned=cfg.DETECTION.ALIGNED,
                    )

    def forward(self, x, bboxes=None, obj_classes=None, obj_classes_lengths=None, trajectories=None, trajectory_boxes=None):
        # x is input images. E.g.,
        # (Pdb) x.shape
        # torch.Size([batch_size, 3, 8, 224, 224])
        
        batch_size = x.shape[0]
        num_frames = x.shape[2]
        feat_dim = 1024
        x = x.permute(0,2,1,3,4).reshape((batch_size*num_frames, x.shape[1], x.shape[3], x.shape[4]))
        
        import pdb; pdb.set_trace()        
        
        x = self.resnext101(x)
        # (Pdb) x.shape
        # torch.Size([batch_size*8, 1024, 14, 14])

        x = x.reshape((batch_size, num_frames, feat_dim, x.shape[2], x.shape[3])).permute(0,2,1,3,4)
        
        # (Pdb) x.shape
        # torch.Size([batch_size, 1024, 8, 14, 14])
        # x passed as list because toi_head follows pathway format
        feats = self.toi_head([x], bboxes, trajectory_boxes)
        
        """
        if self.enable_detection:
            if self.enable_detection_hoi:
                toi_pooled_features = self.toi_head(x, bboxes, trajectory_boxes) if self.enable_toi_pooling else None # need to go before x = self.head(x)
                #import pdb; pdb.set_trace()
                #x = self.head(x) # head only does temporal pooling of features without roi pooling
                # len(x)
                # 16(batch_size)
                # x[0].shape
                # torch.Size([2304,14,14])
                x = self.hoi_head(x, bboxes, obj_classes, obj_classes_lengths, action_labels, gt_obj_classes, gt_obj_classes_lengths, trajectories, human_poses, toi_pooled_features, trajectory_boxes, skeleton_imgs, trajectory_box_masks)
            else:
                x = self.head(x, bboxes)
        else:
            x = self.head(x)
        """
        return feats
@MODEL_REGISTRY.register()
class Lighten(nn.Module):

    def __init__(self, cfg):
        """
        The `__init__` method of any subclass should also contain these
            arguments.
        Args:
            cfg (CfgNode): model building configs, details are in the
                comments of the config file.
        """
        super(Lighten, self).__init__()
       # self.norm_module = get_norm(cfg)
        self.enable_detection = cfg.DETECTION.ENABLE
        self.enable_detection_hoi = cfg.DETECTION.ENABLE_HOI
        self.enable_toi_pooling = cfg.DETECTION.ENABLE_TOI_POOLING
        self.num_pathways = 1
        self.use_saved_feat = cfg.MODEL.USE_SAVED_FEAT
        self._construct_network(cfg)
        
        #init_helper.init_weights(
        #    self, cfg.MODEL.FC_INIT_STD, cfg.RESNET.ZERO_INIT_FINAL_BN, cfg
        #)

    def _construct_network(self, cfg):
        """
        Builds a SlowFast model. The first pathway is the Slow pathway and the
            second pathway is the Fast pathway.
        Args:
            cfg (CfgNode): model building configs, details are in the
                comments of the config file.
        """
        assert cfg.MODEL.ARCH in _POOL1.keys()
        pool_size = _POOL1[cfg.MODEL.ARCH]
        assert len({len(pool_size), self.num_pathways}) == 1
        #assert cfg.RESNET.DEPTH in _MODEL_STAGE_DEPTH.keys()
        
        num_groups = cfg.RESNET.NUM_GROUPS
        width_per_group = cfg.RESNET.WIDTH_PER_GROUP
        dim_inner = num_groups * width_per_group
        out_dim_ratio = (
            cfg.SLOWFAST.BETA_INV // cfg.SLOWFAST.FUSION_CONV_CHANNEL_RATIO
        )
        # RESNEXT OUT FEAT DIM
        self.dim_in = 1024
        
        if not self.use_saved_feat and cfg.LIGHTEN.BACKBONE == 'resnext101':
            self.backbone = torch.hub.load('pytorch/vision:v0.4.0','resnext101_32x8d',pretrained=True)
            self.backbone = nn.Sequential(*(list(self.backbone.children())[:-3]))

        if not self.use_saved_feat and cfg.LIGHTEN.FREEZE_BACKBONE:
            for param in self.backbone.parameters():
                param.requires_grad = False

        if cfg.DETECTION.ENABLE:
            if cfg.DETECTION.ENABLE_HOI:
                if cfg.DETECTION.ENABLE_TOI_POOLING:
                    self.toi_head = head_helper.ResNetToIHead(
                        dim_in=[
                            self.dim_in,
                            #width_per_group * 32, # mention feature dim here
                            #width_per_group * 32 // cfg.SLOWFAST.BETA_INV,
                        ],
                        num_classes=cfg.MODEL.NUM_CLASSES,
                        pool_size=[None] # changed temporal pooling kernel size
                        if cfg.MULTIGRID.SHORT_CYCLE else 
                        [
                            [
                                13,
                                #cfg.DATA.NUM_FRAMES
                                #// cfg.SLOWFAST.ALPHA
                                #// pool_size[0][0],
                                1,
                                1,
                            ],
                        ],
                        resolution=[[cfg.DETECTION.ROI_XFORM_RESOLUTION] * 2] * 2,
                        scale_factor=[cfg.DETECTION.SPATIAL_SCALE_FACTOR] * 2,
                        num_frames=cfg.DATA.NUM_FRAMES,
                        alpha=cfg.SLOWFAST.ALPHA,
                        aligned=cfg.DETECTION.ALIGNED,
                    )
                self.head = head_helper.ResNetPoolHead(
                    dim_in=[
                        self.dim_in,
                        #width_per_group * 32,
                    ],
                    num_classes=cfg.MODEL.NUM_CLASSES,
                    pool_size=[None]
                    if cfg.MULTIGRID.SHORT_CYCLE else 
                    [
                        [
                            cfg.DATA.NUM_FRAMES
                            // cfg.SLOWFAST.ALPHA
                            // pool_size[0][0],
                            1,
                            1,
                        ],
                    ],
                )
                #self.hoi_head = head_helper.LightenHOIHead(cfg, 
                #    resolution=[cfg.DETECTION.ROI_XFORM_RESOLUTION] * 2, scale_factor=cfg.DETECTION.SPATIAL_SCALE_FACTOR, aligned=cfg.DETECTION.ALIGNED)
                if cfg.MODEL.USE_3D_CONV:
                    self.hoi_head = head_helper.Conv3dHOIHead(cfg,
                            resolution=[cfg.DETECTION.ROI_XFORM_RESOLUTION] * 2, scale_factor=cfg.DETECTION.SPATIAL_SCALE_FACTOR, aligned=cfg.DETECTION.ALIGNED)
                    
                elif cfg.MODEL.USE_DEDICATED_RNN:
                    self.hoi_head = head_helper.VanillaDedicatedRnnHOIHead(cfg,
                            resolution=[cfg.DETECTION.ROI_XFORM_RESOLUTION] * 2, scale_factor=cfg.DETECTION.SPATIAL_SCALE_FACTOR, aligned=cfg.DETECTION.ALIGNED)
                else:
                    self.hoi_head = head_helper.VanillaHOIHead(cfg,
                            resolution=[cfg.DETECTION.ROI_XFORM_RESOLUTION] * 2, scale_factor=cfg.DETECTION.SPATIAL_SCALE_FACTOR, aligned=cfg.DETECTION.ALIGNED)
            else:
                self.head = head_helper.ResNetRoIHead(
                    dim_in=[
                        self.dim_in,
                        #width_per_group * 32,
                    ],
                    num_classes=cfg.MODEL.NUM_CLASSES,
                    pool_size=[
                        [
                            cfg.DATA.NUM_FRAMES
                            // cfg.SLOWFAST.ALPHA
                            // pool_size[0][0],
                            1,
                            1,
                        ],
                    ],
                    resolution=[[cfg.DETECTION.ROI_XFORM_RESOLUTION] * 2] * 2,
                    scale_factor=[cfg.DETECTION.SPATIAL_SCALE_FACTOR] * 2,
                    dropout_rate=cfg.MODEL.DROPOUT_RATE,
                    act_func=cfg.MODEL.HEAD_ACT,
                    aligned=cfg.DETECTION.ALIGNED,
                )
        else:
            self.head = head_helper.ResNetBasicHead(
                dim_in=[
                    self.dim_in,
                    #width_per_group * 32,
                ],
                num_classes=cfg.MODEL.NUM_CLASSES,
                pool_size=[None, None]
                if cfg.MULTIGRID.SHORT_CYCLE
                else [
                    [
                        cfg.DATA.NUM_FRAMES
                        // cfg.SLOWFAST.ALPHA
                        // pool_size[0][0],
                        cfg.DATA.CROP_SIZE // 32 // pool_size[0][1],
                        cfg.DATA.CROP_SIZE // 32 // pool_size[0][2],
                    ],
                    [
                        cfg.DATA.NUM_FRAMES // pool_size[1][0],
                        cfg.DATA.CROP_SIZE // 32 // pool_size[1][1],
                        cfg.DATA.CROP_SIZE // 32 // pool_size[1][2],
                    ],
                ],  # None for AdaptiveAvgPool3d((1, 1, 1))
                dropout_rate=cfg.MODEL.DROPOUT_RATE,
                act_func=cfg.MODEL.HEAD_ACT,
            )

    def forward(self, x, bboxes=None, obj_classes=None, obj_classes_lengths=None, action_labels=None, gt_obj_classes=None, gt_obj_classes_lengths=None, trajectories=None, human_poses=None, trajectory_boxes=None, skeleton_imgs=None, trajectory_box_masks=None, heatmaps=None):
        # x is a list of features. E.g.,
        # (Pdb) x[0].shape
        # torch.Size([batch_size, 3, 8, 224, 224])
        # (Pdb) trajectory_boxes.shape
        # torch.Size([17, 129])

        batch_size = x[0].shape[0]
        num_frames = x[0].shape[2]
        feat_dim = 1024
        
        if not self.use_saved_feat:
            x[0] = x[0].permute(0,2,1,3,4).reshape(batch_size*num_frames, x[0].shape[1], x[0].shape[3], x[0].shape[4])
            x[0] = self.backbone(x[0])
            W = x[0].shape[-1]
            H = x[0].shape[-2]
            x[0] = x[0].reshape(batch_size, num_frames, feat_dim, H, W).permute(0,2,1,3,4)
            # (Pdb) x[0].shape
            # torch.Size([batch_size, 1024, 8, 14, 14])
        
        # import pdb; pdb.set_trace()
        if self.enable_detection:
            if self.enable_detection_hoi:
                roi_pooled_features = self.toi_head(x, bboxes, trajectory_boxes) if self.enable_toi_pooling else None # need to go before x = self.head(x)
                # (Pdb) roi_pooled_features.shape
                # torch.Size([17, 1024, 8, 7, 7])->17=len of seq. of objs

                # x = self.head(x) # head only does temporal pooling of features without roi pooling
                # len(x)
                # 16(batch_size)
                # x[0].shape
                # torch.Size([2304,14,14])
                x = self.hoi_head(roi_pooled_features, x, bboxes, obj_classes, obj_classes_lengths, action_labels, gt_obj_classes, gt_obj_classes_lengths, trajectories, human_poses, trajectory_boxes, skeleton_imgs, trajectory_box_masks, heatmaps)
            else:
                x = self.head(x, bboxes)
        else:
            x = self.head(x)
        
        return x