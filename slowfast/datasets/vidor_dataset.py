#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import logging
import numpy as np
import torch
import json
import os
import pandas as pd#added by poorvi

from . import vidor_helper as vidor_helper
from . import cv2_transform as cv2_transform
from . import transform as transform
from . import utils as utils
from .build import DATASET_REGISTRY
from .image import draw_gaussian

logger = logging.getLogger(__name__)

import cv2

@DATASET_REGISTRY.register()
class Vidor(torch.utils.data.Dataset):
    """
    VidOR Dataset
    """

    def __init__(self, cfg, split):
        self.cfg = cfg
        self._split = split
        self._sample_rate = cfg.DATA.SAMPLING_RATE
        self._video_length = cfg.DATA.NUM_FRAMES
        self._seq_len = self._video_length * self._sample_rate
        self._num_classes = cfg.MODEL.NUM_CLASSES
        # Augmentation params.
        self._data_mean = cfg.DATA.MEAN
        self._data_std = cfg.DATA.STD
        self._use_bgr = cfg.VIDOR.BGR
        self.random_horizontal_flip = cfg.DATA.RANDOM_FLIP
        if self._split == "train":
            self._crop_size = cfg.DATA.TRAIN_CROP_SIZE
            self._jitter_min_scale = cfg.DATA.TRAIN_JITTER_SCALES[0]
            self._jitter_max_scale = cfg.DATA.TRAIN_JITTER_SCALES[1]
            self._use_color_augmentation = cfg.VIDOR.TRAIN_USE_COLOR_AUGMENTATION
            self._pca_jitter_only = cfg.VIDOR.TRAIN_PCA_JITTER_ONLY
            self._pca_eigval = cfg.VIDOR.TRAIN_PCA_EIGVAL
            self._pca_eigvec = cfg.VIDOR.TRAIN_PCA_EIGVEC
        else:
            self._crop_size = cfg.DATA.TEST_CROP_SIZE
            self._test_force_flip = cfg.VIDOR.TEST_FORCE_FLIP

        # settings for Baseline
        self.is_baseline = True if cfg.MODEL.ARCH == 'baseline' else False
        self.multigrid_enabled = cfg.MULTIGRID.ENABLE
        
        if self.cfg.MODEL.USE_TRAJECTORIES:
            if split == 'train':
                self.trajectories_path = cfg.VIDOR.TRAIN_GT_TRAJECTORIES 
            elif self.cfg.VIDOR.TEST_WITH_GT: # testing with GT
                self.trajectories_path = cfg.VIDOR.TEST_GT_TRAJECTORIES
            else: # testing with detected boxes
                self.trajectories_path = cfg.VIDOR.TEST_TRAJECTORIES

        if self.cfg.MODEL.USE_SPA_CONF or self.cfg.MODEL.USE_ALPHA_POSES:
            self.human_poses_path = 'human_poses' if split == 'train' or self.cfg.VIDOR.TEST_WITH_GT else 'human_poses_detected-bboxes'
            # self.heatmap_size = cfg.MODEL.SPA_CONF_HEATMAP_SIZE
            self.skeletons = [[16, 14], [14, 12], [17, 15], [15, 13], [12, 13], [6, 12],
                              [7, 13], [6, 7], [6, 8], [7, 9], [8, 10], [9, 11], [2, 3], 
                              [1, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 7]]
        elif self.cfg.MODEL.USE_HUMAN_POSES:
            self.human_poses_path = cfg.VIDOR.TRAIN_GT_HUMAN_POSES if split == 'train' else cfg.VIDOR.TEST_GT_HUMAN_POSES
       
        '''
        if self.cfg.MODEL.USE_SAVED_FEAT:
            self._feat_file = None
            self._feat = None
        '''
        self._load_data(cfg)

    def _load_data(self, cfg):
        """
        Load frame paths and annotations from files

        Args:
            cfg (CfgNode): config
        """
        if not self.cfg.MODEL.USE_SAVED_FEAT:
            # Load frame trajectories
            if self.cfg.MODEL.USE_TRAJECTORIES:
                if cfg.VIDOR.TEST_DEBUG:
                    print('Loading trajectories...')
                with open(os.path.join(cfg.VIDOR.ANNOTATION_DIR, self.trajectories_path), 'r') as f:
                    self._trajectories = json.load(f)

        # Load human pose features (theta; 72 points)
        if self.cfg.MODEL.USE_HUMAN_POSES:
            if cfg.VIDOR.TEST_DEBUG:
                print('Loading human poses...')
            import pickle
            with open(os.path.join(cfg.VIDOR.ANNOTATION_DIR, self.human_poses_path), 'rb') as f:
                self._human_poses = pickle.load(f)
        elif self.cfg.MODEL.USE_SPA_CONF or self.cfg.MODEL.USE_ALPHA_POSES:
            if cfg.VIDOR.TEST_DEBUG:
                print('Loading human poses for spatial configuration module...')
            self._human_poses_root = os.path.join(cfg.VIDOR.ANNOTATION_DIR, self.human_poses_path)
        
        # Loading frame paths.
        (
            self._image_paths, # image paths of all the videos
            self._video_idx_to_name, # in this we will all video index to name mapping. So this order will be fixed
        ) = vidor_helper.load_image_lists(cfg, is_train=(self._split == "train"))
        #self._image_paths(list of list): top-level list is for each video
        #self._image_paths[0]: has all image paths for that video
        #self._video_idx_to_name(list of string): contains video_ids

        # Loading annotations for boxes and labels.
        self._instances = vidor_helper.load_boxes_and_labels(
            cfg, mode=self._split
            )
        #self._instances(dict)
        #self._instances.keys(): video_id 
        #dict_keys(['0085/7002697331', '0085/6885773487', '0085/2851336169', '0085/3001415728'])
        #self._instances['0085/7002697331'].keys(): keyframe timestamp stored in middle_frame_timestamp
        #dict_keys([1, 2, 3, 4, 7, 8, 9, 10, 11, 13, 26, 27, 28])

        # Get indices of keyframes and corresponding boxes and labels.
        (
            self._keyframe_indices,
            self._keyframe_boxes_and_labels,
        ) = vidor_helper.get_keyframe_data(cfg, self._instances, mode=self._split)
        #self._keyframe_indices(list):len->keyframe_count
        #self._keyframe_boxes_and_labels(list):len->video_count
        #sum([len(self._keyframe_boxes_and_labels[i]) for i in range(video_count))]) = keyframe_count
        #self._keyframe_indices[1]:(0, 1, 2, 60, '0085/7002697331')
        #(video_idx_in_instances, sec_idx, sec_of_keyframe, sec*fps, video_id)
        # import pdb; pdb.set_trace()

        # Calculate the number of used boxes.
        self._num_boxes_used = vidor_helper.get_num_boxes_used(
            self._keyframe_indices, self._keyframe_boxes_and_labels
        )
        
        # Calculate label weights for loss function
        if self.cfg.MODEL.USE_LABEL_WEIGHTS:
            import pickle
            with open(os.path.join(cfg.VIDOR.ANNOTATION_DIR, cfg.VIDOR.LABEL_FREQ_FILE),'rb') as f:
                _label_freq = pickle.load(f)

            self._label_pos_weights = self._compute_pos_weights(_label_freq)

        self.print_summary()
        
        def debug(idx):
            pass

        if cfg.VIDOR.TEST_DEBUG:
            debug(0)
            # pass

    def print_summary(self):
        logger.info("=== VidOR dataset summary ===")
        logger.info("Split: {}".format(self._split))
        logger.info("Number of videos: {}".format(len(self._image_paths)))
        total_frames = sum(
            len(video_img_paths) for video_img_paths in self._image_paths
        )
        logger.info("Number of frames: {}".format(total_frames))
        logger.info("Number of key frames: {}".format(len(self)))
        logger.info("Number of boxes: {}.".format(self._num_boxes_used))

    def __len__(self):
        return len(self._keyframe_indices)
    
    def _compute_pos_weights(self, label_freq):
        """
        To compute label positive weights for loss function
        Args:
            label_freq (list): gt label freq. 
            len (label_freq) -> 50
        Returns:
            pos_weights (tensor): positive label weights
        """
        total_interaction = sum(label_freq)
        negatives_freq = [total_interaction - x for x in label_freq]
        pos_weights = [y/x for x,y in zip(label_freq, negatives_freq) ]
        pos_weights = torch.FloatTensor(pos_weights)
        return pos_weights

    
    def _images_and_boxes_preprocessing_cv2(self, imgs, boxes, gt_boxes=None, min_scale=None, crop_size=None, n_imgs=0):
        """
        This function performs preprocessing for the input images and
        corresponding boxes for one clip with opencv as backend.

        Args:
            imgs (tensor): the images.
            boxes (ndarray): the (proposal) boxes for the current clip.
            gt_boxes: the ground truth boxes for the current clip.

        Returns:
            imgs (tensor): list of preprocessed images.
            boxes (ndarray): preprocessed boxes.
        """

        height, width, _ = imgs[0].shape

        # boxes[:, [0, 2]] *= width
        # boxes[:, [1, 3]] *= height
        boxes = cv2_transform.clip_boxes_to_image(boxes, height, width)

        # `transform.py` is list of np.array. However, for AVA, we only have
        # one np.array.
        boxes = [boxes]

        crop_size = crop_size if self.multigrid_enabled and crop_size is not None else self._crop_size
        
        if self._split != 'train':
            assert gt_boxes is not None
            gt_boxes = cv2_transform.clip_boxes_to_image(gt_boxes, height, width)
            gt_boxes = [gt_boxes]
        
        # The image now is in HWC, BGR format.
        if self.cfg.MODEL.SAVE_FEAT:
            # Short side to test_scale. Non-local and STRG uses 256.
            #import pdb; pdb.set_trace()
            if self._split == "train":
                # imgs = [cv2_transform.scale(crop_size, img) for img in imgs]
                # boxes, gt_boxes = cv2_transform.scale_boxes(crop_size, boxes[0], height, width, gt_boxes=None)
                imgs, boxes = cv2_transform.center_crop_fixed(
                    crop_size, imgs, boxes
                )
                #import pdb; pdb.set_trace()
                # gt_boxes should be none for train
                # assert gt_boxes is None
                # boxes = boxes
                #import pdb; pdb.set_trace()
            
            elif self._split == "val" or self._split == "test":
                imgs = [cv2_transform.scale(crop_size, img) for img in imgs]
                boxes, gt_boxes = cv2_transform.scale_boxes(
                        crop_size, boxes[0], height, width, gt_boxes=gt_boxes[0]
                )
                boxes, gt_boxes = [boxes], [gt_boxes]

        elif self._split == "train":  # "train"
           # import pdb; pdb.set_trace()
            imgs, boxes = cv2_transform.random_short_side_scale_jitter_list(
                imgs,
                min_size=self._jitter_min_scale if not self.multigrid_enabled and min_scale is None else min_scale,
                max_size=self._jitter_max_scale,
                boxes=boxes,
            )
            imgs, boxes = cv2_transform.random_crop_list(
                imgs, crop_size, order="HWC", boxes=boxes, n_imgs=n_imgs, USE_SPA_CONF=self.cfg.MODEL.USE_SPA_CONF
            )
            if self.random_horizontal_flip:
                # random flip
                # if self.cfg.MODEL.USE_SPA_CONF and len(imgs[n_imgs].shape) != 3:
                #     for i in range(n_imgs, len(imgs) + 1):
                #         imgs[i] = np.expand_dims(imgs[i], axis=-1)
                imgs, boxes = cv2_transform.horizontal_flip_list(
                    0.5, imgs, order="HWC", boxes=boxes, 
                    n_imgs=n_imgs, USE_SPA_CONF=self.cfg.MODEL.USE_SPA_CONF
                )
            # elif self._split == "val":
            #     # Short side to test_scale. Non-local and STRG uses 256.
            #     imgs = [cv2_transform.scale(self._crop_size, img) for img in imgs]
            #     boxes, gt_boxes = cv2_transform.scale_boxes(
            #             self._crop_size, boxes[0], height, width, gt_boxes=gt_boxes[0]
            #     )
            #     boxes, gt_boxes = [boxes], [gt_boxes]
            #     imgs, boxes, gt_boxes = cv2_transform.spatial_shift_crop_list(
            #         self._crop_size, imgs, 1, boxes=boxes, gt_boxes=gt_boxes
            #     )

            #     if self._test_force_flip:
            #         imgs, boxes = cv2_transform.horizontal_flip_list(
            #             1, imgs, order="HWC", boxes=boxes, gt_boxes=gt_boxes
            #         )
        elif self._split == "val" or self._split == "test":
            # Short side to test_scale. Non-local and STRG uses 256.
            #import pdb; pdb.set_trace()
            imgs = [cv2_transform.scale(crop_size, img) for img in imgs]
            boxes, gt_boxes = cv2_transform.scale_boxes(
                    crop_size, boxes[0], height, width, gt_boxes=gt_boxes[0]
            )
            boxes, gt_boxes = [boxes], [gt_boxes]

            if self._test_force_flip:
                # if self.cfg.MODEL.USE_SPA_CONF and len(imgs[n_imgs].shape) != 3:
                #     imgs[i] = np.expand_dims(imgs[i], axis=-1)
                    # imgs[n_imgs:] = [np.expand_dims(img, axis=-1) for img in imgs[n_imgs:]]
                imgs, boxes = cv2_transform.horizontal_flip_list(
                    1, imgs, order="HWC", boxes=boxes, gt_boxes=gt_boxes,
                    n_imgs=n_imgs, USE_SPA_CONF=self.cfg.MODEL.USE_SPA_CONF
                )
        else:
            raise NotImplementedError(
                "Unsupported split mode {}".format(self._split)
            )

        # Convert image to CHW keeping BGR order.
        if self.cfg.MODEL.USE_SPA_CONF:
            try:
                if len(imgs[n_imgs].shape) == 2:
                    imgs[n_imgs:] = [np.expand_dims(img, axis=-1) for img in imgs[n_imgs:]]
                elif len(imgs[n_imgs].shape) > 3:
                    imgs[n_imgs:] = [np.expand_dims(img.squeeze(), axis=-1) for img in imgs[n_imgs:]]
            except:
                import pdb; pdb.set_trace()
                
            # for i in range(n_imgs, len(imgs) + 1):
            #     imgs[i] = np.expand_dims(imgs[i], axis=-1)
        # try:
        imgs = [cv2_transform.HWC2CHW(img) for img in imgs]
        # except:
        #     print('imgs[n_imgs].shape:', imgs[n_imgs].shape)
        #     print('len(imgs):', len(imgs))
        #     print('n_imgs:', n_imgs)
        #     import pdb; pdb.set_trace()

        # Image [0, 255] -> [0, 1].
        if self.cfg.MODEL.USE_SPA_CONF:
            imgs[:n_imgs] = [img / 255.0 for img in imgs[:n_imgs]]
        else:    
            imgs = [img / 255.0 for img in imgs]

        if self.cfg.MODEL.USE_SPA_CONF:
            imgs[:n_imgs] = [
                np.ascontiguousarray(
                    img.reshape((3, imgs[0].shape[1], imgs[0].shape[2]))
                ).astype(np.float32)
                for img in imgs[:n_imgs]
            ]
            imgs[n_imgs:] = [
                np.ascontiguousarray(
                    img.reshape((1, imgs[0].shape[1], imgs[0].shape[2]))
                ).astype(np.float32)
                for img in imgs[n_imgs:]
            ]
        else:
            imgs = [
                np.ascontiguousarray(
                    # img.reshape((3, self._crop_size, self._crop_size))
                    img.reshape((3, imgs[0].shape[1], imgs[0].shape[2]))
                ).astype(np.float32)
                for img in imgs
            ]

        # Do color augmentation (after divided by 255.0).
        if self.cfg.MODEL.USE_SPA_CONF:
            skeleton_imgs = imgs[n_imgs:]
            imgs = imgs[:n_imgs]
        if self._split == "train" and self._use_color_augmentation: # False
            if not self._pca_jitter_only:
                imgs = cv2_transform.color_jitter_list(
                    imgs,
                    img_brightness=0.4,
                    img_contrast=0.4,
                    img_saturation=0.4,
                )

            imgs = cv2_transform.lighting_list(
                imgs,
                alphastd=0.1,
                eigval=np.array(self._pca_eigval).astype(np.float32),
                eigvec=np.array(self._pca_eigvec).astype(np.float32),
            )

        # Normalize images by mean and std.
        imgs = [
            cv2_transform.color_normalization(
                img,
                np.array(self._data_mean, dtype=np.float32),
                np.array(self._data_std, dtype=np.float32),
            )
            for img in imgs
        ]

        # Concat list of images to single ndarray.
        imgs = np.concatenate(
            [np.expand_dims(img, axis=1) for img in imgs], axis=1
        )

        if not self._use_bgr:
            # Convert image format from BGR to RGB.
            imgs = imgs[::-1, ...]

        imgs = np.ascontiguousarray(imgs)
        imgs = torch.from_numpy(imgs)
        
        if self.cfg.MODEL.USE_SPA_CONF:
            skeleton_imgs = np.concatenate(
                [np.expand_dims(img, axis=1) for img in skeleton_imgs], axis=1
            )
            skeleton_imgs = np.ascontiguousarray(skeleton_imgs)
            skeleton_imgs = torch.from_numpy(skeleton_imgs)

        boxes = cv2_transform.clip_boxes_to_image(
            boxes[0], imgs[0].shape[1], imgs[0].shape[2]
        )
        if gt_boxes is not None:
            gt_boxes = cv2_transform.clip_boxes_to_image(
                gt_boxes[0], imgs[0].shape[1], imgs[0].shape[2]
            )
        if self.cfg.MODEL.USE_SPA_CONF:
            return (imgs, skeleton_imgs, boxes) if gt_boxes is None else (imgs, skeleton_imgs, boxes, gt_boxes)
        else:
            return (imgs, boxes) if gt_boxes is None else (imgs, boxes, gt_boxes)

    def _images_and_boxes_preprocessing(self, imgs, boxes, gt_boxes=None):
        """
        This function performs preprocessing for the input images and
        corresponding boxes for one clip.

        Args:
            imgs (tensor): the images.
            boxes (ndarray): the boxes for the current clip.

        Returns:
            imgs (tensor): list of preprocessed images.
            boxes (ndarray): preprocessed boxes.
        """
        # Image [0, 255] -> [0, 1].
        imgs = imgs.float()
        imgs = imgs / 255.0

        height, width = imgs.shape[2], imgs.shape[3]
        # The format of boxes is [x1, y1, x2, y2]. The input boxes are in the
        # range of [0, 1].
        # boxes[:, [0, 2]] *= width
        # boxes[:, [1, 3]] *= height
        boxes = transform.clip_boxes_to_image(boxes, height, width)

        if self._split == "train":
            # Train split
            import pdb; pdb.set_trace()
            imgs, boxes = transform.random_short_side_scale_jitter(
                imgs,
                min_size=self._jitter_min_scale,
                max_size=self._jitter_max_scale,
                boxes=boxes,
            )# images are cropped in next step
            imgs, boxes = transform.random_crop(
                imgs, self._crop_size, boxes=boxes
            )

            # Random flip.
            imgs, boxes = transform.horizontal_flip(0.5, imgs, boxes=boxes)
        elif self._split == "val":
            # Val split
            # Resize short side to crop_size. Non-local and STRG uses 256.
            imgs, boxes = transform.random_short_side_scale_jitter(
                imgs,
                min_size=self._crop_size,
                max_size=self._crop_size,
                boxes=boxes,
            )

            # Apply center crop for val split
            imgs, boxes = transform.uniform_crop(
                imgs, size=self._crop_size, spatial_idx=1, boxes=boxes
            )
            
            #import pdb; pdb.set_trace()

            if self._test_force_flip:
                imgs, boxes = transform.horizontal_flip(1, imgs, boxes=boxes)
        elif self._split == "test":
            # Test split
            # Resize short side to crop_size. Non-local and STRG uses 256.
            imgs, boxes = transform.random_short_side_scale_jitter(
                imgs,
                min_size=self._crop_size,
                max_size=self._crop_size,
                boxes=boxes,
            )
            
            import pdb; pdb.set_trace()

            if self._test_force_flip:
                imgs, boxes = transform.horizontal_flip(1, imgs, boxes=boxes)
        else:
            raise NotImplementedError(
                "{} split not supported yet!".format(self._split)
            )

        # Do color augmentation (after divided by 255.0).
        if self._split == "train" and self._use_color_augmentation:
            if not self._pca_jitter_only:
                imgs = transform.color_jitter(
                    imgs,
                    img_brightness=0.4,
                    img_contrast=0.4,
                    img_saturation=0.4,
                )

            imgs = transform.lighting_jitter(
                imgs,
                alphastd=0.1,
                eigval=np.array(self._pca_eigval).astype(np.float32),
                eigvec=np.array(self._pca_eigvec).astype(np.float32),
            )

        # Normalize images by mean and std.
        imgs = transform.color_normalization(
            imgs,
            np.array(self._data_mean, dtype=np.float32),
            np.array(self._data_std, dtype=np.float32),
        )

        if not self._use_bgr:
            # Convert image format from BGR to RGB.
            # Note that Kinetics pre-training uses RGB!
            imgs = imgs[:, [2, 1, 0], ...]

        boxes = transform.clip_boxes_to_image(
            boxes, self._crop_size, self._crop_size
        )

        return imgs, boxes

    def get_box_from_pose(self, pose, offset=10):
        '''
            pose: 17 x 2
            returns bbox: 4
        '''
        xmin, ymin = pose.min(0)
        xmax, ymax = pose.max(0)
        return np.array([xmin-offset, ymin-offset, 
                         xmax+offset, ymax+offset])

    def draw_heatmaps(self, human_pose, bbox=None, map_size=7):
        '''
            Takes a 17 x 2 human pose, the bbox of the person, and the map_size.
            The pose is first resized so that it's between [56,56]. Here I'm assuming 
            the bboxes are squares
        '''
        human_pose = np.array(human_pose)
        if bbox is None:
            bbox = self.get_box_from_pose(human_pose)

        n_joints = human_pose.shape[0]
        box_w, box_h = bbox[2] - bbox[0], bbox[3] - bbox[1]
        human_pose = map_size * (human_pose - bbox[:2]) / np.array([box_w, box_h]) 
        ret = np.zeros((n_joints, map_size, map_size))
        for j in range(n_joints):
           ret[j] = draw_gaussian(ret[j], human_pose[j], sigma=1)

        return ret


    def draw_human_skeleton(self, human_pose, orig_width, orig_height):
        human_pose = np.array(human_pose)
        ret = np.zeros((orig_height, orig_width))
        cur_kps = np.zeros((17, 2), dtype=np.int)
        cur_kps[:, 0] = (human_pose[:, 0]).astype(np.int)
        cur_kps[:, 1] = (human_pose[:, 1]).astype(np.int)

        for j, sk in enumerate(self.skeletons):
            sk0 = sk[0] - 1
            sk1 = sk[1] - 1
            ret = cv2.line(ret, tuple(cur_kps[sk0]), tuple(cur_kps[sk1]), 0.05 * (j + 1), 1)
        
        return ret

    # need a map: {0: [1, 4, 59, 198,], ... rare_class_id: [10004]}
    # have a map for each video: {class_id: freq}
    # have a map: 
    # Let's say you have and idx class_i: idx % (max_num_vids)
    # num_classes is 50 <<< batch size (4)
    # Soln: Randomize which classes to choose from
    # Randomly select a class or just do idx % 50
    # For that class, fetch the video using idx % max_num_vids
    # 
    def __getitem__(self, idx):
        """
        Generate corresponding clips, boxes, labels and metadata for given idx.

        Args:
            idx (int): the video index provided by the pytorch sampler.
        Returns:
            frames (tensor): the frames of sampled from the video. The dimension
                is `channel` x `num frames` x `height` x `width`.
            label (ndarray): the label for correspond boxes for the current video.
            idx (int): the video index provided by the pytorch sampler.
            extra_data (dict): a dict containing extra data fields, like "boxes",
                "ori_boxes" and "metadata".
        """
        if self.multigrid_enabled:
            short_cycle_idx = None
            # When short cycle is used, input index is a tupple.
            if isinstance(idx, tuple):
                idx, short_cycle_idx = idx
            
            if self._split == 'train':
                crop_size = self.cfg.DATA.TRAIN_CROP_SIZE
                min_scale = self.cfg.DATA.TRAIN_JITTER_SCALES[0]

                if short_cycle_idx in [0, 1]:
                    crop_size = int(
                        round(
                            self.cfg.MULTIGRID.SHORT_CYCLE_FACTORS[short_cycle_idx]
                            * self.cfg.MULTIGRID.DEFAULT_S
                        )
                    )
                
                if self.cfg.MULTIGRID.DEFAULT_S > 0:
                    # Decreasing the scale is equivalent to using a larger "span"
                    # in a sampling grid.
                    min_scale = int(
                        round(
                            float(min_scale)
                            * crop_size
                            / self.cfg.MULTIGRID.DEFAULT_S
                        )
                    )
                
                self._sample_rate = utils.get_random_sampling_rate(
                    self.cfg.MULTIGRID.LONG_CYCLE_SAMPLING_RATE,
                    self.cfg.DATA.SAMPLING_RATE,
                )
                self._seq_len = self._video_length * self._sample_rate
            elif self._split in ['val', 'test']:
                min_scale = crop_size = None
            else:
                raise NotImplementedError(
                    "Does not support {} mode".format(self.mode)
                )
        else:
            min_scale = crop_size = None

        #import pdb; pdb.set_trace()
        #center_idx = sec*fps
        #added by poorvi
        # act_vid_map = pd.read_pickle(r'/mnt/data/apoorva/vidor-dataset/vidHOI/train/act_vid_map.pkl')#action: array of videos in which it appears
        # vid_act_count_map = pd.read_pickle(r'/mnt/data/apoorva/vidor-dataset/vidHOI/train/vid_act_count_map.pkl')#video: dictionary(action:count....)
        # vid_act_map = pd.read_pickle(r'/mnt/data/apoorva/vidor-dataset/vidHOI/train/vid_act_map.pkl')#video: action list...)

        # action_num = idx % 50 #random action selected
        # video_id_list =act_vid_map[action_num] #list of videos in which the action occurs

        # max_videos = len(video_id_list) 
        # video_num = idx % max_videos
        # video_index = video_id_list[video_num]
        # res = list(vid_act_map.keys()).index(video_index)
        # #print(res)
        # new_idx=res%50
        #added by poorvi

        video_idx, sec_idx, sec, center_idx, orig_video_idx = self._keyframe_indices[idx]#[idx]

        #video_idx, sec_idx, sec, center_idx, orig_video_idx = self._keyframe_indices[idx]
        assert orig_video_idx == '/'.join(self._image_paths[video_idx][0].split('.')[0].split('/')[-3:-1])
        # Get the frame idxs for current clip.
        # frame sampling done here
        # out of 64 frame 32 frames are sampled out
        # out of 32, 16 frames from both sides of center_idx are taken
        if self.is_baseline:
            num_frames = len(self._image_paths[video_idx])
            if center_idx < 0:
                center_idx = 0
                # seq = [0]
            elif center_idx >= num_frames:
                center_idx = num_frames - 1
                # seq = [num_frames - 1]
            # else:
            seq = [center_idx]
        else:
            seq = utils.get_sequence(
                center_idx, #sec*fps
                self._seq_len // 2, #64//2=32 -> half_len
                self._sample_rate, #2
                num_frames=len(self._image_paths[video_idx]),
            )
        
        #import pdb; pdb.set_trace()

        clip_label_list = self._keyframe_boxes_and_labels[video_idx][sec_idx]
        assert len(clip_label_list) > 0

        # Get boxes and labels for current clip.
        boxes = np.array(clip_label_list[0])
        obj_classes = np.array(clip_label_list[1])
        action_labels = np.array(clip_label_list[2])
        gt_ids_to_idxs = clip_label_list[3]
        gt_idxs_to_ids = clip_label_list[4]
        ori_boxes = boxes.copy()

        if self._split != 'train':
            proposal_boxes = np.array(clip_label_list[5])
            proposal_classes = np.array(clip_label_list[6])
            proposal_scores = np.array(clip_label_list[7])
            if not self.cfg.VIDOR.TEST_WITH_GT:
                proposal_ids = clip_label_list[8]
            
            gt_boxes = boxes
            ori_boxes = proposal_boxes.copy()
            boxes = proposal_boxes
        else:
            gt_boxes = None
        # Score is not used.
        # boxes = boxes[:, :4].copy()
        if not self.cfg.MODEL.USE_SAVED_FEAT:
            # Load images of current clip.
            image_paths = [self._image_paths[video_idx][frame] for frame in seq]
            imgs = utils.retry_load_images(
                image_paths, backend=self.cfg.VIDOR.IMG_PROC_BACKEND
            )

            orig_img_width, orig_img_height = imgs[0].shape[1], imgs[0].shape[0]
            n_boxes = boxes.shape[0]
            n_imgs = len(imgs)

            if self.cfg.DETECTION.ENABLE_TOI_POOLING or self.cfg.MODEL.USE_TRAJECTORY_CONV or self.cfg.MODEL.USE_SPA_CONF:
                assert self.cfg.MODEL.USE_TRAJECTORIES
                all_trajectories = [self._trajectories[orig_video_idx][frame] for frame in seq]
                boxes_ids = [gt_idxs_to_ids[i] for i in range(n_boxes)] if self._split == 'train' or self.cfg.VIDOR.TEST_WITH_GT else proposal_ids
                trajectories = []
                for j, frame in enumerate(seq):
                    trajectory = []
                    all_trajectory = all_trajectories[j]
                    for i in boxes_ids:
                        found = False
                        for traj in all_trajectory:
                            if traj['tid'] == i:
                                trajectory.append(list(traj['bbox'].values()))
                                found = True
                                break
                        if not found:
                            trajectory.append([0, 0, imgs[0].shape[1], imgs[0].shape[0]]) # if that object doesn't exist then use whole-img bbox
                    trajectories.append(trajectory)
                #import pdb; pdb.set_trace()
                # (Pdb) np.array(trajectories).shape
                # (32, 2, 4) -> 2 means n_obj
                # if self.cfg.VIDOR.TEST_DEBUG:
                #     import pdb; pdb.set_trace()
                trajectories = np.array(trajectories, dtype=np.float64)
                # trajectories = np.transpose(trajectories, [1, 0, 2])
                trajectories = trajectories.reshape(-1, 4)
                #import pdb; pdb.set_trace()
                boxes = np.concatenate((boxes, trajectories))

            if self.cfg.MODEL.USE_SPA_CONF:
                json_path = os.path.join(self._human_poses_root, orig_video_idx + '.json')
                with open(json_path, 'r') as f:
                    human_poses = json.load(f)
                
                # human_poses = self._human_poses[orig_video_idx]
                boxes_ids = [gt_idxs_to_ids[i] for i in range(n_boxes)] if self._split == 'train' or self.cfg.VIDOR.TEST_WITH_GT else proposal_ids
                try:
                    if self._split == 'train' or self.cfg.VIDOR.TEST_WITH_GT:
                        boxes_ids = [gt_idxs_to_ids[i] for i in range(n_boxes)]
                        human_poses = [human_poses[str(boxes_ids[jdx])] for jdx, obj_class in enumerate(obj_classes) if obj_class == 0]
                    else:
                        boxes_ids = proposal_ids
                        human_poses = [human_poses[str(boxes_ids[jdx])] for jdx, obj_class in enumerate(proposal_classes) if obj_class == 0]
                except:
                    import pdb; pdb.set_trace()
                full_human_pose_maps = np.zeros((len(human_poses), len(seq), orig_img_height, orig_img_width))
                # scale_x = self.heatmap_size / orig_img_width
                # scale_y = self.heatmap_size / orig_img_height
                for i, human_pose in enumerate(human_poses):
                    for j, frame_idx in enumerate(seq):
                        if str(frame_idx) in human_pose:
                            full_human_pose_maps[i][j] = self.draw_human_skeleton(human_pose[str(frame_idx)], orig_img_width, orig_img_height)

                # full_human_pose_maps = shape (n_person, 32, orig_img_height, orig_img_width)
                n_person = full_human_pose_maps.shape[0]
                full_human_pose_maps = full_human_pose_maps.reshape(-1, orig_img_height, orig_img_width)
                
                # import pdb; pdb.set_trace()
                imgs.extend([np.expand_dims(full_human_pose_maps[i], axis=-1) for i in range(n_person * self.cfg.DATA.NUM_FRAMES)])
                
                # box_maps = np.zeros((n_boxes, orig_img_height, orig_img_width))
                # n_person = human_poses.shape[0]
                # pose_maps = np.zeros((n_person, orig_img_height, orig_img_width))

                no_human_pose_used = False
                if n_imgs == len(imgs): # no human pose used!
                    print('skipping 1 img for pose module due to no human pose detected!')
                    no_human_pose_used = True

            if self.cfg.VIDOR.IMG_PROC_BACKEND == "pytorch": # False
                # T H W C -> T C H W.
                imgs = imgs.permute(0, 3, 1, 2)
                # Preprocess images and boxes.
                if gt_boxes is None:
                    imgs, boxes = self._images_and_boxes_preprocessing(
                        imgs, boxes=boxes
                    )
                else:
                    ### NOT IMPLEMENTED! ###
                    imgs, boxes, gt_boxes = self._images_and_boxes_preprocessing(
                        imgs, boxes=boxes, gt_boxes=gt_boxes
                    )
                # T C H W -> C T H W.
                imgs = imgs.permute(1, 0, 2, 3)
            else:
                # Preprocess images and boxes
                if gt_boxes is None:
                    if self.cfg.MODEL.USE_SPA_CONF:
                        imgs, skeleton_imgs, boxes = self._images_and_boxes_preprocessing_cv2(
                            imgs, boxes=boxes, min_scale=min_scale, crop_size=crop_size,
                            n_imgs=n_imgs
                        )
                    else:
                        imgs, boxes = self._images_and_boxes_preprocessing_cv2(
                            imgs, boxes=boxes, min_scale=min_scale, crop_size=crop_size,
                            n_imgs=n_imgs
                        )
                else:
                    if self.cfg.MODEL.USE_SPA_CONF:
                        imgs, skeleton_imgs, boxes, gt_boxes = self._images_and_boxes_preprocessing_cv2(
                            imgs, boxes=boxes, gt_boxes=gt_boxes, n_imgs=n_imgs, 
                        )
                    else:
                        imgs, boxes, gt_boxes = self._images_and_boxes_preprocessing_cv2(
                            imgs, boxes=boxes, gt_boxes=gt_boxes, n_imgs=n_imgs
                        )

            if self.cfg.DETECTION.ENABLE_TOI_POOLING or self.cfg.MODEL.USE_TRAJECTORY_CONV or self.cfg.MODEL.USE_SPA_CONF:
                #import pdb; pdb.set_trace()
                # Bug fixed here boxes were reshaped incorrectly
                trajectory_boxes = boxes[n_boxes:].reshape(-1, n_boxes, 4)
                trajectory_boxes = trajectory_boxes.transpose(1, 0, 2).reshape(n_boxes, -1)
                # trajectory_boxes.shape
                # (32, 2, 4) -> 2 is n_obj
                boxes = boxes[:n_boxes]

            if self.cfg.MODEL.USE_SPA_CONF:
                # import pdb; pdb.set_trace()
                # trajectory_boxes.shape = (2, 128)
                # skeleton_imgs.shape = torch.Size([1, 32, 224, 224])

                trajectory_box_masks = np.zeros((trajectory_boxes.shape[0], self.cfg.DATA.NUM_FRAMES, skeleton_imgs.shape[2], skeleton_imgs.shape[3])) # shape (n_boxes, 32, 224, 224)
                for box_id in range(trajectory_boxes.shape[0]):
                    trajectory_box = trajectory_boxes[box_id].reshape(-1, 4) # shape (32, 4)
                    for frame_id in range(trajectory_box.shape[0]): 
                        x1, y1, x2, y2 = int(round(trajectory_box[frame_id][0])), int(round(trajectory_box[frame_id][1])), int(round(trajectory_box[frame_id][2])), int(round(trajectory_box[frame_id][3]))
                        trajectory_box_masks[box_id][frame_id][y1:y2, x1:x2] = 1

                # import pdb; pdb.set_trace()
            
            if self.cfg.MODEL.USE_TRAJECTORIES:
                all_trajectories = [self._trajectories[orig_video_idx][frame] for frame in seq]
                boxes_ids = [gt_idxs_to_ids[i] for i in range(len(boxes))] if self._split == 'train' or self.cfg.VIDOR.TEST_WITH_GT else proposal_ids
                trajectories = []
                for j, frame in enumerate(seq):
                    trajectory = []
                    all_trajectory = all_trajectories[j]
                    for i in boxes_ids:
                        found = False
                        for traj in all_trajectory:
                            try:
                                if traj['tid'] == i:
                                    trajectory.append(list(traj['bbox'].values()))
                                    found = True
                                    break
                            except:
                                import pdb; pdb.set_trace()
                        if not found:
                            trajectory.append([0, 0, imgs[0].shape[1], imgs[0].shape[0]]) # if that object doesn't exist then use whole-img bbox
                    trajectories.append(trajectory)
                # (Pdb) np.array(trajectories).shape
                # (32, 2, 4)
                # if self.cfg.VIDOR.TEST_DEBUG:
                #     import pdb; pdb.set_trace()
                trajectories = np.array(trajectories, dtype=np.float64)
                trajectories = np.transpose(trajectories, [1, 0, 2])

                width_ratio, height_ratio = imgs[0].shape[1] / orig_img_width, imgs[0].shape[0] / orig_img_height
                trajectories[:, :, 0] *= width_ratio
                trajectories[:, :, 1] *= height_ratio
                trajectories[:, :, 2] *= width_ratio
                trajectories[:, :, 3] *= height_ratio
                
                # no need to flatten trajectory bboxes for resnet feat generation
                if self.cfg.MODEL.ARCH != "resnet50":
                    trajectories = trajectories.reshape(boxes.shape[0], -1)
                # trajectories.shape = (n_trajectories, 32*4)
            
            imgs, trajectories = utils.pack_pathway_output(self.cfg, imgs, trajectories)

        metadata = [[video_idx, sec]] * len(boxes)

        extra_data = {
            "boxes": boxes,
            "ori_boxes": ori_boxes,
            "metadata": metadata,
            "obj_classes": obj_classes,
            "action_labels": action_labels,
        }

        # for debugging remove later
        # video_fld,video_id = self._video_idx_to_name[video_idx].split('/')
        # extra_data['video_folder'] = int(video_fld)
        # extra_data['video_id'] = int(video_id)
        # extra_data['frame_id'] = int(center_idx+1)
        
        if self.cfg.MODEL.ARCH in ["resnet50", "resnext101"]:
            extra_data["video_id"] = self._video_idx_to_name[video_idx]
        
        if not self.cfg.MODEL.USE_SAVED_FEAT:
            if self.cfg.MODEL.USE_TRAJECTORIES:
                extra_data["trajectories"] = trajectories

            if self.cfg.DETECTION.ENABLE_TOI_POOLING or self.cfg.MODEL.USE_TRAJECTORY_CONV:
                extra_data["trajectory_boxes"] = trajectory_boxes
            
            if self.cfg.MODEL.USE_SPA_CONF:
                # if self.cfg.VIDOR.TEST_DEBUG:
                #     import pdb; pdb.set_trace()
                skeleton_imgs = skeleton_imgs.reshape(n_person, self.cfg.DATA.NUM_FRAMES, skeleton_imgs.shape[2], skeleton_imgs.shape[3])
                extra_data["skeleton_imgs"] = torch.nn.functional.interpolate(skeleton_imgs, 32)

                trajectory_box_masks = torch.tensor(trajectory_box_masks)
                extra_data["trajectory_box_masks"] = torch.nn.functional.interpolate(trajectory_box_masks, 32)
            elif self.cfg.MODEL.USE_HUMAN_POSES:
                human_poses = self._human_poses[orig_video_idx]
                human_poses = np.concatenate(([[human_poses[boxes_ids[jdx]]] for jdx, obj_class in enumerate(obj_classes) if obj_class == 0]))
                human_poses = human_poses[:, seq, :]

                human_poses = human_poses.reshape(human_poses.shape[0], -1)
                extra_data["human_poses"] = human_poses
            
        if gt_boxes is not None:
            extra_data['gt_boxes'] = gt_boxes
            extra_data['proposal_classes'] = proposal_classes
            extra_data['proposal_scores'] = proposal_scores

        if self.cfg.DEMO.ENABLE or self.cfg.DEMO.ENABLE_ALL:
            extra_data['orig_video_idx'] = orig_video_idx + '/' + orig_video_idx.split('/')[1] + '_' + f'{center_idx+1:06d}'
        
        if self.cfg.TEST.SAVE_IMGS:
            extra_data['orig_video_idx'] = orig_video_idx + '/' + orig_video_idx.split('/')[1] + '_' + f'{center_idx+1:06d}'
        
        # print('imgs[0].shape:', imgs[0].shape, 'extra_data["boxes"][0].shape:', extra_data['boxes'][0].shape)
        #import pdb; pdb.set_trace()
        if self.cfg.MODEL.USE_ALPHA_POSES:
            json_path = os.path.join(self._human_poses_root, orig_video_idx + '.json')
            with open(json_path, 'r') as f:
                human_poses = json.load(f)
            n_boxes = extra_data['boxes'].shape[0]             
            # heat_maps = draw_heatmaps(human_pose[str(center_idx)]
            # human_poses = self._human_poses[orig_video_idx]
            boxes_ids = [gt_idxs_to_ids[i] for i in range(n_boxes)] if self._split == 'train' or self.cfg.VIDOR.TEST_WITH_GT else proposal_ids
            ######
            # I think the code expects to take all the poses (frames) of a person and stack
            # multiple persons sequentiallly. Instead of that, I'm taking only the center pose
            # for each person
            ######
            try:
                if self._split == 'train' or self.cfg.VIDOR.TEST_WITH_GT:
                    boxes_ids = [gt_idxs_to_ids[i] for i in range(n_boxes)]
                    human_poses = [human_poses[str(boxes_ids[jdx])] for jdx, obj_class in enumerate(obj_classes) if obj_class == 0]
                    # human_poses = [human_poses[str(boxes_ids[jdx])][str(center_idx)] for jdx, obj_class in enumerate(obj_classes) if obj_class == 0]
                    # human_boxes = [extra_data['boxes'][jdx] for jdx, obj_class in enumerate(obj_classes) if obj_class == 0]
                else:
                    boxes_ids = proposal_ids
                    human_poses = [human_poses[str(boxes_ids[jdx])] for jdx, obj_class in enumerate(proposal_classes) if obj_class == 0]
                    # human_poses = [human_poses[str(boxes_ids[jdx])][str(center_idx)] for jdx, obj_class in enumerate(proposal_classes) if obj_class == 0]
                    # human_boxes = [extra_data['boxes'][jdx] for jdx, obj_class in enumerate(proposal_classes) if obj_class == 0]
            except:
                import pdb; pdb.set_trace()


            # import pdb; pdb.set_trace()
            map_size = 7
            # full_human_pose_maps = np.zeros((len(human_poses), len(seq), map_size, map_size))
            full_human_pose_maps = np.zeros((len(human_poses), 17, len(seq[::4]),  map_size, map_size))
            # scale_x = self.heatmap_size / orig_img_width
            # scale_y = self.heatmap_size / orig_img_height
            # for i, human_pose in enumerate(human_poses):
            #     full_human_pose_maps[i] = self.draw_heatmaps(human_pose, human_boxes[i])
            # Will explore the multi-frame pose later
        
            for i, human_pose in enumerate(human_poses):
                for j, frame_idx in enumerate(seq[::4]):
                    if str(frame_idx) in human_pose:
                        full_human_pose_maps[i, :, j] = self.draw_heatmaps(human_pose[str(frame_idx)])
                        # full_human_pose_maps[i][j] = self.draw_human_skeleton(human_pose[str(frame_idx)], orig_img_width, orig_img_height)

            # full_human_pose_maps = shape (n_person, 32, orig_img_height, orig_img_width)
            n_person = full_human_pose_maps.shape[0]
            # full_human_pose_maps = full_human_pose_maps.reshape(-1, orig_img_height, orig_img_width)

            # import pdb; pdb.set_trace()
            # imgs.extend([np.expand_dims(full_human_pose_maps[i], axis=-1) for i in range(n_person * self.cfg.DATA.NUM_FRAMES)])
            extra_data['heatmaps'] = full_human_pose_maps
            
        if self.cfg.MODEL.USE_LABEL_WEIGHTS:
            extra_data['pos_weights'] = self._label_pos_weights

        # to load saved features
        if self.cfg.MODEL.USE_SAVED_FEAT:
            # assumed here that 1 keyframe is there
            # change features lists if len(imgs)>1
            # assert len(imgs) == 1
            # del imgs
            '''
            video_id = self._video_idx_to_name[video_idx]
            filename = self._video_to_file[video_id][sec]
            if filename != self._feat_file:
                import pickle
                self._feat_file = filename
                #with open(os.path.join(self.cfg.VIDOR.FEAT_DIR, self._split, filename), 'rb') as f:
                with open(os.path.join(self.cfg.VIDOR.FEAT_DIR, 'val', filename), 'rb') as f:
                    self._feat = pickle.load(f)
            
            #list because imgs are of list type
            features = [self._feat[video_id][sec]]
            '''
            video_id = self._video_idx_to_name[video_idx]
            filename = f'{sec:04d}.pkl'
            import pickle
            if self._split in ["train"]:
                filepath = os.path.join(self.cfg.VIDOR.FEAT_DIR, self._split, video_id, filename)

            if self._split in ["test", "val"]:
                filepath = os.path.join(self.cfg.VIDOR.FEAT_DIR, "val", video_id, filename)
            
            #if self._split in ["test", "val"] and not self.cfg.VIDOR.TEST_WITH_GT:
            #    det_trajectory_path = os.path.join(self.cfg.VIDOR.TRAJ_DIR, "val", video_id, filename)


            with open(filepath, 'rb') as f:
                data = pickle.load(f)
                # import pdb; pdb.set_trace()
                features = [data['features']]
                #if self.cfg.VIDOR.TEST_WITH_GT:
                extra_data['trajectory_boxes'] = data['trajectory_boxes']
            #if not self.cfg.VIDOR.TEST_WITH_GT:
            #    with open(det_trajectory_path, 'rb') as ft:
            #        det_trajectories = pickle.load(ft)
            #        extra_data['trajectory_boxes']=det_trajectories['trajectory_boxes']
                #features type list
                #len(features)=1
                #features[0].shape = (1024,8,14,14(19 for val)) -> (feat_dim, timestamp, H, W)
                #extra_data['trajectory_boxes'].shape = (3, 129) -> 3 is num_objs


            return features, extra_data
        
        return imgs, extra_data
