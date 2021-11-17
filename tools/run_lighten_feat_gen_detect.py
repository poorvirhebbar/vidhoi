#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import sys
sys.path.append("/mnt/data/apoorva/HOI_vidhoi/")

"""Wrapper to train and test a video classification model."""
from slowfast.utils.misc import launch_job
from slowfast.utils.parser import load_config, parse_args

# from lighten_gen_resnet_feat import generate_features 
from pose_feat_gen_detect import generate_features 


def main():
    """
    Main function to spawn the train and test process.
    """
    args = parse_args()
    cfg = load_config(args)

    # Perform feature generattion.
    launch_job(cfg=cfg, init_method=args.init_method, func=generate_features)

if __name__ == "__main__":
    main()
