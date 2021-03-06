# CUDA_VISIBLE_DEVICES=1,2 python tools/run_net_vidor.py --cfg configs/vidor/SLOWFAST_32x2_R50_SHORT_SCRATCH_EVAL_GT_trajectory-toipool.yaml DATA.PATH_TO_DATA_DIR slowfast/datasets/vidor NUM_GPUS 1 DATA_LOADER.NUM_WORKERS 0 TRAIN.BATCH_SIZE 16 TEST.BATCH_SIZE 2 LOG_MODEL_INFO False TRAIN.ENABLE True TRAIN.CHECKPOINT_FILE_PATH ./checkpoints/SLOWFAST_32x2_R50_SHORT_SCRATCH_EVAL_GT_trajectory-toipool/checkpoint_epoch_00020.pyth TRAIN.CHECKPOINT_TYPE pytorch VIDOR.TEST_DEBUG False TRAIN.AUTO_RESUME False


CUDA_VISIBLE_DEVICES=1 python tools/run_net_vidor.py --cfg configs/vidor/VANILLA_DEDICATED_RNN_TRAIN_RESNEXT101_FEAT.yaml DATA.PATH_TO_DATA_DIR slowfast/datasets/vidor NUM_GPUS 1 DATA_LOADER.NUM_WORKERS 0 TRAIN.BATCH_SIZE 8 TEST.BATCH_SIZE 1 LOG_MODEL_INFO False TRAIN.ENABLE True VIDOR.TEST_DEBUG False TRAIN.AUTO_RESUME True DATA_LOADER.PIN_MEMORY False MODEL.USE_LABEL_WEIGHTS True MODEL.DROPOUT 0.0 MODEL.USE_UNION_BBOX True

