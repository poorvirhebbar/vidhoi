#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Train a video classification model."""
import numpy as np
import pickle
import math
import random
import pprint
import torch
from fvcore.nn.precise_bn import get_bn_modules, update_bn_stats

import slowfast.models.losses as losses
import slowfast.models.optimizer as optim
import slowfast.utils.checkpoint as cu
import slowfast.utils.distributed as du
import slowfast.utils.logging as logging
import slowfast.utils.metrics as metrics
import slowfast.utils.misc as misc
import slowfast.visualization.tensorboard_vis as tb
from slowfast.datasets import loader
from slowfast.models import build_model
from slowfast.utils.meters import AVAMeter, TrainMeter, ValMeter, VidorMeter
from slowfast.utils.multigrid import MultigridSchedule
from focalloss import focal_loss

from torch import nn
import torch.nn.functional as F
from tqdm import tqdm

logger = logging.get_logger(__name__)


def train_epoch(
    train_loader, model, optimizer, train_meter, cur_epoch, cfg, writer=None
):
    """
    Perform the video training for one epoch.
    Args:
        train_loader (loader): video training loader.
        model (model): the video model to train.
        optimizer (optim): the optimizer to perform optimization on the model's
            parameters.
        train_meter (TrainMeter): training meters to log the training performance.
        cur_epoch (int): current epoch of training.
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
        writer (TensorboardWriter, optional): TensorboardWriter object
            to writer Tensorboard log.
    """
    # Enable train mode.
    model.train()
    train_meter.iter_tic()
    data_size = len(train_loader)

    #import pdb; pdb.set_trace()
    for cur_iter, (inputs, meta) in enumerate(train_loader):
        # obj_classes = meta['obj_classes']
        # action_labels = meta['action_labels']
        # import pdb; pdb.set_trace()
        # (Pdb) len(inputs)
        # 2
        # (Pdb) inputs[0].shape
        # torch.Size([16, 3, 8, 224, 224])
        # (Pdb) inputs[1].shape
        # torch.Size([16, 3, 32, 224, 224])
        # (Pdb) labels.shape
        # torch.Size([44, 50])
        # (Pdb) meta.keys()
        # dict_keys(['boxes', 'ori_boxes', 'metadata'])
        # (Pdb) meta['boxes'].shape
        # torch.Size([44, 5]) # the first entry is the number in the mini-batch

        # Transfer the data to the current GPU device.
        if isinstance(inputs, (list,)):
            for i in range(len(inputs)):
                inputs[i] = inputs[i].cuda(non_blocking=True)
        else:
            inputs = inputs.cuda(non_blocking=True)
        # labels = labels.cuda()
        # obj_classes = obj_classes.cuda()
        # action_labels = action_labels.cuda()
        for key, val in meta.items():
            if isinstance(val, (list,)):
                for i in range(len(val)):
                    #import pdb; pdb.set_trace()
                    val[i] = val[i].cuda(non_blocking=True)
            else:
                meta[key] = val.cuda(non_blocking=True)

        # Update the learning rate.
        lr = optim.get_epoch_lr(cur_epoch + float(cur_iter) / data_size, cfg)
        optim.set_lr(optimizer, lr)

        # Compute the predictions.
        trajectories = human_poses = trajectory_boxes = skeleton_imgs = trajectory_box_masks = heatmaps = None
        if cfg.MODEL.USE_TRAJECTORIES:
            trajectories = meta['trajectories']
        if cfg.MODEL.USE_HUMAN_POSES:
            human_poses = meta['human_poses']
        if cfg.DETECTION.ENABLE_TOI_POOLING or cfg.MODEL.USE_TRAJECTORY_CONV:
            trajectory_boxes = meta['trajectory_boxes']
        if cfg.MODEL.USE_SPA_CONF:
            skeleton_imgs = meta['skeleton_imgs']
            trajectory_box_masks = meta['trajectory_box_masks']
        if cfg.MODEL.USE_ALPHA_POSES:
            heatmaps = meta['heatmaps']
        # if cfg.VIDOR.TEST_DEBUG:
        #   import pdb; pdb.set_trace()
        preds, action_labels, bbox_pair_ids = model(inputs, meta["boxes"], meta['obj_classes'], meta['obj_classes_lengths'], meta['action_labels'], trajectories=trajectories, human_poses=human_poses, trajectory_boxes=trajectory_boxes, skeleton_imgs=skeleton_imgs, trajectory_box_masks=trajectory_box_masks, heatmaps=heatmaps)
        
        #import pdb; pdb.set_trace()
        # self._log_accuracy()
        # fetching label weights from the saved label frequencies
        with open('slowfast/datasets/vidor/predicate_frequencies.pkl', 'rb') as f:
            freq = pickle.load(f)
        

        # Propensity score
        pos_weights = None
        if cfg.MODEL.USE_PROPENSITY:
            freq = torch.Tensor(freq)
            A = 0.2 #0.55
            B = 10 #1.0
            N = freq.sum()
            C = (math.log(N) - 1)*(B + 1)**A
            pos_weights = -0.5 + C*(freq + B)**(-A)
            pos_weights = pos_weights.cuda()
            # print(pos_weights)
            # import pdb; pdb.set_trace()
        
        
        alpha = 1.0
        #train_action_freq=pd.read_pickle(r'../../../vidor-dataset/vidHOI/train/train_action_frequency_class.pkl')#action_class: frequency
        if cfg.MODEL.USE_LABEL_WEIGHTS:
            pos_weights = meta['pos_weights'][0]
            # pos_weights = F.softmax(pos_weights)
            pos_weights = pos_weights * 0.0 + 1.0
            pos_weights[[3,7]] = 0.1 # next_to, in_front_of
            if cfg.MODEL.DEC_WEIGHTS_14:
                pos_weights[[1,4]] = 0.5 # watch, behind
            if cfg.MODEL.INC_WEIGHTS:
                pos_weights[10:] = 10
            alpha = pos_weights.unsqueeze(0)

        ''' 
        for j in range(action_labels.shape[0]):
            # This loop is over all the pairs
            if action_labels[j, 3] == 1:
                p = random.random()
                if p > 0.1:
                    action_labels[j, 3] = 0
            if action_labels[j, 7] == 1:
                p = random.random()
                if p > 0.1:
                    action_labels[j, 7] = 0

        '''
        loss = F.binary_cross_entropy_with_logits(
            preds,
            action_labels,
            reduction="mean",
            reduce=True if not cfg.MODEL.USE_FOCAL_LOSS else False,
            pos_weight=pos_weights
        )

        if cfg.MODEL.USE_FOCAL_LOSS:
            pt = torch.exp(-loss)
            #   alpha           gamma
            loss = alpha * (1-pt)**2.0 * loss
            loss = torch.mean(loss)
        #loss = focal_loss(preds, action_labels, reduction='mean')
        #if cur_epoch == 25:
        #    import pdb; pdb.set_trace()

        '''
        # Explicitly declare reduction to mean.
        loss_fun = losses.get_loss_func(cfg.MODEL.LOSS_FUNC)(reduction="mean")

        # Compute the loss.
        loss = loss_fun(preds, meta['action_labels'])
        '''
        try:
            # check Nan Loss.
            misc.check_nan_losses(loss,optimizer)
            # Perform the backward pass.
            optimizer.zero_grad()
            loss.backward()
            #nn.utils.clip_grad_norm_(model.parameters(),max_norm=2)
            # Update the parameters.
            optimizer.step()
            """ 
            if loss.item() < 10000:
                # Perform the backward pass.
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(),max_norm=2)
                # Update the parameters.
                optimizer.step()
            """
        except RuntimeError as e:
            print("caught error : ",repr(e))


        if cfg.NUM_GPUS > 1:
            loss = du.all_reduce([loss])[0]
        loss = loss.item()

        train_meter.iter_toc()

        # EXERIMENTAL CODE TO GET ACCURACY

        preds_score = F.sigmoid(preds).cpu()
        preds = preds_score >= 0.5 # Convert scores into 'True' or 'False'
        action_labels = action_labels.cpu()
        
        # Update and log stats.
        # not using predicted proposals for accuracy calculation
        train_meter.update_stats(
            preds_score.cpu(), preds.cpu(), None, None, None, None, None, 
            action_labels.cpu(), None, None ,loss, lr)

        # Update and log stats.
        #train_meter.update_stats(None, None, None, None, None, 
        #                            None, None, None, None, None, loss, lr)
        
        
        # EXERIMENTAL CODE TO GET ACCURACY END HERE
        
        # write to tensorboard format if available.
        if writer is not None:
            writer.add_scalars(
                {"Train/loss": loss, "Train/lr": lr},
                global_step=data_size * cur_epoch + cur_iter,
            )

        train_meter.log_iter_stats(cur_epoch, cur_iter)
        train_meter.iter_tic()

        # TODO: ONLY FOR DEBUG PURPOSE! #
        if cfg.VIDOR.TEST_DEBUG:
            logger.info('[TEST_DEBUG] Break Training!')
            break

    # Log epoch stats.
    train_meter.log_epoch_stats(cur_epoch)
    train_meter.reset()


@torch.no_grad()
def eval_epoch(val_loader, model, val_meter, cur_epoch, cfg, writer=None):
    """
    Evaluate the model on the val set.
    Args:
        val_loader (loader): data loader to provide validation data.
        model (model): model to evaluate the performance.
        val_meter (ValMeter): meter instance to record and calculate the metrics.
        cur_epoch (int): number of the current epoch of training.
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
        writer (TensorboardWriter, optional): TensorboardWriter object
            to writer Tensorboard log.
    """
    # Evaluation mode enabled. The running stats would not be updated.
    model.eval()
    val_meter.iter_tic()

    for cur_iter, (inputs, meta) in tqdm(enumerate(val_loader)):
        # Transferthe data to the current GPU device.
        if isinstance(inputs, (list,)):
            for i in range(len(inputs)):
                inputs[i] = inputs[i].cuda(non_blocking=True)
        else:
            inputs = inputs.cuda(non_blocking=True)
        
        for key, val in meta.items():
            if isinstance(val, (list,)):
                for i in range(len(val)):
                    val[i] = val[i].cuda(non_blocking=True)
            else:
                meta[key] = val.cuda(non_blocking=True)

        # Compute the predictions.
        trajectories = human_poses = trajectory_boxes = skeleton_imgs = trajectory_box_masks = None
        if cfg.MODEL.USE_TRAJECTORIES:
            trajectories = meta['trajectories']
        if cfg.MODEL.USE_HUMAN_POSES:
            human_poses = meta['human_poses']
        if cfg.DETECTION.ENABLE_TOI_POOLING or cfg.MODEL.USE_TRAJECTORY_CONV:
            trajectory_boxes = meta['trajectory_boxes']
        if cfg.MODEL.USE_SPA_CONF:
            skeleton_imgs = meta['skeleton_imgs']
            trajectory_box_masks = meta['trajectory_box_masks']
        preds, action_labels, bbox_pair_ids, gt_bbox_pair_ids = model(inputs, meta["boxes"], meta['proposal_classes'], meta['proposal_lengths'], meta['action_labels'], meta['obj_classes'], meta['obj_classes_lengths'], trajectories=trajectories, human_poses=human_poses, trajectory_boxes=trajectory_boxes, skeleton_imgs=skeleton_imgs, trajectory_box_masks=trajectory_box_masks)

        preds_score = F.sigmoid(preds).cpu()
        preds = preds_score >= 0.5 # Convert scores into 'True' or 'False'
        action_labels = action_labels.cpu()
        boxes = meta["boxes"].cpu()
        obj_classes = meta['obj_classes'].cpu()
        # obj_classes_lengths = meta['obj_classes_lengths'].cpu()
        bbox_pair_ids = bbox_pair_ids.cpu()
        gt_bbox_pair_ids = gt_bbox_pair_ids.cpu()
        # hopairs = hopairs # .cpu()
        proposal_scores = meta['proposal_scores'].cpu()
        gt_boxes = meta['gt_boxes'].cpu()
        proposal_classes = meta['proposal_classes'].cpu()

        '''
        if cfg.NUM_GPUS > 1:
            preds_score = torch.cat(du.all_gather_unaligned(preds_score), dim=0)
            preds = torch.cat(du.all_gather_unaligned(preds), dim=0)
            action_labels = torch.cat(du.all_gather_unaligned(action_labels), dim=0)
            boxes = torch.cat(du.all_gather_unaligned(boxes), dim=0)
            obj_classes = torch.cat(du.all_gather_unaligned(obj_classes), dim=0)
            # obj_classes_lengths = torch.cat(du.all_gather_unaligned(obj_classes_lengths), dim=0),
            bbox_pair_ids = torch.cat(du.all_gather_unaligned(bbox_pair_ids), dim=0)
            gt_bbox_pair_ids = torch.cat(du.all_gather_unaligned(gt_bbox_pair_ids), dim=0)
            # hopairs = torch.cat(du.all_gather_unaligned(hopairs), dim=0)
            proposal_scores = torch.cat(du.all_gather_unaligned(proposal_scores), dim=0)
            gt_boxes = torch.cat(du.all_gather_unaligned(gt_boxes), dim=0)
            proposal_classes = torch.cat(du.all_gather_unaligned(proposal_classes), dim=0)
        '''

        val_meter.iter_toc()
        # Update and log stats.
        val_meter.update_stats(
            preds_score.cpu(), preds.cpu(), bbox_pair_ids.cpu(),
            proposal_scores.cpu(), boxes.cpu(), 
            proposal_classes.cpu(), gt_boxes.cpu(), action_labels.cpu(), obj_classes.cpu(), gt_bbox_pair_ids.cpu()
        )

        val_meter.log_iter_stats(cur_epoch, cur_iter)
        val_meter.iter_tic()

        # TODO: ONLY FOR DEBUG PURPOSE! #
        if cfg.VIDOR.TEST_DEBUG:
            logger.info('[TEST_DEBUG] Break Testing!')
            break

    # Log epoch stats.
    val_meter.log_epoch_stats(cur_epoch)
    # write to tensorboard format if available.
    if writer is not None:
        writer.add_scalars(
            {
                "Val/mAP": val_meter.map,
                "Val/max_recall": val_meter.m_rec,
                "Val/hd": val_meter.hd,
                "Val/dt": val_meter.dt,
                "Val/one_Dr": val_meter.one_dr
            }, global_step=cur_epoch
        )
        # all_preds_cpu = [
        #     pred.clone().detach().cpu() for pred in val_meter.all_preds
        # ]
        # all_labels_cpu = [
        #     label.clone().detach().cpu() for label in val_meter.all_labels
        # ]
        # writer.plot_eval(
        #     preds=all_preds_cpu, labels=all_labels_cpu, global_step=cur_epoch
        # )

    val_meter.reset()


def calculate_and_update_precise_bn(loader, model, num_iters=200):
    """
    Update the stats in bn layers by calculate the precise stats.
    Args:
        loader (loader): data loader to provide training data.
        model (model): model to update the bn stats.
        num_iters (int): number of iterations to compute and update the bn stats.
    """

    def _gen_loader():
        for inputs, _, _, _ in loader:
            if isinstance(inputs, (list,)):
                for i in range(len(inputs)):
                    inputs[i] = inputs[i].cuda(non_blocking=True)
            else:
                inputs = inputs.cuda(non_blocking=True)
            yield inputs

    # Update the bn stats.
    update_bn_stats(model, _gen_loader(), num_iters)


def build_trainer(cfg):
    """
    Build training model and its associated tools, including optimizer,
    dataloaders and meters.
    USED WHEN USING MULTIGRID
    Args:
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
    Returns:
        model (nn.Module): training model.
        optimizer (Optimizer): optimizer.
        train_loader (DataLoader): training data loader.
        val_loader (DataLoader): validatoin data loader.
        precise_bn_loader (DataLoader): training data loader for computing
            precise BN.
        train_meter (TrainMeter): tool for measuring training stats.
        val_meter (ValMeter): tool for measuring validation stats.
    """
    # Build the video model and print model statistics.
    model = build_model(cfg)
    if du.is_master_proc() and cfg.LOG_MODEL_INFO:
        misc.log_model_info(model, cfg, use_train_input=True)

    # Construct the optimizer.
    optimizer = optim.construct_optimizer(model, cfg)

    # Create the video train and val loaders.
    train_loader = loader.construct_loader(cfg, "train")
    val_loader = loader.construct_loader(cfg, "val")
    precise_bn_loader = loader.construct_loader(
        cfg, "train", is_precise_bn=True
    )
    # Create meters.
    train_meter = VidorMeter(len(train_loader), cfg, mode="train")
    val_meter = VidorMeter(len(val_loader), cfg, mode="val")

    return (
        model,
        optimizer,
        train_loader,
        val_loader,
        precise_bn_loader,
        train_meter,
        val_meter,
    )


def train(cfg):
    """
    Train a video model for many epochs on train set and evaluate it on val set.
    Args:
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
    """
    # assert cfg.NUM_GPUS == cfg.TEST.BATCH_SIZE
    
    # Set up environment.
    du.init_distributed_training(cfg)
    # Set random seed from configs.
    np.random.seed(cfg.RNG_SEED)
    torch.manual_seed(cfg.RNG_SEED)

    # Setup logging format.
    logging.setup_logging(cfg.OUTPUT_DIR)

    # Init multigrid.
    multigrid = None
    if cfg.MULTIGRID.LONG_CYCLE or cfg.MULTIGRID.SHORT_CYCLE:
        multigrid = MultigridSchedule()
        cfg = multigrid.init_multigrid(cfg)
        if cfg.MULTIGRID.LONG_CYCLE:
            cfg, _ = multigrid.update_long_cycle(cfg, cur_epoch=0)
    # Print config.
    logger.info("Train with config:")
    logger.info(pprint.pformat(cfg))

    # Build the video model and print model statistics.
    model = build_model(cfg)
    if du.is_master_proc() and cfg.LOG_MODEL_INFO:
        misc.log_model_info(model, cfg, use_train_input=True)

    # Construct the optimizer.
    optimizer = optim.construct_optimizer(model, cfg)

    # Load a checkpoint to resume training if applicable.
    start_epoch = cu.load_train_checkpoint(cfg, model, optimizer)
    # start_epoch = cu.load_train_checkpoint(cfg, model, None)

    # Create the video train and val loaders.
    # import pdb; pdb.set_trace()
    train_loader = loader.construct_loader(cfg, "train")
    val_loader = []
    if cfg.TRAIN.EVAL:
        val_loader = loader.construct_loader(cfg, "val")
    # val_loader = []
    # precise_bn_loader = loader.construct_loader(
    #    cfg, "train", is_precise_bn=True
    # )

    # Create meters.
    train_meter = VidorMeter(len(train_loader), cfg, mode="train")
    val_meter = VidorMeter(len(val_loader), cfg, mode="val")

    # set up writer for logging to Tensorboard format.
    if cfg.TENSORBOARD.ENABLE and du.is_master_proc(
        cfg.NUM_GPUS * cfg.NUM_SHARDS
    ):
        writer = tb.TensorboardWriter(cfg)
    else:
        writer = None

    # Perform the training loop.
    logger.info("Start epoch: {}".format(start_epoch + 1))

    for cur_epoch in range(start_epoch, cfg.SOLVER.MAX_EPOCH):
        if cfg.MULTIGRID.LONG_CYCLE:
            cfg, changed = multigrid.update_long_cycle(cfg, cur_epoch)
            if changed:
                (
                    model,
                    optimizer,
                    train_loader,
                    val_loader,
                    precise_bn_loader,
                    train_meter,
                    val_meter,
                ) = build_trainer(cfg)
                for k in range(10):
                    print('####################')

                # Load checkpoint.
                if cu.has_checkpoint(cfg.OUTPUT_DIR):
                    last_checkpoint = cu.get_last_checkpoint(cfg.OUTPUT_DIR)
                    assert "{:05d}.pyth".format(cur_epoch) in last_checkpoint
                else:
                    last_checkpoint = cfg.TRAIN.CHECKPOINT_FILE_PATH
                logger.info("Load from {}".format(last_checkpoint))
                cu.load_checkpoint(
                    last_checkpoint, model, cfg.NUM_GPUS > 1, optimizer
                )

        # Shuffle the dataset.
        loader.shuffle_dataset(train_loader, cur_epoch)
        # Train for one epoch.
        train_epoch(
            train_loader, model, optimizer, train_meter, cur_epoch, cfg, writer
        )

        # Compute precise BN stats.
        if cfg.BN.USE_PRECISE_STATS and len(get_bn_modules(model)) > 0:
            calculate_and_update_precise_bn(
                precise_bn_loader,
                model,
                min(cfg.BN.NUM_BATCHES_PRECISE, len(precise_bn_loader)),
            )
        _ = misc.aggregate_sub_bn_stats(model)

        # Save a checkpoint.
        if cu.is_checkpoint_epoch(
            cfg, cur_epoch, None if multigrid is None else multigrid.schedule
        ):
            cu.save_checkpoint(cfg.OUTPUT_DIR, model, optimizer, cur_epoch, cfg)
        
        # ONLY FOR DEBUG PURPOSE! #
        # if cfg.VIDOR.TEST_DEBUG:
        #     eval_epoch(val_loader, model, val_meter, cur_epoch, cfg, writer)
        
        # Evaluate the model on validation set.
        if misc.is_eval_epoch(
            cfg, cur_epoch, None if multigrid is None else multigrid.schedule
        ) and du.is_master_proc():
            logger.info('evaluating...')
            eval_epoch(val_loader, model, val_meter, cur_epoch, cfg, writer)

    if writer is not None:
        writer.close()
