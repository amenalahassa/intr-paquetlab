# ------------------------------------------------------------------------
# INTR
# Copyright (c) 2023 Imageomics Paul. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Copied from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# ------------------------------------------------------------------------
"""
Train and eval functions used in main.py
"""
import math
import os
import sys
from typing import Iterable
import torch
import util.misc as utils

def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0):

    model.train()
    criterion.train()
    args = criterion.args

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10
    labels = []
    preds = []
    losses = []
    input_weights = []

    for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
        samples = samples.to(device)

        if args.loss_as == "weighted":
            weights = torch.tensor([t["weight"] for t in targets]).to(device)
            class_weights = data_loader.dataset.class_weights.to(device)
        else:
            weights = None
            class_weights = None
            
        filenames=[t["file_name"] for t in targets]
        for t in targets:
            del t["file_name"]

        # tg = [t['image_label'].item() for t in targets]
        # counts = {item: tg.count(item) for item in set(tg)}
        # print(f"Number of items per class: {counts}")
        
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        outputs, _ ,_ ,_ ,_  = model(samples)

        
        labels.extend([t["image_label"] for t in targets])
        preds.extend(outputs['query_logits'])
        
        if args.num_queries > 1:
            loss_dict = criterion(outputs, targets, model, weights = class_weights)
        else:
            loss_dict = criterion(outputs, targets, model, weights = weights)
            
        ## INTR uses only one type of loss i.e., CE loss
        loss = sum(loss_dict[k] for k in loss_dict.keys())

        ## reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_value =sum(loss_dict_reduced.values())

        losses.append(loss_value.cpu().detach().numpy())
        input_weights.extend(weights.cpu())
        
        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        loss.backward()
        if max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()

        # acc1, acc5, _ = utils.class_accuracy(outputs, targets, topk=(1, 5))
        # acc = accuracy_score(outputs, targets)
        # acc, f1, precision, recall, b_acc = utils.class_f1_precision_recall(outputs, targets, args, weights = weights.cpu())
        
        # metric_logger.update(loss=loss_value)
        # metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        # metric_logger.update(acc=acc)
        # # metric_logger.update(acc5=acc5)

        # metric_logger.update(f1=f1)
        # metric_logger.update(precision=precision)
        # metric_logger.update(recall=recall)
        # metric_logger.update(binary_acc=b_acc)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    acc, f1, precision, recall, b_acc = utils.class_f1_precision_recall(preds, labels, args, model_output = False, weights = input_weights)
    # print("Averaged stats:", metric_logger)
    print("Stats: Accuracy {acc}, F1 {f1}, Recall {rc}, Precision {pr}, Balanced Accuracy {b_acc}".format(acc=acc, f1=f1, rc=recall, pr=precision, b_acc=b_acc))

    # print("Averaged stats:", metric_logger)
    # stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    # stats["acc_std"] = metric_logger.meters["acc"].std
    stats = {
        "loss":  sum(losses) / len(losses),
        "acc": acc ,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "binary_acc": b_acc,
    }
    return stats


@torch.no_grad()
def evaluate(model, criterion,  data_loader, device, output_dir, plot_confusion = False, labels_name = None):
    model.eval()
    criterion.eval()
    args = criterion.args

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'
    labels = []
    preds = []
    losses = []
    input_weights = []

    for samples, targets in metric_logger.log_every(data_loader, 10, header):
        samples = samples.to(device)

        if args.loss_as == "weighted":
            weights = torch.tensor([t["weight"] for t in targets]).to(device)
            class_weights = data_loader.dataset.class_weights.to(device)
        else:
            weights = None
            class_weights = None
            
        filenames=[t["file_name"] for t in targets]
        for t in targets:
            del t["file_name"]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        outputs,_,_,_,_ = model(samples)
        
        # query_logits = 
        # target_classes = 
    
        # Assuming the highest logit value corresponds to the predicted class
        # _, pred = torch.max(query_logits, dim=1)
    
        # # Convert to numpy arrays for sklearn metrics
        # pred_np = pred.cpu().numpy()
        # target_np = target_classes.cpu().numpy()

        labels.extend([t["image_label"] for t in targets])
        preds.extend(outputs['query_logits'])
        
        if args.num_queries > 1:
            loss_dict = criterion(outputs, targets, model, weights = class_weights)
        else:
            loss_dict = criterion(outputs, targets, model, weights = weights)
            
        # ## reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_value = sum(loss_dict_reduced.values())

        losses.append(loss_value.cpu().detach().numpy())
        input_weights.extend(weights.cpu())
        
        # metric_logger.update(loss=loss_value)                                   
        # acc1, acc5, _ = utils.class_accuracy(outputs, targets, topk=(1, 5))
        # metric_logger.update(acc1=acc1)

        # f1, precision, recall = utils.class_f1_precision_recall(outputs, targets)
        
        # acc, f1, precision, recall, b_acc = utils.class_f1_precision_recall(outputs, targets, args, weights = weights.cpu())

        # metric_logger.update(f1=f1)
        # metric_logger.update(precision=precision)
        # metric_logger.update(recall=recall)
        # metric_logger.update(acc=acc)
        # metric_logger.update(binary_acc=b_acc)

    # acc = accuracy_score(preds, labels)
    # precision, recall, f1, _  = precision_recall_fscore_support(preds, labels, average='macro')
    # acc, f1, precision, recall = utils.class_f1_precision_recall(preds, labels, criterion.args, averager='macro')
    
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    acc, f1, precision, recall, b_acc = utils.class_f1_precision_recall(preds, labels, args, model_output = False, weights = input_weights)
    # print("Averaged stats:", metric_logger)
    print("Stats: Accuracy {acc}, F1 {f1}, Recall {rc}, Precision {pr}, Balanced Accuracy {b_acc}".format(acc=acc, f1=f1, rc=recall, pr=precision, b_acc=b_acc))
    # stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    stats = {
        "loss":  sum(losses) / len(losses),
        "acc": acc ,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "binary_acc": b_acc,
    }

    if plot_confusion: 
        utils.plot_confusion(preds, labels, args, model_output = False, labels_name=labels_name)
    
    return stats