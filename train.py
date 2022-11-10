import torch
import sys
import csv
import time
import argparse
import logging
import os
import numpy as np
from torch import nn, optim
from utils.train_utils import *
from model.planner import MotionPlanner
from model.predictor import Predictor
from torch.utils.data import DataLoader
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler as DSample
from datetime import timedelta

def train_epoch(data_loader, predictor, planner, optimizer, use_planning, epoch):
    epoch_loss = []
    epoch_metrics = []
    current = 0
    size = len(data_loader.dataset)
    predictor.train()
    start_time = time.time()

    for it,batch in enumerate(data_loader):
        # try:
            # prepare data
            ego = batch[0].to(args.local_rank)
            neighbors = batch[1].to(args.local_rank)
            map_lanes = batch[2].to(args.local_rank)
            map_crosswalks = batch[3].to(args.local_rank)
            ref_line_info = batch[4].to(args.local_rank)
            ground_truth = batch[5].to(args.local_rank)
            current_state = torch.cat([ego.unsqueeze(1), neighbors[..., :-1]], dim=1)[:, :, -1]
            weights = torch.ne(ground_truth[:, 1:, :, :3], 0)

            # predict
            optimizer.zero_grad()
            plans, predictions, scores, cost_function_weights = predictor(ego, neighbors, map_lanes, map_crosswalks)
            plan_trajs = torch.stack([bicycle_model(plans[:, i], ego[:, -1])[:, :, :3] for i in range(3)], dim=1)
            loss = MFMA_loss(plan_trajs, predictions, scores, ground_truth, weights, use_planning) # multi-future multi-agent loss
            
            # plan
            if use_planning:
                plan, prediction = select_future(plans, predictions, scores)

                planner_inputs = {
                    "control_variables": plan.view(-1, 100), # initial control sequence
                    "predictions": prediction.detach(), # prediction for surrounding vehicles 
                    "ref_line_info": ref_line_info,
                    "current_state": current_state
                }

                for i in range(cost_function_weights.shape[1]):
                    planner_inputs[f'cost_function_weight_{i+1}'] = cost_function_weights[:, i].unsqueeze(1)

                final_values, info = planner.layer.forward(planner_inputs)
                plan = final_values["control_variables"].view(-1, 50, 2)
                plan = bicycle_model(plan, ego[:, -1])[:, :, :3]

                plan_cost = planner.objective.error_squared_norm().mean() / planner.objective.dim()
                plan_loss = F.smooth_l1_loss(plan, ground_truth[:, 0, :, :3]) 
                plan_loss += F.smooth_l1_loss(plan[:, -1], ground_truth[:, 0, -1, :3])
                loss += plan_loss + 1e-3 * plan_cost # planning loss
            else:
                plan, prediction = select_future(plan_trajs, predictions, scores)

            # loss backward
            loss.backward()
            nn.utils.clip_grad_norm_(predictor.parameters(), 5)
            optimizer.step()

            # compute metrics
            metrics = motion_metrics(plan, prediction, ground_truth, weights)
            epoch_metrics.append(metrics)
            epoch_loss.append(loss.item())

            # show loss
            current += batch[0].shape[0]
            sys.stdout.write(f"\rTrain Progress: [{current:>6d}/{size:>6d}] \
            Loss: {np.mean(epoch_loss):>.4f} \
            {(time.time()-start_time)/current:>.4f}s/sample \
            T2A(epoch): {((time.time()-start_time)/current)*(size-current):>.4f}"
            )
            sys.stdout.flush()
        # except:
        #     print(">>>>skip====>>>>")
            if args.local_rank==0 and use_planning and it%250==0:
                torch.save(predictor.state_dict(), f'training_log/{args.name}/model_{epoch+1}.pth')
                logging.info(f"Model saved in training_log/{args.name}\n")    
            dist.barrier()
    # show metrics
    epoch_metrics = np.array(epoch_metrics)
    plannerADE, plannerFDE = np.mean(epoch_metrics[:, 0]), np.mean(epoch_metrics[:, 1])
    predictorADE, predictorFDE = np.mean(epoch_metrics[:, 2]), np.mean(epoch_metrics[:, 3])
    epoch_metrics = [plannerADE, plannerFDE, predictorADE, predictorFDE]
    logging.info(f'\nplannerADE: {plannerADE:.4f}, plannerFDE: {plannerFDE:.4f}, predictorADE: {predictorADE:.4f}, predictorFDE: {predictorFDE:.4f}')
        
    return np.mean(epoch_loss), epoch_metrics

def valid_epoch(data_loader, predictor, planner, use_planning):
    epoch_loss = []
    epoch_metrics = []
    current = 0
    size = len(data_loader.dataset)
    predictor.eval()
    start_time = time.time()

    for batch in data_loader:
        # prepare data
        ego = batch[0].to(args.local_rank)
        neighbors = batch[1].to(args.local_rank)
        map_lanes = batch[2].to(args.local_rank)
        map_crosswalks = batch[3].to(args.local_rank)
        ref_line_info = batch[4].to(args.local_rank)
        ground_truth = batch[5].to(args.local_rank)
        current_state = torch.cat([ego.unsqueeze(1), neighbors[..., :-1]], dim=1)[:, :, -1]
        weights = torch.ne(ground_truth[:, 1:, :, :3], 0)

        # predict
        with torch.no_grad():
            plans, predictions, scores, cost_function_weights = predictor(ego, neighbors, map_lanes, map_crosswalks)
            plan_trajs = torch.stack([bicycle_model(plans[:, i], ego[:, -1])[:, :, :3] for i in range(3)], dim=1)
            loss = MFMA_loss(plan_trajs, predictions, scores, ground_truth, weights, use_planning) # multi-future multi-agent loss

        # plan 
        if use_planning:
            plan, prediction = select_future(plans, predictions, scores)

            planner_inputs = {
                "control_variables": plan.view(-1, 100), # generate initial control sequence
                "predictions": prediction, # generate predictions for surrounding vehicles 
                "ref_line_info": ref_line_info,
                "current_state": current_state
            }

            for i in range(cost_function_weights.shape[1]):
                planner_inputs[f'cost_function_weight_{i+1}'] = cost_function_weights[:, i].unsqueeze(1)

            with torch.no_grad():
                final_values, info = planner.layer.forward(planner_inputs)

            plan = final_values["control_variables"].view(-1, 50, 2)
            plan = bicycle_model(plan, ego[:, -1])[:, :, :3]

            plan_cost = planner.objective.error_squared_norm().mean() / planner.objective.dim()
            plan_loss = F.smooth_l1_loss(plan, ground_truth[:, 0, :, :3])
            plan_loss += F.smooth_l1_loss(plan[:, -1], ground_truth[:, 0, -1, :3])
            loss += plan_loss + 1e-3 * plan_cost # planning loss
        else:
            plan, prediction = select_future(plan_trajs, predictions, scores)

        # compute metrics
        metrics = motion_metrics(plan, prediction, ground_truth, weights)
        epoch_metrics.append(metrics)
        epoch_loss.append(loss.item())

        # show progress
        current += batch[0].shape[0]
        sys.stdout.write(f"\rValid Progress: [{current:>6d}/{size:>6d}]  Loss: {np.mean(epoch_loss):>.4f}  {(time.time()-start_time)/current:>.4f}s/sample")
        sys.stdout.flush()

    epoch_metrics = np.array(epoch_metrics)
    plannerADE, plannerFDE = np.mean(epoch_metrics[:, 0]), np.mean(epoch_metrics[:, 1])
    predictorADE, predictorFDE = np.mean(epoch_metrics[:, 2]), np.mean(epoch_metrics[:, 3])
    epoch_metrics = [plannerADE, plannerFDE, predictorADE, predictorFDE]
    logging.info(f'\nval-plannerADE: {plannerADE:.4f}, val-plannerFDE: {plannerFDE:.4f}, val-predictorADE: {predictorADE:.4f}, val-predictorFDE: {predictorFDE:.4f}')

    return np.mean(epoch_loss), epoch_metrics

def model_training():
    # Logging
    log_path = f"./training_log/{args.name}/"
    os.makedirs(log_path, exist_ok=True)
    initLogging(log_file=log_path+'train.log')

    logging.info("------------- {} -------------".format(args.name))
    logging.info("Batch size: {}".format(args.batch_size))
    logging.info("Learning rate: {}".format(args.learning_rate))
    logging.info("Use integrated planning module: {}".format(args.use_planning))
    logging.info("Use device: {}".format(args.device))

    args.local_rank = int(os.environ["LOCAL_RANK"]) if "LOCAL_RANK" in os.environ else args.local_rank
    args.world_size = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else args.world_size
    set_seed(args.seed+args.local_rank)

    dist.init_process_group(backend="nccl")
    # set up predictor
    predictor = Predictor(50).to(args.local_rank)
    # predictor = DDP(predictor,device_ids=[args.local_rank],find_unused_parameters=True)
    predictor = DDP(
                    predictor,
                    device_ids=[args.local_rank],
                    output_device=args.local_rank,
                    find_unused_parameters=True
                    )

    # set up planner
    if args.use_planning:
        trajectory_len, feature_len = 50, 9
        planner = MotionPlanner(trajectory_len, feature_len, args.local_rank)
    else:
        planner = None

    if args.ckpt:
        map_location = {'cuda:%d' % 0: 'cuda:%d' % args.local_rank}
        predictor.load_state_dict(torch.load(args.ckpt,map_location = map_location)) 
    
    # set up optimizer
    optimizer = optim.Adam(predictor.parameters(), lr=args.learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.5)

    # training parameters
    train_epochs = args.train_epochs
    batch_size = args.batch_size
    
    # set up data loaders
    train_set = DrivingData(args.train_set+'/*')
    valid_set = DrivingData(args.valid_set+'/*')
    train_sampler = DSample(train_set,shuffle=False)
    valid_sampler = DSample(valid_set,shuffle=False)
    train_loader = DataLoader(train_set, batch_size=batch_size, num_workers=args.num_workers,sampler=train_sampler)
    valid_loader = DataLoader(valid_set, batch_size=batch_size, num_workers=args.num_workers,sampler=valid_sampler)

    logging.info("Dataset Prepared: {} train data, {} validation data\n".format(len(train_set), len(valid_set)))
    # begin training
    for epoch in range(train_epochs):
        logging.info(f"Epoch {epoch+1}/{train_epochs}")
        
        # train 
        if planner:
            if epoch < args.pretrain_epochs:
                args.use_planning = False
            else:
                args.use_planning = True         

        train_loss, train_metrics = train_epoch(train_loader, predictor, planner, optimizer, args.use_planning, epoch)
        val_loss, val_metrics = valid_epoch(valid_loader, predictor, planner, args.use_planning)

        # save to training log
        if args.local_rank==0:
            log = {'epoch': epoch+1, 'loss': train_loss, 'lr': optimizer.param_groups[0]['lr'], 'val-loss': val_loss, 
                'train-plannerADE': train_metrics[0], 'train-plannerFDE': train_metrics[1], 
                'train-predictorADE': train_metrics[2], 'train-predictorFDE': train_metrics[3],
                'val-plannerADE': val_metrics[0], 'val-plannerFDE': val_metrics[1], 
                'val-predictorADE': val_metrics[2], 'val-predictorFDE': val_metrics[3]}

            if epoch == 0:
                with open(f'./training_log/{args.name}/train_log.csv', 'w') as csv_file: 
                    writer = csv.writer(csv_file) 
                    writer.writerow(log.keys())
                    writer.writerow(log.values())
            else:
                with open(f'./training_log/{args.name}/train_log.csv', 'a') as csv_file: 
                    writer = csv.writer(csv_file)
                    writer.writerow(log.values())

        # reduce learning rate
        scheduler.step()

        # save model at the end of epoch
        if args.local_rank==0:
            torch.save(predictor.state_dict(), f'training_log/{args.name}/model_{epoch+1}_{val_metrics[0]:.4f}.pth')
            logging.info(f"Model saved in training_log/{args.name}\n")    
        dist.barrier()
    dist.destroy_process_group()

if __name__ == "__main__":
    # Arguments
    parser = argparse.ArgumentParser(description='Training')
    parser.add_argument('--name', type=str, help='log name (default: "Exp1")', default="Exp1")
    parser.add_argument('--train_set', type=str, help='path to train datasets')
    parser.add_argument('--valid_set', type=str, help='path to validation datasets')
    parser.add_argument('--seed', type=int, help='fix random seed', default=42)
    parser.add_argument("--num_workers", type=int, default=8, help="number of workers used for dataloader")
    parser.add_argument('--pretrain_epochs', type=int, help='epochs of pretraining predictor', default=5)
    parser.add_argument('--train_epochs', type=int, help='epochs of training', default=20)
    parser.add_argument('--batch_size', type=int, help='batch size (default: 32)', default=32)
    parser.add_argument('--learning_rate', type=float, help='learning rate (default: 2e-4)', default=2e-4)
    parser.add_argument('--use_planning', action="store_true", help='if use integrated planning module (default: False)', default=False)
    parser.add_argument('--device', type=str, help='run on which device (default: cuda)', default='cuda')
    parser.add_argument('--ckpt', type=str, default=None)
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--world_size", type=int, default=-1)

    args = parser.parse_args()
    # os.environ["NCCL_DEBUG"] = "INFO"
    # os.environ["TORCH_CPP_LOG_LEVEL"]="INFO"
    # os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"
    # Run
    model_training()
