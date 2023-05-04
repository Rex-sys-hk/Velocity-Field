# %%
import math
import multiprocessing
import torch
import sys
import csv
import time
import argparse
import logging
import os
import shutil
import numpy as np
from tensorboardX import SummaryWriter
from torch import nn, optim
from common_utils import inference, init_planner, load_checkpoint, save_checkpoint, predictor_selection
from utils.data_loading import DrivingData
from utils.riskmap.car import bicycle_model, traj_smooth_mp
from utils.train_utils import *
from utils.riskmap.rm_utils import get_u_from_X, get_yawv_from_pos, has_nan, load_cfg_here
from model.planner import BasePlanner, CostMapPlanner, EularSamplingPlanner, MotionPlanner, Planner, RiskMapPlanner
from model.predictor import CostVolume, Predictor
from model.predictor_base import STCostMap, VectorField
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler as DSample
import matplotlib
import matplotlib.pyplot as plt

os.environ["DIPP_ABS_PATH"] = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.getenv('DIPP_ABS_PATH'))

def train_epoch(data_loader, predictor: Predictor, planner: Planner, optimizer, use_planning, epoch, distributed = False):
    epoch_loss = []
    epoch_metrics = []
    current = 0
    size = len(data_loader.dataset)
    predictor.train()
    start_time = time.time()
    iter_base = (size/data_loader.batch_size)*epoch
    tb_iters = int(iter_base)
    for it,batch in enumerate(data_loader):
        tb_iters += 1
        # prepare data
        ego = batch[0].to(args.local_rank)
        neighbors = batch[1].to(args.local_rank)
        map_lanes = batch[2].to(args.local_rank)
        map_crosswalks = batch[3].to(args.local_rank)
        ref_line_info = batch[4].to(args.local_rank)
        ground_truth = batch[5].to(args.local_rank)
        lattice_sample = batch[6].to(args.local_rank) if batch[6].sum() else None
        current_state = torch.cat([ego.unsqueeze(1), neighbors[..., :-1]], dim=1)[:, :, -1]
        masks = torch.ne(ground_truth[:, 1:, :, :3], 0) # ne is not equal
        # predict
        optimizer.zero_grad()
        us, predictions, scores, identical_output = predictor(ego, neighbors, map_lanes, map_crosswalks)
        plan_trajs = torch.stack([bicycle_model(us[:, i], ego[:, -1])[:, :, :3] for i in range(cfg['model_cfg']['mode_num'])], dim=1)
        # plan_trajs, predictions, scores, cost_function_weights = predictor(ego, neighbors, map_lanes, map_crosswalks)
        # us = torch.stack([get_u_from_X(plan_trajs[:,i], ego[:,-1]) for i in range(cfg['model_cfg']['mode_num'])], dim=1).clone().detach()
        loss, best_mode = MFMA_loss(plan_trajs, predictions, scores, ground_truth, masks, use_planning) # multi-future multi-agent loss
        u, prediction = select_future(us, predictions, best_mode)
        init_guess, prediction = select_future(plan_trajs, predictions, best_mode)
        # TODO GT smoother
        # plan
        if not use_planning:
            plan, prediction = select_future(plan_trajs, predictions, best_mode)
        ## BASELINE
        elif planner.name=='base':
            loss += imitation_loss(init_guess, ground_truth)
            # loss+=0.1*F.smooth_l1_loss(u[...,-1],torch.zeros_like(u[...,-1])) # reduce steering
            plan = init_guess
        ## DIPP
        elif planner.name=='dipp':
            planner: MotionPlanner = planner
            planner_inputs = {
                "control_variables": u.view(-1, 100), # initial control sequence
                "predictions": prediction.detach(), # prediction for surrounding vehicles 
                "ref_line_info": ref_line_info,
                "current_state": current_state # including neighbor cars
            }

            for i in range(identical_output.shape[1]):
                planner_inputs[f'cost_function_weight_{i+1}'] = identical_output[:, i].unsqueeze(1)
            planner.layer.to(init_guess.device)
            final_values, info = planner.layer.forward(planner_inputs)
            u = final_values["control_variables"].view(-1, 50, 2)
            plan = bicycle_model(u, ego[:, -1])[:, :, :3]

            plan_cost = planner.objective.error_squared_norm().mean() / planner.objective.dim()
            loss += imitation_loss(plan, ground_truth)
            loss += 1e-3 * plan_cost # planning loss
        ## EULA
        elif planner.name=='esp':
            planner:EularSamplingPlanner=planner
            loss+=imitation_loss(init_guess, ground_truth)
            # loss+=0.1*F.smooth_l1_loss(u[...,-1],torch.zeros_like(u[...,-1])) # reduce steering
            planner_inputs = {
                # "control_variables": u.view(-1, 100), # initial control sequence
                "predictions": prediction.detach(), # prediction for surrounding vehicles 
                "ref_line_info": ref_line_info,
                "current_state": current_state, # including neighbor cars
                'init_guess_u': u.detach().clone(),
                "lattice_sample": lattice_sample,
            }
            with torch.no_grad():
                plan,u = planner.plan(planner_inputs, genetic = cfg['planner']['train_genetic'])
            loss += planner.get_loss(ground_truth[...,0:1,:,:], tb_iter=tb_iters, tb_writer=tbwriter)
        ## RISK
        elif planner.name=='risk':
            planner:RiskMapPlanner = planner
            vf_map:VectorField = predictor.module.vf_map if distributed else predictor.vf_map
            loss += imitation_loss(init_guess, ground_truth)
            # loss+=0.1*F.smooth_l1_loss(u[...,-1],torch.zeros_like(u[...,-1])) # reduce steering
            planner_inputs = {
                "predictions": prediction, # prediction for surrounding vehicles 
                "ref_line_info": ref_line_info,
                "current_state": current_state,
                "vf_map": vf_map,
                'init_guess_u': u.detach().clone(),
                "lattice_sample": lattice_sample,
            }
            # plan loss
            with torch.no_grad():
                plan, _ = planner.plan(planner_inputs, genetic = cfg['planner']['train_genetic']) # control
            loss += planner.get_loss(ground_truth[...,0:1,:,:],
                                     tb_iters,
                                     tbwriter)
            loss += vf_map.get_loss(ground_truth[...,0:1,:,:], 
                                    planner.gt_sample['X'],
                                    planner_inputs)
        ## Neural Motion Planner
        elif planner.name=='nmp':
            planner:CostMapPlanner = planner
            cost_map:STCostMap = predictor.module.cost_volume if distributed else predictor.cost_volume
            loss+=0.1*F.smooth_l1_loss(u[...,-1],torch.zeros_like(u[...,-1])) # reduce steering
            # loss += imitation_loss(init_guess, ground_truth)
            planner_inputs = {
                "predictions": prediction, # prediction for surrounding vehicles 
                "ref_line_info": ref_line_info,
                "current_state": current_state,
                "cost_map": cost_map,
                'init_guess_u': u.detach().clone(),
                "lattice_sample": lattice_sample,
            }
            # plan loss
            with torch.no_grad():
                plan, _ = planner.plan(planner_inputs, genetic = cfg['planner']['train_genetic']) # control
            loss += planner.get_loss(ground_truth[...,0:1,:,:],
                                     tb_iters,
                                     tbwriter)
        # loss backward
        loss.backward()
        nn.utils.clip_grad_norm_(predictor.parameters(), 5)
        optimizer.step()
        # compute metrics
        if tb_iters%args.img_log_interval==0:
            if not args.imm_show:
                matplotlib.use('Agg')
            plt.title(f'{args.name}')
            plt.autoscale(False)
            ## special output
            if use_planning:
                if planner.name in ['risk','esp','nmp']:
                    for traj in planner.gt_sample['X'][0].cpu().detach():
                        plt.plot(traj[...,0],traj[...,1],'g', lw=0.3, zorder=2, alpha = 0.2)
                    for traj in planner.sample_plan['X'][0,::10].cpu().detach():
                        plt.plot(traj[...,0],traj[...,1],'r--', lw=0.3, zorder=2, alpha = 0.2)
                if planner.name in ['risk']:
                    vf_map:VectorField = predictor.module.vf_map if distributed else predictor.vf_map
                    with torch.no_grad():
                        vf_map.plot(planner.sample_plan['X'][:,::10])
                        # vf_map.plot_gt(ground_truth[...,0:1,:,:],planner.gt_sample['X'][0:1])
                if planner.name in ['nmp']:
                    cost_map:STCostMap = predictor.module.cost_volume if distributed else predictor.cost_volume
                    cost_map.plot(planner.sample_plan['X'])
                else:
                    plt.axis('equal')
            else:
                plt.axis('equal')
                
            ## general output
            # reference lane
            plt.plot(ref_line_info[0,...,0].cpu().detach(),ref_line_info[0,...,1].cpu().detach(), color = 'yellow', lw=4, zorder=0, alpha=0.5, label='reflane')
            # ego history
            plt.plot(ego[0,...,0].cpu().detach(), ego[0,...,1].cpu().detach(), 'g--', marker = '.',  markersize=1, lw=0.5, zorder=1, alpha=0.5, label='history')
            # ground truth
            plt.plot(ground_truth[0,0,...,0].cpu().detach(),ground_truth[0,0,...,1].cpu().detach(), 'g--', zorder=1, lw=3, alpha=0.5, label='GT')
            # result
            plt.plot(plan[0,...,0].cpu().detach(),plan[0,...,1].cpu().detach(), color = 'cyan', marker = '.',  markersize=1.5, lw=0.6, zorder=4, alpha=0.5, label='plan result')
            for mod in plan_trajs[0].cpu().detach():
                plt.plot(mod[...,0], mod[...,1], 'grey', lw=0.5, zorder=2, alpha = 0.5)
            # prediction_t = prediction*masks
            # for nei in range(10):
            #     plt.plot(prediction_t[0,nei,...,0].cpu().detach(),prediction_t[0,nei,...,1].cpu().detach(),'y--')
            #     plt.plot(ground_truth[0,nei+1,...,0].cpu().detach(),ground_truth[0,nei+1,...,1].cpu().detach(),'k')
            plt.xlim(-20,100)
            plt.ylim(-40,40)
            plt.legend()
            if args.imm_show:
                plt.show()
            plt.savefig(f'training_log/{args.name}/images/model_{tb_iters}.png',dpi=400)
            plt.close()
            plt.figure()
        metrics = motion_metrics(plan, prediction, ground_truth, masks)
        epoch_metrics.append(metrics)
        epoch_loss.append(loss.item())
        # logging and show loss
        if args.local_rank==0 or not distributed:
            current += batch[0].shape[0]
            sys.stdout.write(f"\r>>>Train Progress: [{current:>6d}/{size:>6d}] Loss: {np.mean(epoch_loss):>.4f} {(time.time()-start_time)/current:>.4f}s/sample T2A(epoch): {((time.time()-start_time)/current)*(size-current):>.4f}")
            sys.stdout.flush()
            tbwriter.add_scalar('train/'+'iter_loss', loss.mean(), tb_iters)
            tbwriter.add_scalar('train/metrics/'+'planADE', metrics[0], tb_iters)
            tbwriter.add_scalar('train/metrics/'+'planFDE', metrics[1], tb_iters)
            tbwriter.add_scalar('train/metrics/'+'preADE', metrics[2], tb_iters)
            tbwriter.add_scalar('train/metrics/'+'preFDE', metrics[3], tb_iters)

    # show metrics
    epoch_metrics = np.array(epoch_metrics)
    plannerADE, plannerFDE = np.mean(epoch_metrics[:, 0]), np.mean(epoch_metrics[:, 1])
    predictorADE, predictorFDE = np.mean(epoch_metrics[:, 2]), np.mean(epoch_metrics[:, 3])
    epoch_metrics = [plannerADE, plannerFDE, predictorADE, predictorFDE]
    print(f'\n>>>plannerADE: {plannerADE:.4f}, plannerFDE: {plannerFDE:.4f}, predictorADE: {predictorADE:.4f}, predictorFDE: {predictorFDE:.4f}')
    if args.local_rank==0 or not distributed:
        tbwriter.add_scalar('train/'+'epoch_loss', np.mean(epoch_loss), epoch)
        tbwriter.add_scalar('train/'+'plannerADE', np.mean(plannerADE), epoch)
        tbwriter.add_scalar('train/'+'plannerFDE', np.mean(plannerFDE), epoch)
        tbwriter.add_scalar('train/'+'predictorADE', np.mean(predictorADE), epoch)
        tbwriter.add_scalar('train/'+'predictorFDE', np.mean(predictorFDE), epoch)

    
    return np.mean(epoch_loss), epoch_metrics

def valid_epoch(data_loader, predictor, planner: Planner, use_planning, epoch, distributed=False):
    epoch_metrics = []
    current = 0
    size = len(data_loader.dataset)
    predictor.eval()
    start_time = time.time()
    iter_base = (size/data_loader.batch_size)*epoch
    tb_iters = iter_base
    for it, batch in enumerate(data_loader):
        tb_iters+=1
        # prepare data
        plan, prediction = inference(batch, predictor, planner, args, use_planning, distributed=distributed, parallel='none')
        ground_truth = batch[5].to(args.local_rank)
        masks = torch.ne(ground_truth[:, 1:, :, :3], 0)
        # compute metrics
        metrics = motion_metrics(plan, prediction, ground_truth, masks)
        epoch_metrics.append(metrics)
        # show progress
        if args.local_rank==0 or not distributed:
            current += batch[0].shape[0]
            sys.stdout.write(f"\r==Valid Progress: [{current:>6d}/{size:>6d}]  Loss: {np.mean(metrics):>.4f}  {(time.time()-start_time)/current:>.4f}s/sample")
            sys.stdout.flush()
            tbwriter.add_scalar('valid/metrics/'+'planADE', metrics[0], tb_iters)
            tbwriter.add_scalar('valid/metrics/'+'planFDE', metrics[1], tb_iters)
            tbwriter.add_scalar('valid/metrics/'+'preADE', metrics[2], tb_iters)
            tbwriter.add_scalar('valid/metrics/'+'preFDE', metrics[3], tb_iters)
    epoch_metrics = np.array(epoch_metrics)
    plannerADE, plannerFDE = np.mean(epoch_metrics[:, 0]), np.mean(epoch_metrics[:, 1])
    predictorADE, predictorFDE = np.mean(epoch_metrics[:, 2]), np.mean(epoch_metrics[:, 3])
    epoch_metrics = [plannerADE, plannerFDE, predictorADE, predictorFDE]
    print(f'\n==val-plannerADE: {plannerADE:.4f}, val-plannerFDE: {plannerFDE:.4f}, val-predictorADE: {predictorADE:.4f}, val-predictorFDE: {predictorFDE:.4f}')
    if args.local_rank==0 or not distributed:
        tbwriter.add_scalar('valid/'+'plannerADE', np.mean(plannerADE), epoch)
        tbwriter.add_scalar('valid/'+'plannerFDE', np.mean(plannerFDE), epoch)
        tbwriter.add_scalar('valid/'+'predictorADE', np.mean(predictorADE), epoch)
        tbwriter.add_scalar('valid/'+'predictorFDE', np.mean(predictorFDE), epoch)
    return 0., epoch_metrics

def model_training():
    # Logging
    log_path = f"./training_log/{args.name}/"
    os.makedirs(log_path, exist_ok=True)
    os.makedirs(os.path.join(log_path,'images'), exist_ok=True)
    os.makedirs(os.path.join(log_path,'ckpt'), exist_ok=True)
    initLogging(log_file=log_path+'train.log')
    shutil.copyfile(os.getenv('DIPP_CONFIG'), f'{log_path}/config.yaml')

    logging.info("------------- {} -------------".format(args.name))
    logging.info("Predictor Name: {}".format(cfg['model_cfg']['name']))
    logging.info("Planner Name: {}".format(cfg['planner']['name']))
    logging.info("Batch size: {}".format(args.batch_size))
    logging.info("Learning rate: {}".format(args.learning_rate))
    logging.info("Use integrated planning module: {}".format(args.use_planning))
    logging.info("Use device: {}".format(args.device))

    args.local_rank = int(os.environ["LOCAL_RANK"]) if "LOCAL_RANK" in os.environ else args.local_rank
    args.world_size = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else args.world_size
    set_seed(args.seed+args.local_rank)
    start_epoch = 0
    best = math.inf
    try:
        dist.init_process_group(backend="nccl")
        distributed = True
    except:
        distributed = False
        logging.warning('[WARNING] Distributed data parallele initialization failed')

    # %% initializing prediction model 
    logging.info(f"Initializing Model: {cfg['model_cfg']['name']}")
    for key in cfg['model_cfg']:
        logging.info(f"--{key}: {cfg['model_cfg'][key]}")
    if not args.ckpt:
        predictor = predictor_selection[cfg['model_cfg']['name']](**cfg['model_cfg']) #Predictor(50)
    else:
        map_location = {'cuda:%d' % 0: 'cuda:%d' % args.local_rank}
        predictor, start_epoch, lr = load_checkpoint(args.ckpt, map_location)
        logging.info(f'ckpt successful loaded from {args.ckpt}')


    # %% initializing planner
    logging.info(f"Initialize planner {cfg['planner']['name']}")
    planner = init_planner(args, cfg, predictor)

    predictor = predictor.to(args.local_rank)
    # set up optimizer
    if args.ckpt:
        optimizer = optim.AdamW(predictor.parameters(), lr=lr)
    else:
        optimizer = optim.AdamW(predictor.parameters(), lr=args.learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=cfg['model_cfg']['gamma'])

    # training parameters
    train_epochs = args.train_epochs
    batch_size = args.batch_size
    
    # set up data loaders
    train_set = DrivingData(args.train_set+'/*',data_aug=cfg['data_aug'])
    valid_set = DrivingData(args.valid_set+'/*')
    if distributed:
        predictor = DDP(
                predictor,
                device_ids=[args.local_rank],
                output_device=args.local_rank,
                find_unused_parameters=False
                )
        train_sampler = DSample(train_set,shuffle=True)
        valid_sampler = DSample(valid_set,shuffle=False)
    else:
        train_sampler = RandomSampler(train_set)
        valid_sampler = SequentialSampler(valid_set)
    train_loader = DataLoader(train_set, batch_size=batch_size, num_workers=args.num_workers,sampler=train_sampler)
    valid_loader = DataLoader(valid_set, batch_size=batch_size, num_workers=args.num_workers,sampler=valid_sampler)

    logging.info("Dataset Prepared: {} train data, {} validation data\n".format(len(train_set), len(valid_set)))
    # %% begin training
    for epoch in range(start_epoch, train_epochs):
        logging.info(f"Epoch {epoch+1}/{train_epochs}")
        
        # train 
        if planner:
            if epoch < args.pretrain_epochs:
                use_planning = False
            else:
                use_planning = True
        # btsz = batch_size if args.use_planning else batch_size*4
        # train_loader = DataLoader(train_set, batch_size=btsz, num_workers=args.num_workers,sampler=train_sampler)
        # valid_loader = DataLoader(valid_set, batch_size=btsz, num_workers=args.num_workers,sampler=valid_sampler)

        train_loss, train_metrics = train_epoch(train_loader, predictor, planner, optimizer, use_planning, epoch, distributed)
        val_loss, val_metrics = valid_epoch(valid_loader, predictor, planner, use_planning, epoch, distributed)

        # save to training log
        if args.local_rank==0 or not distributed:
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
        if args.local_rank==0 or not distributed:
            ckpt_file_name = f'training_log/{args.name}/ckpt/model_newest.pth.tar'
            save_checkpoint(epoch,ckpt_file_name,cfg,predictor,dist=distributed)
            if val_metrics[0]<best:
                ckpt_file_name = f'training_log/{args.name}/ckpt/model_best_{epoch+1}_{val_metrics[0]:.4f}.pth.tar'
                save_checkpoint(epoch,ckpt_file_name,cfg,predictor,dist=distributed)
                best = val_metrics[0]
                
        if distributed:
            dist.barrier()
    if distributed:
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
    parser.add_argument(
        "-c", "--config", help="Config file with dataset parameters", required=False, default=None, type=str)
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--world_size", type=int, default=1)
    parser.add_argument("--img_log_interval", type=int, default=100)
    parser.add_argument('--imm_show', action="store_true", help='if show image immediately', default=False)
    parser.add_argument('--opt_plan', action="store_true", help='enable optimization based planning', default=False)

    args = parser.parse_args()
    distributed = False
    # os.environ["NCCL_DEBUG"] = "INFO"
    # os.environ["TORCH_CPP_LOG_LEVEL"]="INFO"
    # os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"
    # Run
    cfg_file = args.config if args.config else 'config.yaml'
    os.environ["DIPP_CONFIG"] = str(os.getenv('DIPP_ABS_PATH') + '/' + cfg_file)
    cfg = load_cfg_here()
    tb_iters = 0
    tbwriter = SummaryWriter(
    log_dir=os.path.join(f'training_log/{args.name}', 'tensorboard_logs')
    )
    torch.autograd.set_detect_anomaly(False)
    torch.cuda.empty_cache() 
    logging.basicConfig(level=logging.ERROR)
    model_training()
