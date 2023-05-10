import torch
import logging
import random
import numpy as np
from torch.nn import functional as F

from utils.riskmap.car import WB, pi_2_pi

def initLogging(log_file: str, level: str = "INFO"):
    logging.basicConfig(filename=log_file, filemode='w',
                        level=getattr(logging, level, None),
                        format='[%(levelname)s %(asctime)s] %(message)s',
                        datefmt='%m-%d %H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler())

def set_seed(CUR_SEED):
    random.seed(CUR_SEED)
    np.random.seed(CUR_SEED)
    torch.manual_seed(CUR_SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def MFMA_loss(plans, predictions, scores, ground_truth, weights, use_planning):

    predictions = predictions * weights.unsqueeze(1)
    prediction_distance = torch.norm(predictions[:, :, :, 9::10, :2] - ground_truth[:, None, 1:, 9::10, :2], dim=-1)
    plan_distance = torch.norm(plans[:, :, 9::10, :2] - ground_truth[:, None, 0, 9::10, :2], dim=-1)
    prediction_distance = prediction_distance.mean(-1).sum(-1)
    plan_distance = plan_distance.mean(-1)

    best_mode = torch.argmin(plan_distance+prediction_distance, dim=-1) 
    score_loss = F.cross_entropy(scores, best_mode)
    best_mode_plan = torch.stack([plans[i, m] for i, m in enumerate(best_mode)])
    best_mode_prediction = torch.stack([predictions[i, m] for i, m in enumerate(best_mode)])

    prediction_loss: torch.tensor = 0
    for i in range(10):
        prediction_loss += F.smooth_l1_loss(best_mode_prediction[:, i], ground_truth[:, i+1, :, :3])

    if not use_planning:
        imitation_loss = F.smooth_l1_loss(best_mode_plan, ground_truth[:, 0, :, :3])
        imitation_loss += F.smooth_l1_loss(best_mode_plan[:, -1], ground_truth[:, 0, -1, :3])
        
        return 0.5 * prediction_loss + imitation_loss + score_loss, best_mode
    else:
        return 0.5 * prediction_loss + score_loss, best_mode

def select_future(plans, predictions, scores = None, best_mode = None):
    best_mode = torch.argmax(scores,dim=-1) if best_mode is None else best_mode
    plan = torch.stack([plans[i, m] for i, m in enumerate(best_mode)])
    prediction = torch.stack([predictions[i, m] for i, m in enumerate(best_mode)])

    return plan, prediction

def imitation_loss(plans, ground_truth, FDE = 0.2, SDE = 0):
    loss = F.smooth_l1_loss(plans[...,:2], ground_truth[:, 0, :, :2])
    loss += FDE * F.smooth_l1_loss(plans[..., -1,:2], ground_truth[:, 0, -1, :2]) if FDE > 0 else 0
    loss += SDE * F.smooth_l1_loss(plans[..., 0,:2], ground_truth[:, 0, 0, :2]) if SDE > 0 else 0
    return loss

def motion_metrics(plan_trajectory, prediction_trajectories, ground_truth_trajectories, weights):
    prediction_trajectories = prediction_trajectories * weights
    plan_distance = torch.norm(plan_trajectory[..., :2] - ground_truth_trajectories[..., 0, :, :2], dim=-1)
    prediction_distance = torch.norm(prediction_trajectories[:, :, :, :2] - ground_truth_trajectories[:, 1:, :, :2], dim=-1)

    # planning
    plannerADE = torch.mean(plan_distance)
    plannerFDE = torch.mean(plan_distance[..., -1])
    # prediction
    predictorADE = torch.mean(prediction_distance, dim=-1)
    predictorADE = torch.masked_select(predictorADE, weights[:, :, 0, 0])
    predictorADE = torch.mean(predictorADE)
    predictorFDE = prediction_distance[:, :, -1]
    predictorFDE = torch.masked_select(predictorFDE, weights[:, :, 0, 0])
    predictorFDE = torch.mean(predictorFDE)

    return plannerADE.item(), plannerFDE.item(), predictorADE.item(), predictorFDE.item()

def project_to_frenet_frame(traj, ref_line):
    distance_to_ref = torch.cdist(traj[:, :, :2], ref_line[:, :, :2])
    k = torch.argmin(distance_to_ref, dim=-1).view(-1, traj.shape[1], 1).expand(-1, -1, 3)
    ref_points = torch.gather(ref_line, 1, k)
    x_r, y_r, theta_r = ref_points[:, :, 0], ref_points[:, :, 1], ref_points[:, :, 2] 
    x, y = traj[:, :, 0], traj[:, :, 1]
    s = 0.1 * (k[:, :, 0] - 200)
    l = torch.sign((y-y_r)*torch.cos(theta_r)-(x-x_r)*torch.sin(theta_r)) * torch.sqrt(torch.square(x-x_r)+torch.square(y-y_r))
    sl = torch.stack([s, l], dim=-1)

    return sl

def project_to_cartesian_frame(traj, ref_line, with_yaw = False):
    k = (10 * traj[:, :, 0] + 200).long()
    k = torch.clip(k, 0, 1200-1)
    ref_points = torch.gather(ref_line, 1, k.view(-1, traj.shape[1], 1).expand(-1, -1, 3))
    x_r, y_r, theta_r = ref_points[:, :, 0], ref_points[:, :, 1], ref_points[:, :, 2] 
    x = x_r - traj[:, :, 1] * torch.sin(theta_r)
    y = y_r + traj[:, :, 1] * torch.cos(theta_r)
    xy = torch.stack([x, y], dim=-1)
    if with_yaw:
        yaw = theta_r + traj[:, :, 2]
        dx = traj[:,:,3]*torch.cos(theta_r) - traj[:,:,4]*torch.sin(theta_r)
        dy = traj[:,:,3]*torch.sin(theta_r) + traj[:,:,4]*torch.cos(theta_r)
        xy = torch.stack( [x,y,yaw,dx,dy],dim=-1)

    return xy
