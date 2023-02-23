import sys 
sys.path.append("..") 
from metrics.metric import Metric
from typing import Dict, Union
import torch
from metrics.utils import min_ade
from datasets.nuScenes.nuScenes_graphs import safety_checker, traj2patch, visualization


class ViolationK(Metric):
    """
    Violation rate the top K trajectories.
    """
    def __init__(self, args: Dict):
        self.k = args['k']
        self.name = 'violation_' + str(self.k)
        self.canvas_size = args['canvas_size']

    def compute(self, predictions: Dict, ground_truth: Union[Dict, torch.Tensor], safemap: torch.Tensor, ini_pose: torch.Tensor) -> torch.Tensor:
        """
        Compute MinADEK
        :param predictions: Dictionary with 'traj': predicted trajectories and 'probs': mode probabilities
        :param ground_truth: Either a tensor with ground truth trajectories or a dictionary
        :return:
        """
        # Unpack arguments
        # mu_x = pred_dist[:, :, 0]
        # mu_y = pred_dist[:, :, 1]
        
        if predictions['traj'].shape[-1] == 5:
            traj_pred = predictions['traj'][...,:2] # storch.Size([32, 10, 12, 2])
        else:
            traj_pred = predictions['traj']
        probs = predictions['probs']
        traj_gt = ground_truth['traj'] if type(ground_truth) == dict else ground_truth
        init_pose = ini_pose
        safe_mask = safemap

        # Useful params
        batch_size = probs.shape[0]
        num_pred_modes = traj_pred.shape[1]
        # sequence_length = traj_pred.shape[2]

        # Masks for variable length ground truth trajectories
        # masks = ground_truth['masks'] if type(ground_truth) == dict and 'masks' in ground_truth.keys() \
        #     else torch.zeros(batch_size, sequence_length).to(traj_pred.device)

        # min_k = min(self.k, num_pred_modes)

        # _, inds_topk = torch.topk(probs, min_k, dim=1) # torch.Size([32, 10])
        # batch_inds = torch.arange(batch_size).unsqueeze(1).repeat(1, min_k) # torch.Size([32, 10])
        # traj_topk = traj[batch_inds, inds_topk] # torch.Size([32, 10, 12, 2])

        # errs, _ = min_ade(traj_topk, traj_gt, masks)


        image_center = [self.canvas_size[0]//2, self.canvas_size[1]//2] # image center
        traj_pred_fit = traj2patch(init_pose, traj_pred, image_center) # torch.Size([32, 10, 12, 2])
        # visualization(traj, torch.tensor(safe_mask).unsqueeze(0), idx, sample_idx=0)    
        safety_pred, safety_pred_step, _, _ = safety_checker(traj_pred_fit, safe_mask)

        # traj_gt_fit = traj2patch(init_pose, traj_gt.unsqueeze(1), image_center)
        # # visualization(traj, torch.tensor(safe_mask).unsqueeze(0), idx, sample_idx=0)    
        # safety_gt, safety_gt_step, _, _ = safety_checker(traj_gt_fit, safe_mask)
        
        # print('safety_gt_step', safety_gt_step[0])
        # visualization(traj_gt_fit, safe_mask, 0, sample_idx=0)  
        # print('safety_pred_step', safety_pred_step[0])  
        # visualization(traj_pred_fit, safe_mask, 0, sample_idx=0) 

        return safety_pred.sum() / (batch_size * num_pred_modes)

