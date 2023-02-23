import sys 
sys.path.append("..") 
from metrics.metric import Metric
from typing import Dict, Union
import torch
from metrics.utils import min_ade
from datasets.nuScenes.nuScenes_graphs import safety_checker, traj2patch
from metrics.utils import min_ade, traj_nll

class MLELoss(Metric):
    """
    Violation rate the top K trajectories.
    """
    def __init__(self, args: Dict):
        self.name = 'MLELoss'
        self.log_p_yt_xz_max = args['log_p_yt_xz_max'] if args is not None and 'log_p_yt_xz_max' in args.keys() else 6
        # self.log_p_yt_xz_min = args['log_p_yt_xz_min'] if args is not None and 'log_p_yt_xz_min' in args.keys() else -1e2

        self.alpha = args['alpha'] if args is not None and 'alpha' in args.keys() else 1
        self.beta = args['beta'] if args is not None and 'beta' in args.keys() else 1

    def compute(self, predictions: Dict, ground_truth: Union[Dict, torch.Tensor]) -> torch.Tensor:
        """
        Compute MLELoss
        :param predictions: Dictionary with 'traj': predicted trajectories and 'probs': mode probabilities
        :param ground_truth: Either a tensor with ground truth trajectories or a dictionary
        :return:
        """

        # Unpack arguments
        traj = predictions['traj']
        log_probs = predictions['probs']
        traj_gt = ground_truth['traj'] if type(ground_truth) == dict else ground_truth

        # Useful variables
        batch_size = traj.shape[0]
        sequence_length = traj.shape[2]


        # Masks for variable length ground truth trajectories
        masks = ground_truth['masks'] if type(ground_truth) == dict and 'masks' in ground_truth.keys() \
            else torch.zeros(batch_size, sequence_length).to(traj.device)

        # Obtain mode with minimum ADE with respect to ground truth:
        _, inds = min_ade(traj, traj_gt, masks)
        # Compute classification loss
        # l_class = - torch.squeeze(log_probs.gather(1, inds.unsqueeze(1))).mean()


        # nll loss
        pred_dist = predictions['pred_dist']
        log_p_yt_xz = torch.clamp(pred_dist.log_prob(traj_gt), max=self.log_p_yt_xz_max)
        log_p_y_xz = torch.sum(log_p_yt_xz, dim=2)
        l_nll = -torch.mean(log_p_y_xz)

        # loss = self.beta * l_nll + self.alpha * l_class
        loss = self.beta * l_nll


        return  loss

