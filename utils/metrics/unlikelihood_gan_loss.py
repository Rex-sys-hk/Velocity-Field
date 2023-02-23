import sys 
sys.path.append("..") 
from metrics.metric import Metric
from typing import Dict, Union
import torch
from metrics.utils import min_ade
from datasets.nuScenes.nuScenes_graphs import safety_checker, traj2patch
from metrics.utils import traj_nll

class UnlikelihoodGanLoss(Metric):
    """
    Violation rate the top K trajectories.
    """
    def __init__(self, args: Dict):
        self.k = args['k']
        self.name = 'unlike_' + str(self.k)
        self.canvas_size = args['canvas_size']
        self.log_p_yt_xz_max = args['log_p_yt_xz_max']
        self.alpha = args['alpha'] if args is not None and 'alpha' in args.keys() else 1
        self.beta = args['beta'] if args is not None and 'beta' in args.keys() else 1
        self.gamma = args['gamma'] if args is not None and 'gamma' in args.keys() else 0
        self.log_bias = args['log_bias'] if args is not None and 'log_bias' in args.keys() else 1e-09
        self.loss_times = 0

    def compute(self, predictions: Dict, ground_truth: Union[Dict, torch.Tensor], safemap: torch.Tensor, ini_pose: torch.Tensor, gamma) -> torch.Tensor:
        """
        Compute MinADEK
        :param predictions: Dictionary with 'traj': predicted trajectories and 'probs': mode probabilities
        :param ground_truth: Either a tensor with ground truth trajectories or a dictionary
        :return:
        """
        # Unpack arguments
        log_probs = predictions['probs']
        if predictions['traj'][-1].shape == 5:
            traj_pred = predictions['traj'][...,:2] # storch.Size([32, 10, 12, 2])
        else:
            traj_pred = predictions['traj']
        traj_gt = ground_truth['traj'] if type(ground_truth) == dict else ground_truth
        init_pose = ini_pose
        safe_mask = safemap

        # Useful params
        batch_size = log_probs.shape[0]
        num_pred_modes = traj_pred.shape[1]
        sequence_length = traj_pred.shape[2]

        # Masks for variable length ground truth trajectories
        masks = ground_truth['masks'] if type(ground_truth) == dict and 'masks' in ground_truth.keys() \
            else torch.zeros(batch_size, sequence_length).to(traj_pred.device)

        # Obtain mode with minimum ADE with respect to ground truth:
        # _, inds = min_ade(traj_pred, traj_gt, masks)
        # # Compute classification loss
        # l_class = - torch.squeeze(log_probs.gather(1, inds.unsqueeze(1))).mean()


        image_center = [self.canvas_size[0]//2, self.canvas_size[0]//2] # image center
        traj_pred_fit = traj2patch(init_pose, traj_pred, image_center)
        # visualization(traj, torch.tensor(safe_mask).unsqueeze(0), idx, sample_idx=0)    
        negative_mask_orig, negative_step_mask, negative_feature_orig, negative_step_feature = safety_checker(traj_pred_fit, safe_mask)

        traj_gt_fit = traj2patch(init_pose, traj_gt.unsqueeze(1), image_center)
        gt_mask, _, _, gt_step_feature = safety_checker(traj_gt_fit, safe_mask)
        gt_mask = ~gt_mask
        negative_mask = negative_mask_orig * gt_mask  # filter out the data where gt is dangerous (fault data)

        negative_traj_total = traj_pred[negative_mask.transpose(1,0)]   # need to make the batch dim first to fetch the data
        # step_mask_total = negative_step_mask.transpose(1,0)[negative_mask.transpose(1,0)]
        num_negative = negative_mask.sum(dim=0)
        # num_negative = num_negative[num_negative.nonzero().squeeze(-1).tolist()] # edit by chant, filter 0
        max_num = num_negative.max()

        negative_traj = torch.split(negative_traj_total, num_negative.cpu().tolist())
        # step_mask = torch.split(step_mask_total, num_negative.cpu().tolist())

        repeat_time = torch.true_divide(max_num, num_negative).ceil().long()
        zero = torch.zeros(max_num, *traj_pred.shape[-2:], device=traj_pred.device)

        negative_batch = [traj.repeat(n, 1, 1)[:max_num] if 0 < n <= max_num else zero for n, traj in zip(repeat_time, negative_traj)]
        negative_batch = torch.stack(negative_batch, dim=1)

        # l_reg = traj_nll(negative_batch, traj_gt, masks)

        pred_dist = predictions['pred_dist']
        log_p_yt_xz_neg = torch.clamp(pred_dist.log_prob(negative_batch),
                                max=self.log_p_yt_xz_max)

        log_p_yt_xz_neg = log_p_yt_xz_neg.mean(dim=0, keepdim=True)
        log_p_y_xz_neg = log_p_yt_xz_neg.sum(dim=2)
        unlikelihood = (log_p_y_xz_neg.exp() + self.log_bias).log()

        log_p_yt_xz = torch.clamp(pred_dist.log_prob(traj_gt), max=self.log_p_yt_xz_max)
        log_p_y_xz = torch.sum(log_p_yt_xz, dim=2)
        l_nll = -torch.mean(log_p_y_xz)
        unlikelihood = unlikelihood * (num_negative > 0).float()
        l_unlikelihood = unlikelihood.mean()

        # if curr_epoch < 2:
        #     self.gamma = 0 # don't use unlike in first 2 epoch
        # if curr_epoch == 2 or curr_epoch == 3:
        #     self.loss_times += 1
        #     self.gamma =  self.loss_times / (2 * dl_length) # slow start within epoch 2,3
        # if curr_epoch > 3:
        #     self.gamma = 1
        self.gamma = gamma

        # loss = self.beta * l_nll + self.alpha * l_class + self.gamma * l_unlikelihood
        loss = self.beta * l_nll + self.gamma * l_unlikelihood

        return loss

