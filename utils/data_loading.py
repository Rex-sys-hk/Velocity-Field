import glob
import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset
from utils.data_utils import agent_norm, map_norm, ref_line_norm
from utils.riskmap.torch_lattice import LatticeSampler
from utils.data_augmentation.kinematic_agent_augmentation import KinematicAgentAugmentor
from utils.data_augmentation.nuplan_utils.trajectory import Trajectory
from utils.data_augmentation.nuplan_utils.agents import Agents
from utils.riskmap.car import bicycle_model, pi_2_pi_pos
from utils.test_utils import batch_check_collision, check_collision

class DrivingData(Dataset):
    def __init__(self, data_dir, data_aug = False):
        self.data_list = glob.glob(data_dir)
        self.data_aug = data_aug
        if self.data_aug:
            self.sampler = LatticeSampler()
            N = 50
            dt = 0.1
            augment_prob = 1.0
            mean = [0.3, 0.1, np.pi / 12]
            std = [0.5, 0.1, np.pi / 12]
            low = [-0.1, -0.1, -0.1]
            high = [0.1, 0.1, 0.1]
            self.gaussian_augmentor = KinematicAgentAugmentor(
                N, dt, mean, std, low, high, augment_prob, use_uniform_noise=False
            )
            # self.uniform_augmentor = KinematicAgentAugmentor(
            #     N, dt, mean, std, low, high, augment_prob, use_uniform_noise=True
            # )
    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        try:
            data = np.load(self.data_list[idx],allow_pickle=True)
            if type(data) is str:
                print(type(data))
                print(os.path.isfile(data))
                os.remove(data)
                print("file %s as a pickle failed, removed" % repr(data))
                print(os.path.isfile(data))
                return self.__getitem__(idx+1)
        except:
            print(f'fail loading file {self.data_list[idx]}')
            if os.path.exists(self.data_list[idx]):
                print(f'removing file {self.data_list[idx]}')
                os.remove(self.data_list[idx])
            return self.__getitem__(idx+1)
        ego = data['ego']
        neighbors = data['neighbors']
        ref_line = data['ref_line'] 
        map_lanes = data['map_lanes']
        map_crosswalks = data['map_crosswalks']
        gt_future_states = data['gt_future_states']
        if self.data_aug and np.random.rand(1)<0.6:
            ego_aug, gt_future_states_aug =  self.data_augment(ego, gt_future_states, neighbors)
            ego[...,:5] = ego_aug
            gt_future_states[0,...,:5] = gt_future_states_aug
            (ego, neighbors, 
             map_lanes, map_crosswalks, 
             ref_line, gt_future_states) = self.normalize_data(
                                                                ego[-1,:2].copy(),
                                                                ego[-1,2].copy(),
                                                                ego,
                                                                neighbors,
                                                                map_lanes,
                                                                map_crosswalks,
                                                                ref_line,
                                                                gt_future_states,
                                                                viz=False
                                                            )
            # center, angle, ego, neighbors, map_lanes, map_crosswalks, ref_line, ground_truth, viz=True
            # ego, neighbors, map_lanes, map_crosswalks, ref_line, ground_truth
        # normalize base on the new GT
        # print(ego.shape, gt_future_states.shape)
        return ego, neighbors, map_lanes, map_crosswalks, ref_line, gt_future_states
    
    def data_augment(self, ego, gt, neighbors):
        # ## aug ego his
        # ego = tensor(ego)
        # gt = tensor(gt)
        # s_ego = self.get_sample(
        #     ego[0],
        #     get_u_from_X(ego[1:],ego[0]),
        #     tensor([0.5, 0.3]),
        #     1,
        #     turb_num=19)
        # a_ego = torch.cat([ego[0:1,:5],s_ego[0]],dim=0)
        neighbors = neighbors.copy().transpose(1,0,2)
        features = {
            'agents': Agents(ego=ego[None,:,:3].copy(),
                             agents=neighbors[None,:,:,:].copy(),),
        }
        targets = {
            'trajectory': Trajectory(data=gt[0,...,:3]),
        }
        aug_features, aug_targets = self.gaussian_augmentor.augment(features, targets)
        # print(aug_features['agents'].ego.shape, aug_targets['trajectory'].data.shape)
        # plt.plot(ego[:,0], ego[:,1], 'g')
        # plt.plot(gt[0,:,0], gt[0,:,1], 'g')
        # plt.plot(aug_features['agents'].ego[0,:,0], aug_features['agents'].ego[0,:,1], 'r')
        # plt.plot(aug_targets['trajectory'].data[:,0], aug_targets['trajectory'].data[:,1], 'b')
        # plt.show()
        aug_ego = aug_features['agents'].ego[0]
        v = np.linalg.norm(np.diff(aug_ego[...,:2],axis=0),axis=-1)
        v = np.pad(v, (1,0), 'edge')
        aug_ego = np.concatenate([aug_features['agents'].ego[0],(v*np.cos(aug_ego[...,2]))[:, None], (v*np.sin(aug_ego[...,2]))[:, None]],axis=-1)
        # aug_ego = yawv2yawdxdy(aug_ego)
        aug_gt = aug_targets['trajectory'].data
        aug_v = np.linalg.norm(np.diff(aug_gt[...,:2],axis=0),axis=-1)
        aug_v = np.pad(aug_v, (1,0), 'edge')
        aug_gt = np.concatenate([aug_targets['trajectory'].data,(aug_v*np.cos(aug_gt[...,2]))[:, None], (aug_v*np.sin(aug_gt[...,2]))[:, None]],axis=-1)
        # aug_gt = yawv2yawdxdy(aug_gt)
        size = np.concatenate([ego[-1,None,5:],neighbors[-1,:,5:8]],axis=0)
        if batch_check_collision(torch.tensor(aug_gt[None,:,:]), torch.tensor(gt[None,1:,:,:3]), torch.tensor(size[None,:])):
            return ego[...,:5], gt[0,:,:5]
        return aug_ego, aug_gt
        # if torch.norm(ego[-1,:2]-a_ego[-1,:2],dim=-1)<0.3:
        #     return a_ego, gt[0,:,:5]
        # # ## aug gt TODO
        # # use lattice planner sample trajs
        # current = a_ego[-1].numpy()
        # v = torch.norm(a_ego[-1, 3:5],dim=-1).numpy()
        # l_trajs = self.sampler.sampling(ref_line,
        #                         current[0],
        #                         current[1],
        #                         pi_2_pi_pos(current[2]),
        #                         v)
        # l_trajs = tensor(l_trajs)
        # l_trajs = yawv2yawdxdy(l_trajs)
        # l_trajs = torch.nan_to_num(l_trajs, nan=np.inf)
        # # ## emergency stop
        # # break_u = get_u_from_X(gt[0:1,:,:5],ego[-1:])
        # # break_u[...,0] = -3
        # # l_trajs = bicycle_model(break_u,ego[-1:])
        # # l_trajs = yawv2yawdxdy(l_trajs)

        # # consider gt last point as current state, 
        # # and sampling with gaussian noise, 
        # # find the a_ego nearest traj
        # inv_X_a = gt[0].clone().flip(dims=[0])
        # # cat ego augmented current ?
        # # inv_X = torch.cat([inv_X_a[1:],a_ego[-1:,:5]],dim=0)
        # inv_X = self.inversing_yawv(inv_X_a[1:])
        # inv_current = self.inversing_yawv(inv_X_a[0])
        # inv_u = get_u_from_X(inv_X, inv_current)
        # # inv_u = np.pad(inv_u,pad_width=((0,1),(0,0)),mode='edge')
        # inv_u = torch.cat([inv_u,inv_u[-1:]],dim=0)
        # inv_traj = self.get_sample(
        #     inv_current,
        #     inv_u,
        #     tensor([0.5, 0.3]),
        #     400,
        #     1)
        
        # ind = torch.argmin(
        #             torch.norm(inv_traj[:,-1,:2]-a_ego[-1,:2], dim=-1))
        # s_inv_traj = inv_traj[ind:ind+1].flip(dims=[1])
        # s_inv_traj = self.inversing_yawv(s_inv_traj)
        # if torch.norm(s_inv_traj[0,0,:2]-a_ego[-1,:2],dim=-1)<=0.2:
        #     # find the one closest to gt
        #     s_inv_traj = torch.cat([s_inv_traj[0:1,1:], gt[0:1,-1:]],dim=-2)
        #     trajs = torch.cat([l_trajs, s_inv_traj],dim=0)
        # else:
        #     trajs = l_trajs

        # # plt.scatter(gt[0,:,0], gt[0,:,1], color='green')
        # # plt.plot(ego[...,0], ego[...,1],'r')
        # # plt.plot(a_ego[...,0], a_ego[...,1],'--r')
        # # plt.plot(ref_line[...,0], ref_line[...,1], 'b')
        # # for t in l_trajs:
        # #     plt.plot(t[...,0], t[...,1], 'grey', lw=0.1)
        # # for t in inv_traj:
        # #     plt.plot(t[...,0], t[...,1], 'green', lw=0.1)
        # dis = torch.norm(trajs[...,:2]-gt[0:1][...,:2],dim=-1)
        # index = torch.argmin(dis.mean(dim=-1))
        # if torch.norm(gt[0,-5:,:2]-trajs[index,-5:,:2],dim=-1).mean()<=2:
        #     new_gt = trajs[index]
        # else:
        #     new_gt = gt[0]
        # # plt.scatter(new_gt[...,0], new_gt[...,1], color='cyan')
        # # plt.axis('equal')
        # # plt.show()
        # return a_ego, new_gt
        
    def inversing_yawv(self, X):
        X[...,2] = pi_2_pi_pos(X[...,2]+torch.pi)
        X[...,3] = -X[...,3]
        X[...,4] = -X[...,4]
        return X
    
    def get_sample(self, cur_state, gt_u, cov = torch.tensor([0.5, 0.2]), sample_num=100, turb_num=50):
        init_guess_u = gt_u
        u = (torch.randn([sample_num,turb_num,2])*cov+1.)*init_guess_u
        X = bicycle_model(u,cur_state[None,:])
        X = torch.stack([X[...,0], X[...,1], X[...,2], 
                         X[...,3]*torch.cos(X[...,2]), 
                         X[...,3]*torch.sin(X[...,2])],dim = -1)
        return X #{'X':X,'u':u}
    
    def normalize_data(self, center, angle, ego, neighbors, map_lanes, map_crosswalks, ref_line, ground_truth, viz=True):
        # get the center and heading (local view)
        # center, angle = self.current_xyh[:2], self.current_xyh[2]
        # normalize agent trajectories
        # print('cu_center:', center, 'cu_angle:',angle)
        # print('before normed:', ego[-1], '\n', ground_truth[0,0])
        ego[:, :5] = agent_norm(ego, center, angle)
        ground_truth[0] = agent_norm(ground_truth[0], center, angle) 
        # print('normed:', ego[-1], '\n', ground_truth[0,0])
        for i in range(neighbors.shape[0]):
            if neighbors[i, -1, 0] != 0:
                neighbors[i, :, :5] = agent_norm(neighbors[i], center, angle, impute=True)
                ground_truth[i+1] = agent_norm(ground_truth[i+1], center, angle)            

        # normalize map points
        for i in range(map_lanes.shape[0]):
            lanes = map_lanes[i]
            crosswalks = map_crosswalks[i]

            for j in range(map_lanes.shape[1]):
                lane = lanes[j]
                if lane[0][0] != 0:
                    lane[:, :9] = map_norm(lane, center, angle)

            for k in range(map_crosswalks.shape[1]):
                crosswalk = crosswalks[k]
                if crosswalk[0][0] != 0:
                    crosswalk[:, :3] = map_norm(crosswalk, center, angle)
        # normalize ref line
        ref_line = ref_line_norm(ref_line, center, angle).astype(np.float32)

        # visulization
        if viz:
            rect = plt.Rectangle((ego[-1, 0]-ego[-1, 5]/2, ego[-1, 1]-ego[-1, 6]/2), ego[-1, 5], ego[-1, 6], linewidth=2, color='r', alpha=0.6, zorder=3,
                                transform=mpl.transforms.Affine2D().rotate_around(*(ego[-1, 0], ego[-1, 1]), ego[-1, 2]) + plt.gca().transData)
            plt.gca().add_patch(rect)
            plt.plot(ego[...,0], ego[...,1], 'r', marker = '.', lw=1, zorder=3)
            plt.plot(ref_line[:, 0], ref_line[:, 1], 'y', linewidth=2, zorder=4)

            future = ground_truth[0][ground_truth[0][:, 0] != 0]
            plt.plot(future[:, 0], future[:, 1], 'r', marker = '.', linewidth=1, zorder=3)

            for i in range(neighbors.shape[0]):
                if neighbors[i, -1, 0] != 0:
                    rect = plt.Rectangle((neighbors[i, -1, 0]-neighbors[i, -1, 5]/2, neighbors[i, -1, 1]-neighbors[i, -1, 6]/2), 
                                          neighbors[i, -1, 5], neighbors[i, -1, 6], linewidth=2, color='m', alpha=0.6, zorder=3,
                                          transform=mpl.transforms.Affine2D().rotate_around(*(neighbors[i, -1, 0], neighbors[i, -1, 1]), neighbors[i, -1, 2]) + plt.gca().transData)
                    plt.gca().add_patch(rect)
                    future = ground_truth[i+1][ground_truth[i+1][:, 0] != 0]
                    plt.plot(future[:, 0], future[:, 1], 'm', linewidth=3, zorder=3)

            for i in range(map_lanes.shape[0]):
                lanes = map_lanes[i]
                crosswalks = map_crosswalks[i]

                for j in range(map_lanes.shape[1]):
                    lane = lanes[j]
                    if lane[0][0] != 0:
                        centerline = lane[:, 0:2]
                        centerline = centerline[centerline[:, 0] != 0]
                        left = lane[:, 3:5]
                        left = left[left[:, 0] != 0]
                        right = lane[:, 6:8]
                        right = right[right[:, 0] != 0]
                        plt.plot(centerline[:, 0], centerline[:, 1], 'c', linewidth=3) # plot centerline
                        plt.plot(left[:, 0], left[:, 1], 'k', linewidth=3) # plot left boundary
                        plt.plot(right[:, 0], right[:, 1], 'k', linewidth=3) # plot left boundary

                for k in range(map_crosswalks.shape[1]):
                    crosswalk = crosswalks[k]
                    if crosswalk[0][0] != 0:
                        crosswalk = crosswalk[crosswalk[:, 0] != 0]
                        plt.plot(crosswalk[:, 0], crosswalk[:, 1], 'b', linewidth=4) # plot crosswalk

            plt.gca().set_aspect('equal')
            plt.tight_layout()
            plt.show()
            # plt.show(block=False)
            # plt.pause(1)
            # plt.close()

        return ego, neighbors, map_lanes, map_crosswalks, ref_line, ground_truth