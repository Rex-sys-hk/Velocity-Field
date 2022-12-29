from audioop import bias, cross
import re

from .car import pi_2_pi
from torch import device, nn
import torch
from pyquaternion import Quaternion
# from riskmap_generator.planning.params_generator import ParamGenerator
import matplotlib.pyplot as plt
import time
from torch.utils.tensorboard import SummaryWriter
# from PIL import Image
import torch.optim
import numpy as np
import sys
from .utils import (has_nan, 
                    load_cfg_here,
                    safety,
                    acceleration,
                    jerk,
                    steering,
                    steering_change,
                    speed,
                    lane_xyyaw,
                    # lane_theta,
                    red_light_violation,
                    neighbor_sl_dis
                    )
sys.path.append("..")
CUDA_LAUNCH_BLOCKING=1


# sys.path.append("/media/rex/software/UWLocalShare/uws/PGP_T")
# from torch_gmm import plot_gmm, plot_path_generator_gmm
# import shutil
# import yaml
# import os
# import argparse
# import torch.utils.data as torch_data
# import io
# import torchvision
# from map_info_aggregator import MapAggre
# from nuscenes.map_expansion.map_api import NuScenesMap
# from train_eval.initialization import initialize_prediction_model, initialize_metric,\
#     initialize_dataset, get_specific_args
# from nuscenes.eval.common.utils import quaternion_yaw


class Map(nn.Module):
    def __init__(self, device: str = 'cuda', test = False ,debug = False) -> None:
        """
        risk map manager
        create risk map 
        offering checking methods
        """
        super().__init__()
        cfg = load_cfg_here()['planner']
        self.planning_cfg = cfg
        self.device = device
        self.cost_items = cfg['cost_items']
        self.time_horizon = cfg['time_horizon']
        # self.using_global_ref = cfg['using_global_ref']

        # self.sdf_index_ratio = torch.tensor(
        #     cfg['sdf_index_ratio'], device=device)
        # self.ego_center = torch.tensor(cfg['ego_center'], device=device)
        # self.num_functions = cfg['num_functions']
        # self.interpolating_factor = cfg['reflane_interpolating_factor']
        # self.sdf_interpolating_factor = cfg['sdf_interpolating_factor']

        self.default_lw = cfg['default_lane_width'] if 'default_lane_width' in cfg.keys() else 2.5
        # set a scaler bias switch. Can only be adpted if negative sample in use
        # self.scalar_bias = nn.Sequential(nn.Linear(self.cost_items, self.cost_items, device=self.device, dtype=float))
        # if enable bias, set zeros otherwise TODO
        self.prediction_dict = None
        self.one_d_sqrt2pi = 0.3989422804014327
        self.debug = debug

    def get_meter(self, sample_plan, context, batch):
        self.get_new_data(context, batch)
        return self.get_vec_map_meter(sample_plan)

    def get_new_data(self, context, batch):
        # open_space = data['image'].flip(0)
        # self.sdf = self.getsdf(open_space)
        # agents_num = data['agents_num']
        # print("agents num:", agents_num.shape)
        self.current_state = context['current_state']
        self.ego_current_state = context['current_state'][:,0:1]
        # get reference lane
        self.reflane = context['ref_line_info']
        # get prediction
        self.prediction = context['predictions']

        # get cross walk
        # self.cross_walk = batch[3].to(device)

        # get map lane
        # self.map_lane = batch[4].to(device)

        # get traffic light TODO(find how DIPP do this)
        # lane_pos = map_feature["lane_features"][..., :2]
        # traffic_light = map_feature["lane_features"][..., 0, 6]
        # self.red_light = lane_pos[traffic_light==3]

        # get SDF TODO(need further considreration)

        # if agents_num > 0:
        # self.prediction = SeqMVN2D(prediction['traj'],prediction['probs'])
        # else:
        #     self.prediction = 0

    def get_vec_map_meter(self, traj):
        ref_dis = self.dis2ref(traj)
        pre_dis = self.dis2pre(traj)
        
        # only consider the nearest neighbor
        pre_dis = pre_dis.min(dim=1).values
        
        tl_dis = self.dis2tl(traj)
        
        # curb_dis = self.dis2curb(traj)
        # cross_dis = self.dis2cross(traj)
        speed_dis = self.dis2speed(traj)
        
        return {'ref_dis':ref_dis,
                'pre_dis':pre_dis,
                'tl_dis':tl_dis,
                # 'curb_dis':curb_dis,
                # 'cross_dis':cross_dis,
                'speed_dis': speed_dis
                }

    def dis2ref(self,traj):
        # print(self.reflane.shape) # torch.Size([48, 1200, 5])
        # lane_xy_d = lane_xyyaw(traj['u'], self.reflane, self.ego_current_state)
        # lane_theta_d = lane_theta(traj['u'], self.reflane, self.ego_current_state)
        return lane_xyyaw(traj['u'], self.reflane, self.ego_current_state)

    def dis2pre(self,traj):
        return neighbor_sl_dis(traj['u'],self.reflane,self.current_state,self.prediction)

    def dis2tl(self,traj):
        return red_light_violation(traj['u'], self.reflane, self.ego_current_state) 

    def dis2speed(self,traj):
        return speed(traj['u'], self.ego_current_state)
        
    # def dis2cross(self,traj):
    #     # print(self.reflane.shape) # torch.Size([48, 11, 4, 100, 3])

    #     return 0
    # def dis2curb(self,traj):
    #     print(self.map_lane.shape) # torch.Size([48, 1200, 5])
    #     print(torch.ne(self.map_lane,self.reflane).any())
    #     return 0

    def get_map_cost(self, traj, batch = False):
        """
        traj is [seq,(x,y,yaw)]
        """
        dis_meter = -self.get_dis_by_sdf(traj,batch = batch)
        has_nan(dis_meter)
        ref_meter = self.get_dis_to_reflane(traj,batch = batch)
        has_nan(ref_meter)
        if traj.shape[0]==self.time_horizon or (batch and traj.shape[1]== self.time_horizon):
            col_meter = self.get_collision_prob(traj)
        else:
            col_meter = torch.zeros_like(dis_meter)

        has_nan(col_meter)# col_meter
        meter = torch.cat([dis_meter, ref_meter, col_meter], dim=-1)
        has_nan(meter)
        return meter

    def get_bb_map_cost(self, traj, extend, batch = False):
        """
        traj: is [seq,(x,y,yaw)]
        extend: is [a+1,c,(x,y,r)]
             extend[-1] is ego extend

        return: cost 
            shape: [seq, (sdf, reflane_dis, collison)]
                sdf: occupied place are positive
                reflane: dis to center lane
                col: probability [0,1]
        """
        has_nan(traj)
        extended_pos = self.add_extend(traj,extend[:,-1],batch = batch)
        # drivable area loss 
        # st = time.time()
        # dis_meter = self.get_dis_by_sdf(extended_pos,extend[-1],batch = batch)
        # has_nan(dis_meter)

        # referance lane loss
        st = time.time()
        ref_meter = self.get_dis_to_reflane(traj,batch = batch)
        has_nan(ref_meter)
        dis_meter = torch.zeros_like(ref_meter)

        # Traffic light loss
        # tl_meter = self.get_dis_to_traffic_light(extended_pos,extend[-1],batch = batch)
        # has_nan(tl_meter)
        tl_meter = torch.zeros_like(ref_meter)
        # collision probbilibty loss
        st = time.time()
        if traj.shape[0]==self.time_horizon or (batch and traj.shape[1]== self.time_horizon):
            dx = self.get_bbdx(traj,extend,batch = batch)
            col_meter = self.get_collision_prob(traj,dx, batch = batch)
        else:
            col_meter = torch.zeros_like(dis_meter)
        has_nan(col_meter)
        # print('--collision check time:', time.time()-st)
        meter = torch.cat([dis_meter, ref_meter, tl_meter, col_meter], dim=-1)
        has_nan(meter)
        return meter

    # def getsdf(self, occ_map):
    #     occ_map = torch.where(occ_map > 0.8, -self.sdf_index_ratio.mean(), self.sdf_index_ratio.mean())
    #     try:
    #         sdf = torch.tensor(skfmm.distance(occ_map.cpu(), dx=self.sdf_index_ratio.cpu()), device=self.device)
    #     except:
    #         sdf = torch.ones_like(occ_map, device=self.device)
    #     return torch.nn.functional.interpolate(sdf.unsqueeze(0).unsqueeze(0),
    #                                             scale_factor=self.sdf_interpolating_factor,
    #                                             mode='bicubic').squeeze()

    # def get_image_idx(self, rpos):
    #     rpos = rpos.to(self.device)
    #     mappos = (rpos[..., :2] + self.ego_center)/self.sdf_index_ratio
    #     mappos = mappos*self.sdf_interpolating_factor
    #     mappos = mappos.flip(-1)
    #     index = mappos.clone().floor().long()
    #     meter_diff = mappos-index
    #     return index,meter_diff
        # pass
    def add_extend(self, rpos, ego_extend = None, batch = False):
        if ego_extend is not None:
            if batch:
                btsz,b,th,posdim = rpos.shape
                # ego_extend_valid_num = (ego_extend[...,0]< 20).sum() #TODO 5e2 is prepresenting inf
                # _,cnum,cdim = ego_extend.shape
                # print(rpos.shape)
                # print(ego_extend.shape)
                epos = self.proj_relative_pose(ego_extend.unsqueeze(-3).unsqueeze(-3).repeat(1,b,th,1,1)[...,:2],
                                                rpos.unsqueeze(-2))
            else:
                btsz,th,posdim = rpos.shape
                # ego_extend_valid_num = (ego_extend[...,0]< 20).sum() #TODO 5e2 is prepresenting inf
                btsz,cnum,cdim = ego_extend.shape
                epos = self.proj_relative_pose(ego_extend.unsqueeze(-3).repeat(1,th,1,1)[...,:2],
                                                rpos.unsqueeze(-2))
        if self.debug:
            plt.clf()
            plt.xlim([-10,50])
            plt.ylim([-10,10])
            # plt.title('ego extend')
            # plt.scatter(ego_extend[...,0].cpu().detach(),
            #             ego_extend[...,1].cpu().detach())
            # plt.show()
            plt.title('raw traj')
            plt.scatter(rpos[...,0].cpu().detach(),
                        rpos[...,1].cpu().detach(),
                        # c = np.ones(rpos.shape[0]),
                        cmap='spring',
                        alpha=0.5)
            plt.show()
            plt.title('extended pos')
            for pos in epos.view(-1,3,2):
                plt.plot(pos[...,0].cpu().detach(),
                        pos[...,1].cpu().detach(),
                        # c = np.ones((epos.shape[0],3)),
                        # cmap='summer',
                        # alpha=0.5
                        )
            plt.axis('equal')
            plt.show()
        return epos

    # def get_dis_by_sdf(self, rpos, ego_extend = None, interpolation_by_diff = True, batch = False):
    #     # if ego_extend is not None:
    #     #     if batch:
    #     #         b,th,posdim = rpos.shape
    #     ego_extend_valid_num = (ego_extend[...,0]< 5e3).sum() #TODO 5e2 is prepresenting inf
    #     #         cnum,cdim = ego_extend.shape
    #     #         rpos = self.proj_relative_pose(ego_extend.unsqueeze(-3).repeat(b,th,1,1)[...,:ego_extend_valid_num,:2],
    #     #                                         rpos.unsqueeze(-2).repeat(1,1,ego_extend_valid_num,1))
    #     #     else:
    #     #         th,posdim = rpos.shape
    #     #         ego_extend_valid_num = (ego_extend[...,0]< 5e3).sum() #TODO 5e2 is prepresenting inf
    #     #         cnum,cdim = ego_extend.shape
    #     #         rpos = self.proj_relative_pose(ego_extend.unsqueeze(-3).repeat(th,1,1)[...,:ego_extend_valid_num,:2],
    #     #                                         rpos.unsqueeze(-2).repeat(1,ego_extend_valid_num,1))
    #     idx,meter_diff = self.get_image_idx(rpos)
    #     # linear iterpolation
    #     ymax = self.sdf.shape[0]-2 #later we have yid+1, so we directly clamp it to max-2 
    #     xmax = self.sdf.shape[1]-2
    #     idx = idx.reshape(-1,2)
    #     yid = idx[...,0].clamp(min=0, max=ymax)
    #     xid = idx[...,1].clamp(min=0, max=xmax)
    #     dis = self.sdf[yid,xid]

    #     # DEBUG Vis
    #     if self.debug:
    #         plt.clf()
    #         plt.imshow(self.sdf.cpu().detach(),origin='lower')
    #         plt.scatter(rpos[...,0].cpu().detach(),rpos[...,1].cpu().detach(),alpha=0.5)
    #         plt.scatter(idx[...,1].cpu().detach(),idx[...,0].cpu().detach(),alpha=0.5)
    #         plt.scatter(xid.cpu().detach(),yid.cpu().detach(),alpha=0.5,c = dis.cpu().detach())
    #         meter_diff_v = meter_diff.cpu().detach().reshape(-1,2)
    #         plt.scatter(meter_diff_v[...,0].cpu().detach(), meter_diff_v[...,1].cpu().detach())
    #         plt.show()

    #     if interpolation_by_diff:
    #         rdis = self.sdf[yid,xid+1]
    #         udis = self.sdf[yid+1,xid]
    #         ind_diff = torch.stack([udis-dis,rdis-dis],dim=-1)
    #         dis = dis.unsqueeze(-1)+ torch.sum(ind_diff*meter_diff.view(-1,2),dim=-1).reshape(-1,1)
    #     if ego_extend is not None:
    #         dis = dis.reshape(-1,ego_extend_valid_num,1)
    #         dis = dis.min(dim=-2).values - ego_extend[0,-1]
    #         # dis = dis.mean(dim=-2) - ego_extend[0,-1]
    #     if batch:
    #         dis = dis.reshape(-1,self.time_horizon,1)
    #     return dis

    def get_dis_to_traffic_light(self, rpos, ego_extend = None, interpolation_by_diff = True, batch = False):
        # if ego_extend is not None:
        #     if batch:
        #         b,th,posdim = rpos.shape
        #         ego_extend_valid_num = (ego_extend[...,0]< 5e3).sum() #TODO 5e2 is prepresenting inf
        #         cnum,cdim = ego_extend.shape
        #         rpos = self.proj_relative_pose(ego_extend.unsqueeze(-3).repeat(b,th,1,1)[...,:ego_extend_valid_num,:2],
        #                                         rpos.unsqueeze(-2).repeat(1,1,ego_extend_valid_num,1))
        #     else:
        #         th,posdim = rpos.shape
        #         ego_extend_valid_num = (ego_extend[...,0]< 5e3).sum() #TODO 5e2 is prepresenting inf
        #         cnum,cdim = ego_extend.shape
        #         rpos = self.proj_relative_pose(ego_extend.unsqueeze(-3).repeat(th,1,1)[...,:ego_extend_valid_num,:2],
        #                                         rpos.unsqueeze(-2).repeat(1,ego_extend_valid_num,1))

        if not batch:
            rpos = rpos.unsqueeze(0)
        btsz, path_num, th, circle_num, dim = rpos.shape
        if not self.red_light.shape[0]:
            dis = torch.ones(path_num,th,1,device=self.device)*1e2 # 1e2 represents inf/None
            dis = dis.squeeze(0) if not batch else dis
            return dis
        dis = rpos.reshape(btsz,path_num,th,circle_num,1,1,dim)-self.red_light
        dis = torch.norm(dis,dim=-1)
        dis = dis.view(btsz, path_num,th,-1).min(dim=-1).values - ego_extend[0,-1]
        dis = dis.squeeze(1) if not batch else dis

        return dis.unsqueeze(-1)


    def get_policy(self,policy):
        """
        only the nearest reflane length count
        """
        ref = policy['reference_lines']
        masks = policy['reference_lines_mask'] # True is valid
        try:
            msk = masks.sum(dim=-1)>=30
            reflanes = ref
            # reflanes = ref[masks.min(dim=-1).values.bool()]
            # reflanes = ref[msk]
            # if reflanes.shape[0]==0:
            #     x = torch.linspace(-5,5,30,device=self.device).unsqueeze(-1)
            #     y = torch.zeros(30,1,device=self.device)
            #     yaw = y.clone()
            #     reflanes = torch.cat([x,y,yaw],dim=-1).unsqueeze(0)
            # print("trying")
        except:
            reflanes = []
            for i in range(ref.shape[0]):
                if masks[i]>=29:
                    # ref[i,masks[i].int():] = 1e8
                    reflanes.append(ref[i])
            reflanes = torch.stack(reflanes, dim=0)
            # pass
        has_nan(reflanes)
        diff = reflanes[..., 1:, :2] - reflanes[..., :-1, :2]
        has_nan(diff)
        yaw = pi_2_pi(torch.atan2(diff[..., 1:2], diff[..., 0:1]))
        has_nan(yaw)
        reflanes = torch.cat([reflanes[..., :2], torch.cat(
            [yaw, yaw[..., -1:, :]], dim=-2)], dim=-1)
        if self.interpolating_factor:
            num,pt,dim = reflanes.shape
            reflanes = reflanes.reshape(-1,pt,dim)
            inter_reflane_x = torch.nn.functional.interpolate(reflanes.unsqueeze(0)[...,0],scale_factor = self.interpolating_factor,mode='linear')
            inter_reflane_y = torch.nn.functional.interpolate(reflanes.unsqueeze(0)[...,1],scale_factor = self.interpolating_factor,mode='linear')
            inter_reflane_yaw = torch.nn.functional.interpolate(reflanes.unsqueeze(0)[...,2],scale_factor = self.interpolating_factor,mode='linear')
            reflanes = torch.stack([inter_reflane_x,inter_reflane_y,inter_reflane_yaw],dim=-1).squeeze(0)
            reflanes = reflanes.reshape(num,-1,dim)
        has_nan(reflanes)
        # for i in range(reflanes.shape[0]):
        #     plt.plot(reflanes[i,0,:,0].cpu(),reflanes[i,0,:,1].cpu())
        # plt.show()
        return reflanes

    def get_reflane(self, lane_feature):
        diff = lane_feature[..., 1:, :2] - lane_feature[..., :-1, :2]
        yaw = pi_2_pi(torch.atan2(diff[..., 1:2], diff[..., 0:1]))
        ref_lane = torch.cat([lane_feature[..., :2], torch.cat(
            [yaw, yaw[..., -1:, :]], dim=-2)], dim=-1)
        return ref_lane

    def get_dis_to_reflane(self, rpos, get_ind = False, batch = False):
        """
        check the closest lane point of ego center
        rpos: should in shape [b*, seq, dim]
        """
        if batch:
            # only using by lattice
            rpos = rpos.unsqueeze(-2).unsqueeze(-2)
        else:
            rpos = rpos.reshape(rpos.shape[0], 1, -1, 1, 1, 3)
        diff = rpos - self.reflane.unsqueeze(-4).unsqueeze(-4)
        has_nan(diff)
        # diff[..., 2] = pi_2_pi(diff[..., 2])
        diff = torch.norm(diff[...,:2], dim=-1)# only x,y
        diff = torch.nan_to_num(diff,posinf=1e2)#.clamp(max = 1e2)
        min_c_dis = diff.min(dim=-1).values
        # dis = min_c_dis.min(dim=-1, keepdim=True).values #min center dis
        dis = min_c_dis.mean(dim=-1, keepdim=True) #min center dis
        # DEBUG Vis
        if self.debug:
            print('in reflane')
            print(dis)
            print(dis.max(),dis.min())
            plt.clf()
            for refi in self.reflane.cpu().detach():
                plt.plot(refi[...,0],refi[...,1])
            print('dis:', dis.shape)
            print('rpos:', rpos.shape)
            plt.scatter(rpos[...,0].cpu().detach().squeeze(),rpos[...,1].cpu().detach().squeeze(),c=dis.cpu().detach().squeeze(),alpha=0.3)
            plt.show()
        return dis

    def get_bbdx(self,rpos,extend,batch = False):
        """
        get the x,y diff between bounding circles
        extend[-1] is ego 
        """
        if not self.prediction:
            return torch.zeros(self.time_horizon, 1, device=self.device, dtype=float)
        # TODO
        agents_predict = self.prediction.mus

        diff = torch.diff(agents_predict,dim=-2)
        yaw = torch.atan2(diff[...,1:2],diff[...,0:1])
        yaw = torch.cat([yaw,yaw[...,-1,:].unsqueeze(-2)],dim=-2)
        agents_predict = torch.cat([agents_predict,yaw],dim=-1)
        # usable params
        anum,modal,th,dim = agents_predict.shape # agent num + ego
        anum_and_ego,cnum,cdim = extend.shape
        agents_occ = self.proj_relative_pose(extend[:anum,:,:2].reshape(anum,1,1,cnum,2).repeat(1,modal,th,1,1),
                                            agents_predict.reshape(anum,modal,th,1,dim))
        ext_ind = (extend[-1,:,0]<2e2).sum()
        if batch:
            ego_occ = self.proj_relative_pose(extend[-1:,:ext_ind,:2].reshape(1,1,1,1,ext_ind,2).repeat(rpos.shape[0],1,1,th,1,1),
                                                rpos.reshape(-1,1,1,th,1,dim))
        else:
            ego_occ = self.proj_relative_pose(extend[-1:,:ext_ind,:2].reshape(1,1,1,ext_ind,2).repeat(1,1,th,1,1),
                                                rpos.reshape(1,1,th,1,dim))
        # TODO enable individual extend circle size
        dxy = agents_occ.unsqueeze(-2)-ego_occ.unsqueeze(-3) # [N,t,c,c,(x,y)]
        # dxy = dxy[torch.logical_and(torch.logical_not(dxy.isnan()), torch.logical_not(dxy.isinf()))]#  = math.inf 
        # # cause calculation between inf will produce none
        if batch:
            mindis2ego,inds = torch.norm(dxy,dim=-1).reshape(rpos.shape[0],anum,modal,th,-1).min(dim=-1)# [a,m,t,1]
            inds = inds.repeat(1,2,1,1,1,1).permute(2,3,4,5,0,1)
            dxy2ego = dxy.reshape(rpos.shape[0],anum,modal,th,-1,2).gather(-2,inds)
            # scale dx and dy by similar triangle
            dratio  = (mindis2ego-(extend[:-1,0,2]+extend[-1:,0,2]).reshape(1,anum,1,1))/mindis2ego.clamp(min=1e-2)
            sd = dxy2ego.squeeze(-2)*dratio.unsqueeze(-1) #the signed distance
            return sd #sign of this need to be used in sequential process TODO
        else:
            mindis2ego,inds = torch.norm(dxy,dim=-1).reshape(anum,modal,th,-1).min(dim=-1)# [a,m,t,1]
            inds = inds.repeat(1,2,1,1,1).permute(2,3,4,0,1)
            dxy2ego = dxy.reshape(anum,modal,th,-1,2).gather(-2,inds)
            # scale dx and dy by similar triangle
            dratio  = (mindis2ego-(extend[:-1,0,2]+extend[-1:,0,2]).reshape(anum,1,1))/mindis2ego.clamp(min=1e-2)
            sd = dxy2ego.squeeze(-2)*dratio.unsqueeze(-1) #the signed distance
            return sd #sign of this need to be used in sequential process TODO
    
    def get_collision_prob(self, rpos, dx = None, batch = False):
        if not self.prediction:
            if batch:
                return torch.zeros(rpos.shape[0],self.time_horizon, 1, device=self.device, dtype=float)
            else:    
                return torch.zeros(self.time_horizon, 1, device=self.device, dtype=float)
        # now is sum of occupancy probability
        # the torch sum can be norm to maxmize a certain risk
        if batch:
            meter = torch.sum(torch.exp(self.prediction.log_prob(rpos.reshape(
                    -1, 1, 1, self.time_horizon, 3)[..., :2],dx = dx)), dim=[1, 2]).unsqueeze(-1)  # .clamp(min=0)
        else:
            meter = torch.sum(torch.exp(self.prediction.log_prob(rpos.reshape(
                    1, 1, self.time_horizon, 3)[..., :2],dx = dx)), dim=[0, 1]).unsqueeze(-1)  # .clamp(min=0)
        has_nan(meter)
        return meter




    def visulize(self):
        plt.gcf()
        W = 25
        H = 15
        xs = torch.linspace(-30, 69, steps=W)
        ys = torch.linspace(-35, 34, steps=H)
        x = xs.repeat(H, 1).permute(1, 0).reshape(-1)
        y = ys.repeat(W, 1).reshape(-1)
        mesh = torch.zeros(W*H, 3)
        mesh[..., 0] = x
        mesh[..., 1] = y
        cost = self.get_map_cost(mesh.to(device=self.device)).cpu().detach()
        # plt.imshow(self.sdf)
        plt.scatter(mesh[..., 0], mesh[..., 1], c=cost.sum(
            dim=-1), cmap='cool', alpha=0.5)
        plt.colorbar()
        plt.axis('equal')
        plt.axis([-30, 70, -35, 35])
        # plt.show()
    def proj_relative_pose(self,state,transf):
        yaw = pi_2_pi(transf[...,2])
        cos = torch.cos(yaw)
        sin = torch.sin(yaw)
        state_t = torch.zeros_like(state,device=self.device)
        state_t[...,0] = state[...,0]*cos -state[...,1]*sin +transf[...,0]
        state_t[...,1] = state[...,0]*sin + state[...,1]*cos +transf[...,1]
        if state_t.shape[-1]==3:
            state_t[...,2] = pi_2_pi(state[...,2]+transf[...,2])
        if state_t.shape[-1]==5:
            state_t[...,2:] = state[...,2:]
        return state_t