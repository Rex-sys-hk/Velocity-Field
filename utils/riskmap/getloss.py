# from statistics import variance
from operator import gt
from matplotlib import pyplot as plt
import torch
import torch.functional as F
from model.meter2risk import Meter2Risk

from .utils import has_nan
from .torch_lattice import torchLatticePlanner
from utils.riskmap.utils import load_cfg_here
# from .utils import convert2detail_state
from .map import Map
from .car import (move,
                  steering_to_yawrate,
                  rad_2_degree,
                  degree_2_rad,
                  check_car_collision_dis,
                  plot_car,
                  plot_arrow,
                  WB,
                  W,
                  pi_2_pi,
                  pi_2_pi_pos,
                  reduce_way_point
                  )


class GetLoss():
    def __init__(self, meter2risk: Meter2Risk, vecmap: Map, device: str = 'cuda', debug=False, writer=None):
        """
        lagrange term
        r-term
        meyer term
        *forward->*get_loss_by_Xu-->
                    |->get_(bb)_map_loss|
                    |->get_control_loss |>loss_process> return
                    |->get_meyer_loss   |
        """
        super().__init__()
        # init params
        cfg = load_cfg_here()['planner']
        self.cfg = cfg
        self.device = device
        self.ti = cfg['time_interval']
        self.th = cfg['time_horizon']
        # self.verbose = cfg['verbose']
        # self.using_global_ref = cfg['using_global_ref']
        # self.bb_col_check = cfg['bb_col_check']
        # self.max_num_bc = cfg['max_circle_num']
        # self.default_circle_interval = cfg['default_circle_interval']
        lcfg = cfg['loss']
        self.loss_CE = lcfg['loss_CE'] if 'loss_CE' in lcfg.keys() else True
        self.loss_var = lcfg['loss_var'] if 'loss_var' in lcfg.keys() else True
        self.loss_reg = lcfg['loss_reg'] if 'loss_reg' in lcfg.keys() else True
        # self.loss_GT_l1 = lcfg['loss_GT_l1'] if 'loss_GT_l1' in lcfg.keys() else True
        self.loss_idv = lcfg['loss_idv'] if 'loss_idv' in lcfg.keys() else True

        # self.loss_END = lcfg['loss_END'] if 'loss_END' in lcfg.keys() else True
        self.loss_cost_GT = lcfg['loss_cost_GT'] if 'loss_cost_GT' in lcfg.keys() else True

        # self.loss_goal:

        # init modules
        self.crossE = torch.nn.CrossEntropyLoss() if self.loss_CE else None

        # init variables
        self.extend_circles = None
        # self.lattice_sampler = torchLatticePlanner(self.cfg, self.device,self.training)

        self.debug = debug
        self.writer = writer
        self.meter2risk: Meter2Risk = None
        self.riskmap = vecmap

    def get_loss(self,
                 samples,
                 sample_risks,
                 traj_result,
                 gt,
                 detailed_gt,
                 gt_risk,
                 tb_iters=0,
                 tb_writer=None
                 ):
        """
        riskmap: include sdf, reflane and dynamic prediction and their measures
        fut_traj: [time, dim]
                dim: [x,y,yaw,v,yawrate,a,steer]

        """
        diffXd = torch.norm(
            samples['X'][..., :2] - gt[..., :2], dim=-1)
        sample_risks = sample_risks.mean(dim=-1)

        loss = 0
        if self.loss_cost_GT:
            # TODO find why gt risk is negtive
            loss_cost_GT = gt_risk.mean()
            loss += loss_cost_GT**2
            if tb_writer:
                tb_writer.add_scalar('loss/'+'loss_cost_GT', loss_cost_GT.mean(), tb_iters)

        if self.loss_CE:
            prob = torch.softmax(-sample_risks[..., 10:], dim=1)
            dis_prob = torch.softmax(-diffXd[..., 10:], dim=1)
            cls_loss = self.crossE(prob, dis_prob)
            loss += cls_loss
            if tb_writer:
                tb_writer.add_scalar('loss/'+'loss_CE', cls_loss.mean(), tb_iters)

        # the compliance of L2 diff between Xd and cost diff GT and Xd and GT
        if self.loss_reg:            
            closest_T = torch.min(diffXd.mean(dim=-1),dim=-1,keepdim=True)
            # remove gather
            loss_diff = (torch.gather(sample_risks,1,closest_T.indices.unsqueeze(dim=-1).repeat(1,1,self.th)) \
                        -gt_risk.mean(dim=-1))**2
            loss += loss_diff.mean()
            if tb_writer:
                tb_writer.add_scalar('loss/'+'loss_reg', loss_diff.mean(), tb_iters)

        # the cost variance of samples
        if self.loss_var:
            variance = torch.var(sample_risks.mean(dim=-1), dim=1).mean()
            loss_var = 1/variance.clamp(1e-1)
            loss += loss_var
            if tb_writer:
                tb_writer.add_scalar('loss/'+'loss_var', loss_var.mean(), tb_iters)

        return loss.mean()  # try max

    def get_extend_circles(self, data):
        """
        ego: [c,(x,y,r)]
        agents: [n,c,(x,y,r)]
        internal params: 
            - max_num_bc: 20
            - default_interval: 1
        return: circles[n+1,c,(x,y,r)] the last one is ego extend
        """

        agents_num = data['agents_num']
        egoex = data['ego_extent']  # (L,W,H)
        agentsex = data['agents_extend'][..., -1, :]  # [N,t,(L,W,H)]
        agentsex = torch.cat([agentsex, egoex.unsqueeze(dim=-2)], dim=-2)
        circles = torch.zeros(data['agents_num'].shape[0], 30+1, self.max_num_bc,
                              3, device=self.device)  # ! no inf !!!!
        circles[..., 0] = 1e5  # default x
        circles[..., 1] = 0  # default y
        circles[..., 2] = 0   # default r

        ahalfL = agentsex[..., 0]/2
        ahalfW = agentsex[..., 1]/2
        h_bounding_circle_num = torch.ceil(
            agentsex[..., 0]/self.default_circle_interval-2).clamp(min=1)
        h_bounding_circle_num[h_bounding_circle_num >
                              self.max_num_bc] = self.max_num_bc
        h_bounding_circle_interval = agentsex[..., 0]/(h_bounding_circle_num+1)
        for bz in range(data['agents_num'].shape[0]):
            for i, (cn, bi, R, Ld2) in enumerate(zip(h_bounding_circle_num[bz], h_bounding_circle_interval[bz], ahalfW[bz], ahalfL[bz])):
                circles[bz, i, :, -1] = R
                for n in range(cn.long()):
                    circles[bz, i, n, 0] = -Ld2+(n+1)*bi
        self.extend_circles = circles
        return circles

    def set_goal(self, goal):
        """
        set goal
        """
        # pass
        self.goal = goal

    def get_cross_entropy(self, fut_traj, fu, ego_feature, BBcollison=False, extend=None, writer=None, tb_iters=0):
        # print('>>>in get_cross_entropy>>>')
        has_nan(fut_traj)
        # Xsigma = 1.
        # usigma = 0.25
        # alpha = 1.
        # sampling_num = 12
        Btsz, GTts, GTdim = fut_traj.shape
        # uts, udim = fu.shape
        s0 = self.get_init_state_from_ego_feature(ego_feature)
        # sampling around reflane
        reflane = self.riskmap.reflane
        self.lattice_sampler.set_init_state(s0, reflane)
        Xd, ud = self.lattice_sampler.make_sample()
        Xd = torch.cat([Xd[..., :4], fut_traj.unsqueeze(1)], dim=1)
        ud = torch.cat([ud, fu.unsqueeze(1)], dim=1)
        costs = self.get_loss_by_Xu(Xd,
                                    ud,
                                    BBcollison=BBcollison,
                                    extend=extend,
                                    vis_loss=True,
                                    batch=True,
                                    writer=writer,
                                    tb_iters=tb_iters)

        loss = 0
        # # cross entropy
        prob = torch.softmax(-costs[..., 10:], dim=0)
        diffXd = torch.norm(
            Xd[..., :2] - fut_traj.reshape(Btsz, 1, GTts, GTdim)[..., :2], dim=-1)
        if self.loss_CE:
            dis_prob = torch.softmax(-diffXd[..., 10:], dim=0)
            cls_loss = self.crossE((prob).unsqueeze(0), dis_prob.unsqueeze(0))
            loss += cls_loss

        # the compliance of L2 diff between Xd and cost diff GT and Xd and GT
        if self.loss_reg:
            closest_T = torch.argmin(diffXd[:, :-1].mean(dim=-1))
            loss_diff = (costs[:, closest_T]-costs[:, -1])**2
            loss += loss_diff.mean()

        # the cost variance of samples
        if self.loss_var:
            variance = torch.var(costs.mean(dim=-1), dim=1).mean()
            loss += 1/variance.clamp(1e-1)

        # velocity difference between GT and NN generated
        try:
            if self.loss_idv:
                dv = torch.pow(self.meter2risk.get_target_v(
                ).squeeze() - fut_traj[..., -1], 2).sum()
                loss += dv
                writer.add_scalar('loss_ele/' + 'dv', dv, self.tb_iters)
        except:
            print('failed to calculate dv')
            pass

        # log
        writer.add_scalar('raw_meter/' + 's0', s0.mean(), self.tb_iters)
        writer.add_scalar('raw_meter/' + 'v_raw',
                          fut_traj[..., -1].mean(), self.tb_iters)
        writer.add_scalar('raw_meter/' + 'sample diffXd.mean()',
                          diffXd.mean(), self.tb_iters)
        writer.add_scalar('raw_meter/' + 'sample costs.mean()',
                          costs.mean(), self.tb_iters)
        writer.add_scalar('loss_ele/' + 'traj_cost',
                          costs[:, -1].mean(), self.tb_iters)

        if self.loss_CE:
            writer.add_scalar('loss_ele/' + 'cls_loss',
                              cls_loss, self.tb_iters)
        if self.loss_reg:
            writer.add_scalar('loss_ele/' + 'reg_loss',
                              loss_diff.mean(), self.tb_iters)
        if self.loss_var:
            writer.add_scalar('loss_ele/' + 'variance',
                              variance, self.tb_iters)
        ind = torch.argmin(costs[:, :-1].mean(dim=-1), dim=-1)
        validXd = torch.gather(Xd, dim=1, index=ind.reshape(
            Btsz, 1, 1, 1).repeat(1, 1, 30, 4)).squeeze(1)
        valid = torch.norm(validXd[..., :2]-fut_traj[..., :2], dim=-1)
        writer.add_scalar('train/' + 'validate', valid.mean(), self.tb_iters)
        # print('===get_cross_entropy end===')
        return loss

    # the main function getting flexible loss
    def get_loss_by_Xu(self,
                       fut_traj,
                       u,
                       BBcollison=False,
                       extend=None,
                       vis_loss=False,
                       batch=False,
                       writer=None,
                       tb_iters=0
                       ):
        """
        fut_traj: future trajectory in form [t,dim]
        u: future control in form [t,dim]
        CKL: indicate whether calculating KL divergnece
        BBcollison: indicates whether using circle bounding box
        extend: the raw extend circles in ego view, *only functional when BBcollision is Ture
        """
        # Lagrangian costs
        has_nan(fut_traj)
        # get costs including sdf, reflane, collision prob
        if BBcollison:
            # check is extend circles initialized
            if extend is None:
                if self.extend_circles is None:
                    raise ValueError(
                        "You should initialize extend first by function 'get_extend_circles()'")
                extend = self.extend_circles
            map_measurement = self.riskmap.get_bb_map_cost(
                fut_traj[..., :3], extend, batch=batch)
        else:
            map_measurement = self.riskmap.get_map_cost(
                fut_traj[..., :3], batch=batch)

        #####
        # vis map measurement
        #####
        axis = torch.linspace(0.1, 3, 30)
        if self.debug:
            for i in range(4):
                plt.close()
                fig, ax = plt.subplots(2, 1)
                plt.title(f'i is {i}')
                print(f"ploting {i}")
                # ax[0].set_ylim(-10,10)
                ax[1].set_ylim(-5, 5)
                for mm, traj in zip(map_measurement.cpu().detach().view(-1, 30, 4),
                                    fut_traj[..., :3].cpu().detach().view(-1, 30, 3)):
                    ax[0].plot(axis, mm[..., i])
                    ax[1].scatter(traj[..., 0], traj[..., 1],
                                  c=mm[..., i], alpha=0.5, cmap='spring')
                plt.axis('equal')
                plt.show()
        # get dv
        v = fut_traj[..., -1:]
        # raw_meters: [sdf, ref, tl, collide, a, s, delta_v, v]
        raw_meters = torch.cat([map_measurement.squeeze(), u, v], dim=-1)
        has_nan(raw_meters)
        costs = self.meter2risk(raw_meters.reshape(raw_meters.shape[0], -1,
                                self.th,
                                raw_meters.shape[-1]),
                                self.riskmap.prediction_dict,
                                writer=writer,
                                tb_iters=tb_iters)
        # print(costs[-1].mean(dim=-1))
        has_nan(costs)
        if vis_loss:
            return costs.mean(dim=-1)  # try max
        regulator = self.meter2risk.regulator() if self.training else 0
        # L_cost.sum()+R_cost.sum()+M_cost
        return costs.mean()+regulator, map_measurement

    def get_init_state_from_ego_feature(self, ego_feature: torch.Tensor):
        # print(ego_feature)
        s0 = torch.zeros_like(ego_feature, device=self.device)
        s = torch.norm(torch.diff(ego_feature[..., :2], dim=1), dim=-1)
        v = s/self.ti
        v = torch.cat([v[:, 0:1], v], dim=1)
        s0[..., :3] = ego_feature[..., :3]
        s0[..., 3] = v
        a = torch.diff(v, dim=1)/self.ti
        # a = torch.nan_to_num(a,0)
        # a = a.clamp(min=-5,max=5)
        a = torch.cat([a[:, 0:1], a], dim=1).unsqueeze(-1)
        s0 = torch.cat([s0, a], dim=-1)
        return s0[:, -1]
