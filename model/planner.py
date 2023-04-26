from copy import deepcopy
import logging
import sys
import os
from turtle import TurtleScreenBase
from attr import has
from matplotlib.pyplot import flag
from numpy import cross
import torch
import numpy as np
from utils.riskmap.car import MAX_ACC, MAX_STEER, bicycle_model, pi_2_pi, pi_2_pi_pos
from utils.riskmap.torch_lattice import LatticeSampler
from utils.test_utils import batch_check_collision, batch_check_traffic, batch_sample_check_collision, batch_sample_check_traffic
try:
    import theseus as th
except:
    logging.warning('[WARNING] theseus is not implemented')
from utils.train_utils import project_to_frenet_frame
from model.meter2risk import Meter2Risk
from utils.riskmap.rm_utils import get_u_from_X, has_nan, load_cfg_here, yawv2yawdxdy

sys.path.append(os.getenv('DIPP_ABS_PATH'))

def get_sample(context, gt_u = None, cov = torch.tensor([0.2, 0.1]), sample_num=100, turb_num=1, no_ref=True, use_lattice=False, add_noref:int=0):
    cov = cov.to(context['init_guess_u'].device)
    lim = torch.tensor([MAX_ACC, MAX_STEER],device = context['init_guess_u'].device)
    cov = cov*lim
    btsz = context['init_guess_u'].shape[0]
    init_guess_u = context['init_guess_u'] if gt_u==None else gt_u
    init_guess_u = init_guess_u if not no_ref else torch.zeros_like(init_guess_u)
    init_guess_u=init_guess_u.unsqueeze(1) if len(init_guess_u.shape)==3 else init_guess_u
    init_guess_u[...,1] = pi_2_pi(init_guess_u[...,1])
    if init_guess_u.shape[1]>1:
        # implement same noise to topk initguess
        u = (torch.randn([btsz,init_guess_u.shape[1],sample_num,turb_num,2],
                         device=init_guess_u.device)*cov) \
            +init_guess_u.unsqueeze(2).repeat(1,1,sample_num,1,1)
        u = u.reshape(btsz,-1,50,2)
    else:
        u = (torch.randn([btsz,sample_num,turb_num,2], 
                         device=init_guess_u.device)*cov)+init_guess_u
    if add_noref and add_noref<u.shape[-3]:
        u = torch.cat([u,cov*torch.randn_like(u[:,0:add_noref,:,:])],dim=-3)
    u = u.clamp(min=-lim, max=lim)
    cur_state = context['current_state'][:,0:1]
    X = bicycle_model(u,cur_state)
    u = torch.cat([init_guess_u,u],dim=-3)
    X = torch.cat([bicycle_model(init_guess_u,cur_state),X],dim=-3)
    if 'lattice_sample' not in context or not use_lattice:
        return {'X':X,'u':u} 
    if context['lattice_sample'] is not None:
        lim = torch.tensor([MAX_ACC,MAX_STEER],device = X.device)
        l_trajs = context['lattice_sample']
        if add_noref and add_noref<=u.shape[-3]:
            l_trajs = torch.cat([l_trajs,torch.nan*torch.ones_like(l_trajs[:,0:add_noref,:,:])],dim=-3)
        # l_trajs = torch.nan_to_num(l_trajs, nan=np.inf)
        l_u = get_u_from_X(l_trajs, cur_state.repeat(1,l_trajs.shape[1],1))
        # True standfor padded place
        masks = l_trajs.sum(dim=[-2,-1]).isnan() + l_u.sum(dim=[-2,-1]).isnan() \
            + l_trajs.sum(dim=[-2,-1]).isinf() + l_u.sum(dim=[-2,-1]).isinf() \
            + (l_u>lim).amax(dim=[-2,-1]) + (l_u<-lim).amax(dim=[-2,-1])
        masks = masks.bool()
        l_trajs[masks] = 0.
        X[:,1:][~masks] = 0.
        X[:,1:] = l_trajs + X[:,1:]
        l_u[masks] = 0.
        u[:,1:][~masks] = 0.
        u[:,1:] = l_u + u[:,1:]
    return {'X':X,'u':u}

def sample_loss(CE, sample, gt, costs, cfg, tk=0, interval=10):
    loss = 0
    traj_risk = costs[:,::interval]
    if not(cfg['loss']['dis_prob'] or cfg['loss']['gt_label']):
        return loss
    diffXd = torch.norm(
        sample['X'][:,::interval,..., :3] - gt[..., :3], dim=-1)
    diffXd = torch.log(torch.nn.functional.normalize(diffXd,dim=1)+1)
    
    if tk:
        diffXd, Xd_id_best = torch.topk(diffXd.mean(dim=-1), k=tk, dim=1, largest=False, sorted=False)
        traj_risk = torch.mean(traj_risk,dim=[-1])
        traj_risk = torch.gather(traj_risk,1,Xd_id_best)
    traj_risk = torch.log(torch.nn.functional.normalize(traj_risk,dim=1)+1)
    
    prob = torch.softmax(-traj_risk, dim=1)
    dis_prob = torch.softmax(-diffXd, dim=1)
    if cfg['loss']['dis_prob']:
        # smoothed distance loss
        loss += CE(prob, dis_prob)
    if cfg['loss']['gt_label']:
        # label loss
        # can only be enbaled when gt is provided
        label = torch.zeros_like(prob[:,0]).long()
        loss += CE(prob, label)
    return loss

def selector(raw_costs, sample_plan, k=1, hand_prefer=torch.ones([15])):
    costs = raw_costs*hand_prefer[:raw_costs.shape[-1]].to(raw_costs.device)
    i = torch.topk(costs.mean(dim=[-1,-2]),k=k,dim=1,largest=False).indices
    sampleX = sample_plan['X']
    sampleu = sample_plan['u']
    _,_,t,d = sampleX.shape
    _,_,_,du = sampleu.shape
    X = torch.gather(sampleX,1,i.unsqueeze(-1).unsqueeze(-1).repeat(1,1,t,d))
    u = torch.gather(sampleu,1,i.unsqueeze(-1).unsqueeze(-1).repeat(1,1,t,du))
    return X.squeeze(1),u.squeeze(1) #torch.gather(sample_plan['X'],1,i),torch.gather(sample_plan['u'],1,i[...,:2])

def sampling_plan(context, meter2risk, measure, 
                  genetic=0, 
                  plan_sample_num=200, 
                  cov_base=torch.tensor([0.2,0.2]), 
                  turb_num=50, 
                  no_ref=False, 
                  use_lattice=True, 
                  add_noref=30, 
                  collision_check=False):
    new_context = {'init_guess_u': context['init_guess_u'].clone().detach(),
                    'lattice_sample': context['lattice_sample'].clone().detach(),
                    'current_state': context['current_state'].clone().detach(),
                    }
    
    if genetic:
        for i in range(genetic):
            sample_plan = get_sample(new_context, 
                                        sample_num=plan_sample_num if i==0 else 30, 
                                        cov=cov_base,
                                        turb_num=turb_num if i==0 else 50,
                                        no_ref=no_ref,
                                        use_lattice=use_lattice if i==0 else False,
                                        add_noref=add_noref,
                                        )
            meter = measure(
                            sample_plan, 
                            context['current_state'], 
                            context['predictions'], 
                            context['ref_line_info'], 
                            context['vf_map']
                            )
            costs = meter2risk(meter) if meter2risk is not None else meter
            # if not self.training:
            if collision_check:
                collide = batch_sample_check_collision(sample_plan['X'], context['predictions'], context['current_state'][..., 5:]).bool()
                collide = collide.unsqueeze(-1).unsqueeze(-1)*torch.inf
                collide = torch.nan_to_num(collide, nan=0)
            else:
                collide = torch.zeros([1],device=sample_plan['X'].device)
            plan_result = selector(costs, sample_plan, k=10)
            new_context['init_guess_u'] = plan_result[1]
            # new_context.pop('lattice_sample', None)
    sample_plan = get_sample(new_context, 
                                    sample_num=plan_sample_num if not genetic else 30, 
                                    cov=cov_base,
                                    turb_num=turb_num if not genetic else 50,
                                    no_ref=no_ref,
                                    use_lattice=use_lattice if not genetic else False,
                                    add_noref=30,
                                    )
    # if not self.training:
    if collision_check:
        collide = batch_sample_check_collision(sample_plan['X'],context['predictions'], context['current_state'][..., 5:]).bool()
        collide = collide.unsqueeze(-1).unsqueeze(-1)*torch.inf
        collide = torch.nan_to_num(collide, nan=0)
    else:
        collide = torch.zeros([1],device=sample_plan['X'].device)
    # else:
    # collide = torch.zeros_like(self.sample_plan['X'][...,0:1],device=self.device)
    meter = measure(
                sample_plan, 
                context['current_state'], 
                context['predictions'], 
                context['ref_line_info'], 
                context['vf_map']
                )
    costs = meter2risk(meter) if meter2risk is not None else meter
    plan_result = selector(costs+collide, sample_plan)
    return plan_result, sample_plan

class Planner:
    def __init__(self, device='cuda:0', test=False) -> None:
        self.name = None
        self.device = device
        self.cfg = load_cfg_here()['planner']
        self.test = test



class BasePlanner(Planner):
    def __init__(self,device, test=False) -> None:
        super(BasePlanner, self).__init__(device, test)
        self.name = 'base'

class MotionPlanner(Planner):
    def __init__(self, trajectory_len, feature_len, device='cuda:0', test=False):
        super(MotionPlanner, self).__init__(device, test)
        self.name = 'dipp'

        self.device = device

        # define cost function
        cost_function_weights = [th.ScaleCostWeight(th.Variable(torch.rand(
            1), name=f'cost_function_weight_{i+1}')) for i in range(feature_len)]

        # define control variable
        control_variables = th.Vector(dof=100, name="control_variables")

        # define prediction variable
        predictions = th.Variable(torch.empty(
            1, 10, trajectory_len, 3), name="predictions")

        # define ref_line_info
        ref_line_info = th.Variable(
            torch.empty(1, 1200, 5), name="ref_line_info")

        # define current state
        current_state = th.Variable(
            torch.empty(1, 11, 8), name="current_state")

        # set up objective
        objective = th.Objective()
        self.objective = cost_function(
            objective, control_variables, current_state, predictions, ref_line_info, cost_function_weights)

        # set up optimizer
        if test:
            self.optimizer = th.GaussNewton(
                objective, th.CholeskyDenseSolver, vectorize=False, max_iterations=50, step_size=0.2, abs_err_tolerance=1e-2)
        else:
            self.optimizer = th.GaussNewton(
                objective, th.CholeskyDenseSolver, vectorize=False, max_iterations=2, step_size=0.4)

        # set up motion planner
        self.layer = th.TheseusLayer(self.optimizer, vectorize=False)
        self.layer.to(device=device)

class EularSamplingPlanner(Planner):
    def __init__(self, meter2risk: Meter2Risk, device='cuda:0', test=False):
        super(EularSamplingPlanner, self).__init__(device, test)
        self.name = 'esp'
        self.device = device
        self.crossE = torch.nn.CrossEntropyLoss() #if self.loss_CE else None
        self.cost_function_weights = meter2risk
        self.cov_base = torch.tensor([0.2, 0.1])
        self.cov_inc = torch.tensor([1+2e-4, 1+2e-4])
        self.turb_num = 50
        
        self.gt_sample_num = self.cfg['gt_sample_num']
        self.plan_sample_num = self.cfg['plan_sample_num']
        try:
            self.cov_base = torch.tensor(self.cfg['cov_base'])
            self.cov_inc = torch.tensor(self.cfg['cov_inc']) + 1.
            self.turb_num = 1 if self.cfg['const_turb'] else 50 
        except:
            logging.warning('cov_base amd conv_inc not define')
    
    def plan(self, context, genetic=0):
        self.context = context
        self.plan_result, self.sample_plan = sampling_plan(self.context, 
                             self.cost_function_weights,
                             cost_function_sample,
                             genetic=genetic,
                             plan_sample_num=self.plan_sample_num,
                             cov_base=self.cov_base,
                             turb_num=self.turb_num,
                             use_lattice=True,
                             collision_check=self.collision_check,
                            )
        return self.plan_result

    def get_loss(self, gt, tb_iter=0, tb_writer=None):
        """
        must be called after forward
        """
        self.gt_sample = {'X':self.sample_plan['X'][:,::2], 'u':self.sample_plan['u'][:,::2]}
        cost = cost_function_sample(self.gt_sample['u'], 
                                    self.context['current_state'], 
                                    self.context['predictions'], 
                                    self.context['ref_line_info'], 
                                    )
        cost = self.cost_function_weights(cost)
        
        loss = 0
        if self.cfg['loss']['nmp_loss']:
            gt_x = self.gt_sample['X'][:,0:1]
            sample_x = self.gt_sample['X'][:,1:]
            gt_cost = cost[:,0:1]
            sample_cost = cost[:,1:]
            collide = batch_sample_check_collision(sample_x,self.context['predictions'], self.context['current_state'][:, :, 5:])
            tl, offroad = batch_sample_check_traffic(sample_x, self.context['ref_line_info'])
            L = gt_cost \
                -sample_cost+torch.norm(gt_x[...,:2]-sample_x[...,:2],dim=-1, keepdim=True) \
                +torch.logical_or(collide, torch.logical_or(tl, offroad)).unsqueeze(-1).unsqueeze(-1)*1000
            L = L.clamp(min=0, max=1000)
            L = L.sum(dim=-1)
            L = torch.amax(L,dim=-1)
            loss += L.mean()
        else:
            loss += sample_loss(self.crossE, self.gt_sample, gt, cost, self.cfg)
        if tb_writer and loss.device==torch.device('cuda:0'):
            tb_writer.add_scalar('train/'+'plan_loss', loss.mean(), tb_iter)

        return loss



class RiskMapPlanner(Planner):
    def __init__(self, meter2risk: Meter2Risk, device, test=False) -> None:
        super(RiskMapPlanner, self).__init__(device, test)
        self.name = 'risk'
        # self.lattice_planner = torchLatticePlanner(device, test=test)
        self.hand_prefer = torch.softmax(
            torch.tensor(self.cfg['risk_preference']), dim=0
        )  # handcratfed preference
        self.meter2risk = meter2risk
        self.crossE = torch.nn.CrossEntropyLoss(label_smoothing=0.05) #if self.loss_CE else None

        self.cov_base = torch.tensor([0.1, 0.005])
        self.cov_inc = torch.tensor([1+2e-4, 1+2e-4])
        self.turb_num = 50
        
        self.gt_sample_num = self.cfg['gt_sample_num']
        self.plan_sample_num = self.cfg['plan_sample_num']
        self.collision_check = self.cfg['collision_check']
        try:
            self.cov_base = torch.tensor(self.cfg['cov_base'])
            self.cov_inc = torch.tensor(self.cfg['cov_inc']) + 1.
            self.turb_num = 1 if self.cfg['const_turb'] else 50 
        except:
            logging.warning('cov_base amd conv_inc not define')

    def plan(self, context, genetic:int = 0):
        self.context = context
        self.plan_result, self.sample_plan = sampling_plan(self.context, 
                             self.meter2risk,
                             risk_cost_function_sample,
                             genetic=genetic,
                             plan_sample_num=self.plan_sample_num,
                             cov_base=self.cov_base,
                             turb_num=self.turb_num,
                             use_lattice=True,
                             collision_check=self.collision_check,
                            )
        return self.plan_result

    def get_loss(self, gt, tb_iter=0, tb_writer=None):
        """
        must be called after forward
        """
        self.gt_sample = {'X':self.sample_plan['X'][:,::2], 'u':self.sample_plan['u'][:,::2]}
        raw_meter = risk_cost_function_sample(
                                            self.gt_sample, 
                                            self.context['current_state'], 
                                            self.context['predictions'], 
                                            self.context['ref_line_info'], 
                                            self.context['vf_map']
                                            )
        gt_risk = self.meter2risk(raw_meter)
        gt_risk = gt_risk.mean(dim=-1)

        loss = 0
        if self.cfg['loss']['nmp_loss']:
            dxdy = self.context['vf_map'].vector_field_diff(self.gt_sample['X'])
            gt_x = self.gt_sample['X'][:,0:1]
            sample_x = self.gt_sample['X'][:,1:]
            gt_dxdy = dxdy[:,0:1]
            sample_dxdy = dxdy[:,1:]
            collide = batch_sample_check_collision(sample_x,self.context['predictions'], self.context['current_state'][:, :, 5:])
            tl, offroad = batch_sample_check_traffic(sample_x, self.context['ref_line_info'])
            L = torch.norm(gt_dxdy-sample_dxdy,dim=-1, keepdim=True) \
                +torch.norm(gt_x[...,:2]-sample_x[...,:2],dim=-1, keepdim=True) \
                +torch.logical_or(collide, torch.logical_or(tl, offroad)).unsqueeze(-1).unsqueeze(-1)*1000
            L = L.clamp(min=0, max=1000)
            L = L.sum(dim=-1)
            L = torch.amax(L,dim=-1)
            loss += L.mean()
        else:
            loss += sample_loss(self.crossE, self.gt_sample, gt, gt_risk, self.cfg)
        if tb_writer and loss.device==torch.device('cuda:0'):
            tb_writer.add_scalar('train/'+'plan_loss', loss.mean(), tb_iter)

        return loss

class CostMapPlanner(Planner):
    def __init__(self, meter2risk: Meter2Risk, device, test=False) -> None:
        super(CostMapPlanner, self).__init__(device, test)
        self.name = 'nmp'
        # self.lattice_planner = torchLatticePlanner(device, test=test)
        self.hand_prefer = torch.softmax(
            torch.tensor(self.cfg['risk_preference']), dim=0
        )  # handcratfed preference
        self.meter2risk = meter2risk
        self.crossE = torch.nn.CrossEntropyLoss(label_smoothing=0.3) #if self.loss_CE else None

        self.cov_base = torch.tensor([0.1, 0.005])
        self.cov_inc = torch.tensor([1+2e-4, 1+2e-4])
        self.turb_num = 50
        
        self.gt_sample_num = self.cfg['gt_sample_num']
        self.plan_sample_num = self.cfg['plan_sample_num']

        try:
            self.cov_base = torch.tensor(self.cfg['cov_base'])
            self.cov_inc = torch.tensor(self.cfg['cov_inc']) + 1.
            self.turb_num = 1 if self.cfg['const_turb'] else 50 
        except:
            logging.warning('cov_base amd conv_inc not define')

    def plan(self, context, genetic:int = 0):
        self.context = context
        self.plan_result, self.sample_plan = sampling_plan(self.context, 
                             self.meter2risk,
                             self.context['cost_map'].get_cost_by_pos,
                             genetic=genetic,
                             plan_sample_num=self.plan_sample_num,
                             cov_base=self.cov_base,
                             turb_num=self.turb_num,
                             use_lattice=True,
                             collision_check=self.collision_check,
                            )
        return self.plan_result

    def get_loss(self, gt, tb_iter=0, tb_writer=None):
        """
        must be called after forward
        """
        loss = 0
        self.gt_sample = {'X':self.sample_plan['X'][:,::2], 'u':self.sample_plan['u'][:,::2]}
        sample_costs = self.context['cost_map'].get_cost_by_pos(self.gt_sample['X'])
        if self.cfg['loss']['nmp_loss']:
            cost = self.context['cost_map'].get_cost_by_pos(self.gt_sample['X'])
            gt_x = self.gt_sample['X'][:,0:1]
            sample_x = self.gt_sample['X'][:,1:]
            gt_cost = cost[:,0:1]
            sample_cost = cost[:,1:]
            collide = batch_sample_check_collision(sample_x,self.context['predictions'], self.context['current_state'][:, :, 5:])
            tl, offroad = batch_sample_check_traffic(sample_x,self.context['ref_line_info'])
            L = gt_cost-sample_cost+torch.norm(gt_x[...,:2]-sample_x[...,:2],dim=-1, keepdim=True)+torch.logical_or(collide, torch.logical_or(tl, offroad)).unsqueeze(-1).unsqueeze(-1)*1000
            L = L.clamp(min=0, max=1000)
            L = L.sum(dim=-1)
            L = torch.amax(L,dim=-1)
            loss += L.mean()
        else:
            loss += sample_loss(self.crossE, self.gt_sample, gt, sample_costs, self.cfg)
        if tb_writer and loss.device==torch.device('cuda:0'):
            tb_writer.add_scalar('train/'+'plan_loss', loss.mean(), tb_iter)
        return loss



# cost functions


def acceleration(optim_vars, aux_vars):
    control = optim_vars[0].tensor.view(-1, 50, 2)
    acc = control[:, :, 0]

    return acc


def jerk(optim_vars, aux_vars):
    control = optim_vars[0].tensor.view(-1, 50, 2)
    acc = control[:, :, 0]
    jerk = torch.diff(acc) / 0.1

    return jerk


def steering(optim_vars, aux_vars):
    control = optim_vars[0].tensor.view(-1, 50, 2)
    steering = control[:, :, 1]

    return steering


def steering_change(optim_vars, aux_vars):
    control = optim_vars[0].tensor.view(-1, 50, 2)
    steering = control[:, :, 1]
    steering_change = torch.diff(steering) / 0.1

    return steering_change


def speed(optim_vars, aux_vars):
    control = optim_vars[0].tensor.view(-1, 50, 2)
    current_state = aux_vars[1].tensor[:, 0]
    velocity = torch.hypot(current_state[:, 3], current_state[:, 4])
    dt = 0.1

    acc = control[:, :, 0]
    speed = velocity.unsqueeze(1) + torch.cumsum(acc * dt, dim=1)
    speed = torch.clamp(speed, min=0)
    speed_limit = torch.max(
        aux_vars[0].tensor[:, :, -1], dim=-1, keepdim=True)[0]
    speed_error = speed - speed_limit

    return speed_error


def lane_xy(optim_vars, aux_vars):
    global ref_points

    control = optim_vars[0].tensor.view(-1, 50, 2)
    ref_line = aux_vars[0].tensor
    current_state = aux_vars[1].tensor[:, 0]

    traj = bicycle_model(control, current_state)
    distance_to_ref = torch.cdist(traj[:, :, :2], ref_line[:, :, :2])
    k = torch.argmin(distance_to_ref, dim=-1).view(-1,
                                                   traj.shape[1], 1).expand(-1, -1, 3)
    ref_points = torch.gather(ref_line, 1, k)
    lane_error = torch.cat([traj[:, 1::2, 0]-ref_points[:, 1::2, 0],
                           traj[:, 1::2, 1]-ref_points[:, 1::2, 1]], dim=1)

    return lane_error


def lane_theta(optim_vars, aux_vars):
    control = optim_vars[0].tensor.view(-1, 50, 2)
    current_state = aux_vars[1].tensor[:, 0]

    traj = bicycle_model(control, current_state)
    theta = traj[:, :, 2]
    lane_error = theta[:, 1::2] - ref_points[:, 1::2, 2]

    return lane_error


def red_light_violation(optim_vars, aux_vars):
    control = optim_vars[0].tensor.view(-1, 50, 2)
    current_state = aux_vars[1].tensor[:, 0]
    ref_line = aux_vars[0].tensor
    red_light = ref_line[..., -1]
    dt = 0.1

    velocity = torch.hypot(current_state[:, 3], current_state[:, 4])
    acc = control[:, :, 0]
    speed = velocity.unsqueeze(1) + torch.cumsum(acc * dt, dim=1)
    speed = torch.clamp(speed, min=0)
    s = torch.cumsum(speed * dt, dim=-1)

    stop_point = torch.max(red_light[:, 200:] == 0, dim=-1)[1] * 0.1
    stop_distance = stop_point.view(-1, 1) - 3
    red_light_error = (s - stop_distance) * \
        (s > stop_distance) * (stop_point.unsqueeze(-1) != 0)

    return red_light_error


def safety(optim_vars, aux_vars):
    control = optim_vars[0].tensor.view(-1, 50, 2)
    neighbors = aux_vars[0].tensor.permute(0, 2, 1, 3)
    current_state = aux_vars[1].tensor
    ref_line = aux_vars[2].tensor

    actor_mask = torch.ne(current_state, 0)[:, 1:, -1]
    ego_current_state = current_state[:, 0]
    ego = bicycle_model(control, ego_current_state)
    ego_len, ego_width = ego_current_state[:, -3], ego_current_state[:, -2]
    neighbors_current_state = current_state[:, 1:]
    neighbors_len, neighbors_width = neighbors_current_state[..., -3], neighbors_current_state[..., -2]

    l_eps = (ego_width.unsqueeze(1) + neighbors_width)/2 + 0.5
    frenet_neighbors = torch.stack([project_to_frenet_frame(
        neighbors[:, :, i].detach(), ref_line) for i in range(neighbors.shape[2])], dim=2)
    frenet_ego = project_to_frenet_frame(ego.detach(), ref_line)

    safe_error = []
    for t in [0, 2, 5, 9, 14, 19, 24, 29, 39, 49]:  # key frames
        # find objects of interest
        l_distance = torch.abs(frenet_ego[:, t, 1].unsqueeze(
            1) - frenet_neighbors[:, t, :, 1])
        s_distance = frenet_neighbors[:, t, :,
                                      0] - frenet_ego[:, t, 0].unsqueeze(-1)
        interactive = torch.logical_and(
            s_distance > 0, l_distance < l_eps) * actor_mask

        # find closest object
        distances = torch.norm(ego[:, t, :2].unsqueeze(
            1) - neighbors[:, t, :, :2], dim=-1).squeeze(1)
        distances = torch.masked_fill(
            distances, torch.logical_not(interactive), 100)
        distance, index = torch.min(distances, dim=1)
        s_eps = (ego_len + torch.index_select(neighbors_len, 1, index)
                 [:, 0])/2 + 5

        # calculate cost
        error = (s_eps - distance) * (distance < s_eps)
        safe_error.append(error)

    safe_error = torch.stack(safe_error, dim=1)

    return safe_error


def  cost_function(objective, control_variables, current_state, predictions, ref_line, cost_function_weights, vectorize=True):
    # travel efficiency
    speed_cost = th.AutoDiffCostFunction([control_variables], speed, 50, cost_function_weights[0], aux_vars=[
                                         ref_line, current_state], autograd_vectorize=vectorize, name="speed")
    objective.add(speed_cost)

    # comfort
    acc_cost = th.AutoDiffCostFunction([control_variables], acceleration, 50,
                                       cost_function_weights[1], autograd_vectorize=vectorize, name="acceleration")
    objective.add(acc_cost)
    jerk_cost = th.AutoDiffCostFunction(
        [control_variables], jerk, 49, cost_function_weights[2], autograd_vectorize=vectorize, name="jerk")
    objective.add(jerk_cost)
    steering_cost = th.AutoDiffCostFunction(
        [control_variables], steering, 50, cost_function_weights[3], autograd_vectorize=vectorize, name="steering")
    objective.add(steering_cost)
    steering_change_cost = th.AutoDiffCostFunction(
        [control_variables], steering_change, 49, cost_function_weights[4], autograd_vectorize=vectorize, name="steering_change")
    objective.add(steering_change_cost)

    # lane
    lane_xy_cost = th.AutoDiffCostFunction([control_variables], lane_xy, 50, cost_function_weights[5], aux_vars=[
                                           ref_line, current_state], autograd_vectorize=vectorize, name="lane_xy")
    objective.add(lane_xy_cost)
    lane_theta_cost = th.AutoDiffCostFunction([control_variables], lane_theta, 25, cost_function_weights[6], aux_vars=[
                                              ref_line, current_state], autograd_vectorize=vectorize, name="lane_theta")
    objective.add(lane_theta_cost)

    # traffic rules
    red_light_cost = th.AutoDiffCostFunction([control_variables], red_light_violation, 50, cost_function_weights[7], aux_vars=[
                                             ref_line, current_state], autograd_vectorize=vectorize, name="red_light")
    objective.add(red_light_cost)
    safety_cost = th.AutoDiffCostFunction([control_variables], safety, 10, cost_function_weights[8], aux_vars=[
                                          predictions, current_state, ref_line], autograd_vectorize=vectorize, name="safety")
    objective.add(safety_cost)

    return objective

def _acceleration(optim_vars, aux_vars):
    control = optim_vars[0].tensor#.view(-1, 50, 2)
    acc = control[..., 0]

    return acc


def _jerk(optim_vars, aux_vars):
    control = optim_vars[0].tensor#.view(-1, 50, 2)
    acc = control[..., 0]
    zeros = torch.zeros_like(acc[...,-1:],device=acc.device)
    jerk = torch.diff(acc, dim=-1, append=zeros) / 0.1

    return jerk


def _steering(optim_vars, aux_vars):
    control = optim_vars[0].tensor#.view(-1, 50, 2)
    steering = control[..., 1]

    return steering


def _steering_change(optim_vars, aux_vars):
    control = optim_vars[0].tensor#.view(-1, 50, 2)
    steering = control[..., 1]
    zeros = torch.zeros_like(steering[...,-1:],device=steering.device)
    steering_change = torch.diff(steering, dim=-1, append=zeros) / 0.1

    return steering_change


def _speed(optim_vars, aux_vars):
    control = optim_vars[0].tensor#.view(-1, 50, 2)
    current_state = aux_vars[1].tensor[:, 0:1]
    velocity = torch.hypot(current_state[..., 3], current_state[..., 4])
    dt = 0.1

    acc = control[..., 0]
    speed = velocity.unsqueeze(-2) + torch.cumsum(acc * dt, dim=-1)
    speed = torch.clamp(speed, min=0)
    speed_limit = torch.max(
        aux_vars[0].tensor[..., -1], dim=-1, keepdim=True)[0]
    speed_error = speed - speed_limit.unsqueeze(-1)

    return speed_error


def _lane_xy(optim_vars, aux_vars):
    global ref_points
    control = optim_vars[0].tensor#.view(-1, 50, 2)
    ref_line = aux_vars[0].tensor
    current_state = aux_vars[1].tensor[:, 0:1]
    
    traj = bicycle_model(control, current_state)
    btsz,mod,th,dim = traj.shape
    distance_to_ref = torch.cdist(traj[..., :2].reshape(btsz,-1,2), ref_line[:, :, :2])
    distance_to_ref = distance_to_ref.reshape(btsz,mod,th,-1)
    k = torch.argmin(distance_to_ref, dim=-1).view(btsz,
                                                   mod, th, 1).expand(-1, -1, -1, 3)
    ref_points = torch.gather(ref_line.unsqueeze(-3).repeat(1,mod,1,1), 2, k)
    lane_error = torch.cat([traj[..., 0:1]-ref_points[..., 0:1],
                           traj[..., 1:2]-ref_points[..., 1:2], 
                           ],  dim=-1)
    lane_error = torch.norm(lane_error,dim=-1)
    return lane_error

def _lane_theta(optim_vars, aux_vars):
    control = optim_vars[0].tensor#.view(-1, 50, 2)
    current_state = aux_vars[1].tensor[:, 0:1]

    traj = bicycle_model(control, current_state)
    theta = traj[..., 2]
    lane_error = theta[...] - ref_points[..., 2]

    return lane_error


def _red_light_violation(optim_vars, aux_vars):
    control = optim_vars[0].tensor#.view(-1, 50, 2)
    current_state = aux_vars[1].tensor[:, 0:1]
    ref_line = aux_vars[0].tensor
    red_light = ref_line[..., -1]
    dt = 0.1

    velocity = torch.hypot(current_state[..., 3], current_state[..., 4])
    acc = control[..., 0]
    speed = velocity.unsqueeze(-1) + torch.cumsum(acc * dt, dim=-1)
    speed = torch.clamp(speed, min=0)
    s = torch.cumsum(speed * dt, dim=-1)
    stop_point = torch.max(red_light[:, 200:] == 0, dim=-1)[1] * 0.1
    stop_distance = stop_point.view(-1, 1, 1) - 3
    red_light_error = (s - stop_distance) * \
        (s > stop_distance) * (stop_point.unsqueeze(-1).unsqueeze(-1) != 0)
    return red_light_error


def _safety(optim_vars, aux_vars):
    control = optim_vars[0].tensor#.view(-1, 50, 2)
    neighbors = aux_vars[0].tensor.permute(0, 2, 1, 3)
    current_state = aux_vars[1].tensor
    ref_line = aux_vars[2].tensor
    actor_mask = torch.ne(current_state, 0)[:, 1:, -1]
    ego_current_state = current_state[:, 0:1]
    ego = bicycle_model(control, ego_current_state)
    ego_len, ego_width = ego_current_state[..., -3], ego_current_state[..., -2]
    neighbors_current_state = current_state[:, 1:]
    neighbors_len, neighbors_width = neighbors_current_state[..., -3], neighbors_current_state[..., -2]

    l_eps = (ego_width + neighbors_width)/2 + 0.5
    frenet_neighbors = torch.stack([
        project_to_frenet_frame(neighbors[:, :, i].detach(), ref_line) 
        for i in range(neighbors.shape[2])], dim=2)
    frenet_ego = torch.stack([
        project_to_frenet_frame(ego[:,i].detach(), ref_line) 
        for i in range(ego.shape[1])], dim=2)

    safe_error = []
    for t in range(ego.shape[2]):#[0, 2, 5, 9, 14, 19, 24, 29, 39, 49]:  # key frames
        # find objects of interest
        l_distance = torch.abs(frenet_ego[:, t, :, 1:] - frenet_neighbors[:, t, :, 1].unsqueeze(-2))
        s_distance = frenet_neighbors[:, t, :, 0].unsqueeze(-2) - frenet_ego[:, t, :, 0:1]
        interactive = torch.logical_and(
            s_distance > 0, l_distance < l_eps.unsqueeze(-2)) * actor_mask.unsqueeze(-2)

        # find closest object
        distances = torch.norm(ego[:, :, t:t+1, :2] - neighbors[:, t:t+1, :, :2], dim=-1)
        distances = torch.masked_fill(
            distances, torch.logical_not(interactive), 100)
        distance, index = torch.min(distances, dim=-1)
        s_eps = (ego_len + torch.gather(neighbors_len.unsqueeze(-2).repeat(1,index.shape[-1],1), 
                                        2, 
                                        index.unsqueeze(-1))
                 [..., 0])/2 + 5

        # calculate cost
        error = (s_eps - distance) * (distance < s_eps)
        safe_error.append(error)

    safe_error = torch.stack(safe_error, dim=-1)
    return safe_error

class TmpContainer():
    def __init__(self, tensor=None) -> None:
        self.tensor = tensor

def cost_function_sample(control_variables, current_state, predictions, ref_line):
    control_variables = TmpContainer(control_variables)
    current_state = TmpContainer(current_state)
    predictions = TmpContainer(predictions)
    ref_line = TmpContainer(ref_line)
    cost = {
        'speed':_speed([control_variables],[ref_line, current_state]).unsqueeze(-1),
        'acceleration':_acceleration([control_variables],[ref_line, current_state]).unsqueeze(-1),
        'jerk':_jerk([control_variables],[ref_line, current_state]).unsqueeze(-1),
        'steering':_steering([control_variables],[ref_line, current_state]).unsqueeze(-1),
        'steering_change':_steering_change([control_variables],[ref_line, current_state]).unsqueeze(-1),
        'lane_xy':_lane_xy([control_variables],[ref_line, current_state]).unsqueeze(-1),
        'lane_theta':_lane_theta([control_variables],[ref_line, current_state]).unsqueeze(-1),
        'red_light_violation':_red_light_violation([control_variables],[ref_line, current_state]).unsqueeze(-1),
        'safety':_safety([control_variables],[predictions, current_state, ref_line]).unsqueeze(-1),
        }
    return cost


def risk_cost_function_sample(control_variables, current_state, predictions, ref_line, vf_map):
    _control_variables = TmpContainer(control_variables['u'])
    _current_state = TmpContainer(current_state)
    _predictions = TmpContainer(predictions)
    _ref_line = TmpContainer(ref_line)
    measure = {
        # 'speed':_speed([_control_variables],[_ref_line, _current_state]).unsqueeze(-1),
        'acceleration':_acceleration([_control_variables],[_ref_line, _current_state]).unsqueeze(-1),
        'jerk':_jerk([_control_variables],[_ref_line, _current_state]).unsqueeze(-1),
        'steering':_steering([_control_variables],[_ref_line, _current_state]).unsqueeze(-1),
        'steering_change':_steering_change([_control_variables],[_ref_line, _current_state]).unsqueeze(-1),
        # 'lane_xy':_lane_xy([_control_variables],[_ref_line, _current_state]).unsqueeze(-1),
        # 'lane_theta':_lane_theta([_control_variables],[_ref_line, _current_state]).unsqueeze(-1),
        # 'safety':_safety([_control_variables],[_predictions, _current_state, _ref_line]).unsqueeze(-1),
        'vf_map':vf_map.vector_field_diff(control_variables['X'])
        }
    return measure