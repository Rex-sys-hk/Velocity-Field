from random import sample
import sys
import os
import torch
try:
    import theseus as th
except:
    # print('[WARNING] theseus is not implemented')
    pass
from utils.riskmap.getloss import GetLoss
from utils.train_utils import bicycle_model, project_to_frenet_frame
from utils.riskmap.torch_lattice import torchLatticePlanner
from utils.riskmap.map import Map
from model.meter2risk import Meter2Risk
from utils.riskmap.utils import get_u_from_X, load_cfg_here
import matplotlib.pyplot as plt
sys.path.append(os.getenv('DIPP_ABS_PATH'))

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


class RiskMapPlanner(Planner):
    def __init__(self, meter2risk: Meter2Risk, device, test=False) -> None:
        super(RiskMapPlanner, self).__init__(device, test)
        self.name = 'risk'
        # self.lattice_planner = torchLatticePlanner(device, test=test)
        self.map = Map(device, test=False)
        self.hand_prefer = torch.softmax(
            torch.tensor(self.cfg['risk_preference'], device=device), dim=0
        )  # handcratfed preference
        self.meter2risk = meter2risk
        self.loss_calculator = GetLoss(meter2risk,self.map,device)

    def get_sample(self, context, gt_u = None, cov = 0.2):
        btsz = context['init_guess_u'].shape[0]
        init_guess_u = context['init_guess_u'] if gt_u==None else gt_u
        u = (torch.randn([btsz,40,50,2],device = init_guess_u.device)*cov+1.)*init_guess_u.unsqueeze(1)
        cur_state = context['current_state'][:,0:1]
        X = bicycle_model(u,cur_state)
        return {'X':X,'u':u}

    def plan(self, context):
        self.context = context
        self.sample_plan = self.get_sample(context)
        self.meter = self.map.get_meter(self.sample_plan, context)
        self.risks = self.meter2risk(self.meter)
        self.plan_result = self.selector(self.risks, self.sample_plan)
        return self.plan_result

    def selector(self, risks, sample_plan):
        costs = risks*self.hand_prefer[:risks.shape[-1]]
        i = torch.argmin(costs.mean(dim=[-1,-2]),dim=1)
        X,u = [],[]
        sampleX = sample_plan['X']
        sampleu = sample_plan['u']
        for ii,m in enumerate(i.cpu()):
            X.append(sampleX[ii,m])
            u.append(sampleu[ii,m])
        X = torch.stack(X,dim=0)
        u = torch.stack(u,dim=0)
        return X,u#torch.gather(sample_plan['X'],1,i),torch.gather(sample_plan['u'],1,i[...,:2])

    def get_loss(self, gt, tb_iter=0, tb_writer=None):
        """
        must be called after forward
        """
        # detailed_gt, u_gt = convert2detail_state(gt)
        u = get_u_from_X(gt,self.map.ego_current_state)
        self.gt_sample = self.get_sample(self.context,u.squeeze(1))
        raw_meter = self.map.get_vec_map_meter(self.gt_sample)
        gt_risk = self.meter2risk(raw_meter)
        
        return self.loss_calculator.get_loss(self.gt_sample,
                                             self.risks,
                                             self.plan_result,
                                             gt,
                                             gt,
                                             gt_risk,
                                             tb_iters=tb_iter,
                                             tb_writer=tb_writer
                                             )


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
