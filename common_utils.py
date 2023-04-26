'''
Author: rxin rxin@connect.ust.hk
Date: 2023-02-16 15:19:45
LastEditors: rxin rxin@connect.ust.hk
LastEditTime: 2023-02-22 14:32:41
FilePath: /DIPP/common_utils.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import torch
import numpy as np
import multiprocessing
from model.predictor import BasePre, CostVolume, EularPre, Predictor, RiskMapPre
from model.predictor_base import STCostMap
from model.planner import BasePlanner, CostMapPlanner, EularSamplingPlanner, MotionPlanner, RiskMapPlanner
from utils.riskmap.car import bicycle_model, pi_2_pi, traj_smooth, TrajSmooth, traj_smooth_mp
from utils.riskmap.rm_utils import get_u_from_X, load_cfg_here
from utils.train_utils import select_future
predictor_selection = {'base': BasePre,
                       'dipp': Predictor,
                       'risk': RiskMapPre,
                       'esp': EularPre,
                       'nmp': CostVolume,
                       }


planner_selection = {'base': BasePlanner,
                     'dipp': MotionPlanner,
                     'risk': RiskMapPlanner,
                     'esp': EularSamplingPlanner,
                     'nmp': CostMapPlanner,
                     }

def save_checkpoint(epoch, save_name, cfg, model, lr=1e-4, dist=False):
    """ Save model to file. """
    if dist:
        model_dict = {'epoch': epoch+1,
                    'state_dict': model.module.state_dict(),
                    'model_cfg': cfg['model_cfg'],
                    'lr': lr,
                    }
    else:
        model_dict = {'epoch': epoch+1,
                    'state_dict': model.state_dict(),
                    'model_cfg': cfg['model_cfg'],
                    'lr': lr,
                    }        

    print("Saving model to {}".format(save_name))
    torch.save(model_dict, save_name)

def load_checkpoint(model_file, map_location = 'cpu'):
    """ Load a model from a file. """
    print("Loading model from {}".format(model_file))
    model_dict = torch.load(model_file,map_location=map_location)
    ## save loaded model params
    print('>>>> Model config is:')
    for k,v in model_dict['model_cfg'].items():
        print('--',k,v)
    predictor = predictor_selection[f"{model_dict['model_cfg']['name']}"](**model_dict['model_cfg'])
    predictor.load_state_dict(state_dict=model_dict['state_dict'])
    epoch = model_dict['epoch']
    lr = model_dict['lr']  if 'lr' in model_dict.keys() else 1e-4
    print('load succeed')
    return predictor, epoch, lr

def init_planner(args, cfg, predictor):
    if args.use_planning:
        if cfg['planner']['name'] == 'dipp':
            trajectory_len, feature_len = 50, 9
            planner = MotionPlanner(trajectory_len, feature_len, device= args.device)
        if cfg['planner']['name'] == 'risk':
            planner = RiskMapPlanner(predictor.meter2risk, device= args.device)
        if cfg['planner']['name'] == 'base':
            planner = BasePlanner(device= args.device)
        if cfg['planner']['name'] == 'esp':
            planner = EularSamplingPlanner(predictor.meter2risk, device= args.device)
        if cfg['planner']['name'] == 'nmp':
            planner = CostMapPlanner(predictor.meter2risk, device=args.device)
    else:
        planner = None
    return planner

def inference(batch, predictor, planner, args, use_planning, distributed=False, parallel='none'):
    try:
        args.device=args.local_rank if args.local_rank else args.device
    except:
        pass
    ego = batch[0].to(args.device)
    neighbors = batch[1].to(args.device)
    map_lanes = batch[2].to(args.device)
    map_crosswalks = batch[3].to(args.device)
    ref_line_info = batch[4].to(args.device)
    lattice_sample = batch[6].to(args.device) if batch[6].sum() else None
    current_state = torch.cat([ego.unsqueeze(1), neighbors[..., :-1]], dim=1)[:, :, -1]
    with torch.no_grad():
        us, predictions, scores, cost_function_weights = predictor(ego, neighbors, map_lanes, map_crosswalks)
        plan_trajs = torch.stack([bicycle_model(us[:, i], ego[:, -1])[:, :, :3] for i in range(scores.shape[1])], dim=1)
        u, prediction = select_future(us, predictions, scores)
        init_guess, prediction = select_future(plan_trajs, predictions, scores)
    if not use_planning:
        plan, prediction = select_future(plan_trajs, predictions, scores)

    elif planner.name=='base':
        u, prediction = select_future(us, predictions, scores)
        plan = init_guess

    elif planner.name=='dipp':
        u, prediction = select_future(us, predictions, scores)

        planner_inputs = {
            "control_variables": u.view(-1, 100), # generate initial control sequence
            "predictions": prediction, # generate predictions for surrounding vehicles 
            "ref_line_info": ref_line_info,
            "current_state": current_state
        }

        for i in range(cost_function_weights.shape[1]):
            planner_inputs[f'cost_function_weight_{i+1}'] = cost_function_weights[:, i].unsqueeze(1)

        with torch.no_grad():
            final_values, info = planner.layer.forward(planner_inputs)

            plan = final_values["control_variables"].view(-1, 50, 2)
            plan = bicycle_model(plan, ego[:, -1])[:, :, :3]

    elif planner.name=='esp':
        planner:EularSamplingPlanner=planner
        planner_inputs = {
            "predictions": prediction.detach(), # prediction for surrounding vehicles 
            "ref_line_info": ref_line_info,
            "current_state": current_state, # including neighbor cars
            'init_guess_u': u,
            'lattice_sample': lattice_sample,
        }
        plan,u = planner.plan(planner_inputs, genetic=planner.cfg['inference_genetic'])
    
    elif planner.name=='risk':
        # u, prediction = select_future(us, predictions, scores)
        planner:RiskMapPlanner=planner
        vf_map = predictor.module.vf_map if distributed else predictor.vf_map
        planner_inputs = {
            "predictions": prediction, # prediction for surrounding vehicles 
            "ref_line_info": ref_line_info,
            "current_state": current_state,
            "vf_map": vf_map,
            'init_guess_u': u,
            'lattice_sample': lattice_sample,
        }
        with torch.no_grad():
            plan, u = planner.plan(planner_inputs, genetic=planner.cfg['inference_genetic']) # control
        # plan = bicycle_model(u, ego[:, -1])[:, :, :3]
    elif planner.name=='nmp':
        planner:CostMapPlanner = planner
        cost_map:STCostMap = predictor.module.cost_volume if distributed else predictor.cost_volume
        u, prediction = select_future(us, predictions, scores)
        # guess loss
        planner_inputs = {
            "predictions": prediction, # prediction for surrounding vehicles 
            "ref_line_info": ref_line_info,
            "current_state": current_state,
            "cost_map": cost_map,
            'init_guess_u': u.detach().clone(),
            'lattice_sample': lattice_sample,
        }
        # plan loss
        with torch.no_grad():
            plan, _ = planner.plan(planner_inputs, genetic=planner.cfg['inference_genetic']) # control
    ## smoothing
    if parallel=='none':
        return plan, prediction
    plan = plan.cpu().numpy()
    c_state = current_state.cpu().numpy()
    threads = []
    if parallel=='mp' and plan.shape[0]>1:
        ## multiprocess 114*80 5:23
        manager = multiprocessing.Manager()
        return_dict = manager.dict()
        for i in range(plan.shape[0]):
            p = multiprocessing.Process(target=traj_smooth_mp,args = (plan[i], c_state[i,0:1], return_dict, i))
            threads.append(p)
            p.start()
        for i in range(plan.shape[0]):
            threads[i].join()
            plan[i] = return_dict[i]
    if parallel=='mt' and plan.shape[0]>1:
        ## multithreads 144*80 6:55
        for i in range(plan.shape[0]):
            threads.append(TrajSmooth(plan[i], c_state[i,0:1]))
            threads[i].start()
        for i in range(plan.shape[0]):
            threads[i].join()
            plan[i] = threads[i].fut_traj
    if parallel=='single' or plan.shape[0]<=1:
        ## single threads 144*80 6:46
        for i in range(plan.shape[0]):
            plan[i] = traj_smooth(plan[i], c_state[i,0:1])
    plan = torch.tensor(plan,device=prediction.device)
    return plan, prediction

def cellect_results(results):
    collisions, progress = [], []
    # zero = []
    traffic_light, off_routes = [], []
    Accs, Jerks, Lat_Accs = [], [], []
    Human_Accs, Human_Jerks, Human_Lat_Accs = [], [], []
    similarity_3s, similarity_5s, similarity_10s = [], [], []
    # prediction_ADE, prediction_FDE = [], []
    data={
          'collision':collisions, 'off_route':off_routes, 'traffic_light':traffic_light, 
          'progress': progress,
        'Acc':Accs, 'Jerk':Jerks, 'Lat_Acc':Lat_Accs, 
        'Human_Acc':Human_Accs, 'Human_Jerk':Human_Jerks, 'Human_Lat_Acc':Human_Lat_Accs,
        'Human_L2_3s':similarity_3s, 'Human_L2_5s':similarity_5s, 'Human_L2_10s':similarity_10s}
    for d in results:
        for k in d.keys():
            if k!= 'Unnamed: 0':
                data[k].extend(np.nan_to_num(d[k],nan=0))
    return data
