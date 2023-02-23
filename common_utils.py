'''
Author: rxin rxin@connect.ust.hk
Date: 2023-02-16 15:19:45
LastEditors: rxin rxin@connect.ust.hk
LastEditTime: 2023-02-22 14:32:41
FilePath: /DIPP/common_utils.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import torch
from model.predictor import Predictor, RiskMapPre
from model.planner import BasePlanner, MotionPlanner, RiskMapPlanner
from utils.train_utils import bicycle_model, select_future
predictor_selection = {'base': Predictor,
                       'dipp': Predictor,
                       'risk': RiskMapPre}


planner_selection = {'base': BasePlanner,
                     'dipp': MotionPlanner,
                     'risk': RiskMapPlanner}

def save_checkpoint(epoch, save_name, cfg, model):
    """ Save model to file. """
    model_dict = {'epoch': epoch+1,
                  'state_dict': model.state_dict(),
                  'model_cfg': cfg['model_cfg'],
                  }

    print("Saving model to {}".format(save_name))
    torch.save(model_dict, save_name)

def load_checkpoint(model_file, map_location):
    """ Load a model from a file. """
    print("Loading model from {}".format(model_file))
    model_dict = torch.load(model_file,map_location='cpu')
    ## save loaded model params
    print('>>>> Model config is:')
    for k,v in model_dict['model_cfg'].items():
        print('--',k,v)
    predictor = predictor_selection[f"{model_dict['model_cfg']['name']}"](**model_dict['model_cfg'])
    predictor.load_state_dict(state_dict=model_dict['state_dict'])
    epoch = model_dict['epoch']
    print('load succeed')
    return predictor, epoch

def inference(batch, predictor, planner, args, use_planning):
    ego = batch[0].to(args.device)
    neighbors = batch[1].to(args.device)
    map_lanes = batch[2].to(args.device)
    map_crosswalks = batch[3].to(args.device)
    ref_line_info = batch[4].to(args.device)
    current_state = torch.cat([ego.unsqueeze(1), neighbors[..., :-1]], dim=1)[:, :, -1]
    with torch.no_grad():
        plans, predictions, scores, cost_function_weights = predictor(ego, neighbors, map_lanes, map_crosswalks)
        plan_trajs = torch.stack([bicycle_model(plans[:, i], ego[:, -1])[:, :, :3] for i in range(3)], dim=1)

    if not use_planning:
        plan, prediction = select_future(plan_trajs, predictions, scores)
    elif planner.name=='dipp':
        plan, prediction = select_future(plans, predictions, scores)

        planner_inputs = {
            "control_variables": plan.view(-1, 100), # generate initial control sequence
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
    elif planner.name=='risk':
        u, prediction = select_future(plans, predictions, scores)
        plan = bicycle_model(u, ego[:, -1])
        planner_inputs = {
            "predictions": prediction, # prediction for surrounding vehicles 
            "ref_line_info": ref_line_info,
            "current_state": current_state,
            "vf_map": predictor.vf_map,
            'init_guess_u': u,
        }
        with torch.no_grad():
            plan, u = planner.plan(planner_inputs, batch) # control
    elif planner.name=='base':
        plan, prediction = select_future(plans, predictions, scores)
        plan = bicycle_model(plan, ego[:, -1])[:, :, :3]

    return plan, prediction