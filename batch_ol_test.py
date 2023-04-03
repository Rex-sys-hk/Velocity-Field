import logging
import os, sys
import argparse
import pandas as pd
import numpy as np
import torch
import tensorflow as tf
from tqdm import tqdm
from torch.utils.data import DataLoader, SequentialSampler
from common_utils import inference, init_planner, load_checkpoint
from model.planner import Planner
from statistic import static_result
from utils.riskmap.rm_utils import load_cfg_here
from utils.test_utils import batch_check_collision, batch_check_dynamics, batch_check_prediction, batch_check_similarity, batch_check_traffic
from utils.data_loading import DrivingData


def batch_op_test(data_loader, predictor, planner: Planner, use_planning, epoch, distributed=False):
    collisions = []
    red_light, off_route = [], []
    Accs, Jerks, Lat_Accs = [], [], []
    Human_Accs, Human_Jerks, Human_Lat_Accs = [], [], []
    similarity_1s, similarity_3s, similarity_5s = [], [], []
    prediction_ADE, prediction_FDE = [], []

    predictor.eval()
    for batch in tqdm(data_loader):
        # prepare data
        ego = batch[0]
        neighbors = batch[1]
        # map_lanes = batch[2]
        # map_crosswalks = batch[3]
        ref_line = batch[4]
        norm_gt_data = batch[5]
        current_state = torch.cat([ego.unsqueeze(1), neighbors[..., :-1]], dim=1)[:, :, -1]
        # inference
        if not args.gt_replay:
            plan, prediction = inference(batch, predictor, planner, args, use_planning, distributed=distributed)
            plan = plan.cpu()
        else:
            plan = norm_gt_data[:,0]
            prediction = norm_gt_data[:,1:]
        # ground_truth = batch[5].to(args.device)
        # masks = torch.ne(ground_truth[:, 1:, :, :3], 0)
        # # compute metrics
        # metrics = motion_metrics(plan, prediction, ground_truth, masks)
        # epoch_metrics.append(metrics)
        # show progress
        # epoch_metrics = np.array(epoch_metrics)
        # plannerADE, plannerFDE = np.mean(epoch_metrics[:, 0]), np.mean(epoch_metrics[:, 1])
        # predictorADE, predictorFDE = np.mean(epoch_metrics[:, 2]), np.mean(epoch_metrics[:, 3])
        # epoch_metrics = [plannerADE, plannerFDE, predictorADE, predictorFDE]
        # logging.info(f'\n==val-plannerADE: {plannerADE:.4f}, val-plannerFDE: {plannerFDE:.4f}, val-predictorADE: {predictorADE:.4f}, val-predictorFDE: {predictorFDE:.4f}')
        # compute metrics
        # logging.info(f"Results:")
        # TODO should be 0.01583
        collision = batch_check_collision(plan, norm_gt_data[:,1:], current_state[:, :, 5:])
        collisions.extend(collision.cpu())

        traffic = batch_check_traffic(plan, ref_line)
        red_light.extend(traffic[0].cpu())
        off_route.extend(traffic[1].cpu())
        # logging.info(f"Collision: {collision}, Red light: {traffic[0]}, Off route: {traffic[1]}")

        Acc, Jerk, Lat_Acc = batch_check_dynamics(plan)
        Accs.extend(Acc.cpu())
        Jerks.extend(Jerk.cpu()) 
        Lat_Accs.extend(Lat_Acc.cpu())
        # logging.info(f"Acceleration: {Acc}, Jerk: {Jerk}, Lateral_Acceleration: {Lat_Acc}")

        Acc, Jerk, Lat_Acc = batch_check_dynamics(norm_gt_data[:,0])
        Human_Accs.extend(Acc.cpu())
        Human_Jerks.extend(Jerk.cpu()) 
        Human_Lat_Accs.extend(Lat_Acc.cpu())
        # logging.info(f"Human: Acceleration: {Acc}, Jerk: {Jerk}, Lateral_Acceleration: {Lat_Acc}")

        similarity = batch_check_similarity(plan, norm_gt_data[:,0])
        similarity_1s.extend(similarity[:, 9].cpu())
        similarity_3s.extend(similarity[:, 29].cpu())
        similarity_5s.extend(similarity[:, 49].cpu())
        # logging.info(f"Similarity@1s: {similarity[:,9].mean().cpu()}, Similarity@3s: {similarity[:,29].mean().cpu()}, Similarity@5s: {similarity[:,49].mean().cpu()}")

        prediction_error = batch_check_prediction(prediction[...,:3], norm_gt_data[:,1:])
        prediction_ADE.extend(prediction_error[0].cpu())
        prediction_FDE.extend(prediction_error[1].cpu())
        # logging.info(f"Prediction ADE: {prediction_error[0]}, FDE: {prediction_error[1]}")


    # save results
    data = {'collision':collisions, 'red_light':red_light, 'off_route':off_route, 
                            'Acc':Accs, 'Jerk':Jerks, 'Lat_Acc':Lat_Accs, 
                            'Human_Acc':Human_Accs, 'Human_Jerk':Human_Jerks, 'Human_Lat_Acc':Human_Lat_Accs,
                            'Prediction_ADE':prediction_ADE, 'Prediction_FDE':prediction_FDE,
                            'Human_L2_1s':similarity_1s, 'Human_L2_3s':similarity_3s, 'Human_L2_5s':similarity_5s}
    df = pd.DataFrame(data=data)
    df.to_csv(f'./testing_log/{args.name}/testing_log.csv')
    logging.info(f'file results saved in ./testing_log/{args.name}/testing_log.csv')
    static_result(df, args.name)

def op_test():
    
    test_set = DrivingData(args.test_processed+'/*')
    test_sampler = SequentialSampler(test_set)
    valid_loader = DataLoader(test_set, batch_size=args.batch_size, num_workers=args.num_workers,sampler=test_sampler)
    
    logging.info(f"Initializing Model: {cfg['model_cfg']['name']}")
    for key in cfg['model_cfg']:
        logging.info(f"--{key}: {cfg['model_cfg'][key]}")
    # map_location = {'cuda:%d' % 0: 'cuda:%d' % args.device}
    predictor, _, _ = load_checkpoint(args.ckpt)
    predictor = predictor.to(args.device)
    logging.info(f'ckpt successful loaded from {args.ckpt}')

    # %% initializing planner
    logging.info(f"Initialize planner {cfg['planner']['name']}")
    planner = init_planner(args, cfg, predictor)


    batch_op_test(valid_loader,predictor.eval(),planner,args.use_planning,epoch=0,distributed=False)


if __name__ == "__main__":
    # Arguments
    parser = argparse.ArgumentParser(description='Training')
    parser.add_argument('--name', type=str, help='log name (default: "Test1")', default="Test1")
    parser.add_argument('--config', type=str, help='path to the config file', default= None)
    # parser.add_argument('--test_set', type=str, help='path to testing datasets')
    parser.add_argument('--test_processed', type=str, help='path to processed testing datasets')
    parser.add_argument('--test_pkg_sid', type=int, help='start package counts', default=0)
    parser.add_argument('--test_pkg_num', type=int, help='test package counts', default=3)
    parser.add_argument('--batch_size', type=int, help='test package counts', default=3)
    parser.add_argument('--num_workers', type=int, help='test process counts', default=48)
    parser.add_argument('--ckpt', type=str, help='path to saved model')
    parser.add_argument('--use_planning', action="store_true", help='if use integrated planning module (default: False)', default=False)
    parser.add_argument('--render', action="store_true", help='if render the scenario (default: False)', default=False)
    parser.add_argument('--save', action="store_true", help='if save the rendered images (default: False)', default=False)
    parser.add_argument('--gt_replay', action="store_true", help='if replay ground truth (default: False)', default=False)
    parser.add_argument('--device', type=str, help='run on which device (default: cuda)', default='cuda')
    args = parser.parse_args()
    os.environ["DIPP_ABS_PATH"] = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(os.getenv('DIPP_ABS_PATH'))
    cfg_file = args.config if args.config else 'config.yaml'
    os.environ["DIPP_CONFIG"] = str(os.getenv('DIPP_ABS_PATH') + '/' + cfg_file)
    cfg = load_cfg_here()
    logging.basicConfig()
    logging.getLogger().setLevel(logging.WARNING)
    
    log_dir=os.path.join(f'testing_log/{args.name}')
    os.makedirs(log_dir,exist_ok=True)

    op_test()
