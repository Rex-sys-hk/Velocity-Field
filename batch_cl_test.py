import torch
import argparse
import os
import sys
import logging
import pandas as pd
import tensorflow as tf
from statistic import static_result
from utils.riskmap.utils import load_cfg_here
from utils.simulator import *
from utils.test_utils import *
from common_utils import *
from model.planner import MotionPlanner
from waymo_open_dataset.protos import scenario_pb2
import ray
import math
os.environ["DIPP_ABS_PATH"] = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.getenv('DIPP_ABS_PATH'))

@ray.remote
def closed_loop_test(test_pkg_sid=0, test_pkg_eid=100, pid=0):
    # logging
    log_path = f"./testing_log/{args.name}/"
    os.makedirs(log_path, exist_ok=True)
    os.makedirs(os.path.join(log_path,'video'), exist_ok=True)
    initLogging(log_file=log_path+'test.log')
    
    logging.info("------------- {} -------------".format(args.name))
    logging.info("Use integrated planning module: {}".format(args.use_planning))
    logging.info("Use device: {}".format(args.device))

    # test file

    # set up simulator
    simulator = Simulator(150) # temporal horizon 15s    

    # load model
    predictor, start_epoch, _ = load_checkpoint(args.model_path, map_location='cpu')
    predictor.to(args.device)
    logging.info(f'ckpt successful loaded from {args.model_path}')
    predictor.eval()

    # cache results
    collisions, off_routes, progress = [], [], []
    Accs, Jerks, Lat_Accs = [], [], []
    Human_Accs, Human_Jerks, Human_Lat_Accs = [], [], []
    similarity_3s, similarity_5s, similarity_10s = [], [], []

    # set up planner
    if args.use_planning:
        if cfg['planner']['name'] == 'dipp':
            trajectory_len, feature_len = 50, 9
            planner = MotionPlanner(trajectory_len, feature_len, device= args.device)
        if cfg['planner']['name'] == 'risk':
            planner = RiskMapPlanner(predictor.meter2risk, device= args.device)
        if cfg['planner']['name'] == 'base':
            planner = BasePlanner(device= args.device)
        if cfg['planner']['name'] == 'esp':
            planner = EularSamplingPlanner(predictor.meter2risk, device= args.local_rank)
    else:
        planner = None
    for i,file in enumerate(files[test_pkg_sid: test_pkg_eid]):
        scenarios = tf.data.TFRecordDataset(file)
        logging.info(f'\n >>>[{i}/{len(files)}]:')
        # iterate scenarios in the test file
        length = [1 for s in scenarios]
        for ii, scenario in enumerate(scenarios):
            logging.info(f'\n >>>scenario {ii}/{len(length)}')
            parsed_data = scenario_pb2.Scenario()
            parsed_data.ParseFromString(scenario.numpy())
            simulator.load_scenario(parsed_data)
            logging.info(f'Scenario: {simulator.scenario_id}')

            obs = simulator.reset()
            done = False

            while not done:
                logging.info(f'Time: {simulator.timestep-19}')
                batch = []
                for i in range(len(obs)):
                    batch.append(torch.from_numpy(obs[i]))
                plan_traj,prediction = inference(batch, predictor, planner, args, args.use_planning)
                plan_traj = plan_traj.cpu().numpy()[0]
                prediction = prediction.cpu().numpy()[0]

                # take one step
                obs, done, info = simulator.step(plan_traj, prediction)
                logging.info(f'Collision: {info[0]}, Off-route: {info[1]}')

                # render
                if args.render:
                    simulator.render()

            # calculate metrics
            collisions.append(info[0])
            off_routes.append(info[1])
            progress.append(simulator.calculate_progress())

            dynamics = simulator.calculate_dynamics()
            acc = np.mean(np.abs(dynamics[0]))
            jerk = np.mean(np.abs(dynamics[1])) 
            lat_acc = np.mean(np.abs(dynamics[2]))
            Accs.append(acc)
            Jerks.append(jerk)
            Lat_Accs.append(lat_acc)

            error, human_dynamics = simulator.calculate_human_likeness()
            h_acc = np.mean(np.abs(human_dynamics[0]))
            h_jerk = np.mean(np.abs(human_dynamics[1])) 
            h_lat_acc = np.mean(np.abs(human_dynamics[2]))
            Human_Accs.append(h_acc)
            Human_Jerks.append(h_jerk)
            Human_Lat_Accs.append(h_lat_acc)

            similarity_3s.append(error[29])
            similarity_5s.append(error[49])
            similarity_10s.append(error[99])

            # save animation
            if args.save:
                simulator.save_animation(os.path.join(log_path,'video'))

    # save metircs
    data={'collision':collisions, 'off_route':off_routes, 'progress': progress,
                            'Acc':Accs, 'Jerk':Jerks, 'Lat_Acc':Lat_Accs, 
                            'Human_Acc':Human_Accs, 'Human_Jerk':Human_Jerks, 'Human_Lat_Acc':Human_Lat_Accs,
                            'Human_L2_3s':similarity_3s, 'Human_L2_5s':similarity_5s, 'Human_L2_10s':similarity_10s}
    df = pd.DataFrame(data=data)
    df.to_csv(f"./testing_log/{args.name}/testing_log_cl_{pid}.csv")
    # static_result(df, args.name, True)
    return data

def cellect_results(results):
    collisions, progress = [], []

    red_light, off_routes = [], []
    Accs, Jerks, Lat_Accs = [], [], []
    Human_Accs, Human_Jerks, Human_Lat_Accs = [], [], []
    similarity_3s, similarity_5s, similarity_10s = [], [], []
    # prediction_ADE, prediction_FDE = [], []
    data={'collision':collisions, 'off_route':off_routes, 'progress': progress,
        'Acc':Accs, 'Jerk':Jerks, 'Lat_Acc':Lat_Accs, 
        'Human_Acc':Human_Accs, 'Human_Jerk':Human_Jerks, 'Human_Lat_Acc':Human_Lat_Accs,
        'Human_L2_3s':similarity_3s, 'Human_L2_5s':similarity_5s, 'Human_L2_10s':similarity_10s}
    for d in results:
        for k in d.keys():
            data[k].extend(d[k])
    return data

if __name__ == "__main__":
    # Arguments
    parser = argparse.ArgumentParser(description='Closed-loop Test')
    parser.add_argument('--name', type=str, help='log name (default: "Test1")', default="Test1")
    parser.add_argument('--config', type=str, help='path to the config file', default= None)
    # parser.add_argument('--batch_size', type=int, help='test package counts', default=3)
    parser.add_argument('--num_workers', type=int, help='test process counts', default=48)
    parser.add_argument('--test_set', type=str, help='path to testing datasets')
    parser.add_argument('--test_pkg_sid', type=int, help='start package counts', default=0)
    parser.add_argument('--test_pkg_num', type=int, help='test package counts', default=3)
    parser.add_argument('--model_path', type=str, help='path to saved model')
    parser.add_argument('--use_planning', action="store_true", help='if use integrated planning module (default: False)', default=False)
    parser.add_argument('--render', action="store_true", help='if render the scene (default: False)', default=False)
    parser.add_argument('--save', action="store_true", help='if save animation (default: False)', default=False)
    parser.add_argument('--device', type=str, help='run on which device (default: cuda)', default='cuda')
    args = parser.parse_args()
    cfg_file = args.config if args.config else 'config.yaml'
    os.environ["DIPP_CONFIG"] = str(os.getenv('DIPP_ABS_PATH') + '/' + cfg_file)
    cfg = load_cfg_here()
    # Run
    files = glob.glob(args.test_set+'/*')[args.test_pkg_sid:args.test_pkg_sid+args.test_pkg_num]
    num_files = len(files)
    num_cpus = args.num_workers
    workload = math.ceil(num_files/num_cpus)
    ray.init(num_cpus=num_cpus)
    ids = [closed_loop_test.remote(i*workload, min((i+1)*workload, num_files), i) for i in range(num_cpus)]
    logging.info('process ids', ids)
    results = ray.get(ids)

    all_data = cellect_results(results)
    all_df = pd.DataFrame(data=all_data)

    static_result(all_df, args.name, True)    
    ray.shutdown()