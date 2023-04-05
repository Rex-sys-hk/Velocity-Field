import torch
import argparse
import glob
import os
import sys
import logging
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from utils.riskmap.rm_utils import load_cfg_here
from utils.test_utils import *
from common_utils import *
from model.planner import MotionPlanner
from waymo_open_dataset.protos import scenario_pb2
os.environ["DIPP_ABS_PATH"] = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.getenv('DIPP_ABS_PATH'))

def open_loop_test():
    # logging
    log_path = f"./testing_log/{args.name}/"
    os.makedirs(log_path, exist_ok=True)
    initLogging(log_file=log_path+'test.log')

    logging.info("------------- {} -------------".format(args.name))
    logging.info("Use integrated planning module: {}".format(args.use_planning))
    logging.info("Use device: {}".format(args.device))

    # test file
    files = glob.glob(args.test_set+'/*')
    processor = TestDataProcess()

    # cache results
    collisions = []
    red_light, off_route = [], []
    Accs, Jerks, Lat_Accs = [], [], []
    Human_Accs, Human_Jerks, Human_Lat_Accs = [], [], []
    similarity_1s, similarity_3s, similarity_5s = [], [], []
    prediction_ADE, prediction_FDE = [], []

    # load model
    predictor, start_epoch, _ = load_checkpoint(args.model_path, map_location='cpu')
    predictor.to(args.device)
    logging.info(f'ckpt successful loaded from {args.model_path}')
    predictor.eval()

    # set up planner
    planner = init_planner(args, cfg, predictor)

    # iterate test files
    for i,file in enumerate(files[args.test_pkg_sid:args.test_pkg_num]):
        scenarios = tf.data.TFRecordDataset(file)
        logging.warning(f'>>>file [{i}/{len(files)}]:')
        # iterate scenarios in the test file
        length = [1 for s in scenarios]
        for ii, scenario in enumerate(scenarios):
            logging.info(f'\n >>>scenario {ii}/{len(length)}')
            parsed_data = scenario_pb2.Scenario()
            parsed_data.ParseFromString(scenario.numpy())

            scenario_id = parsed_data.scenario_id
            sdc_id = parsed_data.sdc_track_index
            timesteps = parsed_data.timestamps_seconds

            # build map
            processor.build_map(parsed_data.map_features, parsed_data.dynamic_map_states)

            # get a testing scenario
            for timestep in tqdm(range(20, len(timesteps)-50, 10), desc=f'>>>scenario {ii}/{len(length)}'):
                logging.info(f"Scenario: {scenario_id} Time: {timestep}")
                
                # prepare data
                # if not os.path.exists(f'{args.test_processed}/{scenario_id}_{timestep}.npz') or args.render:
                # try:
                input_data = processor.process_frame(timestep, sdc_id, parsed_data.tracks)
                # except:
                #     print(f'Scenario: {scenario_id} Time: {timestep} is not usable, skipped')
                #     continue
                ego = torch.from_numpy(input_data[0]).to(args.device)
                neighbors = torch.from_numpy(input_data[1]).to(args.device)
                lanes = torch.from_numpy(input_data[2]).to(args.device)
                crosswalks = torch.from_numpy(input_data[3]).to(args.device)
                ref_line = torch.from_numpy(input_data[4]).to(args.device)
                neighbor_ids, norm_gt_data, gt_data = input_data[5], input_data[6], input_data[7]
                current_state = torch.cat([ego.unsqueeze(1), neighbors[..., :-1]], dim=1)[:, :, -1]
                batch = []
                for i in range(5):
                    batch.append(torch.from_numpy(input_data[i]))
                # else:
                #     data = np.load(f'{args.test_processed}/{scenario_id}_{timestep}.npz',allow_pickle=True)
                #     logging.info(f'Read data from {args.test_processed}/{scenario_id}_{timestep}.npz')
                #     ego = torch.from_numpy(data['ego']).unsqueeze(0)
                #     neighbors = torch.from_numpy(data['neighbors']).unsqueeze(0)
                #     ref_line = torch.from_numpy(data['ref_line'] ).unsqueeze(0)
                #     lanes = torch.from_numpy(data['map_lanes']).unsqueeze(0)
                #     crosswalks = torch.from_numpy(data['map_crosswalks']).unsqueeze(0)
                #     current_state = torch.cat([ego.unsqueeze(1), neighbors[..., :-1]], dim=1)[:, :, -1]
                #     norm_gt_data = data['gt_future_states']
                #     batch = [ego,neighbors,lanes,crosswalks,ref_line]
                plan, prediction = inference(batch, predictor, planner, args, args.use_planning, parallel='none')
                plan = plan.cpu().numpy()[0]

                # compute metrics
                # logging.info(f"Results:")
                # collision = check_collision(plan, norm_gt_data[1:], current_state.cpu().numpy()[0, :, 5:])
                # collisions.append(collision)
                # traffic = check_traffic(plan, ref_line.cpu().numpy()[0])
                # red_light.append(traffic[0])
                # off_route.append(traffic[1])
                # logging.info(f"Collision: {collision}, Red light: {traffic[0]}, Off route: {traffic[1]}")

                # Acc, Jerk, Lat_Acc = check_dynamics(plan)
                # Accs.append(Acc)
                # Jerks.append(Jerk) 
                # Lat_Accs.append(Lat_Acc)
                # logging.info(f"Acceleration: {Acc}, Jerk: {Jerk}, Lateral_Acceleration: {Lat_Acc}")

                # Acc, Jerk, Lat_Acc = check_dynamics(norm_gt_data[0])
                # Human_Accs.append(Acc)
                # Human_Jerks.append(Jerk) 
                # Human_Lat_Accs.append(Lat_Acc)
                # logging.info(f"Human: Acceleration: {Acc}, Jerk: {Jerk}, Lateral_Acceleration: {Lat_Acc}")

                # similarity = check_similarity(plan, norm_gt_data[0])
                # similarity_1s.append(similarity[9])
                # similarity_3s.append(similarity[29])
                # similarity_5s.append(similarity[49])
                # logging.info(f"Similarity@1s: {similarity[9]}, Similarity@3s: {similarity[29]}, Similarity@5s: {similarity[49]}")

                # prediction_error = check_prediction(prediction[0].cpu().numpy(), norm_gt_data[1:])
                # prediction_ADE.append(prediction_error[0])
                # prediction_FDE.append(prediction_error[1])
                # logging.info(f"Prediction ADE: {prediction_error[0]}, FDE: {prediction_error[1]}")

                ### plot scenario ###
                if args.render:
                    # visualization
                    plt.ion()

                    # map
                    for vector in parsed_data.map_features:
                        vector_type = vector.WhichOneof("feature_data")
                        vector = getattr(vector, vector_type)
                        polyline = map_process(vector, vector_type)

                    # sdc
                    agent_color = ['r', 'm', 'b', 'g'] # [sdc, vehicle, pedestrian, cyclist]
                    color = agent_color[0]
                    track = parsed_data.tracks[sdc_id].states[timestep]
                    curr_state = (track.center_x, track.center_y, track.heading)
                    plan = transform(plan, curr_state, include_curr=True)

                    rect = plt.Rectangle((track.center_x-track.length/2, track.center_y-track.width/2), 
                                        track.length, track.width, linewidth=2, color=color, alpha=0.6, zorder=3,
                                        transform=mpl.transforms.Affine2D().rotate_around(*(track.center_x, track.center_y), track.heading) + plt.gca().transData)
                    plt.gca().add_patch(rect)
                    plt.plot(plan[::5, 0], plan[::5, 1], linewidth=2, color=color, marker='.', markersize=6, zorder=4)
                    ego_gt = np.insert(gt_data[0, :, :3], 0, curr_state, axis=0)
                    plt.plot(ego_gt[:, 0], ego_gt[:, 1], 'k--', linewidth=2, zorder=4)

                    # neighbors
                    for i, id in enumerate(neighbor_ids):
                        track = parsed_data.tracks[id].states[timestep]
                        color = agent_color[parsed_data.tracks[id].object_type]
                        rect = plt.Rectangle((track.center_x-track.length/2, track.center_y-track.width/2), 
                                            track.length, track.width, linewidth=2, color=color, alpha=0.6, zorder=3,
                                            transform=mpl.transforms.Affine2D().rotate_around(*(track.center_x, track.center_y), track.heading) + plt.gca().transData)
                        plt.gca().add_patch(rect)
                        predict_traj = prediction.cpu().numpy()[0, i]
                        predict_traj = transform(predict_traj, curr_state)
                        predict_traj = np.insert(predict_traj, 0, (track.center_x, track.center_y), axis=0)
                        plt.plot(predict_traj[::5, 0], predict_traj[::5, 1], linewidth=2, color=color, marker='.', markersize=6, zorder=3)
                        
                        other_gt = np.insert(gt_data[i+1, :, :3], 0, (track.center_x, track.center_y, track.heading), axis=0)
                        other_gt = other_gt[other_gt[:, 0] != 0]
                        plt.plot(other_gt[:, 0], other_gt[:, 1], 'k--', linewidth=2, zorder=3)         

                    for i, track in enumerate(parsed_data.tracks):
                        if i not in [sdc_id] + neighbor_ids and track.states[timestep].valid:
                            rect = plt.Rectangle((track.states[timestep].center_x-track.states[timestep].length/2, track.states[timestep].center_y-track.states[timestep].width/2), 
                                                track.states[timestep].length, track.states[timestep].width, linewidth=2, color='k', alpha=0.6, zorder=3,
                                                transform=mpl.transforms.Affine2D().rotate_around(*(track.states[timestep].center_x, track.states[timestep].center_y), track.states[timestep].heading) + plt.gca().transData)
                            plt.gca().add_patch(rect)

                    # dynamic_map_states
                    for signal in parsed_data.dynamic_map_states[timestep].lane_states:
                        traffic_signal_process(processor.lanes, signal)

                    # show plot
                    plt.gca().axis([-100 + plan[0, 0], 100 + plan[0, 0], -100 + plan[0, 1], 100 + plan[0, 1]])
                    plt.gca().set_facecolor('xkcd:grey')
                    plt.gca().margins(0)  
                    plt.gca().set_aspect('equal')
                    plt.gca().axes.get_yaxis().set_visible(False)
                    plt.gca().axes.get_xaxis().set_visible(False)
                    plt.tight_layout()

                    # save image
                    if args.save:
                        save_path = f"./testing_log/{args.name}/images"
                        os.makedirs(save_path, exist_ok=True)
                        plt.savefig(f'{save_path}/{scenario_id}_{timestep}.png')

                    # clear
                    plt.pause(0.1)
                    plt.clf()

    # save results
    # data = {'collision':collisions, 'red_light':red_light, 'off_route':off_route, 
    #                         'Acc':Accs, 'Jerk':Jerks, 'Lat_Acc':Lat_Accs, 
    #                         'Human_Acc':Human_Accs, 'Human_Jerk':Human_Jerks, 'Human_Lat_Acc':Human_Lat_Accs,
    #                         'Prediction_ADE':prediction_ADE, 'Prediction_FDE':prediction_FDE,
    #                         'Human_L2_1s':similarity_1s, 'Human_L2_3s':similarity_3s, 'Human_L2_5s':similarity_5s}
    # df = pd.DataFrame(data=data)
    # df.to_csv(f'./testing_log/{args.name}/testing_log.csv')
    # logging.info(f'file results saved in ./testing_log/{args.name}/testing_log.csv')
    # static_result(df, args.name)

if __name__ == "__main__":
    # Arguments
    parser = argparse.ArgumentParser(description='Training')
    parser.add_argument('--name', type=str, help='log name (default: "Test1")', default="Test1")
    parser.add_argument('--config', type=str, help='path to the config file', default= None)
    parser.add_argument('--test_set', type=str, help='path to testing datasets')
    parser.add_argument('--test_processed', type=str, help='path to processed testing datasets')
    parser.add_argument('--test_pkg_sid', type=int, help='start package counts', default=0)
    parser.add_argument('--test_pkg_num', type=int, help='test package counts', default=3)
    parser.add_argument('--model_path', type=str, help='path to saved model')
    parser.add_argument('--use_planning', action="store_true", help='if use integrated planning module (default: False)', default=False)
    parser.add_argument('--render', action="store_true", help='if render the scenario (default: False)', default=False)
    parser.add_argument('--save', action="store_true", help='if save the rendered images (default: False)', default=False)
    parser.add_argument('--device', type=str, help='run on which device (default: cuda)', default='cuda')
    args = parser.parse_args()
    cfg_file = args.config if args.config else 'config.yaml'
    os.environ["DIPP_CONFIG"] = str(os.getenv('DIPP_ABS_PATH') + '/' + cfg_file)
    cfg = load_cfg_here()
    logging.basicConfig()
    logging.getLogger().setLevel(logging.WARNING)
    # Run
    open_loop_test()
