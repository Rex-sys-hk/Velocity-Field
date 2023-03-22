import torch
import numpy as np
import matplotlib.pyplot as plt
import time
from .rm_utils import load_cfg_here
try:
    from .mpp.lib.frenet import FrenetSampler
except:
    print('[WARNING] FrenetSampler is not implemented')
    pass
from .car import (move,
                  steering_to_yawrate,
                  rad_2_degree,
                  degree_2_rad,
                  check_car_collision_dis,
                  plot_car,
                  plot_arrow,
                  WB,  # wheel base
                  W,
                  pi_2_pi,
                  pi_2_pi_pos,
                  degree_2_rad,
                  torch_yawrate_to_steering,
                  get_arc_length,
                  reduce_way_point
                  )
MAX_ACC = 5
MAX_STEER = 30

show_animation = False



class torchLatticePlanner:
    def __init__(self, device = 'cuda:0', test = False) -> None:
        cfg = load_cfg_here()['planner']
        self.cfg = cfg
        self.device = device
        self.ti = cfg['time_interval']
        self.th = cfg['time_horizon']
        self.T = self.th*self.ti
        self.ts = np.linspace(self.ti, self.T, self.th)
        # if not test:
        #     self.c_sampler = FrenetSampler(os.getenv('DIPP_ABS_PATH')+'/'+cfg['train_sampler_config'])
        # else:
        #     self.c_sampler = FrenetSampler(os.getenv('DIPP_ABS_PATH')+'/'+cfg['test_sampler_config'])
        # self.c_sampler.get_max_num_of_samples()
        self.sampled_traj = None

        self.strict_filter = cfg['strict_filter']['enable'] if 'strict_filter' in cfg.keys() else False
        if self.strict_filter:
            self.v_lim = cfg['strict_filter']['v_lim']
            self.a_lim = cfg['strict_filter']['a_lim']
        self.making_sample = cfg['making_sample']['enable'] if 'making_sample' in cfg.keys() else False
        if self.making_sample:
            self.make_range_x = cfg['making_sample']['make_range_x']
            self.make_range_y = cfg['making_sample']['make_range_y']
            self.make_num_x = cfg['making_sample']['make_num_x']
            self.make_num_y = cfg['making_sample']['make_num_y']

    def get_sample(self, context):
        # self.set_init_state(context['current_state'],context['ref_line_info'])
        # X,u = self.make_sample()
        btsz = context['init_guess_u'].shape[0]
        init_guess_u = context['init_guess_u']
        u = (torch.randn([btsz,400,50,2],device = init_guess_u.device)*0.2+1)*init_guess_u.unsqueeze(1)
        # u = torch.cat([torch.rand([btsz,400,50,1],device=context['current_state'].device)*10-5, 
        #                torch.rand([btsz,400,50,1],device=context['current_state'].device)*1.2-0.6],
        #                dim=-1)
        cur_state = context['current_state'][:,0:1]
        X = bicycle_model(u,cur_state)
        return {'X':X,'u':u}

    def set_init_state(self, s0, reflanes):
        self.init_X_state = s0 # the current state including neighbor vehicles
        # print('initial state:', s0)
        # self.c_sampler.set_init_state(s0[0],s0[1],s0[2],s0[3],s0[4])
        self.reflanes = reflanes
        # self.lane_num = reflanes.shape[0] if len(reflanes.shape)==3 else 1

    def new_frenet(self,waypoint):
        # extend way points
        # ext_p_num = 10
        # wpext_x = torch.linspace(-ext_p_num,0,ext_p_num,device=self.device)
        # wpext_y =torch.zeros(wpext_x.shape[0],device=self.device)
        # comp = torch.stack([waypoint[0:ext_p_num,1],wpext_x],axis=0)
        # waypoint[0:ext_p_num,0] = torch.min(comp,axis=0).values

        w = reduce_way_point(waypoint[...,:2].cpu().detach().numpy())
        wx = w[...,0] #[0.0, 10.0, 20.5, 35.0, 70.5]
        wy = w[...,1] #[0.0, -6.0, 5.0, 6.5, 0.0]
        ss = get_arc_length(w[...,:2])
        # ss = np.stack(ss,axis=0)
        self.c_sampler.set_reference_line(wx,wy,ss)

        # self.tx, self.ty, tyaw, tc, self.csp = generate_target_course(wx, wy)
        # if show_animation:
        #     print("init state in new_frenet:",self.c_sampler.get_init_state())

        #     plt.scatter(wx,wy)
        #     ref = self.c_sampler.get_ref_line()

        #     plt.scatter(ref[...,0],ref[...,1])
        #     plt.axis('equal')
        #     plt.show()



    def get_valid_path_sample(self, training = True):
        st = time.time()
        self.c_sampler.sample(self.T)
        self.c_sampler.sample_global_state(self.ts)
        trajs = self.c_sampler.get_trajectories(exclude_invalid=True)

        if show_animation:
            plt.clf()
            for traj in trajs:
                plt.plot(traj[...,0],traj[...,1])
            # plt.plot(fp.x,fp.y)
            plt.axis('equal')
            plt.show()
        """
        # template <typename T>
        # Eigen::MatrixXd FrenetTrajectory<T>::ToNumpy() {
        # const double init_s = traj_[0].s[0];
        # Eigen::MatrixXd numpy(traj_.size(), kTrajFeatureDim);

        # for (int i = 0; i < traj_.size(); ++i) {
        #     numpy(i, 0) = traj_[i].pos[0];
        #     numpy(i, 1) = traj_[i].pos[1];
        #     numpy(i, 2) = std::cos(traj_[i].yaw);
        #     numpy(i, 3) = std::sin(traj_[i].yaw);
        #     numpy(i, 4) = traj_[i].s[0] - init_s;
        #     numpy(i, 5) = traj_[i].s[1];
        #     numpy(i, 6) = traj_[i].s[2];
        #     numpy(i, 7) = traj_[i].d[0];
        #     numpy(i, 8) = traj_[i].d[1];
        #     numpy(i, 9) = traj_[i].d[2];
        #     numpy(i, 10) = traj_[i].vel;
        #     numpy(i, 11) = traj_[i].acc;
        # }
        # return numpy;
        # }
        """
        # transform traj to X and u
        st = time.time()
        ## recording traj and X
        trajs = np.stack(trajs,axis=0)
        trajs = torch.from_numpy(trajs).to(self.device)
        self.sampled_traj = trajs
        # [x,y,yaw,v,acc]
        fut_traj = [trajs[...,0:2], torch.atan2(trajs[...,3:4],trajs[...,2:3])]
        fut_traj = torch.cat(fut_traj,dim=-1)
        X,u = convert2detail_state(fut_traj=fut_traj,ti = self.ti,device = self.device)
        
        # # get u from X
        # iniX = self.init_X_state.unsqueeze(0).unsqueeze(0).repeat(X.shape[0],1,1)
        # X2u = torch.cat([iniX,X],dim=1)
        # dX2u = torch.diff(X2u,n=1,dim=1)/self.ti
        # a = trajs[...,11:12].nan_to_num(0)
        # steer = torch_yawrate_to_steering(dX2u[...,2:3],X[...,3:4])
        # u = torch.cat([a,steer],dim=-1)
        if self.strict_filter:
            indnn = torch.min(torch.logical_not(torch.isnan(X)),dim=-1).values
            indv = X[...,3]<=self.v_lim
            inda = torch.logical_and((u[...,0]<=self.a_lim), (u[...,0]>=-self.a_lim))
            ind = torch.logical_and(indv, inda)
            ind = torch.logical_and(ind, indnn)
            ind = torch.min(ind.view(X.shape[0],-1),dim=-1).values
            X = X[ind]
            u = u[ind]

        if show_animation:
            for xi in X:
                plt.plot(xi[...,0].cpu().detach(),xi[...,1].cpu().detach())
                # plt.plot(trajs[...,0].cpu().detach(),trajs[...,1].cpu().detach())
            plt.show()
        return X, u

    def make_sample(self):
        btsz = self.init_X_state.shape[0]
        x = torch.linspace(self.ti,self.th*self.ti,self.th,device=self.device).unsqueeze(-1) \
            *torch.norm(self.init_X_state[:,0,3:5],dim=-1,keepdim=True).unsqueeze(-1)
        x = x.unsqueeze(-3).repeat(1,self.make_num_x,1,1)
        dx = torch.linspace(0, (self.th-1)*self.ti, self.th, device=self.device).unsqueeze(-1) # dm/s
        dx = dx.repeat(btsz,self.make_num_x,1,1)*torch.linspace(0,
                                                            self.make_range_x,
                                                            self.make_num_x,
                                                            device=self.device).reshape(self.make_num_x,1,1)
        x = x+dx
        # x_no0 = x.view(-1,self.th)[...,-1]>=0
        # x = x[x_no0]
        # xn = x_no0.sum()
        y = torch.zeros(btsz,self.make_num_y,self.th,1,device=self.device)
        dy = torch.linspace(0.1,self.make_range_y ,self.th,device=self.device).unsqueeze(-1)**2
        dy = dy.repeat(btsz,self.make_num_y,1,1)*torch.linspace(-1,1,self.make_num_y,device=self.device).reshape(self.make_num_y,1,1)
        y = y+dy
        x = x.repeat(1,self.make_num_y,1,1)
        y = y.unsqueeze(2).repeat(1,1,self.make_num_x,1,1).reshape(btsz,-1,self.th,1)
        gent = torch.cat([x,y],dim=-1)
        # gent = gent[gent[...,-1,0]>=torch.abs(gent[...,-1,1])].reshape()
        # print(gent.shape)
        # yaw = torch.atan2(x,y)
        X,u = convert2detail_state(fut_traj=gent, ti = self.ti, from_init_guess=True)

        return X,u

    def sample_X_by_u(self):
        traj = 0
        control = 0
        # control = 
        # traj = bicycle_model(control, current_state)
        return traj, control
    
class LatticeSampler():
    def __init__(self) -> None:
        self.T = 5.0
        self.Ts = np.linspace(0.1,5.0,50)
        self.c_sampler = FrenetSampler('/mnt/c/Users/87649/Desktop/working_space/DIPP/utils/riskmap/mpp/frenet_sampler.train.json')
        self.ref_length_lim = 80
        
    def sampling(self,
                 ref_info: np.ndarray,
                 x: float=0.,y: float=0.,yaw: float=0.,v: float=0.,acc: float=0.,
                 ref_check = True
                 ):
        """
        # template <typename T>
        # Eigen::MatrixXd FrenetTrajectory<T>::ToNumpy() {
        # const double init_s = traj_[0].s[0];
        # Eigen::MatrixXd numpy(traj_.size(), kTrajFeatureDim);

        # for (int i = 0; i < traj_.size(); ++i) {
        #     numpy(i, 0) = traj_[i].pos[0];
        #     numpy(i, 1) = traj_[i].pos[1];
        #     numpy(i, 2) = std::cos(traj_[i].yaw);
        #     numpy(i, 3) = std::sin(traj_[i].yaw);
        #     numpy(i, 4) = traj_[i].s[0] - init_s;
        #     numpy(i, 5) = traj_[i].s[1];
        #     numpy(i, 6) = traj_[i].s[2];
        #     numpy(i, 7) = traj_[i].d[0];
        #     numpy(i, 8) = traj_[i].d[1];
        #     numpy(i, 9) = traj_[i].d[2];
        #     numpy(i, 10) = traj_[i].vel;
        #     numpy(i, 11) = traj_[i].acc;
        # }
        # return numpy;
        # }
        """
        # check total ref points num
        # check max ref interval
        # trim reflane
        if ref_check:
            ref_info = self.trim_reflane(ref_info,x,y)
        wx = ref_info[...,0]
        wy = ref_info[...,1]
        ss = get_arc_length(ref_info[...,:2])
        
        self.c_sampler.set_init_state(x,y,yaw,v,acc)
        self.c_sampler.set_reference_line(wx,wy,ss)
        self.c_sampler.sample(self.T)
        self.c_sampler.sample_global_state(self.Ts)
        trajs = self.c_sampler.get_trajectories(exclude_invalid=True)
        
        trajs = np.array(trajs)
        fut_traj = [trajs[...,0:2], 
                    np.arctan2(trajs[...,3:4],trajs[...,2:3]),
                    trajs[...,10:11]]
        fut_traj = np.concatenate(fut_traj,axis=-1)
        return fut_traj
    
    def trim_reflane(self, ref_info: np.ndarray, x:float, y:float):
        ss = get_arc_length(ref_info[...,:2])
        if np.amax(ss)>self.ref_length_lim:
            # find -5, 70 reflane
            current = np.array([[x,y]])
            front_distance = 5./np.diff(ss,axis=0).mean()
            back_distance = 70./np.diff(ss,axis=0).mean()
            distances = np.sqrt(np.sum((ref_info[...,:2] - current)**2, axis=-1))
            nearest_index = np.argmin(distances)
            start_index = max(0, nearest_index - int(front_distance))
            end_index = min(len(ref_info) - 1, nearest_index + int(back_distance))
            ref_info = ref_info[start_index:end_index+1]
        if len(ref_info)>250:
            interval = int(np.round(len(ref_info)/250))
            ref_info = ref_info[::interval]
        # plt.scatter(ref_info[...,0], ref_info[...,1])
        return ref_info
        
if __name__=="__main__":
    x = torch.linspace(-10,80,1200)
    # y = torch.zeros_like(x)
    # y = torch.sin(x/20)
    y = (x/5)**2
    w = torch.stack([x,y],dim=-1).numpy()
    sampler = LatticeSampler()
    trajs = sampler.sampling(w,0.,2.,0.8,2.,0.)
    for traj in trajs:
        plt.plot(traj[...,0], traj[...,1])
    plt.plot(x,y)
    plt.axis('equal')
    plt.show()