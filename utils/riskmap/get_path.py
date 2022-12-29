from soupsieve import select
import torch
from torch import nn

from src.models.decoders.utils import has_nan

from .torch_mpc import torchMPC
from .torch_lattice import torchLatticePlanner
import numpy as np
from .getloss import GetLoss
import time
import matplotlib.pyplot as plt
# torch.autograd.set_detect_anomaly(True)


class GetPath:
    def __init__(self, cfg, getloss: GetLoss = None, device: str = 'cuda:0', debug=False) -> None:
        self.cfg = cfg
        if getloss is not None:
            self.getloss = getloss.eval()
        self.device = device
        self.ti = cfg['time_interval']
        self.bb_col_check = cfg['bb_col_check']
        self.lattice_flag = cfg['lattice'] if 'lattice' in cfg.keys() else False
        # initialize planning mode
        if self.lattice_flag:
            self.lattice = torchLatticePlanner(cfg,self.device,training=False)
        else:
            self.mpc = torchMPC(cfg)
            # self.optimizer = torch.optim.LBFGS(self.mpc.parameters(),cfg['mpc_opt_rate']) # torch.optim.LSR1
            self.optimizer = torch.optim.AdamW(self.mpc.parameters(), cfg['mpc_opt_rate'])
            self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=1, gamma=0.9)
            self.num_iters = cfg['max_num_iters']

        self.risk_tolerance = cfg['risk_tolerance'] if 'risk_tolerance' in cfg.keys() else 1.
        self.debug = debug

    def get_path(self):
        start = time.time()
        if self.lattice_flag:
            # Using Lattice
            # with torch.no_grad():
            X,u = self.get_lattice_path()
            print('opt_time is :', time.time()-start)
            return X, u
        else:
            # Using mpc
            self.get_mpc_path()
            print('opt_time is :', time.time()-start)
            return self.mpc.X, self.mpc.u

    def new_map(self, getloss: GetLoss = None, s0=None, ego_feature=None, extend_data = None):
        if getloss is not None:
            self.getloss = getloss.eval()

        if s0 is None:
            s0 = self.get_init_state_from_ego_feature(ego_feature)

        if extend_data is not None:
            self.getloss.get_extend_circles(extend_data)

        if self.lattice_flag:
            self.lattice.set_init_state(s0,reflanes=self.getloss.riskmap.reflane)
        else:
            self.mpc.set_init_state(s0)

    def get_init_state_from_ego_feature(self, ego_feature: torch.Tensor):
        # print(ego_feature)
        s0 = ego_feature.clone()
        ds = torch.norm(torch.diff(s0[..., :2],dim=0),dim=-1)
        v = ds/self.ti
        v = torch.cat([v[0:1], v], dim=0).clamp(min=0,max = 25)
        a = torch.diff(v,dim=0)/self.ti
        a = torch.cat([a[0:1],a],dim=0)
        v = v+a.clamp(max = 5, min=-5)*self.ti/2
        s0[..., 3] = v
        s0 = torch.cat([s0,a.unsqueeze(-1)],dim=-1)
        return s0[-1]
    
    def set_initial_guess(self,ego_guess,prob,data):
        """
        set initial guess form ego prediction
        """
        loss = []
        u_set=[]
        self.getloss.get_extend_circles(data)
        for traj in ego_guess[...,:2]:
            X,u = self.getloss.convert2detail_state(traj,from_init_guess=True)
            u_set.append(u)
            loss.append(self.getloss.get_loss_by_Xu(
                        X, u, vis_loss=False, BBcollison=True))
        loss = torch.stack(loss,dim=0)
        u = torch.stack(u_set,dim=0)
        loss_exp = loss#*(-torch.log(prob))
        iniu = u[torch.argmin(loss_exp)]
        iniX = ego_guess[torch.argmin(loss_exp)]
        s0 = self.mpc.s0
        self.mpc = torchMPC(self.cfg,device=self.device,ini_u=iniu,s0 = s0)
        return iniX,iniu

    def get_mpc_path(self):
        if self.getloss is None:
            raise ValueError(
                'get raise is None, please use new_map initialize self.getloss first')
        # self.optimizer.zero_grad()
        # loss = self.get_forward_loss()
        # loss.backward()
        total_iterations = 0
        du = 1e5
        last_u = self.mpc.u.clone()
        while du>1e-3: #abs(dloss)<1e-4:
            # print('\n total_iterations: ', total_iterations)
            # with torch.no_grad():
            #     dloss = self.get_forward_loss().item()
            # self.optimizer.step(self.closure)
            self.optimizer.zero_grad()
            loss = self.get_forward_loss()
            has_nan(loss)
            loss.backward()
            # self.optimizer.step(self.closure)
            self.optimizer.step()

            # self._train_step(expert_x, expert_u, extra_info, theta,
            #                  mu_r, lam_r, optimizer, pbar)
            # self.lr_scheduler.step()

            # with torch.no_grad():
            #     theta_r = np.exp(theta[-1].item())
            #     lam_r -= mu_r*theta_r
            #     if abs(theta_r) > 0.5*abs(prev_theta_r) and abs(theta_r) > 1e-5:
            #         mu_r *= 10.0
            du = torch.max(torch.abs(self.mpc.u-last_u))
            last_u = self.mpc.u.clone()
            total_iterations += 1
            if total_iterations >= self.num_iters:
                break

    def get_forward_loss(self):
        self.mpc.forward_simu()
        loss,_ = self.getloss.get_loss_by_Xu(self.mpc.X, self.mpc.u, BBcollison=self.bb_col_check)
        has_nan(loss)
        return loss

    def get_lattice_path(self):
        st = time.time()
        # get values
        Xs = []
        us = []
        reflanes = []
        # three reflane at most
        for i in range(self.lattice.lane_num):
            self.lattice.new_frenet(self.lattice.reflanes[i])
            X,u = self.lattice.get_valid_path_sample()
            Xs.append(X)
            us.append(u)
            reflanes.append(
                torch.from_numpy(
                    self.lattice.c_sampler.get_ref_line()
                    ).to(self.device)
                    )
        if self.lattice.making_sample:
            aX,au = self.lattice.make_sample()
            Xs.append(aX)
            us.append(au)
        X = torch.cat(Xs,dim=0)
        u = torch.cat(us,dim=0)
        reflanes = torch.stack(reflanes,dim=0)
        self.X = X.clone()
        self.u = u.clone()
        self.smoothed_reflanes = reflanes.clone()
        print('generate time', time.time()-st)
        st = time.time()
        # reflane dis filter
        if X.shape[0]>0:
            X,u = self.ref_lane_dis_filter(X,u,reflanes)
        # deal with problems with all nan
        if X.shape[0]==0:
            sn,tn,dn = X.shape
            x = torch.linspace(0.1,3,30,device=self.device)*self.lattice.init_X_state[3]
            dcc = torch.linspace(0.1,3,30,device=self.device)*-1 # dm/s
            x = x+dcc
            X = torch.zeros(1,tn,dn,device=self.device)
            u = X.clone()[...,:2]
            X[...,0] = x.clamp(min = 0.1)
        c = self.getloss.get_loss_by_Xu(X,
                                        u,
                                        BBcollison=self.getloss.bb_col_check,
                                        extend=self.getloss.extend_circles,
                                        batch=True,
                                        vis_loss=True)
        # selector
        X,u = self.selection_from_traj_set(X,u,c)
        print('cost cal time', time.time()-st)
        return X,u

    def ref_lane_dis_filter(self, X, u, ref):  
        refn,pts,rdim = ref.shape
        sampn,th,xdim = X.shape
        diff = torch.norm(X[...,:2].unsqueeze(-2)-ref.reshape(-1,rdim),dim=-1)
        diff = torch.min(diff.reshape(sampn,th,-1),dim=-1).values
        if (diff[:,0]>=2).all():
            return X,u
        ind = torch.min(diff<2,dim=-1).values #TODO
        X = X[ind]
        u = u[ind]

        return X,u


    def selection_from_traj_set(self,X,u,costs):
        # costs = self.getloss.get_loss_by_Xu(X,u,BBcollison=True,vis_loss=True,batch = False)
        # costs = torch.sum(costs,dim=-1)
        # costs = torch.sum(costs,dim=-1)
        # print(costs)
        if self.debug:
            # for Xi in X.detach():
            plt.scatter(X[...,0].detach().cpu(),
                        X[...,1].detach().cpu(),
                        c = costs.detach().cpu())
            plt.axis('equal')
            plt.show()
        i = torch.argmin(costs.sum(dim=-1))
        return X[i],u[i]

    def closure(self):
        self.optimizer.zero_grad()

        # loss = -self.env.log_likelihood(x=expert_x,
        #                                 u=expert_u,
        #                                 extra_info=extra_info,
        #                                 theta=theta,
        #                                 mu_r=mu_r,
        #                                 lam_r=lam_r)
        loss = self.get_forward_loss()

        # Keeping track of training losses
        # if self.verbose:
        #     self._training_losses.append(loss.item())

        # with torch.no_grad():
        #     curr_theta = theta.numpy()
        #     reg_val = np.exp(curr_theta[-1])

        # Logging
        # logger_dict = {'nll': [loss.item()], 'theta_r': [reg_val]}
        # for i in range(curr_theta.shape[0]-1):
        #     logger_dict[f'theta_{i}'] = [curr_theta[i]]
        # self.logger.add_rows(logger_dict)
        # self.logger.increment('lbfgs_iter')
        # self.logger.increment('total_iter')

        # Progress Bar
        # pbar.set_description(f'NLL: {loss.item():.2f}, reg: {reg_val:.2g}, theta: {curr_theta[:-1]}')

        loss.backward()
        # optimizer.step()
        return loss

    # def _train_step(self,):

    #     def closure():
    #         self.optimizer.zero_grad()

    #         # loss = -self.env.log_likelihood(x=expert_x,
    #         #                                 u=expert_u,
    #         #                                 extra_info=extra_info,
    #         #                                 theta=theta,
    #         #                                 mu_r=mu_r,
    #         #                                 lam_r=lam_r)
    #         loss = self.get_loss()

    #         # Keeping track of training losses
    #         # if self.verbose:
    #         #     self._training_losses.append(loss.item())

    #         # with torch.no_grad():
    #         #     curr_theta = theta.numpy()
    #         #     reg_val = np.exp(curr_theta[-1])

    #             # Logging
    #             # logger_dict = {'nll': [loss.item()], 'theta_r': [reg_val]}
    #             # for i in range(curr_theta.shape[0]-1):
    #             #     logger_dict[f'theta_{i}'] = [curr_theta[i]]
    #             # self.logger.add_rows(logger_dict)
    #             # self.logger.increment('lbfgs_iter')
    #             # self.logger.increment('total_iter')

    #             # Progress Bar
    #             # pbar.set_description(f'NLL: {loss.item():.2f}, reg: {reg_val:.2g}, theta: {curr_theta[:-1]}')

    #         loss.backward()
    #         # optimizer.step()
    #         return loss

    #     # self.logger.update_indices({'lbfgs_iter': 0})
    #     self.optimizer.step(closure)
    #     # closure()
    #     # self.logger.increment('outer_iter')

    #     # pbar.update()

    #     # Keeping track of training thetas
    #     # self._track_theta(theta)
