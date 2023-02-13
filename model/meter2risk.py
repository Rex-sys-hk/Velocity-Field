# select Cost mode here
# tvc is a module can be cused in each mode
from difflib import context_diff
from turtle import forward
import torch
from torch import nn
import torch.nn.functional as F
from utils.riskmap.utils import has_nan,load_cfg_here
debug = False

class Meter2Risk(nn.Module):
    def __init__(self, device: str = 'cuda') -> None:
        super().__init__()
        ## define some common params here
        self.device = device
        cfg = load_cfg_here()['planner']['meter2risk']
        self.TVC=cfg['TVC']
        self.V_dim=['V_dim']
        self.th=cfg['th']
        self.idv= cfg['idv']
        self.latent_feature = None

    def forward(self):
        raise NotImplementedError(Meter2Risk)
    
    def get_risk(self):
        raise NotImplementedError(Meter2Risk)

    def set_latent_feature(self, latent_feature):
        self.latent_feature = latent_feature



def get_pos_def_matrix(mat:torch.Tensor):
    mat = torch.bmm(mat,mat.transpose(dim0=-1,dim1=-2))
    mat = mat+mat.transpose(dim0=-1,dim1=-2)
    return mat

class UnifiedGaussianCost(Meter2Risk):
    def __init__(self, device: str = 'cuda') -> None:
        super().__init__()
        self.device = device
        th = th if TVC else 1
        self.lmda = nn.Parameter(torch.ones(th, V_dim, device=self.device))
        self.beta = nn.Parameter(torch.ones(th, V_dim, device=self.device))

    def forward(self,raw_meter):
        """
        raw_meter: (th, 7)
        """
        has_nan(self.lmda)
        norm_meter = (raw_meter - torch.mean(raw_meter, dim=-1, keepdim=True)) / \
            (torch.var(raw_meter, dim=-1,keepdim=True)+1e-6)
        self.K = (self.lmda*norm_meter).unsqueeze(-2)*(norm_meter).unsqueeze(-1)
        cost = torch.exp(self.K.clamp(max=1e2))
        # print(self.K.shape)
        # print(cost.shape)
        return cost.sum(dim=-1)

    def regulator(self):
        # regulator
        cost = 0
        # trace
        # cost += torch.diagonal(self.K, dim1=-2, dim2=-1).sum(-1)
        # has_nan(torch.det(K))
        # cost += torch.det(K).sum()
        # cost += -trace.sum()
        cost += (self.lmda**2).sum(dim=-1)
        return cost.sum()


class SimpleSignedRisk(Meter2Risk):
    def __init__(self, TVC=False,V_dim=2, th=1, device: str = 'cuda') -> None:
        super().__init__()
        self.device = device
        th = th if TVC else 1
        self.lmda = nn.Parameter(torch.ones(1, V_dim, device=self.device))
        self.beta = nn.Parameter(torch.ones(1, V_dim, device=self.device))
        self.R_base = nn.Parameter(torch.ones(1, 2, device=self.device))

        self.mask = torch.tensor([[1, 1, 0]], device=self.device)
        self.weight = nn.Parameter(torch.ones(th, 7, device=self.device))


    def forward(self,raw_meters):
        has_nan(raw_meters)
        has_nan(self.beta)
        has_nan(self.lmda)
        cost = torch.zeros_like(raw_meters,device=self.device)
        R_L_cost = torch.exp(
            self.beta**2)*torch.exp(self.lmda*raw_meters[...,0:2])
        # L_cost = R_L_cost*self.mask+raw_meters[...,0:3]*(1-self.mask)
        cost[...,0:2] = R_L_cost

        R_mat = torch.softmax(self.R_base, dim=-1)
        cost[...,3:5] = torch.pow(raw_meters[...,3:5], 2)*R_mat
        return  cost*torch.softmax(self.weight,dim=-1)

    def regulator(self):
        cost = 0
        cost += (self.beta**2).sum()
        return cost


class OriginalMeters(Meter2Risk):
    def __init__(self, TVC = False, V_dim=7, th=1, device: str = 'cuda') -> None:
        super().__init__()
        self.device = device
        print("Using OriginalMeters, No training should be conducted")
        self.sign = torch.tensor([[-1,1,1,1,1,1,-1]],device=self.device)

    def forward(self, raw_meters):
        return raw_meters*self.sign
    
    def regulator(self):
        return 0


class SimpleWeights(Meter2Risk):
    def __init__(self,TVC=False ,V_dim=7, th=1, device: str = 'cuda') -> None:
        super().__init__()
        self.device = device
        self.weights = nn.Parameter(torch.ones(th,V_dim,device=self.device))
        self.sign = torch.tensor([[-1,1,1,1,1,1,-1]],device=self.device)
        # [sdf,ref,coll,a,s,dv,v]

    def forward(self,raw_meters):
        # TODO sdf and v is not completely functional
        self.raw_meters = raw_meters
        self.cost = raw_meters**2*self.sign*self.weights**2

        return self.cost.sum(dim=-1)

    def regulator(self):
        reg = torch.sum(self.weights**2)
        var = torch.var(self.cost)
        return reg+var

class PositiveDeterministicWeights(Meter2Risk):
    def __init__(self,TVC=False ,V_dim=7, th=1, device: str = 'cuda') -> None:
        super().__init__()
        self.device = device
        th = th if TVC else 1
        self.L_mat = nn.Parameter(torch.ones(4,4,device=self.device,dtype=float))
        self.R_mat = nn.Parameter(torch.ones(2,2,device=self.device,dtype=float))

    def forward(self, raw_meters):
        map_meter = torch.cat([raw_meters[...,:3],raw_meters[...,-1:]],dim=-1)
        L_mat = self.L_mat
        L_mat = get_pos_def_matrix(L_mat)
        has_nan(L_mat)
        L_cost = torch.matmul(torch.matmul(
            map_meter.unsqueeze(-2), L_mat.clamp(max=1e2)), map_meter.unsqueeze(-1))

        control_meter = raw_meters[...,3:5]
        R_mat = self.R_mat
        R_mat = get_pos_def_matrix(R_mat)
        R_cost = torch.matmul(torch.matmul(
            control_meter.unsqueeze(-2), R_mat.clamp(max=1e2)), control_meter.unsqueeze(-1))

        return L_cost.sum(dim=-1)+R_cost.sum(dim=-1)


    def regulator(self):
        L_mat = get_pos_def_matrix(self.L_mat)
        R_mat = get_pos_def_matrix(self.R_mat)
        L_t = torch.diagonal(L_mat, dim1=-2, dim2=-1).sum(-1)
        R_t = torch.diagonal(R_mat, dim1=-2, dim2=-1).sum(-1)
        
        return -(L_t+R_t).sum()


class CIOC(Meter2Risk):
    def __init__(self,TVC=False ,V_dim=7, th=1, device: str = 'cuda') -> None:
        super().__init__()
        self.device = device
        th = th if TVC else 1
        self.th = th
        # clastering of expected feature
        self.expected_feature = nn.Parameter(torch.rand(th,V_dim,device=self.device))
        self.lamda = nn.Parameter(torch.ones(1,V_dim,device=self.device))
        self.beta = nn.Parameter(torch.ones(1,device=self.device))
        self.y = nn.Parameter(torch.zeros(th,1,device=self.device))
        self.feature_playback=[]
        self.zeros = torch.zeros(th,V_dim,device=self.device)
        self.linear_cost = SimpleWeights(TVC=TVC ,V_dim=V_dim, th=th, device=device)
        

    def forward(self,raw_meters):
        self.raw_meters = raw_meters#.clone()
        if self.training:
            cost = self.zeros
            # cost += self.linear_cost.forward(self.raw_meters)
        else:
            cost = self.get_reward()
        return cost

    def regulator(self):
        """
        Only callable after forward is called
        """
        if self.raw_meters is None:
            raise ValueError("This regulator is only callable after forward function ")
        statistic_ave = torch.sum((self.expected_feature-self.raw_meters)**2)
        NLL = -self.GPLikelihood()
        LFC = self.linear_cost.forward(self.raw_meters).sum() +self.linear_cost.regulator()
        self.raw_meters = None
        return statistic_ave + NLL + LFC

    def get_reward(self):
        K = self.get_K()
        kt = self.raw_meters.unsqueeze(1) - self.expected_feature.unsqueeze(0)
        kt = kt.sum(dim=-1)
        reward = kt*(torch.linalg.inv(K)*self.y.squeeze(-1)).sum(dim=-1)
        return reward.sum(dim=-1) + self.linear_cost(self.raw_meters.detach()).sum(dim=-1)

    def GPLikelihood(self):
        K = self.get_K()
        P = -0.5*(self.y.T@torch.linalg.inv(K)*self.y)
        P += -0.5*torch.log(torch.det(K))
        P += -0.5*torch.diagonal(torch.linalg.inv(K))
        P += -torch.log(self.lamda+1).sum()
        return P

    def get_K(self):
        # indc = torch.ones(self.th,self.th,7)-torch.diag(torch.ones(self.th)).unsqueeze(-1)
        # randmat = indc*torch.randn(self.th,self.th,7)*0.02
        # randmat = get_pos_def_matrix(randmat.permute(2,0,1)).permute(1,2,0)
        # randmat = randmat.to(self.device)
        K = self.beta**2*torch.exp(
                -0.5*(
                (
                (self.expected_feature.unsqueeze(0)-self.expected_feature.unsqueeze(1))**2 #+randmat*indc.to(self.device)\
                )*self.lamda
                ).sum(dim=-1)
            )
        return K
        

class Enssemble(Meter2Risk):
    def __init__(self) -> None:
        super().__init__()


class DeepCost(Meter2Risk):
    def __init__(self, TVC=False ,V_dim=8, th=1, device: str = 'cuda', idv = True) -> None:
        super().__init__()
        self.device = device
        th = th if TVC else 1
        self.th = th
        # beta, lambda, Rmat, weight, vfilter, traget_v
        # self.para_dim = 4+4+2+6+4+1
        self.map_elements = 4
        self.u_dim = 2
        self.weight_dim = 7
        self.vfilter_dim = self.map_elements
        self.target_v_dim = 1
        # self.decoder = nn.Sequential(Mlp(192, 192, out_features= self.para_dim*th,act_layer=nn.GELU),
        #                             # nn.LayerNorm(self.para_dim*th),
        #                             Mlp(self.para_dim*th, act_layer=nn.Tanh),
        #                             # nn.Softmax(dim=0)
        #                             )

        # No soft max in mlp as the th dim is not seperated
        input_shape=256
        self.beta = nn.Sequential(
                                    nn.Linear(input_shape,64),
                                    nn.LayerNorm(64),
                                    nn.GELU(),
                                    nn.Linear(64,32),
                                    nn.GELU(),
                                    nn.Linear(32,self.th*self.map_elements),
                                    # nn.GELU(),
                                    # nn.Softmax(dim=-1),
                                    nn.Sigmoid(),
                                    ).to(device)
        self.lmda = nn.Sequential(
                                    nn.Linear(input_shape,64),
                                    nn.LayerNorm(64),
                                    nn.GELU(),
                                    nn.Linear(64,32),
                                    nn.GELU(),
                                    nn.Linear(32,self.th*self.map_elements),
                                    # nn.Softmax(dim=-1),
                                    nn.Softsign(),
                                    # nn.Sigmoid()
                                    ).to(device)       
        self.Rmat = nn.Sequential(
                                    nn.Linear(input_shape,64),
                                    nn.LayerNorm(64),
                                    nn.GELU(),
                                    nn.Linear(64,32),
                                    nn.GELU(),
                                    # nn.Tanhshrink(),
                                    nn.Linear(32,self.th*self.u_dim),
                                    nn.Sigmoid()
                                    # nn.Softmax(dim=-1)
                                    ).to(device)
        self.target_v = nn.Sequential(
                                    nn.Linear(input_shape,64),
                                    nn.LayerNorm(64),
                                    nn.GELU(),
                                    nn.Linear(64,32),
                                    nn.GELU(),
                                    # nn.Tanhshrink(),
                                    nn.Linear(32,self.th*self.target_v_dim),
                                    nn.GELU(),
                                    ).to(device)


    def forward(self, raw_meters:torch.Tensor, prediction = None, writer = None, tb_iters = 0):
        # print("===in cost forward===")
        #     # 0-sdf 1-ref 2-tl 3-coll 4-a 5-s 6-v

        # useful parameters
        raw_meter=[]
        for key in raw_meters.keys():
            raw_meter.append(raw_meters[key])
        raw_meters=torch.cat(raw_meter,dim=-1)
        batch, sample_num, th, items = raw_meters.shape

        # create cost container
        cost = torch.zeros((batch,sample_num,th,self.map_elements+self.u_dim+self.target_v_dim),device=self.device)
        if len(raw_meters.shape)!=3:
            ValueError('raw_meter in shape', raw_meters.shape, 'which should be in 3 dim')

        # get ego context encoding
        # context_enc = prediction['context_enc']['agents2graph_enc'][-1]
        context_enc = self.latent_feature['agent_map'][...,0,:].max(dim=1).values

        # raw_meters Normalization
        # raw_meters = F.normalize(raw_meters,dim=-1)
        # raw_meters = F.normalize(raw_meters,dim=-2)
        raw_meters[...,:self.map_elements] = F.normalize(raw_meters[...,:self.map_elements],dim= -1)
        
        # get params wrt context enc
        beta = self.beta(context_enc).view(batch,-1,self.th,self.map_elements)
        beta = torch.softmax(beta,dim=-1)
        lmda = self.lmda(context_enc).view(batch,-1,self.th,self.map_elements)
        # lmda = torch.softmax(lmda,dim=-1)*3
        R_mat = self.Rmat(context_enc).reshape(batch,-1,self.th,self.u_dim)
        R_mat = torch.softmax(R_mat,dim=-1)
        target_v = self.target_v(context_enc).reshape(batch,-1,self.th,self.target_v_dim)
        self.target_v_value = target_v

        L_cost = beta*torch.exp(lmda*raw_meters[...,:self.map_elements])
        
        # sdf, reflane, tl
        cost[...,0:self.map_elements] = L_cost
        # # coll
        # cost[...,self.map_elements] = raw_meters[...,self.map_elements]
        # u
        cost[...,self.map_elements:self.map_elements+self.u_dim] = \
            torch.pow(raw_meters[...,self.map_elements:self.map_elements+self.u_dim], 2)\
                *R_mat
        # dv
        cost[...,-1:] = (raw_meters[...,-1:]-target_v.detach())**2
        has_nan(cost)

        if writer :
            writer.add_scalar('raw_meter/' + 'raw3', raw_meters[...,3].mean(), tb_iters)
            writer.add_scalar('cost/' + 'c0', cost[...,0].mean(), tb_iters)
            writer.add_scalar('cost/' + 'c1', cost[...,1].mean(), tb_iters)
            writer.add_scalar('cost/' + 'c2', cost[...,2].mean(), tb_iters)
            writer.add_scalar('cost/' + 'c3', cost[...,3].mean(), tb_iters)
            writer.add_scalar('cost/' + 'c4', cost[...,4].mean(), tb_iters)
            writer.add_scalar('cost/' + 'c5', cost[...,5].mean(), tb_iters)
            writer.add_scalar('cost/' + 'c6', cost[...,6].mean(), tb_iters)


        return cost

    def get_target_v(self):
        return self.target_v_value

    def regulator(self):
        return 0


class DeepMapOnly(Meter2Risk):
    def __init__(self, TVC=False ,V_dim=8, th=1, device: str = 'cuda', idv = True) -> None:
        super().__init__()
        self.device = device
        th = th if TVC else 1
        self.th = th
        # beta, lambda, Rmat, traget_v
        # self.para_dim = 4+4+2+6+4+1
        self.map_elements = 3
        self.collision = 1
        self.u_dim = 2
        self.target_v_dim = 1
        self.loss_idv = idv
        # No soft max in mlp as the th dim is not seperated  
        self.beta = nn.Sequential(
                                    nn.Linear(192,64),
                                    nn.LayerNorm(64),
                                    nn.GELU(),
                                    nn.Linear(64,32),
                                    nn.GELU(),
                                    nn.Linear(32,self.th*self.map_elements),
                                    # nn.GELU(),
                                    # nn.Softmax(dim=-1),
                                    nn.Sigmoid(),
                                    ).to(device)     
        self.Rmat = nn.Sequential(
                                    nn.Linear(192,64),
                                    nn.LayerNorm(64),
                                    nn.GELU(),
                                    nn.Linear(64,32),
                                    nn.GELU(),
                                    # nn.Tanhshrink(),
                                    nn.Linear(32,self.th*self.u_dim),
                                    nn.Sigmoid()
                                    # nn.Softmax(dim=-1)
                                    ).to(device)
        self.target_v = nn.Sequential(
                                    nn.Linear(192,64),
                                    nn.LayerNorm(64),
                                    nn.GELU(),
                                    nn.Linear(64,32),
                                    nn.GELU(),
                                    # nn.Tanhshrink(),
                                    nn.Linear(32,self.th*self.target_v_dim),
                                    nn.GELU(),
                                    ).to(device)

        self.map_mapper = nn.Sequential(
                                    nn.Linear(192,192),
                                    # nn.Linear(self.th*self.map_elements,128),
                                    # nn.LayerNorm(128),
                                    nn.GELU(),
                                    nn.Linear(192,128),
                                    nn.GELU(),
                                    nn.Linear(128,128),
                                    nn.GELU(),
                                    # nn.Tanhshrink(),
                                    nn.Linear(128,self.th*self.map_elements),
                                    nn.Tanh(),
                                    ).to(device)


    def forward(self, raw_meters:torch.Tensor, prediction = None, writer = None, tb_iters = 0):
        # print("===in cost forward===")
        #     # 0-sdf 1-ref 2-tl 3-coll 4-a 5-s 6-v

        # useful parameters
        sample_num, th, items = raw_meters.shape

        # create cost container
        cost = torch.zeros((sample_num,th,self.map_elements+self.collision+self.u_dim+self.target_v_dim),device=self.device)
        if len(raw_meters.shape)!=3:
            ValueError('raw_meter in shape', raw_meters.shape, 'which should be in 3 dim')

        # get ego context encoding
        context_enc = prediction['context_enc']['agents2graph_enc'][-1]

        # raw_meters Normalization
        # raw_meters = F.normalize(raw_meters,dim=-1)
        # raw_meters = F.normalize(raw_meters,dim=-2)
        # raw_meters[...,:self.map_elements] = F.normalize(raw_meters[...,:self.map_elements],dim= -1)
        
        # get params wrt context enc
        beta = self.beta(context_enc).view(-1,self.th,self.map_elements)
        beta = torch.softmax(beta,dim=-1)
        # lmda = self.lmda(context_enc).view(-1,self.th,self.map_elements)
        # lmda = torch.softmax(lmda,dim=-1)
        R_mat = self.Rmat(context_enc).reshape(-1,self.th,self.u_dim)
        # R_mat = torch.softmax(R_mat,dim=-1)
        target_v = self.target_v(context_enc).reshape(-1,self.th,self.target_v_dim)
        self.target_v_value = target_v
        
        # L_cost = beta*torch.exp(lmda*raw_meters[...,:self.map_elements])
        map_mapper = self.map_mapper(context_enc).reshape(-1,self.th,self.map_elements)
        # L_cost = self.map_mapper(raw_meters[...,:self.map_elements].reshape(sample_num,-1).float())
        L_cost = map_mapper*raw_meters[...,:self.map_elements]
        
        # sdf, reflane, tl
        cost[...,:self.map_elements] = beta*torch.exp(L_cost)#.reshape(sample_num,th,-1)
        # coll
        cost[...,self.map_elements] = raw_meters[...,self.map_elements]
        # u
        us = self.map_elements+self.collision
        ue = self.map_elements+self.collision+self.u_dim
        # should not be normalize, use soft max at most
        # cost[...,us:ue] = \
        #     torch.pow(F.normalize(raw_meters[...,us:ue],dim=0), 2)\
        #         *R_mat
        #
        cost[...,us:ue] = \
            torch.pow(raw_meters[...,us:ue], 2)*R_mat
        # dv
        if not self.loss_idv:
            cost[...,-1:] = (raw_meters[...,-1:]-target_v)**2
        else:
            cost[...,-1:] = (raw_meters[...,-1:]-target_v.detach())**2
        has_nan(cost)

        if writer :
            writer.add_scalar('raw_meter/' + 'raw0', raw_meters[...,0].mean(), tb_iters)
            writer.add_scalar('raw_meter/' + 'raw1', raw_meters[...,1].mean(), tb_iters)
            writer.add_scalar('raw_meter/' + 'raw2', raw_meters[...,2].mean(), tb_iters)
            writer.add_scalar('raw_meter/' + 'raw3', raw_meters[...,3].mean(), tb_iters)
            writer.add_scalar('raw_meter/' + 'raw4', raw_meters[...,4].mean(), tb_iters)
            writer.add_scalar('raw_meter/' + 'raw5', raw_meters[...,5].mean(), tb_iters)
            writer.add_scalar('raw_meter/' + 'raw6', raw_meters[...,6].mean(), tb_iters)

            writer.add_scalar('cost/' + 'c0', cost[...,0].mean(), tb_iters)
            writer.add_scalar('cost/' + 'c1', cost[...,1].mean(), tb_iters)
            writer.add_scalar('cost/' + 'c2', cost[...,2].mean(), tb_iters)
            writer.add_scalar('cost/' + 'c3', cost[...,3].mean(), tb_iters)
            writer.add_scalar('cost/' + 'c4', cost[...,4].mean(), tb_iters)
            writer.add_scalar('cost/' + 'c5', cost[...,5].mean(), tb_iters)
            writer.add_scalar('cost/' + 'c6', cost[...,6].mean(), tb_iters)

        return cost

    def get_target_v(self):
        return self.target_v_value

    def regulator(self):
        return 0


class DeepMapOnly_vv(Meter2Risk):
    def __init__(self, TVC=False ,V_dim=8, th=1, device: str = 'cuda', idv= True) -> None:
        super().__init__()
        self.device = device
        th = th if TVC else 1
        self.th = th
        self.thv = 30 # the velocity diff should always be time variable
        # beta, lambda, Rmat, traget_v
        # self.para_dim = 4+4+2+6+4+1
        self.map_elements = 2
        self.collision = 1
        self.u_dim = 2
        self.target_v_dim = 1
        self.u_weight_dim = 2
        # self.total = 3+1+2+1

        # No soft max in mlp as the th dim is not seperated  
        self.beta = nn.Sequential(
                                    nn.Linear(256,128),
                                    nn.LayerNorm(128),
                                    nn.GELU(),
                                    nn.Linear(128,128),
                                    nn.GELU(),
                                    nn.Linear(128,self.th*self.map_elements),
                                    # nn.GELU(),
                                    # nn.Softmax(dim=-1),
                                    nn.Sigmoid(),
                                    ).to(device)     
        self.Rmat = nn.Sequential(
                                    nn.Linear(256,128),
                                    nn.LayerNorm(128),
                                    nn.GELU(),
                                    nn.Linear(128,128),
                                    nn.GELU(),
                                    # nn.Tanhshrink(),
                                    nn.Linear(128,self.th*self.u_dim),
                                    nn.Sigmoid()
                                    # nn.Softmax(dim=-1)
                                    ).to(device)
        self.target_v = nn.Sequential(
                                    nn.Linear(256,64),
                                    nn.LayerNorm(64),
                                    nn.GELU(),
                                    nn.Linear(64,32),
                                    nn.GELU(),
                                    # nn.Tanhshrink(),
                                    nn.Linear(32,self.thv*self.target_v_dim),
                                    nn.GELU(),
                                    ).to(device)

        self.map_mapper = nn.Sequential(
                                    nn.Linear(256,192),
                                    # nn.Linear(self.th*self.map_elements,128),
                                    # nn.LayerNorm(128),
                                    nn.GELU(),
                                    nn.Linear(192,128),
                                    nn.GELU(),
                                    nn.Linear(128,128),
                                    nn.GELU(),
                                    # nn.Tanhshrink(),
                                    nn.Linear(128,self.th*self.map_elements),
                                    nn.Tanh(),
                                    ).to(device)
        self.m_weight = nn.Parameter(torch.ones(3,device=self.device))

        # self.u_weight = nn.Sequential(
        #                             nn.Linear(192,64),
        #                             nn.LayerNorm(64),
        #                             nn.GELU(),
        #                             nn.Linear(64,32),
        #                             nn.GELU(),
        #                             # nn.Tanhshrink(),
        #                             nn.Linear(32,self.u_weight_dim),
        #                             nn.Sigmoid(),
        #                             ).to(device)
        # self.m_weight = torch.tensor([0.5,0.5],device=self.device)


    def forward(self, raw_meters:torch.Tensor, prediction = None, writer = None, tb_iters = 0):
        # print("===in cost forward===")
        #     # 0-sdf 1-ref 2-tl 3-coll 4-a 5-s 6-v
        # get ego context encoding
        # print(self.latent_feature['agent_map'].shape)
        # torch.Size([48, 3, 11, 256])
        context_enc = self.latent_feature['agent_map'][...,0,:] #prediction['context_enc']['agents2graph_enc'][:,-1]
        
        btsz,mode,dim = context_enc.shape
        #select useful params
        raw_meters = torch.cat([raw_meters[key] for key in raw_meters.keys()],dim=-1)
        # useful parameters
        _,sample_num, th, items = raw_meters.shape
        # create cost container
        cost = torch.zeros((btsz,sample_num,th,self.map_elements+self.collision+self.u_dim+self.target_v_dim),device=self.device)


        # raw_meters Normalization
        # raw_meters = F.normalize(raw_meters,dim=-1)
        # raw_meters = F.normalize(raw_meters,dim=-2)
        # raw_meters[...,:self.map_elements] = F.normalize(raw_meters[...,:self.map_elements],dim= -1)
        m_weight = torch.pow(self.m_weight,2)
        # get params wrt context enc
        beta = self.beta(context_enc).view(btsz,-1,self.th,self.map_elements)
        beta = torch.softmax(beta,dim=-1)
        # lmda = self.lmda(context_enc).view(-1,self.th,self.map_elements)
        # lmda = torch.softmax(lmda,dim=-1)
        R_mat = self.Rmat(context_enc).reshape(btsz,-1,self.th,self.u_dim)
        # R_mat = torch.softmax(R_mat,dim=-1)
        target_v = self.target_v(context_enc).reshape(btsz,-1,self.thv,self.target_v_dim)
        self.target_v_value = target_v
        
        # weight = self.weight(context_enc).reshape(-1,self.th,self.total)

        # L_cost = beta*torch.exp(lmda*raw_meters[...,:self.map_elements])
        map_mapper = self.map_mapper(context_enc).reshape(btsz,-1,self.th,self.map_elements)
        # L_cost = self.map_mapper(raw_meters[...,:self.map_elements].reshape(sample_num,-1).float())
        L_cost = map_mapper*raw_meters[...,:self.map_elements]
        
        # sdf, reflane, tl
        cost[...,:self.map_elements] = beta*torch.exp(L_cost)*m_weight[0]#.reshape(sample_num,th,-1)
        # coll
        cost[...,self.map_elements] = raw_meters[...,self.map_elements]
        # u
        us = self.map_elements+self.collision
        ue = self.map_elements+self.collision+self.u_dim
        # should not be normalize, use softmax at most
        cost[...,us:ue] = \
            torch.pow(raw_meters[...,us:ue], 2)*R_mat*m_weight[1]
        # dv
        cost[...,-1:] = (raw_meters[...,-1:]-target_v.detach())**2*m_weight[2]
        # cost = cost*weight
        has_nan(cost)

        if writer :
            writer.add_scalar('raw_meter/' + 'raw0', raw_meters[...,0].mean(), tb_iters)
            writer.add_scalar('raw_meter/' + 'raw1', raw_meters[...,1].mean(), tb_iters)
            writer.add_scalar('raw_meter/' + 'raw2', raw_meters[...,2].mean(), tb_iters)
            writer.add_scalar('raw_meter/' + 'raw3', raw_meters[...,3].mean(), tb_iters)
            writer.add_scalar('raw_meter/' + 'raw4', raw_meters[...,4].mean(), tb_iters)
            writer.add_scalar('raw_meter/' + 'raw5', raw_meters[...,5].mean(), tb_iters)
            # writer.add_scalar('raw_meter/' + 'raw6', raw_meters[...,6].mean(), tb_iters)

            writer.add_scalar('cost/' + 'c0', cost[...,0].mean(), tb_iters)
            writer.add_scalar('cost/' + 'c1', cost[...,1].mean(), tb_iters)
            writer.add_scalar('cost/' + 'c2', cost[...,2].mean(), tb_iters)
            writer.add_scalar('cost/' + 'c3', cost[...,3].mean(), tb_iters)
            writer.add_scalar('cost/' + 'c4', cost[...,4].mean(), tb_iters)
            writer.add_scalar('cost/' + 'c5', cost[...,5].mean(), tb_iters)
            # writer.add_scalar('cost/' + 'c6', cost[...,6].mean(), tb_iters)

        return cost

    def get_target_v(self):
        return self.target_v_value

    def regulator(self):
        return 0

class LinearMap_vv(Meter2Risk):
    def __init__(self, TVC=False ,V_dim=8, th=1, device: str = 'cuda', idv = True) -> None:
        super().__init__()
        self.device = device
        th = th if TVC else 1
        self.th = th
        self.thv = 30 # the velocity diff should always be time variable
        # beta, lambda, Rmat, traget_v
        # self.para_dim = 4+4+2+6+4+1
        self.map_elements = 2
        self.collision = 1
        self.u_dim = 2
        self.target_v_dim = 1
        self.u_weight_dim = 2
        # self.total = 3+1+2+1

        # No soft max in mlp as the th dim is not seperated  
        self.beta = nn.Sequential(
                                    nn.Linear(192,128),
                                    nn.LayerNorm(128),
                                    nn.GELU(),
                                    nn.Linear(128,128),
                                    nn.GELU(),
                                    nn.Linear(128,self.th*self.map_elements),
                                    # nn.GELU(),
                                    # nn.Softmax(dim=-1),
                                    nn.Softsign(),
                                    ).to(device)     
        self.Rmat = nn.Sequential(
                                    nn.Linear(192,128),
                                    nn.LayerNorm(128),
                                    nn.GELU(),
                                    nn.Linear(128,128),
                                    nn.GELU(),
                                    # nn.Tanhshrink(),
                                    nn.Linear(128,self.th*self.u_dim),
                                    nn.Sigmoid()
                                    # nn.Softmax(dim=-1)
                                    ).to(device)
        self.target_v = nn.Sequential(
                                    nn.Linear(192,64),
                                    nn.LayerNorm(64),
                                    nn.GELU(),
                                    nn.Linear(64,32),
                                    nn.GELU(),
                                    # nn.Tanhshrink(),
                                    nn.Linear(32,self.thv*self.target_v_dim),
                                    nn.GELU(),
                                    ).to(device)

        self.map_mapper = nn.Sequential(
                                    nn.Linear(192,192),
                                    # nn.Linear(self.th*self.map_elements,128),
                                    # nn.LayerNorm(128),
                                    nn.GELU(),
                                    nn.Linear(192,128),
                                    nn.GELU(),
                                    nn.Linear(128,128),
                                    nn.GELU(),
                                    # nn.Tanhshrink(),
                                    nn.Linear(128,self.th*self.map_elements),
                                    nn.Softsign(),
                                    ).to(device)
        self.m_weight = nn.Parameter(torch.ones(3,device=self.device))
        # self.u_weight = nn.Sequential(
        #                             nn.Linear(192,64),
        #                             nn.LayerNorm(64),
        #                             nn.GELU(),
        #                             nn.Linear(64,32),
        #                             nn.GELU(),
        #                             # nn.Tanhshrink(),
        #                             nn.Linear(32,self.u_weight_dim),
        #                             nn.Sigmoid(),
        #                             ).to(device)
        # self.m_weight = torch.tensor([0.5,0.5],device=self.device)

    def forward(self, raw_meters:torch.Tensor, prediction = None, writer = None, tb_iters = 0):
        # print("===in cost forward===")
        #     # 0-sdf 1-ref 2-tl 3-coll 4-a 5-s 6-v
        #select usful params
        raw_meters = torch.cat([raw_meters[...,0:2],raw_meters[...,3:]],dim=-1)
        # useful parameters
        sample_num, th, items = raw_meters.shape
        # create cost container
        cost = torch.zeros((sample_num,th,self.map_elements+self.collision+self.u_dim+self.target_v_dim),device=self.device)
        if len(raw_meters.shape)!=3:
            ValueError('raw_meter in shape', raw_meters.shape, 'which should be in 3 dim')

        # get ego context encoding
        context_enc = prediction['context_enc']['agents2graph_enc'][-1]

        # raw_meters Normalization
        # raw_meters = F.normalize(raw_meters,dim=-1)
        # raw_meters = F.normalize(raw_meters,dim=-2)
        # raw_meters[...,:self.map_elements] = F.normalize(raw_meters[...,:self.map_elements],dim= -1)
        m_weight = torch.pow(self.m_weight,2)
        # get params wrt context enc
        beta = self.beta(context_enc).view(-1,self.th,self.map_elements)
        # beta = torch.softmax(beta,dim=-1)
        # lmda = self.lmda(context_enc).view(-1,self.th,self.map_elements)
        # lmda = torch.softmax(lmda,dim=-1)
        R_mat = self.Rmat(context_enc).reshape(-1,self.th,self.u_dim)
        # R_mat = torch.softmax(R_mat,dim=-1)
        target_v = self.target_v(context_enc).reshape(-1,self.thv,self.target_v_dim)
        self.target_v_value = target_v
        
        # weight = self.weight(context_enc).reshape(-1,self.th,self.total)

        # L_cost = beta*torch.exp(lmda*raw_meters[...,:self.map_elements])
        map_mapper = self.map_mapper(context_enc).reshape(-1,self.th,self.map_elements)
        # L_cost = self.map_mapper(raw_meters[...,:self.map_elements].reshape(sample_num,-1).float())
        L_cost = map_mapper+raw_meters[...,:self.map_elements]
        
        # sdf, reflane, tl
        cost[...,:self.map_elements] = beta*L_cost*m_weight[0]#.reshape(sample_num,th,-1)
        # coll
        cost[...,self.map_elements] = raw_meters[...,self.map_elements]
        # u
        us = self.map_elements+self.collision
        ue = self.map_elements+self.collision+self.u_dim
        # should not be normalize, use softmax at most
        cost[...,us:ue] = \
            torch.pow(raw_meters[...,us:ue], 2)*R_mat*m_weight[1]
        # dv
        cost[...,-1:] = (raw_meters[...,-1:]-target_v.detach())**2*m_weight[2]
        # cost = cost*weight
        has_nan(cost)

        if writer :
            writer.add_scalar('raw_meter/' + 'raw0', raw_meters[...,0].mean(), tb_iters)
            writer.add_scalar('raw_meter/' + 'raw1', raw_meters[...,1].mean(), tb_iters)
            writer.add_scalar('raw_meter/' + 'raw2', raw_meters[...,2].mean(), tb_iters)
            writer.add_scalar('raw_meter/' + 'raw3', raw_meters[...,3].mean(), tb_iters)
            writer.add_scalar('raw_meter/' + 'raw4', raw_meters[...,4].mean(), tb_iters)
            writer.add_scalar('raw_meter/' + 'raw5', raw_meters[...,5].mean(), tb_iters)
            # writer.add_scalar('raw_meter/' + 'raw6', raw_meters[...,6].mean(), tb_iters)

            writer.add_scalar('cost/' + 'c0', cost[...,0].mean(), tb_iters)
            writer.add_scalar('cost/' + 'c1', cost[...,1].mean(), tb_iters)
            writer.add_scalar('cost/' + 'c2', cost[...,2].mean(), tb_iters)
            writer.add_scalar('cost/' + 'c3', cost[...,3].mean(), tb_iters)
            writer.add_scalar('cost/' + 'c4', cost[...,4].mean(), tb_iters)
            writer.add_scalar('cost/' + 'c5', cost[...,5].mean(), tb_iters)
            # writer.add_scalar('cost/' + 'c6', cost[...,6].mean(), tb_iters)

        return cost

    def get_target_v(self):
        return self.target_v_value

    def regulator(self):
        return 0
    
class SimpleCostCoef(Meter2Risk):
    def __init__(self, device: str = 'cuda') -> None:
        super().__init__(device)
        self.th = 50
        self.map_elements = 7
        self.get_coeff = nn.Sequential(
                                nn.Linear(256,128),
                                nn.LayerNorm(128),
                                nn.GELU(),
                                nn.Linear(128,128),
                                nn.GELU(),
                                nn.Linear(128,self.th*self.map_elements),
                                # nn.GELU(),
                                # nn.Softmax(dim=-1),
                                nn.Softsign(),
                                ).to(device)
    
    def forward(self, raw_meters):
        raw_meters = torch.cat([raw_meters[key] for key in raw_meters.keys()],dim=-1)
        coeff = self.get_coeff(self.latent_feature['agent_map'][...,0,:].max(dim=1).values).reshape(-1,1,self.th,self.map_elements)
        return coeff*raw_meters
        
CostModules = {
    'deep_cost':DeepCost,
    'deep_map':DeepMapOnly,
    'deep_map_vv':DeepMapOnly_vv,
    'linear_map_vv':LinearMap_vv,
    'simple_coef':SimpleCostCoef,
}