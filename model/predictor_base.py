import torch
from torch import embedding, int64, long, nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from utils.riskmap.car import MAX_ACC, MAX_STEER
from utils.riskmap.rm_utils import has_nan, load_cfg_here, standardize_vf, yawv2yawdxdy
from utils.test_utils import batch_sample_check_collision

def time_embeding(traj):
    b,s,th,d = traj.shape
    t = torch.linspace(0, 1, th, device=traj.device).reshape(1, 1, th, 1).repeat(b, s, 1, 1)
    st = torch.sin(2 * torch.pi * t)
    ct = torch.cos(2 * torch.pi * t)
    return torch.cat([traj, t, st, ct], dim=-1)
# Agent history encoder
class AgentEncoder(nn.Module):
    def __init__(self):
        super(AgentEncoder, self).__init__()
        self.motion = nn.LSTM(8, 256, 2, batch_first=True)

    def forward(self, inputs):
        traj, _ = self.motion(inputs[:, :, :8])
        output = traj[:, -1]

        return output

# Local context encoders
class LaneEncoder(nn.Module):
    def __init__(self):
        super(LaneEncoder, self).__init__()
        # encdoer layer
        self.self_line = nn.Linear(3, 128)
        self.left_line = nn.Linear(3, 128)
        self.right_line = nn.Linear(3, 128)
        self.speed_limit = nn.Linear(1, 64)
        self.self_type = nn.Embedding(4, 64, padding_idx=0)
        self.left_type = nn.Embedding(11, 64, padding_idx=0)
        self.right_type = nn.Embedding(11, 64, padding_idx=0)
        self.traffic_light_type = nn.Embedding(9, 64, padding_idx=0)
        self.interpolating = nn.Embedding(2, 64)
        self.stop_sign = nn.Embedding(2, 64)
        self.stop_point = nn.Embedding(2, 64)

        # hidden layers
        self.pointnet = nn.Sequential(nn.Linear(512, 384), nn.ReLU(), nn.Linear(384, 256), nn.ReLU())

    def forward(self, inputs):
        # embedding
        self_line = self.self_line(inputs[..., :3])
        left_line = self.left_line(inputs[..., 3:6])
        right_line = self.right_line(inputs[...,  6:9])
        speed_limit = self.speed_limit(inputs[..., 9].unsqueeze(-1))
        self_type = self.self_type(inputs[..., 10].int())
        left_type = self.left_type(inputs[..., 11].int())
        right_type = self.right_type(inputs[..., 12].int()) 
        traffic_light = self.traffic_light_type(inputs[..., 13].int())
        stop_point = self.stop_point(inputs[..., 14].int())
        interpolating = self.interpolating(inputs[..., 15].int()) 
        stop_sign = self.stop_sign(inputs[..., 16].int())

        lane_attr = self_type + left_type + right_type + traffic_light + stop_point + interpolating + stop_sign
        lane_embedding = torch.cat([self_line, left_line, right_line, speed_limit, lane_attr], dim=-1)
    
        # process
        output = self.pointnet(lane_embedding)

        return output

class CrosswalkEncoder(nn.Module):
    def __init__(self):
        super(CrosswalkEncoder, self).__init__()
        self.point_net = nn.Sequential(nn.Linear(3, 64), nn.ReLU(), nn.Linear(64, 128), nn.ReLU(), nn.Linear(128, 256), nn.ReLU())
    
    def forward(self, inputs):
        output = self.point_net(inputs)

        return output
# VAE
class VAEcore(nn.Module):
    def __init__(self, encoding_dim, latent_dim=64) -> None:
        super().__init__()
                # 定义均值层和方差层
        self.z_mean = nn.Linear(encoding_dim, latent_dim)
        self.z_log_var = nn.Linear(encoding_dim, latent_dim)
        self.decoder = nn.Sequential(nn.Linear(latent_dim, encoding_dim), 
                                     nn.ReLU(), 
                                     nn.Linear(encoding_dim, encoding_dim), 
                                     nn.ReLU(),
                                    nn.Linear(encoding_dim, encoding_dim),
                                    nn.Sigmoid(),
                                     )
        
    def reparameterize(self, z_mean, z_log_var):
        epsilon = torch.randn(z_mean.size()).to(z_mean.device)
        z = z_mean + torch.exp(0.5 * z_log_var) * epsilon
        return z
    
    def loss_function(self, z_mean, z_log_var):
        loss = -0.5 * torch.sum(1 + z_log_var - z_mean ** 2 - z_log_var.exp())
        
        return loss
    
    def forward(self, x):
        z_mean = self.z_mean(x)
        z_log_var = self.z_log_var(x)
        z = self.reparameterize(z_mean, z_log_var)
        z = self.decoder(z)
        loss = self.loss_function(z_mean, z_log_var)
        # loss += F.binary_cross_entropy(z, x, reduction='sum')
        return z, loss
        

# Transformer modules
class CrossTransformer(nn.Module):
    def __init__(self):
        super(CrossTransformer, self).__init__()
        self.cross_attention = nn.MultiheadAttention(256, 8, 0.1, batch_first=True)
        self.transformer = nn.Sequential(nn.LayerNorm(256), nn.Linear(256, 1024), nn.ReLU(), nn.Dropout(0.1), nn.Linear(1024, 256), nn.LayerNorm(256))

    def forward(self, query, key, value, mask=None):
        attention_output, _ = self.cross_attention(query, key, value, key_padding_mask=mask)
        output = self.transformer(attention_output)

        return output

class MultiModalTransformer(nn.Module):
    def __init__(self, modes=3, output_dim=256):
        super(MultiModalTransformer, self).__init__()
        # self.modes = modes
        cfg = load_cfg_here()
        self.mode_num = cfg['model_cfg']['mode_num'] if cfg['model_cfg']['mode_num'] else 3
        self.attention = nn.ModuleList([nn.MultiheadAttention(256, 4, 0.1, batch_first=True) for _ in range(self.mode_num)])
        self.ffn = nn.Sequential(nn.LayerNorm(256), nn.Linear(256, 1024), nn.ReLU(), nn.Dropout(0.1), nn.Linear(1024, output_dim), nn.LayerNorm(output_dim))

    def forward(self, query, key, value, mask=None):
        attention_output = []
        for i in range(self.mode_num):
            attention_output.append(self.attention[i](query, key, value, key_padding_mask=mask)[0])
        attention_output = torch.stack(attention_output, dim=1)
        output = self.ffn(attention_output)

        return output

# Transformer-based encoders
class Agent2Agent(nn.Module):
    def __init__(self):
        super(Agent2Agent, self).__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model=256, nhead=8, dim_feedforward=1024, activation='relu', batch_first=True)
        self.interaction_net = nn.TransformerEncoder(encoder_layer, num_layers=2, enable_nested_tensor = False)

    def forward(self, inputs, mask=None):
        output = self.interaction_net(inputs, src_key_padding_mask=mask)

        return output

class Agent2Map(nn.Module):
    def __init__(self):
        super(Agent2Map, self).__init__()
        self.lane_attention = CrossTransformer()
        self.crosswalk_attention = CrossTransformer()
        self.map_attention = MultiModalTransformer() 

    def forward(self, actor, lanes, crosswalks, mask):
        query = actor.unsqueeze(1)
        # TODO use map_actor
        lanes_actor = [self.lane_attention(query, lanes[:, i], lanes[:, i]) for i in range(lanes.shape[1])]
        crosswalks_actor = [self.crosswalk_attention(query, crosswalks[:, i], crosswalks[:, i]) for i in range(crosswalks.shape[1])]
        map_actor = torch.cat(lanes_actor+crosswalks_actor, dim=1)
        output = self.map_attention(query, map_actor, map_actor, mask).squeeze(2)
        return map_actor, output 

# Decoders
class AgentDecoder(nn.Module):
    def __init__(self, future_steps):
        super(AgentDecoder, self).__init__()
        self._future_steps = future_steps 
        cfg = load_cfg_here()
        self.mode_num = cfg['model_cfg']['mode_num'] if cfg['model_cfg']['mode_num'] else 3
        self.decode = nn.Sequential(nn.Dropout(0.1), nn.Linear(512, 256), nn.ELU(), nn.Linear(256, future_steps*3))

    def transform(self, prediction, current_state):
        x = current_state[:,None,:, 0] 
        y = current_state[:,None,:, 1]
        theta = current_state[:,None,:, 2]
        
        delta_x = prediction[..., 0]
        delta_y = prediction[..., 1]
        delta_theta = prediction[..., 2]
        # _sin = torch.sin(theta)
        # _cos = torch.cos(theta)
        # new_x = x.unsqueeze(-2) + delta_x*_cos-delta_y*_sin
        # new_y = y.unsqueeze(-2) + delta_x*_sin+delta_y*_cos
        new_x = x.unsqueeze(-1) + delta_x
        new_y = y.unsqueeze(-1) + delta_y
        new_theta = theta.unsqueeze(-1) + delta_theta
        traj = torch.stack([new_x, new_y, new_theta], dim=-1)

        return traj
       
    def forward(self, agent_map, agent_agent, current_state):
        feature = torch.cat([agent_map, agent_agent.unsqueeze(1).repeat(1, self.mode_num, 1, 1)], dim=-1)
        decoded = self.decode(feature).view(-1, self.mode_num, 10, self._future_steps, 3)
        # trajs = torch.stack([self.transform(decoded[:, i, j], current_state[:, j]) for i in range(self.mode_num) for j in range(10)], dim=1)
        trajs = self.transform(decoded, current_state)
        trajs = torch.reshape(trajs, (-1, self.mode_num, 10, self._future_steps, 3))

        return trajs

class AVDecoder(nn.Module):
    def __init__(self, future_steps=50, feature_len=9):
        super(AVDecoder, self).__init__()
        self._future_steps = future_steps
        cfg = load_cfg_here()
        self.mode_num = cfg['model_cfg']['mode_num'] if cfg['model_cfg']['mode_num'] else 3
        self.control = nn.Sequential(nn.Dropout(0.1), nn.Linear(512, 256), nn.ELU(), nn.Linear(256, future_steps*2))
        self.cost = nn.Sequential(nn.Linear(1, 128), nn.ReLU(), nn.Linear(128, feature_len), nn.Softmax(dim=-1))
        self.register_buffer('scale', torch.tensor([1, 1, 1, 1, 1, 10, 100]))
        self.register_buffer('constraint', torch.tensor([[10, 10]]))

    def forward(self, agent_map, agent_agent):
        feature = torch.cat([agent_map, agent_agent.unsqueeze(1).repeat(1, self.mode_num, 1)], dim=-1)
        actions = self.control(feature).view(-1, self.mode_num , self._future_steps, 2)
        dummy = torch.ones(1, 1).to(self.cost[0].weight.device)
        cost_function_weights = torch.cat([self.cost(dummy)[:, :7] * self.scale, self.constraint], dim=-1)

        return actions, cost_function_weights
    
class AVDecoderNc(nn.Module):
    def __init__(self, future_steps=50, feature_len=9):
        super(AVDecoderNc, self).__init__()
        self._future_steps = future_steps
        cfg = load_cfg_here()
        self.mode_num = cfg['model_cfg']['mode_num'] if cfg['model_cfg']['mode_num'] else 3
        self.control = nn.Sequential(nn.Dropout(0.1), nn.Linear(512, 256), nn.ELU(), nn.Linear(256, future_steps*2))
        a_max = MAX_ACC
        s_max = MAX_STEER
        self.mean_control = torch.tensor([a_max,s_max])

    def forward(self, agent_map, agent_agent):
        feature = torch.cat([agent_map, agent_agent.unsqueeze(1).repeat(1, self.mode_num, 1)], dim=-1)
        actions = self.control(feature).view(-1, self.mode_num , self._future_steps, 2)
        self.mean_control = self.mean_control.to(actions.device)
        actions = (torch.sigmoid(actions)-0.5)*2*self.mean_control
        return actions, 0

class Score(nn.Module):
    def __init__(self):
        super(Score, self).__init__()
        self.reduce = nn.Sequential(nn.Dropout(0.1), nn.Linear(512, 256), nn.ELU())
        self.decode = nn.Sequential(nn.Dropout(0.1), nn.Linear(512, 128), nn.ELU(), nn.Linear(128, 1))
        cfg = load_cfg_here()
        self.mode_num = cfg['model_cfg']['mode_num'] if cfg['model_cfg']['mode_num'] else 3

    def forward(self, map_feature, agent_agent, agent_map):
        # pooling
        map_feature = map_feature.view(map_feature.shape[0], -1, map_feature.shape[-1])
        map_feature = torch.max(map_feature, dim=1)[0]
        agent_agent = torch.max(agent_agent, dim=1)[0]
        agent_map = torch.max(agent_map, dim=2)[0]

        feature = torch.cat([map_feature, agent_agent], dim=-1)
        feature = self.reduce(feature.detach())
        feature = torch.cat([feature.unsqueeze(1).repeat(1, self.mode_num, 1), agent_map.detach()], dim=-1)
        scores = self.decode(feature).squeeze(-1)

        return scores
    
class PreABC(nn.Module):
    def __init__(self, name:str = 'pre_abc', future_steps = 50, mode_num = 3, gamma = 1.) -> None:
        super().__init__()
        self.name = 'pre_abc'
        self._future_steps = future_steps
        cfg = load_cfg_here()
        self.cfg = cfg
        # agent layer
        self.vehicle_net = AgentEncoder()
        self.pedestrian_net = AgentEncoder()
        self.cyclist_net = AgentEncoder()

        # map layer
        self.lane_net = LaneEncoder()
        self.crosswalk_net = CrosswalkEncoder()
        
        # attention layers
        self.agent_map = Agent2Map()
        self.agent_agent = Agent2Agent()

        # decode layers
        # self.plan = AVDecoderNc(self._future_steps)
        self.predict = AgentDecoder(self._future_steps)
        self.score = Score()
        
    def forward_base(self, ego, neighbors, map_lanes, map_crosswalks):
        # actors
        ego_actor = self.vehicle_net(ego)
        vehicles = torch.stack([self.vehicle_net(neighbors[:, i]) for i in range(10)], dim=1) 
        pedestrians = torch.stack([self.pedestrian_net(neighbors[:, i]) for i in range(10)], dim=1) 
        cyclists = torch.stack([self.cyclist_net(neighbors[:, i]) for i in range(10)], dim=1)
        neighbor_actors = torch.where(neighbors[:, :, -1, -1].unsqueeze(2)==2, pedestrians, vehicles)
        neighbor_actors = torch.where(neighbors[:, :, -1, -1].unsqueeze(2)==3, cyclists, neighbor_actors)
        actors = torch.cat([ego_actor.unsqueeze(1), neighbor_actors], dim=1)
        actor_mask = torch.eq(torch.cat([ego.unsqueeze(1), neighbors[..., :-1]], dim=1), 0)[:, :, -1, -1]

        # maps
        lane_feature = self.lane_net(map_lanes)
        crosswalk_feature = self.crosswalk_net(map_crosswalks)
        lane_mask = torch.eq(map_lanes, 0)[:, :, :, 0, 0]
        crosswalk_mask = torch.eq(map_crosswalks, 0)[:, :, :, 0, 0]
        map_mask = torch.cat([lane_mask, crosswalk_mask], dim=2)
        map_mask[:, :, 0] = False # prevent nan
        
        # actor to actor
        agent_agent = self.agent_agent(actors, actor_mask)
        
        # map to actor
        map_feature = []
        agent_map = []
        for i in range(actors.shape[1]):
            output = self.agent_map(agent_agent[:, i], lane_feature[:, i], crosswalk_feature[:, i], map_mask[:, i])
            map_feature.append(output[0])
            agent_map.append(output[1])

        map_feature = torch.stack(map_feature, dim=1)
        agent_map = torch.stack(agent_map, dim=2)
        plans, cost_function_weights = self.plan(agent_map[:, :, 0], agent_agent[:, 0])
        predictions = self.predict(agent_map[:, :, 1:], agent_agent[:, 1:], neighbors[:, :, -1])
        scores = self.score(map_feature, agent_agent, agent_map)
        return map_feature, agent_map, agent_agent, plans, predictions, scores, cost_function_weights
    
# %% VectorField
class VFMapDecoder(nn.Module):
    def __init__(self) -> None:
        super(VFMapDecoder, self).__init__()
        cfg = load_cfg_here()
        self.time_embedding = cfg['model_cfg']['time_embedding'] if 'time_emedding' in cfg['model_cfg'] else False
        self.time_embdim = 5 if self.time_embedding else 2
        self.mode_num = cfg['model_cfg']['mode_num'] if cfg['model_cfg']['mode_num'] else 3
        self.reduce = nn.Sequential(nn.Dropout(0.1), nn.Linear(512, 256), nn.ReLU())
        self.map_feat_emb = nn.Sequential(nn.Linear(512, 256), nn.ReLU(), nn.Dropout(0.1), nn.Linear(256, 256),nn.ReLU(),nn.Dropout(0.1),nn.Linear(256, 64))
        self.emb = nn.Sequential(nn.Linear(self.time_embdim, 32),nn.ReLU(), nn.Dropout(0.1), nn.Linear(32, 64),nn.ReLU(), nn.Dropout(0.1),nn.Linear(64, 64))
        self.cross_attention = nn.MultiheadAttention(64, 4, 0.1, batch_first=True)
        self.transformer = nn.Sequential(nn.Linear(64, 64), nn.GELU(), nn.Dropout(0.1), nn.Linear(64, 32), nn.LayerNorm(32), nn.GELU(), nn.Linear(32,2))
        
    def set_latent_feature(self, latent_feature):
        self._map_feature = latent_feature['map_feature']
        self._agent_map = latent_feature['agent_map']
        self._agent_agent = latent_feature['agent_agent']
        # self._masks = masks[:,0]

    def forward(self, sample):
        # cross attention of self.grid and map_feature
        # TODO : add time embedding here
        b,s,th,d = sample.shape
        query = time_embeding(sample[...,:2]) if self.time_embedding else sample[...,:2]
        query = self.emb(query.reshape(b,-1,self.time_embdim))
        map_feature = self._map_feature.view(self._map_feature.shape[0], -1, self._map_feature.shape[-1])
        map_feature = torch.amax(map_feature, dim=-2)
        agent_agent = torch.amax(self._agent_agent, dim=-2)
        agent_map = torch.amax(self._agent_map, dim=-2)

        feature = torch.cat([map_feature, agent_agent], dim=-1)
        feature = self.reduce(feature.detach())
        feature = torch.cat([feature.unsqueeze(1).repeat(1, self.mode_num, 1), agent_map.detach()], dim=-1)
        feature = self.map_feat_emb(feature)
        x,_ = self.cross_attention(query,
                                   feature,
                                   feature,
                                #    key_padding_mask = self._masks
                                   )
        dx_dy = self.transformer(x)
        return dx_dy
    
    def disable_grad(self):
        self.reduce.requires_grad_(False)
        self.map_feat_emb.requires_grad_(False)
        self.emb.requires_grad_(False)
        self.cross_attention.requires_grad_(False)
        self.transformer.requires_grad_(False)
        
    def enable_grad(self):
        self.reduce.requires_grad_(True)
        self.map_feat_emb.requires_grad_(True)
        self.emb.requires_grad_(True)
        self.cross_attention.requires_grad_(True)
        self.transformer.requires_grad_(True)

class VectorField(nn.Module):
    def __init__(self) -> None:
        super(VectorField, self).__init__()
        self.vf_inquery = VFMapDecoder()
        self.rear_range = 10.
        self.front_range = 90.
        self.side_range = 30.
        self.steps_s = int(100)
        self.steps_l = int(60)
        self.resolution_s = (self.front_range+self.rear_range)/self.steps_s
        self.resolution_l = 2*self.side_range/self.steps_l
        # make grid
        s = torch.linspace(-self.rear_range,self.front_range,self.steps_s)
        l = torch.linspace(-self.side_range,self.side_range,self.steps_l)
        x,y = torch.meshgrid(s,l,indexing='xy')
        self.grid_points = torch.stack([x,y],dim=-1).reshape(1, -1, 2)
    
    def plot(self, samples):
        b,s,t,d = samples.shape
        sample_dx_dy = self.vf_inquery(samples[...,:2])#.reshape(b,s,t,2)
        sample_dx_dy, m = standardize_vf(sample_dx_dy)
        plt.quiver(samples.reshape(b,-1,d)[0,::7,0].cpu().detach(), 
            samples.reshape(b,-1,d)[0,::7,1].cpu().detach(),
            sample_dx_dy[0,::7,0].cpu().detach(),
            sample_dx_dy[0,::7,1].cpu().detach(),
            label=f'Plan Sampled Vector, max={m}',
            width=0.001,
            scale = 5,
            color='red',
            zorder=1,
            alpha=0.2,
            scale_units='inches',
            )
        # vis vector field at time 0
        dx_dy = self.vf_inquery(self.grid_points.to(samples.device).repeat(samples.shape[0],1,1).unsqueeze(-2))[0]
        dx_dy, m = standardize_vf(dx_dy)
        plt.quiver(self.grid_points[0,...,0].cpu().detach(), 
                   self.grid_points[0,...,1].cpu().detach(),
                   dx_dy[...,0].cpu().detach(),
                   dx_dy[...,1].cpu().detach(),
                   label=f'Vis Vector Field, max={m}',
                   width=0.001,
                   scale = 5,
                   color='k',
                   zorder= 1,
                   alpha= 0.2,
                   scale_units='inches',
                   )
        
    def plot_gt(self, gt, samples):
        diff_sample_gt = 0
        dis_diff = gt[...,:2]-torch.cat([torch.zeros_like(samples[...,0:1,:2],device=samples.device), 
                                        samples[...,:-1,:2]],
                                        dim=-2)
        d_dis_diff = dis_diff/0.1 #Time interval
        diff_sample_gt+=d_dis_diff
        sample_dxy = torch.stack([torch.cos(samples[...,2])*samples[...,3], 
                                  torch.sin(samples[...,2])*samples[...,3]],
                                  dim=-1)
        diff_sample_gt += gt[...,3:5] - sample_dxy
        diff_sample_gt, m = standardize_vf(diff_sample_gt)
        plt.quiver(samples[0,...,0].cpu().detach(),
            samples[0,...,1].cpu().detach(),
            diff_sample_gt[0,...,0].cpu().detach(),
            diff_sample_gt[0,...,1].cpu().detach(), 
            color = 'green',
            width=0.001,
            scale = 5,
            label=f'GT Vector, max={m}',
            zorder= 1,
            alpha = 0.2,
            scale_units='inches',
            )
        
    def get_loss(self, gt, sample, context=None, k = 30):
        # convert to vx,vy
        loss = 0
        # get dx_dy
        query = torch.cat([gt[...,:2],sample[...,:2]],dim=1)
        b,s,t,d = query.shape
        dx_dy = self.vf_inquery(query[...,:2]).reshape(b,s,t,2)
        dx_dy_gt = dx_dy[:,0:1]#.clamp(max = 30, min = 0)
        dx_dy_samp = dx_dy[:,1:]
        # dx_dy_grid = self.vf_inquery(self.grid_points.to(sample.device).repeat(sample.shape[0],1,1))
        
        # imitation loss
        # GT velocity prior
        loss += torch.nn.functional.smooth_l1_loss(dx_dy_gt, gt[...,3:5]) # TODO gt smoothing is not solved
        # Smaple velocity priorl
        dis = torch.norm(sample[...,:3]-gt[...,:3],dim=-1)
        # TODO tuning k number
        nearest_id = torch.topk(dis.mean(dim=-1), k=k, dim=-1, largest=False, sorted=False).indices
        # gather debug， follow the integrated terminal
        nearest_sample = torch.gather(sample,-3,nearest_id.unsqueeze(-1).unsqueeze(-1).repeat(1,1,t,4))
        nearest_sample = yawv2yawdxdy(nearest_sample)
        dx_dy_nearest = torch.gather(dx_dy_samp,-3,nearest_id.unsqueeze(-1).unsqueeze(-1).repeat(1,1,t,2))
        # weighted loss
        # farther sample contributs less prior
        if context:
            collision = batch_sample_check_collision(nearest_sample, 
                                                     context['predictions'], 
                                                     context['current_state'][:, :, 5:],
                                                     t_stamp=True)
            nearest_sample[collision][...,3:5] = 0.#-nearest_sample[collision][...,3:5]
        # core vf map construction loss  
        nearest_dis = gt[...,:3]-nearest_sample[...,:3]
        discount = torch.exp(-0.5*torch.norm(nearest_dis, dim=-1, keepdim=True)**2)
        correction = nearest_dis[...,:2]/0.1 # dt
        correction_discount = torch.exp(-0.5*torch.norm(nearest_dis[...,:2], dim=-1, keepdim=True)**2)
        # loss += torch.nn.functional.smooth_l1_loss(dx_dy_nearest, 
        #                                             correction+discount*nearest_sample[...,3:5])
        loss += (correction_discount \
                *torch.norm(
                dx_dy_nearest-(correction+discount*nearest_sample[...,3:5]), 
                dim=-1, 
                keepdim=True)
            ).mean()
        return loss
    
    def vector_field_diff(self, traj):
        b,s,t,d = traj.shape
        if d != 4:
            raise ValueError('Trajectory should be in shape of (batch, sample, time, 4)')
        traj_dxy = torch.stack([torch.cos(traj[...,2])*traj[...,3],
                            torch.sin(traj[...,2])*traj[...,3]],
                            dim=-1)
        dx_dy = self.vf_inquery(traj).reshape(b,s,t,2)
        diff = traj_dxy - dx_dy
        diff = torch.abs(diff)
        # diff = torch.nn.functional.smooth_l1_loss(diff, torch.zeros_like(diff), reduction='none')
        return diff
    
    def disable_grad(self):
        self.vf_inquery.disable_grad()
    
    def enable_grad(self):
        self.vf_inquery.enable_grad()
# %% cost volume
class CostMapDecoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        cfg = load_cfg_here()
        self.mode_num = cfg['model_cfg']['mode_num'] if cfg['model_cfg']['mode_num'] else 3
        # self.map_feat_emb = nn.Sequential(nn.Linear(256*9, 256), nn.ReLU(), nn.Dropout(0.1), nn.Linear(256, 256),nn.ReLU(),nn.Dropout(0.1),nn.Linear(256, 64))
        self.emb = nn.Sequential(nn.Linear(5, 32),nn.ReLU(), nn.Dropout(0.1), nn.Linear(32, 64),nn.ReLU(), nn.Dropout(0.1),nn.Linear(64, 64))
        self.cross_attention = nn.MultiheadAttention(64, 4, 0.1, batch_first=True)
        self.transformer = nn.Sequential(nn.Linear(64, 256), nn.ReLU(), nn.Dropout(0.1), nn.Linear(256, 64), nn.LayerNorm(64), nn.ReLU(), nn.Linear(64,1))
        self.reduce = nn.Sequential(nn.Dropout(0.1), nn.Linear(512, 256), nn.ELU())
        self.embedding = nn.Sequential(nn.Dropout(0.1), nn.Linear(512*self.mode_num, 256), nn.ELU(), nn.Linear(256, 64))
        self._map_feature = None
        self._agent_feature = None
        self._masks = None
        
    def set_latent_feature(self, map_feature, agent_map, agent_agent):
        map_feature = map_feature.view(map_feature.shape[0], -1, map_feature.shape[-1])
        map_feature = torch.max(map_feature, dim=1)[0]
        agent_agent = torch.max(agent_agent, dim=1)[0]
        agent_map = torch.max(agent_map, dim=2)[0]
        feature = torch.cat([map_feature, agent_agent], dim=-1)
        feature = self.reduce(feature.detach())
        feature = torch.cat([feature.unsqueeze(1).repeat(1, self.mode_num, 1), agent_map.detach()], dim=-1)
        self.x = self.embedding(feature.reshape(feature.shape[0], -1))
        

    def forward(self, sample):
        # cross attention of self.grid and map_feature
        query = self.emb(sample)
        x,_ = self.cross_attention(query,
                                   self.x.unsqueeze(1),
                                   self.x.unsqueeze(1),
                                   )
        cost = self.transformer(x)
        return cost

class TConvCostVolumeDecoder(nn.Module):
    def __init__(self, bev_channels, cost_filter_sizes, cost_filter_nums, planning_horizon):
        super(TConvCostVolumeDecoder, self).__init__()
        cfg = load_cfg_here()
        self.bev_channels = bev_channels
        self.cost_filter_sizes = cost_filter_sizes
        self.cost_filter_nums = cost_filter_nums
        self.planning_horizon = planning_horizon
        self.mode_num = cfg['model_cfg']['mode_num'] if cfg['model_cfg']['mode_num'] else 3

        # embedding layers
        self.reduce = nn.Sequential(nn.Dropout(0.1), nn.Linear(512, 256), nn.ELU())
        self.embedding = nn.Sequential(nn.Dropout(0.1), nn.Linear(512*self.mode_num, 11000), nn.ELU())

        # Two add on deconvolution layers with stride 2
        self.deconv01 = nn.ConvTranspose2d(10, 64, 3, stride=2)
        self.bn01 = nn.BatchNorm2d(64)
        self.deconv02 = nn.ConvTranspose2d(64, 256, 3, stride=2)
        self.bn02 = nn.BatchNorm2d(256)

        # Two deconvolution layers with stride 2
        self.deconv1 = nn.ConvTranspose2d(self.bev_channels, self.cost_filter_nums[0], self.cost_filter_sizes[0], stride=2)
        self.bn1 = nn.BatchNorm2d(self.cost_filter_nums[0])
        self.deconv2 = nn.ConvTranspose2d(self.cost_filter_nums[0], self.cost_filter_nums[1], self.cost_filter_sizes[0], stride=2)
        self.bn2 = nn.BatchNorm2d(self.cost_filter_nums[1])

        # # Two convolution layers with stride 1
        # self.conv1 = nn.Conv2d(self.cost_filter_nums[1], self.cost_filter_nums[0], self.cost_filter_sizes[0], stride=1, padding=1)
        # self.bn3 = nn.BatchNorm2d(self.cost_filter_nums[0])
        # self.conv2 = nn.Conv2d(self.cost_filter_nums[0], self.cost_filter_nums[1], self.cost_filter_sizes[1], stride=1, padding=1)
        # self.bn4 = nn.BatchNorm2d(self.cost_filter_nums[1])

        # Final convolution layer with filter number T
        # self.cost_layer = nn.Conv2d(self.cost_filter_nums[1], self.planning_horizon, 1, stride=1, padding=0)
        self.cost_layer = nn.Conv2d(64, self.planning_horizon, 1, stride=1, padding=0)
        
        
    def set_latent_feature(self, map_feature, agent_map, agent_agent):
        # self._map_feature = map_feature
        # self._masks = map_masks[:,0]
        # self._agent_feature = agent_feature
        map_feature = map_feature.view(map_feature.shape[0], -1, map_feature.shape[-1])
        map_feature = torch.max(map_feature, dim=1)[0]
        agent_agent = torch.max(agent_agent, dim=1)[0]
        agent_map = torch.max(agent_map, dim=2)[0]
        feature = torch.cat([map_feature, agent_agent], dim=-1)
        feature = self.reduce(feature.detach())
        feature = torch.cat([feature.unsqueeze(1).repeat(1, self.mode_num, 1), agent_map.detach()], dim=-1)
        x = self.embedding(feature.reshape(feature.shape[0], -1))
        x = x.reshape(x.shape[0],10,44,25)
        
        self.cost_volume = self.forward(x)
        
    def forward(self, x):
        # add on decovolution layers
        x = self.bn01(self.deconv01(x))
        x = self.bn02(self.deconv02(x))
        # Deconvolution layers
        x = self.bn1(self.deconv1(x))
        x = self.bn2(self.deconv2(x))

        # # Convolution layers
        # x = self.bn3(self.conv1(x))
        # x = F.relu(x)
        # x = self.bn4(self.conv2(x))
        # x = F.relu(x)

        # Final cost volume layer
        c = self.cost_layer(x)

        # Clip cost volume values between -1000 to +1000
        c = torch.clamp(c, -1000, 1000)
        # directly trim the cost volume be in the same shape with the Lidar data
        return c[...,8:712,8:408]

class STCostMap(nn.Module):
    def __init__(self, th = 50) -> None:
        super(STCostMap, self).__init__()
        """
        t resolution is 0.1
        """
        self.th = th
        self.real_t = 5.0
        self.resolution_t = 0.1
        self.rear_range = 70.4
        self.front_range = 70.4
        self.side_range = 40.
        self.steps_s = int(50)
        self.steps_l = int(25)
        self.resolution_s = (self.front_range+self.rear_range)/self.steps_s
        self.resolution_l = 2*self.side_range/self.steps_l
        # make grid
        s = torch.linspace(-self.rear_range,self.front_range,self.steps_s)
        l = torch.linspace(-self.side_range,self.side_range,self.steps_l)
        t = torch.linspace(self.resolution_t, self.real_t, self.th) # 5.0 s-->50 steps
        x,y,t = torch.meshgrid(s,l,t,indexing='xy')
        emb_t = (t/self.th)*2*torch.pi
        self.grid_points = torch.stack([ x, y, emb_t, torch.sin(emb_t), torch.sin(emb_t)],dim=-1)
        self.cost_map = None
        self.cost = None #CostMapDecoder()

    def metric2index(self, s, l, t):
        idx_s = torch.floor((s+self.rear_range)/self.resolution_s).clip(0,self.steps_s-1)
        idx_l = torch.floor((l+self.side_range)/self.resolution_l).clip(0,self.steps_l-1)
        idx_t = torch.floor((t-self.resolution_t)/self.resolution_t)
        idx = idx_s+idx_l*self.steps_s+idx_t*self.steps_l*self.steps_s
        return idx.to(dtype=long).clip(0,self.steps_s*self.steps_l-1)
    
    def get_cost_by_pos(self,samples, *args):
        samples = samples['X']
        btsz, sample, th, dim = samples.shape
        query = time_embeding(samples[...,:2])
        cost = self.cost(query.reshape(btsz, -1, query.shape[-1]))
        return cost.reshape(btsz, sample, th, 1)
    
    def plot(self, samples = None, interval = 1):
        L, S, T, d = self.grid_points.shape
        with torch.no_grad():
            cost = self.cost(self.grid_points.to(samples.device).view(1,-1,5).repeat(samples.shape[0],1,1))[0]
        # fig = plt.figure()
        # ax = fig.add_subplot(projection='3d')
        # ax.scatter(self.grid_points[0,::interval,0].cpu().detach(), 
        #            self.grid_points[0,::interval,1].cpu().detach(),
        #            self.grid_points[0,::interval,2].cpu().detach(),
        #            c = cost.reshape(-1)[::interval].cpu().detach(),
        #            alpha=0.1,
        #            label='cost value')
        grid_points = self.grid_points[:,:,::interval,:].cpu().detach()
        cost = cost.reshape(L,S,T,1)[:,:,::interval,:].cpu().detach()
        for t in range(T):
            cost_t = cost[:,:,t,0].cpu().detach()
            c = cost_t[cost_t<cost_t.mean()]
            plt.scatter(grid_points[...,t,0][cost_t<cost_t.mean()].cpu().detach(), 
                        grid_points[...,t,1][cost_t<cost_t.mean()].cpu().detach(),
                        c = c,
                        alpha=0.05,
                        label=f'time = {t*self.resolution_t}'
                        )
        
    # def plot_gt(self, gt, samples):
    #     diff_sample_gt = 0
    #     dis_diff = gt[...,:2]-torch.cat([torch.zeros_like(samples[...,0:1,:2],device=samples.device), 
    #                                     samples[...,:-1,:2]],
    #                                     dim=-2)
    #     d_dis_diff = dis_diff/0.1 #Time interval
    #     diff_sample_gt+=d_dis_diff
    #     sample_cost = torch.stack([torch.cos(samples[...,2])*samples[...,3], 
    #                               torch.sin(samples[...,2])*samples[...,3]],
    #                               dim=-1)
    #     # diff_sample_gt += gt[...,3:5] - sample_dxy
    #     # plt.scatter(samples[0,...,0].cpu().detach(),
    #     #     samples[0,...,1].cpu().detach())
    #     plt.scatter3D(samples[0,...,0].cpu().detach(),
    #         samples[0,...,1].cpu().detach(),
    #         samples[0,...,2].cpu().detach(),
    #         # diff_sample_gt[0,...,0].cpu().detach(),
    #         # diff_sample_gt[0,...,1].cpu().detach(), 
    #         c = sample_cost[...,0],
    #         # color = 'green',
    #         label='GT Vector')


class AttCostMap(STCostMap):
    def __init__(self, th=50) -> None:
        super().__init__(th)
        """
        t resolution = 0.5s
        """
        self.resolution_t = 0.5
        self.th = int(self.real_t/self.resolution_t)
        self.steps_s = int(50)
        self.steps_l = int(25)
        self.cost = CostMapDecoder()
        s = torch.linspace(-self.rear_range,self.front_range,self.steps_s)
        l = torch.linspace(-self.side_range,self.side_range,self.steps_l)
        t = torch.linspace(self.resolution_t, self.real_t, self.th) # 5.0 s-->50 steps
        x,y,t = torch.meshgrid(s,l,t,indexing='xy')
        emb_t = (t/self.th)*2*torch.pi
        self.grid_points = torch.stack([ x, y, emb_t, torch.sin(emb_t), torch.sin(emb_t)],dim=-1)
        
        
        
class TConvCostMap(STCostMap):
    def __init__(self, th=50) -> None:
        super().__init__(th)
        self.resolution_t = 0.5
        self.th = int(self.real_t/self.resolution_t)
        self.steps_s = int(704)
        self.steps_l = int(400)
        self.cost = TConvCostVolumeDecoder(
                            bev_channels=256, # 176*100
                            cost_filter_sizes=[3,3],
                            cost_filter_nums=[128,64],
                            planning_horizon=self.th,
                            )
        # only for visulization
        s = torch.linspace(-self.rear_range,self.front_range,self.steps_s)[::10]
        l = torch.linspace(-self.side_range,self.side_range,self.steps_l)[::10]
        t = torch.linspace(self.resolution_t, self.real_t, self.th)
        x,y,t = torch.meshgrid(s,l,t,indexing='xy')
        emb_t = (t/self.th)*2*torch.pi
        self.grid_points = torch.stack([ x, y, emb_t, torch.sin(emb_t), torch.sin(emb_t)],dim=-1)

    def metric2index(self, s, l, t):
        idx_s = torch.round((s+self.rear_range)/self.resolution_s).clip(0,self.steps_s-1)
        idx_l = torch.round((l+self.side_range)/self.resolution_l).clip(0,self.steps_l-1)
        idx_t = torch.round((t-self.resolution_t)/self.resolution_t)
        return torch.stack([idx_t,idx_s,idx_l],dim=-1).to(dtype=long)

    def get_cost_by_pos(self,samples, *args):
        samples = samples['X']
        btsz, sample, th, dim = samples.shape
        # t = self.t.unsqueeze(-1).repeat(btsz,sample,1,1).reshape(btsz,-1).to(samples.device)
        t = torch.linspace(self.real_t/th, self.real_t, th).repeat(btsz,sample,1,1).reshape(btsz,-1).to(samples.device)
        # inquiry utils
        samples = samples.view(btsz,-1,dim)
        query = self.metric2index(samples[...,0], samples[...,1], t)
        cost = []
        for i in range(query.shape[0]):
            cost.append(self.cost.cost_volume[i,query[i,...,0],query[i,...,1],query[i,...,2]])
        cost = torch.stack(cost, dim=0)
        return cost.view(btsz, sample, th, 1)
    
    def plot(self, samples = None, interval = 1):
        L, S, T, d = self.grid_points[:,:,::interval,:].shape
        grid_points = self.grid_points[:,:,::interval,:].cpu().detach()
        cost = self.cost.cost_volume[0].cpu().detach().permute(2,1,0)[::10,::10,::interval]
        for t in range(T):
            cost_t = cost[:,:,t].cpu().detach()
            c = cost_t[cost_t<cost_t.mean()]
            plt.scatter(grid_points[...,t,0][cost_t<cost_t.mean()].cpu().detach(), 
                        grid_points[...,t,1][cost_t<cost_t.mean()].cpu().detach(),
                        c = c,
                        alpha=0.05,
                        label=f'time = {t*self.resolution_t}'
                        )
