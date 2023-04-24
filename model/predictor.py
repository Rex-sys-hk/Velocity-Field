from model.predictor_base import AVDecoder, AVDecoderNc, AttCostMap, PreABC, TConvCostMap, VAEcore, VectorField
from .meter2risk import CostModules, Meter2Risk 

# Build predictor
class BasePre(PreABC):
    def __init__(self, name:str = 'base', future_steps = 50, mode_num = 3, gamma = 1.):
        super().__init__(name, future_steps, mode_num, gamma)
        self.name = 'base'
        if self.name != name:
            print('[WARNING]: Module name and config name different')
            print(f'set name is {name}')
            print(f'module name is {self.name}')
        # decode layers
        self.plan = AVDecoderNc(self._future_steps)

    def forward(self, ego, neighbors, map_lanes, map_crosswalks):
        (map_feature, agent_map, 
        agent_agent, plans, 
        predictions, scores, 
        cost_function_weights) = self.forward_base(ego, neighbors, map_lanes, map_crosswalks)
        
        return plans, predictions, scores, cost_function_weights

# Build predictor
class Predictor(PreABC):
    def __init__(self, name:str = 'dipp', future_steps = 50, mode_num = 3, gamma = 1.):
        super().__init__(name, future_steps, mode_num, gamma)
        self.name = 'dipp'
        if self.name != name:
            print('[WARNING]: Module name and config name different')
            print(f'set name is {name}')
            print(f'module name is {self.name}')

        # decode layers
        self.plan = AVDecoder(self._future_steps)

    def forward(self, ego, neighbors, map_lanes, map_crosswalks):
        (map_feature, agent_map, 
        agent_agent, plans, 
        predictions, scores, 
        cost_function_weights) = self.forward_base(ego, neighbors, map_lanes, map_crosswalks)
        
        return plans, predictions, scores, cost_function_weights

    
class RiskMapPre(PreABC):
    def __init__(self, name:str = 'risk', future_steps = 50, mode_num = 3, gamma = 1.):
        super().__init__(name, future_steps, mode_num, gamma)

        self.name = 'risk'
        if self.name != name:
            print('[WARNING]: Module name and config name different')
            print(f'set name is {name}')
            print(f'module name is {self.name}')
        # decode layers
        self.plan = AVDecoderNc(self._future_steps)

        self.meter2risk: Meter2Risk = CostModules[self.cfg['planner']['meter2risk']['name']]()
        self.vf_map = VectorField()

    def forward(self, ego, neighbors, map_lanes, map_crosswalks):
        (map_feature, agent_map, 
        agent_agent, plans, 
        predictions, scores, 
        cost_function_weights) = self.forward_base(ego, neighbors, map_lanes, map_crosswalks)
        # to be compatible with risk map
        latent_feature = {'map_feature':map_feature,'agent_map':agent_map,'agent_agent':agent_agent}
        self.meter2risk.set_latent_feature(latent_feature)
        self.vf_map.vf_inquery.set_latent_feature(latent_feature)
        
        return plans, predictions, scores, cost_function_weights
    
class EularPre(PreABC):
    def __init__(self, name:str = 'esp', future_steps = 50, mode_num = 3, gamma = 1.):
        super().__init__(name, future_steps, mode_num, gamma)

        self.name = 'esp'
        if self.name != name:
            print('[WARNING]: Module name and config name different')
            print(f'set name is {name}')
            print(f'module name is {self.name}')

        # decode layers
        self.plan = AVDecoderNc(self._future_steps)

        self.meter2risk: Meter2Risk = CostModules[self.cfg['planner']['meter2risk']['name']]()

    def forward(self, ego, neighbors, map_lanes, map_crosswalks):
        (map_feature, agent_map, 
        agent_agent, plans, 
        predictions, scores, 
        cost_function_weights) = self.forward_base(ego, neighbors, map_lanes, map_crosswalks)
        # to be compatible with risk map
        latent_feature = {'map_feature':map_feature,'agent_map':agent_map}
        self.meter2risk.set_latent_feature(latent_feature)
        
        return plans, predictions, scores, cost_function_weights

class CostVolume(PreABC):
    def __init__(self, name:str = 'nmp', future_steps = 50, mode_num = 3, gamma = 1.):
        super().__init__(name, future_steps, mode_num, gamma)

        self.name = 'nmp'
        if self.name != name:
            print('[WARNING]: Module name and config name different') 
            print(f'set name is {name}')
            print(f'module name is {self.name}')

        # decode layers
        self.plan = AVDecoderNc(self._future_steps)

        if self.cfg['planner']['conv_decoder']:
            self.cost_volume = TConvCostMap(self._future_steps)
        else:
            self.cost_volume = AttCostMap(self._future_steps)
        self.meter2risk: Meter2Risk = None#CostModules[cfg['planner']['meter2risk']['name']]()

    def forward(self, ego, neighbors, map_lanes, map_crosswalks):
        (map_feature, agent_map, 
        agent_agent, plans, 
        predictions, scores, 
        cost_function_weights) = self.forward_base(ego, neighbors, map_lanes, map_crosswalks)
        # plan + prediction 
        self.cost_volume.cost.set_latent_feature(map_feature, agent_map, agent_agent)
        return plans, predictions, scores, cost_function_weights

if __name__ == "__main__":
    # set up model
    model = Predictor(50)
    print(model)
    print('Model Params:', sum(p.numel() for p in model.parameters()))
