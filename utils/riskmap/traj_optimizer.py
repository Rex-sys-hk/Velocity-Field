 ## DEPRACTED
from casadi import *
import ml_casadi.torch as mc
from typing import Sequence, List
import sys
import torch

sys.path.append('/host/DIPP')
from utils.riskmap.rm_utils import Mlp
from utils.data_augmentation.data_augmentation_util import ConstrainedNonlinearSmoother, Pose
from model.predictor_base import VFMapDecoder, VectorField
import numpy as np
os.environ["DIPP_CONFIG"] = str('/host/DIPP' + '/' + 'config_risk.yaml')


class TrajOptimizer(ConstrainedNonlinearSmoother):
    def __init__(self, trajectory_len: int, dt: float, cost_map: VectorField):

        self._cost_map = cost_map
        super().__init__(trajectory_len, dt)
        
    
    def _init_optimization(self) -> None:
        """
        Initialize related variables and constraints for optimization.
        """
        self.nx = 4  # state dim
        self.nu = 2  # control dim

        self._optimizer = Opti()  # Optimization problem
        self._create_decision_variables()
        self._create_parameters()
        self._set_dynamic_constraints()
        self._set_state_constraints()
        self._set_control_constraints()
        self._set_torch_model() # TODO
        self._set_objective()

        # Set default solver options (quiet)
        self._optimizer.solver("ipopt", {"ipopt.print_level": 0, "print_time": 0, "ipopt.sb": "yes"})
    def _set_objective(self) -> None:
        """Set the objective function. Use care when modifying these weights."""
        # Follow reference, minimize control rates and absolute inputs
        alpha_xy = 1.0
        alpha_yaw = 0.1
        alpha_rate = 0.08
        alpha_abs = 0.08
        alpha_lat_accel = 0.06
        # dx_dy = self.f_map(self.state[:2, :])
        dx_dy = self.cm_output
        vf_error = (dx_dy[:,0].T-cos(self.yaw[1:])*self.speed[1:])**2 \
            + (dx_dy[:,1].T-sin(self.yaw[1:])*self.speed[1:])**2
        cost_stage = (
            # alpha_xy * sumsqr(self.ref_traj[:2, :] - vertcat(self.position_x, self.position_y))
            # + alpha_yaw * sumsqr(self.ref_traj[2, :] - self.yaw)
            sumsqr(vf_error)
            + alpha_rate * (sumsqr(self.curvature_rate) + sumsqr(self.jerk))
            + alpha_abs * (sumsqr(self.curvature) + sumsqr(self.accel))
            + alpha_lat_accel * sumsqr(self.lateral_accel)
        )
        
        # Take special care with the final state
        alpha_terminal_xy = 1.0
        alpha_terminal_yaw = 40.0  # really care about final heading to help with lane changes
        cost_terminal = 0
        # cost_terminal = alpha_terminal_xy * sumsqr(
        #     self.ref_traj[:2, -1] - vertcat(self.position_x[-1], self.position_y[-1])
        # ) + alpha_terminal_yaw * sumsqr(self.ref_traj[2, -1] - self.yaw[-1])

        self._optimizer.minimize(cost_stage)
        
    def _set_initial_guess(self, x_curr: Sequence[float], initial_guess: Sequence[Pose]) -> None:
        """Set a warm-start for the solver based on the reference trajectory."""
        self._check_inputs(x_curr, initial_guess)

        # Initialize state guess based on reference
        self._optimizer.set_initial(self.state[:3, :], DM(initial_guess).T)  # (x, y, yaw)
        self._optimizer.set_initial(self.state[3, :], DM(x_curr[3]))  # speed

        # I think initializing the controls would be quite noisy, so using default zero init
        
    def set_reference_trajectory(self, x_curr: Sequence[float], init_guess: Sequence[Pose]) -> None:
        """
        Set the reference trajectory that the smoother is trying to loosely track.

        :param x_curr: current state of size nx (x, y, yaw, speed)
        :param reference_trajectory: N+1 x 3 reference, where the second dim is for (x, y, yaw)
        """
        self._check_inputs(x_curr, init_guess)

        self._optimizer.set_value(self.x_curr, DM(x_curr))
        # self._optimizer.set_value(self.ref_traj, DM(init_guess).T)
        self._set_initial_guess(x_curr, init_guess)
        
    def _create_parameters(self) -> None:
        """
        Define the expert trjactory and current position for the trajectory optimizaiton.
        """
        # self.ref_traj = self._optimizer.parameter(3, self.trajectory_len + 1)  # (x, y, yaw)
        self.x_curr = self._optimizer.parameter(self.nx, 1)
        
        
    # def _pytorch_to_casadi(self, traj):
    #     traj = torch.tensor(traj.full())
    #     traj = traj.permute(1,0)
    #     dx_dy = self._cost_map.vf_inquery(traj)
    #     dm_dx_dy = DM(dx_dy.detach().numpy())
    #     return dm_dx_dy
    
    def _set_torch_model(self):
        self._cm_modle = mc.TorchMLCasadiModuleWrapper(
            self._cost_map,
            input_size = 50*2,
            output_size = 50*2,
        )
        self.cm_output = self._optimizer.variable(50,2)
        # TODO here is to approximate the MPC model, not applicatable to our case
        # ERROR   Opti parameter 'opti1_p_2' of shape 100x1, defined at /host/DIPP/utils/riskmap/ml_casadi/common/module.py:81 in get_sym_approx_params_list
        # a = self._optimizer.parameter( self.input_size, 1)
        self.cm_output = self._cm_modle.approx(self.state[:2,1:].T.reshape((100,1)),
                                               order=1,
                                               optimizer=self._optimizer).reshape((50,2))
        # self._f_map = Function('f_map', 
        #                        [self.state], [self.cm_output])
        
if __name__ == "__main__":
    # print(traj.shape)
    model = VectorField()
    map_feature = torch.randn([1,256])
    agent_map = torch.randn([1,256])
    agent_agent = torch.randn([1,256])
    model.vf_inquery.set_latent_feature({'map_feature':map_feature,'agent_map':agent_map,'agent_agent':agent_agent})
    
    to = TrajOptimizer(50 ,0.1, model)
    x = np.linspace(0, 50, 51)[:,None]
    y = np.zeros_like(x)
    yaw = np.zeros_like(x)
    traj = np.concatenate([x, y, yaw],axis = -1)

    to.set_reference_trajectory([0, 0, 0, 1], traj)
    sol = to.solve()
    ego_perturb: List[np.float32] = np.vstack(
    [
        sol.value(to.position_x),
        sol.value(to.position_y),
        sol.value(to.yaw),
    ])
    print(ego_perturb)