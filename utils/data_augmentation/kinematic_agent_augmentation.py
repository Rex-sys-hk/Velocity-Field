import logging
from typing import Any, List, Optional, Tuple, TypeVar, cast
import copy
import numpy as np
import torch
from model.planner import get_sample
# from nuplan_utils.abstract_scenario import AbstractScenario
from utils.data_augmentation.abstract_data_augmentation import (AbstractAugmentor,
                                        FeaturesType, 
                                        TargetsType)
from utils.data_augmentation.data_augmentation_util import (
    ConstrainedNonlinearSmoother,
    GaussianNoise,
    ParameterToScale,
    ScalingDirection,
    UniformNoise,
)
from utils.riskmap.rm_utils import get_u_from_X
# from nuplan.planning.training.modeling.types import FeaturesType, TargetsType

logger = logging.getLogger(__name__)


class KinematicAgentAugmentor(AbstractAugmentor):
    """
    Data augmentation that perturbs the current ego position and generates a feasible future trajectory that
    satisfies a set of kinematic constraints.

    This involves constrained minimization of the following objective:
    * minimize dist(perturbed_trajectory, ground_truth_trajectory)
    """

    def __init__(
        self,
        trajectory_length: int,
        dt: float,
        mean: List[float],
        std: List[float],
        low: List[float],
        high: List[float],
        augment_prob: float,
        use_uniform_noise: bool = False,
    ) -> None:
        """
        Initialize the augmentor.
        :param trajectory_length: Length of trajectory to be augmented.
        :param dt: Time interval between trajecotry points.
        :param mean: Parameter to set mean vector of the Gaussian noise on [x, y, yaw].
        :param std: Parameter to set standard deviation vector of the Gaussian noise on [x, y, yaw].
        :param low: Parameter to set lower bound vector of the Uniform noise on [x, y, yaw]. Used only if use_uniform_noise == True.
        :param high: Parameter to set upper bound vector of the Uniform noise on [x, y, yaw]. Used only if use_uniform_noise == True.
        :param augment_prob: Probability between 0 and 1 of applying the data augmentation.
        :param use_uniform_noise: Parameter to decide to use uniform noise instead of gaussian noise if true.
        """
        self._random_offset_generator = UniformNoise(low, high) if use_uniform_noise else GaussianNoise(mean, std)
        self._augment_prob = augment_prob
        self._optimizer = ConstrainedNonlinearSmoother(trajectory_length, dt)
        self.dt = dt

    def augment(
        self, features: FeaturesType, targets: TargetsType, scenario = None
    ) -> Tuple[FeaturesType, TargetsType]:
        """Inherited, see superclass."""
        if np.random.rand() >= self._augment_prob:
            return features, targets
        gt_feature = copy.deepcopy(features)
        # Perturb the current position
        # augment the last frame of ego history and future
        # last time step only
        # features['agents'].ego[0][-1] += self._random_offset_generator.sample()
        # all time steps
        ego_his = torch.tensor(features['agents'].ego)
        v = torch.diff(ego_his,dim=-2)/self.dt
        v = torch.cat([v[:,0:1], v], dim=-2) # padding
        ego_his = torch.cat([ego_his, v],dim=-1)
        context = {'init_guess_u': get_u_from_X(ego_his[:,1:], ego_his[:,0]),
                   'current_state': ego_his[:,0:1]}
        aug_his = get_sample(context, cov = torch.tensor([.4,.2]), sample_num=1, turb_num=19)
        features['agents'].ego[:,1:] = aug_his['X'][:,0,:,:3].numpy()
        

        ego_trajectory = np.concatenate(
            [features['agents'].ego[0][-1:, :3], targets['trajectory'].data]
        )
        ego_x, ego_y, ego_yaw = ego_trajectory.T
        ego_velocity = np.linalg.norm(np.diff(ego_trajectory[:, :2], axis=0), axis=1)/self.dt # didn't divided by dt brefore

        # Define the 'current state' as a boundary condition, and reference trajectory
        x_curr = [ego_x[0], ego_y[0], ego_yaw[0], ego_velocity[0]]
        ref_traj = ego_trajectory

        # Set reference and solve
        self._optimizer.set_reference_trajectory(x_curr, ref_traj)

        try:
            sol = self._optimizer.solve()
        except RuntimeError:
            logger.info("Smoothing failed! Use G.T. instead" )
            return gt_feature, targets

        if not sol.stats()['success']:
            logger.info(f"Smoothing failed with status {sol.stats()['return_status']}! Use G.T. instead")
            return gt_feature, targets

        ego_perturb: List[np.float32] = np.vstack(
            [
                sol.value(self._optimizer.position_x),
                sol.value(self._optimizer.position_y),
                sol.value(self._optimizer.yaw),
            ]
        )
        ego_perturb = ego_perturb.T

        features["agents"].ego[0][-1] = np.float32(ego_perturb[0])
        targets["trajectory"].data = np.float32(ego_perturb[1:])

        return features, targets

    @property
    def required_features(self) -> List[str]:
        """Inherited, see superclass."""
        return ['agents']

    @property
    def required_targets(self) -> List[str]:
        """Inherited, see superclass."""
        return ['trajectory']

    @property
    def augmentation_probability(self) -> ParameterToScale:
        """Inherited, see superclass."""
        return ParameterToScale(
            param=self._augment_prob,
            param_name=f'{self._augment_prob=}'.partition('=')[0].split('.')[1],
            scaling_direction=ScalingDirection.MAX,
        )

    @property
    def get_schedulable_attributes(self) -> List[ParameterToScale]:
        """Inherited, see superclass."""
        return cast(List[ParameterToScale], self._random_offset_generator.get_schedulable_attributes())
