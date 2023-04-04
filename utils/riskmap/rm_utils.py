from re import L
from matplotlib.patches import Ellipse
import matplotlib.pyplot as plt
import torch.distributions as td
from scipy.spatial.distance import cdist
import psutil
from sklearn.cluster import KMeans
import numpy as np
import torch
import yaml
import os
import sys
import math
from torch import nn, true_divide

from utils.train_utils import project_to_frenet_frame
from .car import WB, bicycle_model, pi_2_pi, pi_2_pi_pos
from math import cos, sin, tan, pi
# Initialize device:
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# Initialize ray:
num_cpus = psutil.cpu_count(logical=False)
# ray.init(num_cpus=num_cpus, log_to_driver=False)
def correct_yaw(yaw: float) -> float:
    """
    nuScenes maps were flipped over the y-axis, so we need to
    add pi to the angle needed to rotate the heading.
    :param yaw: Yaw angle to rotate the image.
    :return: Yaw after correction.
    """
    if yaw <= 0:
        yaw = -np.pi - yaw
    else:
        yaw = np.pi - yaw

    return yaw

def k_means_anchors(k, ds):
    """
    Extracts anchors for multipath/covernet using k-means on train set trajectories
    """
    prototype_traj = ds[0]['ground_truth']['traj']
    traj_len = prototype_traj.shape[0]
    traj_dim = prototype_traj.shape[1]
    ds_size = len(ds)
    trajectories = np.zeros((ds_size, traj_len, traj_dim))
    for i, data in enumerate(ds):
        trajectories[i] = data['ground_truth']['traj']
    clustering = KMeans(n_clusters=k).fit(trajectories.reshape((ds_size, -1)))
    anchors = np.zeros((k, traj_len, traj_dim))
    for i in range(k):
        anchors[i] = np.mean(trajectories[clustering.labels_ == i], axis=0)
    anchors = torch.from_numpy(anchors).float().to(device)
    return anchors


def bivariate_gaussian_activation(ip: torch.Tensor) -> torch.Tensor:
    """
    Activation function to output parameters of bivariate Gaussian distribution
    """
    mu_x = ip[..., 0:1]
    mu_y = ip[..., 1:2]
    sig_x = ip[..., 2:3]
    sig_y = ip[..., 3:4]
    sig_x = torch.exp(sig_x)
    sig_y = torch.exp(sig_y)
    rho = ip[..., 4:5]
    rho = torch.tanh(rho)
    out = torch.cat([mu_x, mu_y, sig_x, sig_y, rho], dim=-1)
    return out


# @ray.remote
def cluster_and_rank(k: int, data: np.ndarray):
    """
    Combines the clustering and ranking steps so that ray.remote gets called just once
    """

    def cluster(n_clusters: int, x: np.ndarray):
        """
        Cluster using Scikit learn
        """
        clustering_op = KMeans(n_clusters=n_clusters,
                               n_init=1, max_iter=100, init='random').fit(x)
        return clustering_op.labels_, clustering_op.cluster_centers_

    def rank_clusters(cluster_counts, cluster_centers):
        """
        Rank the K clustered trajectories using Ward's criterion. Start with K cluster centers and cluster counts.
        Find the two clusters to merge based on Ward's criterion. Smaller of the two will get assigned rank K.
        Merge the two clusters. Repeat process to assign ranks K-1, K-2, ..., 2.
        """

        num_clusters = len(cluster_counts)
        cluster_ids = np.arange(num_clusters)
        ranks = np.ones(num_clusters)

        for i in range(num_clusters, 0, -1):
            # Compute Ward distances:
            centroid_dists = cdist(cluster_centers, cluster_centers)
            n1 = cluster_counts.reshape(
                1, -1).repeat(len(cluster_counts), axis=0)
            n2 = n1.transpose()
            wts = n1 * n2 / (n1 + n2)
            dists = wts * centroid_dists + \
                np.diag(np.inf * np.ones(len(cluster_counts)))

            # Get clusters with min Ward distance and select cluster with fewer counts
            c1, c2 = np.unravel_index(dists.argmin(), dists.shape)
            c = c1 if cluster_counts[c1] <= cluster_counts[c2] else c2
            c_ = c2 if cluster_counts[c1] <= cluster_counts[c2] else c1

            # Assign rank i to selected cluster
            ranks[cluster_ids[c]] = i

            # Merge clusters and update identity of merged cluster
            cluster_centers[c_] = (cluster_counts[c_] * cluster_centers[c_] + cluster_counts[c] * cluster_centers[c]) /\
                                  (cluster_counts[c_] + cluster_counts[c])
            cluster_counts[c_] += cluster_counts[c]

            # Discard merged cluster
            cluster_ids = np.delete(cluster_ids, c)
            cluster_centers = np.delete(cluster_centers, c, axis=0)
            cluster_counts = np.delete(cluster_counts, c)

        return ranks

    cluster_lbls, cluster_ctrs = cluster(k, data)
    cluster_cnts = np.unique(cluster_lbls, return_counts=True)[1]
    cluster_ranks = rank_clusters(cluster_cnts.copy(), cluster_ctrs.copy())
    return {'lbls': cluster_lbls, 'ranks': cluster_ranks, 'counts': cluster_cnts}

# TODO
def cluster_traj(k: int, traj: torch.Tensor):
    """
    clusters sampled trajectories to output K modes.
    :param k: number of clusters
    :param traj: set of sampled trajectories, shape [batch_size, num_samples, traj_len, 2]
    :return: traj_clustered:  set of clustered trajectories, shape [batch_size, k, traj_len, 2]
             scores: scores for clustered trajectories (basically 1/rank), shape [batch_size, k]
    """

    # Initialize output tensors
    batch_size = traj.shape[0]
    num_samples = traj.shape[1]
    traj_len = traj.shape[2]

    # Down-sample traj along time dimension for faster clustering
    data = traj[:, :, 0::3, :]
    data = data.reshape(batch_size, num_samples, -1).detach().cpu().numpy()

    # Cluster and rank
    cluster_ops = ray.get([cluster_and_rank.remote(k, data_slice)
                          for data_slice in data])
    cluster_lbls = [cluster_op['lbls'] for cluster_op in cluster_ops]
    cluster_counts = [cluster_op['counts'] for cluster_op in cluster_ops]
    cluster_ranks = [cluster_op['ranks'] for cluster_op in cluster_ops]

    # Compute mean (clustered) traj and scores
    lbls = torch.as_tensor(cluster_lbls, device=device).unsqueeze(
        -1).unsqueeze(-1).repeat(1, 1, traj_len, 2).long()
    traj_summed = torch.zeros(batch_size, k, traj_len,
                              2, device=device).scatter_add(1, lbls, traj)
    cnt_tensor = torch.as_tensor(
        cluster_counts, device=device).unsqueeze(-1).unsqueeze(-1).repeat(1, 1, traj_len, 2)
    traj_clustered = traj_summed / cnt_tensor
    scores = 1 / torch.as_tensor(cluster_ranks, device=device)
    scores = scores / torch.sum(scores, dim=1)[0]

    return traj_clustered, scores


def has_nan(input:torch.Tensor):
    # if (input==0).all():
    #     raise ValueError('all 0')
    if torch.isnan(input).all():
        raise ValueError('all nan')
    if torch.isinf(input).all():
        raise ValueError('all inf')
    if torch.isnan(input).any():
        raise ValueError('has nan')
    if torch.isinf(input).any():
        raise ValueError('has inf')
    # return torch.isnan(input).any()

class GMM2D(td.Distribution):
    r"""
    Gaussian Mixture Model using 2D Multivariate Gaussians each of as N components:
    Cholesky decompesition and affine transformation for sampling:

    .. math:: Z \sim N(0, I)

    .. math:: S = \mu + LZ

    .. math:: S \sim N(\mu, \Sigma) \rightarrow N(\mu, LL^T)

    where :math:`L = chol(\Sigma)` and

    .. math:: \Sigma = \left[ {\begin{array}{cc} \sigma^2_x & \rho \sigma_x \sigma_y \\ \rho \sigma_x \sigma_y & \sigma^2_y \\ \end{array} } \right]

    such that

    .. math:: L = chol(\Sigma) = \left[ {\begin{array}{cc} \sigma_x & 0 \\ \rho \sigma_y & \sigma_y \sqrt{1-\rho^2} \\ \end{array} } \right]

    :param log_pis: Log Mixing Proportions :math:`log(\pi)`. [..., N]
    :param mus: Mixture Components mean :math:`\mu`. [..., N * 2]
    :param log_sigmas: Log Standard Deviations :math:`log(\sigma_d)`. [..., N * 2]
    :param corrs: Cholesky factor of correlation :math:`\rho`. [..., N]
    :param clip_lo: Clips the lower end of the standard deviation.
    :param clip_hi: Clips the upper end of the standard deviation.
    """

    def __init__(self, log_pis, mus, log_sigmas, corrs):
        super(GMM2D, self).__init__(
            batch_shape=log_pis.shape[0], event_shape=log_pis.shape[1:], validate_args=False)
        self.components = log_pis.shape[-1]
        self.dimensions = 2
        self.device = log_pis.device

        log_pis = torch.clamp(log_pis, min=-1e5)
        self.log_pis = log_pis - \
            torch.logsumexp(log_pis, dim=-1, keepdim=True)  # [..., N]
        self.log_sigmas = self.reshape_to_components(log_sigmas)  # [..., N, 2]
        self.pis = torch.exp(self.log_pis)                      # [..., N]
        self.mus = self.reshape_to_components(mus)         # [..., N, 2]
        # [..., N, 2]
        self.sigmas = torch.clamp(torch.exp(self.log_sigmas),min=1e-8,max=1e6)
        self.one_minus_rho2 = 1 - corrs**2                        # [..., N]
        self.one_minus_rho2 = torch.clamp(
            self.one_minus_rho2, min=1e-5, max=1)  # otherwise log can be nan
        self.corrs = torch.clamp(corrs,min=-1,max=1)  # [..., N]

        self.L = torch.stack([torch.stack([self.sigmas[..., 0], torch.zeros_like(self.log_pis)], dim=-1),
                              torch.stack([self.sigmas[..., 1] * self.corrs,
                                           self.sigmas[..., 1] * torch.sqrt(self.one_minus_rho2 + 1e-8)],
                                          dim=-1)],
                             dim=-2)

        self.pis_cat_dist = td.Categorical(logits=log_pis)
        self.cov = self.get_covariance_matrix()

    @classmethod
    def from_log_pis_mus_cov_mats(cls, log_pis, mus, cov_mats):
        corrs_sigma12 = cov_mats[..., 0, 1]
        sigma_1 = torch.clamp(cov_mats[..., 0, 0], min=1e-8)
        sigma_2 = torch.clamp(cov_mats[..., 1, 1], min=1e-8)
        sigmas = torch.stack(
            [torch.sqrt(sigma_1 + 1e-8), torch.sqrt(sigma_2 + 1e-8)], dim=-1)
        log_sigmas = torch.log(sigmas)
        corrs = corrs_sigma12 / (torch.prod(sigmas, dim=-1))
        return cls(log_pis, mus, log_sigmas, corrs)

    def rsample(self, sample_shape=torch.Size()):
        """
        Generates a sample_shape shaped reparameterized sample or sample_shape
        shaped batch of reparameterized samples if the distribution parameters
        are batched.

        :param sample_shape: Shape of the samples
        :return: Samples from the GMM.
        """
        mvn_samples = (self.mus +
                       torch.squeeze(
                           torch.matmul(self.L,
                                        torch.unsqueeze(
                                            torch.randn(
                                                size=sample_shape + self.mus.shape, device=self.device),
                                            dim=-1)
                                        ),
                           dim=-1))
        component_cat_samples = self.pis_cat_dist.sample(sample_shape)
        selector = torch.unsqueeze(to_one_hot(
            component_cat_samples, self.components), dim=-1)
        return torch.sum(mvn_samples*selector, dim=-2)

    def  log_prob(self, value, time_pri = False):
        r"""
        Calculates the log probability of a value using the PDF for bivariate normal distributions:

        .. math::
            f(x | \mu, \sigma, \rho)={\frac {1}{2\pi \sigma _{x}\sigma _{y}{\sqrt {1-\rho ^{2}}}}}\exp
            \left(-{\frac {1}{2(1-\rho ^{2})}}\left[{\frac {(x-\mu _{x})^{2}}{\sigma _{x}^{2}}}+
            {\frac {(y-\mu _{y})^{2}}{\sigma _{y}^{2}}}-{\frac {2\rho (x-\mu _{x})(y-\mu _{y})}
            {\sigma _{x}\sigma _{y}}}\right]\right)

        :param value: The log probability density function is evaluated at those values.
        :return: Log probability
        """
        # x: [..., 2]
        value = torch.unsqueeze(value, dim=-2).repeat(1,1,1,1,self.mus.shape[-2],1)       # [..., 1, 2]

        # if value.shape[2] == self.mus.shape[2] or (time_pri and value.shape[1] == self.mus.shape[1]):  # position 2 is the time duration
        #     dx = value - self.mus                       # [..., N, 2]
        # else:
        dx = value[...,:self.mus.shape[2],:] - self.mus

        exp_nominator = ((torch.sum((dx/self.sigmas)**2, dim=-1)  # first and second term of exp nominator
                          - 2*self.corrs*torch.prod(dx, dim=-1)/torch.prod(self.sigmas, dim=-1)))    # [..., N]

        component_log_p = -(2*np.log(2*np.pi)
                            + torch.log(self.one_minus_rho2)
                            + 2*torch.sum(self.log_sigmas, dim=-1)
                            + exp_nominator/self.one_minus_rho2) / 2

        return torch.logsumexp(self.log_pis + component_log_p, dim=-1)

    def get_for_node_at_time(self, n, t):
        return self.__class__(self.log_pis[:, n:n+1, t:t+1], self.mus[:, n:n+1, t:t+1],
                              self.log_sigmas[:, n:n+1, t:t+1], self.corrs[:, n:n+1, t:t+1])

    def mode(self):
        """
        Calculates the mode of the GMM by calculating probabilities of a 2D mesh grid

        :param required_accuracy: Accuracy of the meshgrid
        :return: Mode of the GMM
        """
        if self.mus.shape[-2] > 1:
            samp, bs, time, comp, _ = self.mus.shape
            assert samp == 1, "WARNING: GMM2D.mode()->For taking the mode only one sample makes sense."
            mode_node_list = []
            for n in range(bs):
                mode_t_list = []
                for t in range(time):
                    nt_gmm = self.get_for_node_at_time(n, t)
                    x_min = self.mus[:, n, t, :, 0].min(
                    ).cpu().detach()  # min of all the samples
                    x_max = self.mus[:, n, t, :, 0].max().cpu().detach()
                    y_min = self.mus[:, n, t, :, 1].min().cpu().detach()
                    y_max = self.mus[:, n, t, :, 1].max().cpu().detach()
                    grid = torch.meshgrid([torch.arange(x_min, x_max, 0.01).to(self.device),
                                           torch.arange(y_min, y_max, 0.01).to(self.device)])
                    search_grid = torch.stack(grid, dim=2
                                              ).view(-1, 2).float().to(self.device)
                    ll_score = nt_gmm.log_prob(search_grid)
                    argmax = torch.argmax(ll_score.squeeze(), dim=0)
                    mode_t_list.append(search_grid[argmax])
                mode_node_list.append(torch.stack(mode_t_list, dim=0))
            return torch.stack(mode_node_list, dim=0).unsqueeze(dim=0)
        return torch.squeeze(self.mus, dim=-2)

    def reshape_to_components(self, tensor):
        if len(tensor.shape) == 5:
            return tensor
        return torch.reshape(tensor, list(tensor.shape[:-1]) + [self.components, self.dimensions])

    def get_covariance_matrix(self):
        cov = self.corrs * torch.prod(self.sigmas, dim=-1)
        E = torch.stack([torch.stack([self.sigmas[..., 0]**2, cov], dim=-1),
                         torch.stack([cov, self.sigmas[..., 1]**2], dim=-1)],
                        dim=-2)
        return E

    def get_vis(self, ax, time_id=None, sample_id=None, transf=None):
        def get_rotation_matrix(yaw):
            return np.array([[math.cos(yaw), -math.sin(yaw)], [math.sin(yaw), math.cos(yaw)]])

        def get_cov_matrix(corr, sigmas):
            sigxy = corr*sigmas[0]*sigmas[1]
            return np.array([[sigmas[0], sigxy], [sigxy, sigmas[1]]])

        def draw_ellipse(position, covariance, ax, node=0, transf=None, **kwargs):
            """Draw an ellipse with a given position and covariance"""
            # Convert covariance to principal axes
            if covariance.shape == (2, 2):
                U, s, Vt = np.linalg.svd(covariance)
                angle = np.arctan2(U[1, 0], U[0, 0])
                width, height = 2 * np.sqrt(s)
            else:
                angle = 0
                # width, height = 2 * np.sqrt(covariance)
                width = covariance[0]
                height = covariance[1]
            if transf is not None:
                # here cuz model default to output variance in two direction rather than covariance,
                # the angle ought to be zero
                angle = angle+transf['rot'][node -
                                            1].detach().cpu().numpy()+transf['center_yaw']
                position = get_rotation_matrix(
                    transf['rot'][node-1].detach().cpu().numpy()).dot(position.T)
                position = transf['tran'][node -
                                          1][:2].detach().cpu().numpy()+position
                position = get_rotation_matrix(
                    math.pi/2-transf['center_yaw']).dot(position.T)
            # Draw the Ellipse
            for nsig in range(1, 5):
                ax.add_patch(Ellipse(position, nsig * width, nsig * height,
                                     np.degrees(correct_yaw(angle)), **kwargs))
        samplen, noden, ts, comp = self.pis.shape
        if sample_id is not None:
            samplen = sample_id
            samp = sample_id
        for samp in range(samplen):
            for node in range(noden):
                for t in range(ts):
                    node_t = self.get_for_node_at_time(node, t)
                    for c in range(comp):
                        cov = get_cov_matrix(node_t.corrs[samp, 0, 0, c].cpu().detach().numpy(),
                                             node_t.sigmas[samp, 0, 0, c].cpu().detach().numpy())
                        draw_ellipse(node_t.mus[samp, 0, 0, c].cpu().detach().numpy(),
                                     cov, ax, transf=transf, node=node,
                                     alpha=float(
                                         node_t.pis[samp, 0, 0, c].cpu().detach().numpy())
                                     )
    def project_to_global(self, transf):
        """
        transf is in shape [b,n,[x,y,yaw]]
        """
        self.mus = self.proj_mus_to_global_pose(self.mus,transf)
        
        cov = self.get_covariance_matrix()
        rot_m = self.get_rotation_mat(
                 transf[...,2])
        # print(rot_m.shape)
        # torch.Size([96, 31, 1, 1, 2, 2])
        # for n, c in enumerate(cov):
        #     rot_m = self.get_rotation_matrix(
        #         self.global_transf['rot'][n])
            # rot_m = rot_m.unsqueeze(0).unsqueeze(
            #     0).repeat(c.shape[0], c.shape[1], 1, 1)
            # L,V = torch.linalg.eig(c)
            # t_L = torch.zeros_like(V)
            # t_L[...,0,0] = L[...,0]
            # t_L[...,1,1] = L[...,1]
            # L = t_L.float()
            # L = (torch.sqrt(torch.abs(L))+self.car_shape_scaler)**2
            # V = V.float()
            # c = torch.matmul(torch.matmul(V,L),V.inverse())
            # t_cov.append(torch.matmul(torch.matmul(
            #     rot_m, c), rot_m.permute(0, 1, 3, 2)).unsqueeze(0))
        # self.gmm_cov = torch.cat(t_cov, dim=0).permute(1, 0, 2, 3, 4).cuda()
        # rot_m = rot_m.unsqueeze(2).unsqueeze(3)
        cov = torch.matmul(torch.matmul(
            rot_m, cov), rot_m.permute(0, 1, 2, 3, 5, 4))
        return self.from_log_pis_mus_cov_mats(self.log_pis,self.mus,cov)

    def proj_mus_to_global_pose(self,state, transf):
        """
        from [0,0,0] view
        """
        cos = torch.cos(transf[...,2])
        sin = torch.sin(transf[...,2])

        state_t = torch.zeros_like(state)
        state_t[...,0] = state[...,0]*cos -state[...,1]*sin +transf[...,0]
        state_t[...,1] = state[...,0]*sin + state[...,1]*cos +transf[...,1]
        if state_t.shape[-1]==3:
            state_t[...,2] = state[...,2]+transf[...,2]
        return state_t

    def get_rotation_mat(self, yaw):
        cos = torch.cos(yaw)
        sin = torch.sin(yaw)
        return torch.stack(
            [torch.stack([cos,-sin],dim=-1),
            torch.stack([sin,cos],dim=-1)],
            dim=-2
        )

def to_one_hot(labels, n_labels):
    return torch.eye(n_labels, device=labels.device)[labels]


def attach_dim(v, n_dim_to_prepend=0, n_dim_to_append=0):
    return v.reshape(
        torch.Size([1] * n_dim_to_prepend)
        + v.shape
        + torch.Size([1] * n_dim_to_append))


def block_diag(m):
    """
    Make a block diagonal matrix along dim=-3
    EXAMPLE:
    block_diag(torch.ones(4,3,2))
    should give a 12 x 8 matrix with blocks of 3 x 2 ones.
    Prepend batch dimensions if needed.
    You can also give a list of matrices.
    :type m: torch.Tensor, list
    :rtype: torch.Tensor
    """
    if type(m) is list:
        m = torch.cat([m1.unsqueeze(-3) for m1 in m], -3)

    d = m.dim()
    n = m.shape[-3]
    siz0 = m.shape[:-3]
    siz1 = m.shape[-2:]
    m2 = m.unsqueeze(-2)
    eye = attach_dim(torch.eye(n, device=m.device).unsqueeze(-2), d - 3, 1)
    return (m2 * eye).reshape(siz0 + torch.Size(torch.tensor(siz1) * n))


def tile(a, dim, n_tile, device='cpu'):
    init_dim = a.size(dim)
    repeat_idx = [1] * a.dim()
    repeat_idx[dim] = n_tile
    a = a.repeat(*(repeat_idx))
    order_index = torch.LongTensor(np.concatenate(
        [init_dim * np.arange(n_tile) + i for i in range(init_dim)])).to(device)
    return torch.index_select(a, dim, order_index)


class SeqMVN2D(td.Distribution):
    def __init__(self,dist_params: torch.Tensor, log_probs: torch.Tensor = None):
        """
        dist params should be in shape [...,p], typically is [b,n,m,s,p]
        where p is [weight, mux, muy, sigma_x, sigma_y, rho]
        """
        # self.raw = dist_params.clone()
        if dist_params.shape[-1]==5:
            if log_probs is None:
                log_probs = torch.log_softmax(torch.ones_like(dist_params)[...,0:1],dim=-3)
                ext_dist_params = torch.cat([log_probs,dist_params],dim=-1)
            else:
                log_probs = log_probs.unsqueeze(-1).unsqueeze(-1).expand_as(dist_params[...,0:1])
                ext_dist_params = torch.cat([log_probs,dist_params],dim=-1)
        else:
            ext_dist_params = dist_params
        self.log_weight = ext_dist_params[...,0:1]
        self.weight = torch.exp(self.log_weight)
        self.mus = ext_dist_params[...,1:3]
        self.sigmas = torch.abs(ext_dist_params[...,3:5]).clamp(1e-5)
        self.log_sigmas = torch.log(self.sigmas)
        self.corr = ext_dist_params[...,5]
        self.one_minus_rho2 = 1-self.corr**2
        # self.cov = self.get_cov_from_corr()
        self.log2pi = 1.8379 # log(1/2pi)


    def log_prob(self, value, dx = None):
        r"""
        Calculates the log probability of a value using the PDF for bivariate normal distributions:

        .. math::
            f(x | \mu, \sigma, \rho)={\frac {1}{2\pi \sigma _{x}\sigma _{y}{\sqrt {1-\rho ^{2}}}}}\exp
            \left(-{\frac {1}{2(1-\rho ^{2})}}\left[{\frac {(x-\mu _{x})^{2}}{\sigma _{x}^{2}}}+
            {\frac {(y-\mu _{y})^{2}}{\sigma _{y}^{2}}}-{\frac {2\rho (x-\mu _{x})(y-\mu _{y})}
            {\sigma _{x}\sigma _{y}}}\right]\right)

        :param value: The log probability density function is evaluated at those values.
        :return: Log probability
        """
        # x: [..., 2]
        # value = torch.unsqueeze(value, dim=-2).repeat(1,1,1,1,self.mus.shape[-2],1)       # [..., 1, 2]
        # value = value[...,:self.mus.shape[-2],:].unsqueeze(-3).expand_as(self.mus)

        # if value.shape[2] == self.mus.shape[2] or (time_pri and value.shape[1] == self.mus.shape[1]):  # position 2 is the time duration
        #     dx = value - self.mus                       # [..., N, 2]
        # else:
        signed_dis = True
        if dx is None:
            signed_dis = False
            if len(value.shape)!=len(self.mus.shape):
                raise ValueError('value shape not aligned')
            dx = value - self.mus
        exp_nominator = ((torch.sum((dx/self.sigmas)**2, dim=-1)  # first and second term of exp nominator
                          - 2*self.corr*torch.prod(dx, dim=-1)/torch.prod(self.sigmas, dim=-1)))    # [..., N]

        component_log_p = -(2*self.log2pi
                            + torch.log(torch.pow((self.one_minus_rho2), 0.5))
                            + torch.sum(self.log_sigmas, dim=-1)
                            + 0.5*exp_nominator/self.one_minus_rho2)
        ll = self.log_weight[...,0] + component_log_p
        ll[ll.isinf()] = 0
        if signed_dis:
            ind = dx[...,0]<0
            ll[ind] = ll[ind]*torch.norm(dx[ind],dim=-1)
        return ll



    def get_cov_from_corr(self):
        """
        corr is [sigma_x, sigma_y, rho]
        """
        cov = self.corr * torch.prod(self.sigmas, dim=-1)
        E = torch.stack([torch.stack([self.sigmas[..., 0]**2, cov], dim=-1),
                         torch.stack([cov, self.sigmas[..., 1]**2], dim=-1)],
                        dim=-2)
        self.cov = E
        return E

    def mus_transform(self,state, transf):
        """
        from [0,0,0] view
        """
        cos = torch.cos(transf[...,2])
        sin = torch.sin(transf[...,2])

        state_t = torch.zeros_like(state)
        state_t[...,0] = state[...,0]*cos -state[...,1]*sin +transf[...,0]
        state_t[...,1] = state[...,0]*sin + state[...,1]*cos +transf[...,1]
        if state_t.shape[-1]==3:
            state_t[...,2] = state[...,2]+transf[...,2]
        return state_t

    def cov_transform(self,cov,yaw):
        rot_m = self.get_rotation_mat(yaw)
        return torch.matmul(torch.matmul(
            rot_m, cov), rot_m.transpose(-1,-2))#.permute(0, 1, 2, 3, 5, 4))
        

    def get_rotation_mat(self, yaw):
        cos = torch.cos(yaw)
        sin = torch.sin(yaw)
        return torch.stack(
            [torch.stack([cos,-sin],dim=-1),
            torch.stack([sin,cos],dim=-1)],
            dim=-2
        )

    # def get_distribution(self):
    #     mix = td.Categorical(self.weight)
    #     mvn = td.MultivariateNormal(self.mus, self.cov, validate_args=False)
    #     # mvn = td.Independent(mvn, -2)
    #     dist = td.MixtureSameFamily(mix, mvn)
    #     return dist

class Mlp(nn.Module):
    def __init__(
        self,
        in_features,
        hiddent_feature=None,
        out_features=None,
        act_layer=nn.ReLU,
        drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hiddent_feature = hiddent_feature or in_features
        self.fc1 = nn.Linear(in_features, hiddent_feature)
        self.act = act_layer()
        self.fc2 = nn.Linear(hiddent_feature, out_features)
        self.drop = nn.Dropout(drop)
        self.layernorm = nn.LayerNorm(hiddent_feature)

    def forward(self, x):
        x = self.fc1(x)
        x = self.layernorm(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

def load_cfg_here():
    with open(os.getenv('DIPP_CONFIG'), 'r') as yaml_file:
        cfg = yaml.safe_load(yaml_file)
    return cfg


# def project_to_frenet_frame(traj, ref_line):
#     # distance_to_ref = torch.cdist(traj[..., :2], ref_line[..., :2])
#     # k = torch.argmin(distance_to_ref, dim=-1).view(-1, traj.shape[1], 1).expand(-1, -1, 3)
#     # ref_points = torch.gather(ref_line, 1, k)
#     # traj = bicycle_model(control, current_state)
#     btsz,mod,th,dim = traj.shape
#     distance_to_ref = torch.cdist(traj[..., :2].reshape(btsz,-1,2), ref_line[..., :2])
#     distance_to_ref = distance_to_ref.reshape(btsz,mod,th,-1)
#     k = torch.argmin(distance_to_ref, dim=-1).view(btsz,
#                                                    mod, th, 1).expand(-1, -1, -1, 3)
#     ref_points = torch.gather(ref_line.unsqueeze(-3).repeat(1,mod,1,1), 2, k)
#     x_r, y_r, theta_r = ref_points[..., 0], ref_points[..., 1], ref_points[..., 2] 
#     x, y = traj[..., 0], traj[..., 1]
#     s = 0.1 * (k[..., 0] - 200)
#     l = torch.sign((y-y_r)*torch.cos(theta_r)-(x-x_r)*torch.sin(theta_r)) * torch.sqrt(torch.square(x-x_r)+torch.square(y-y_r))
#     sl = torch.stack([s, l], dim=-1)
#     return sl

def get_u_from_X(X, init_state, dt = 0.1, L = WB):
    # extend fut_traj
    _X = torch.cat([init_state[...,:5].unsqueeze(-2),X[...,:5]],dim = -2)
    v = torch.hypot(_X[..., 3], _X[..., 4]) # vehicle's velocity [m/s]
    a = torch.diff(v,dim=-1)/dt
    d_theta = pi_2_pi(torch.diff(_X[...,2],dim=-1)/dt)
    steering = torch.atan2(L*d_theta,v[...,1:])
    steering = torch.nan_to_num(steering,nan=0.)
    steering = pi_2_pi(steering)
    u = torch.stack([a,steering],dim=-1)
    return u

def yawv2yawdxdy(X):
    if X.shape[-1] != 4:
        raise ValueError('X last dim should be [x, y, yaw, velocity]')
    x = X[...,0:1]
    y = X[...,1:2]
    yaw = X[...,2:3]
    v = X[...,3:4]
    dx = v*torch.cos(yaw)
    dy = v*torch.sin(yaw)
    return torch.cat([x,y,yaw,dx,dy],dim=-1)

# %% cost functions


def acceleration(control):
    acc = control[:, :, 0]
    return acc


def jerk(control):
    acc = control[:, :, 0]
    jerk = torch.diff(acc) / 0.1
    return jerk


def steering(control):
    steering = control[:, :, 1]
    return steering


def steering_change(control):
    steering = control[:, :, 1]
    steering_change = torch.diff(steering) / 0.1
    return steering_change


def speed(control, current_state):
    velocity = torch.hypot(current_state[..., 3], current_state[..., 4])
    dt = 0.1

    acc = control[..., 0]
    speed = velocity.unsqueeze(-1) + torch.cumsum(acc * dt, dim=-1)
    speed = torch.clamp(speed, min=0)
    speed_limit = torch.max(
        control[..., -1], dim=-1, keepdim=True)[0]
    speed_error = speed - speed_limit

    return speed_error.unsqueeze(-1)


def lane_xyyaw(control, ref_line, current_state):
    # global ref_points

    # control = u[0].tensor.view(-1, 50, 2)
    # ref_line = aux_vars[0].tensor
    # current_state = aux_vars[1].tensor[:, 0]
    traj = bicycle_model(control, current_state)
    btsz,mod,th,dim = traj.shape
    distance_to_ref = torch.cdist(traj[..., :2].reshape(btsz,-1,2), ref_line[:, :, :2])
    distance_to_ref = distance_to_ref.reshape(btsz,mod,th,-1)
    k = torch.argmin(distance_to_ref, dim=-1).view(btsz,
                                                   mod, th, 1).expand(-1, -1, -1, 3)
    ref_points = torch.gather(ref_line.unsqueeze(-3).repeat(1,mod,1,1), 2, k)
    lane_error = torch.cat([traj[..., 0:1]-ref_points[..., 0:1],
                           traj[..., 1:2]-ref_points[..., 1:2], 
                           traj[..., 2:3]-ref_points[..., 2:3]],  dim=-1)
    return lane_error


# def lane_theta(control, ref_line, current_state):
#     # control = optim_vars[0].tensor.view(-1, 50, 2)
#     # current_state = aux_vars[1].tensor[:, 0]

#     traj = bicycle_model(control, current_state)
#     btsz,mod,th,dim = traj.shape
#     # distance_to_ref = torch.cdist(traj[:, :, :2], ref_line[:, :, :2])
#     distance_to_ref = torch.cdist(traj[..., :2].reshape(btsz,-1,2), ref_line[:, :, :2])
#     distance_to_ref = distance_to_ref.reshape(btsz,mod,th,-1)
#     k = torch.argmin(distance_to_ref, dim=-1).view(btsz,
#                                                    mod, th, 1).expand(-1, -1, -1, 3)
#     ref_points = torch.gather(ref_line.unsqueeze(-3).repeat(1,mod,1,1), 1, k)
#     # theta = traj[..., 2:3]
#     lane_error = traj[..., 1::2, 2:3] - ref_points[..., 1::2, 2:3]
#     return lane_error


def red_light_violation(control, ref_line, current_state):
    # control = optim_vars[0].tensor.view(-1, 50, 2)
    # current_state = aux_vars[1].tensor[:, 0]
    # ref_line = aux_vars[0].tensor
    red_light = ref_line[..., -1]
    dt = 0.1

    velocity = torch.hypot(current_state[..., 3], current_state[..., 4])
    acc = control[..., 0]
    speed = velocity.unsqueeze(-1) + torch.cumsum(acc * dt, dim=-1)
    speed = torch.clamp(speed, min=0)
    s = torch.cumsum(speed * dt, dim=-1)
    stop_point = torch.max(red_light[:, 200:] == 0, dim=-1)[1] * 0.1
    stop_distance = stop_point.view(-1, 1, 1) - 3
    red_light_error = (s - stop_distance) * \
        (s > stop_distance) * (stop_point.unsqueeze(-1).unsqueeze(-1) != 0)

    return red_light_error.unsqueeze(-1)


def neighbor_sl_dis(control, ref_line, current_state, neighbors):
    actor_mask = torch.ne(current_state, 0)[:, 1:, -1]
    ego_current_state = current_state[:, 0:1]
    ego = bicycle_model(control, ego_current_state)
    ego_len, ego_width = ego_current_state[..., -3], ego_current_state[..., -2]
    neighbors_current_state = current_state[:, 1:]
    neighbors_len, neighbors_width = neighbors_current_state[..., -
                                                             3], neighbors_current_state[..., -2]

    l_eps = (ego_width + neighbors_width)/2 + 0.5
    # print('l_eps', l_eps.shape)
    # frenet_neighbors = torch.stack([project_to_frenet_frame(
    #     neighbors[:, :, i].detach(), ref_line) for i in range(neighbors.shape[2])], dim=2)
    frenet_neighbors = project_to_frenet_frame(neighbors,ref_line)
    frenet_ego = project_to_frenet_frame(ego.detach(), ref_line)
    # safe_error = []
    # for t in [0, 2, 5, 9, 14, 19, 24, 29, 39, 49]:  # key frames
        # find objects of interest
    sl_dis = torch.abs(frenet_ego[...,0:2].unsqueeze(
        1) - frenet_neighbors[..., 0:2].unsqueeze(2))
    # l_distance = torch.abs(frenet_ego[...,1].unsqueeze(
    #     1) - frenet_neighbors[..., 1].unsqueeze(2))
    # s_distance = torch.abs(frenet_ego[...,0].unsqueeze(
    #     1) - frenet_neighbors[..., 0].unsqueeze(2))
    # btsz,nbnum = actor_mask.shape
    # interactive = torch.logical_and(
    #     s_distance > 0, l_distance < l_eps.reshape(btsz, nbnum,1,1,1)) * actor_mask.reshape(btsz, nbnum,1,1,1)

    # # find closest object
    # distances = torch.norm(ego[..., :2].unsqueeze(-4) - neighbors[..., :2].unsqueeze(-3), dim=-1, keepdim=True)
    # distances = torch.masked_fill(
    #     distances, torch.logical_not(interactive), 100)
    # distance, index = torch.min(distances, dim=-4)
    # s_eps = (ego_len + torch.index_select(neighbors_len, 1, index)
    #             [:, 0])/2 + 5

    # # calculate cost
    # error = (s_eps - distance) * (distance < s_eps)
        # safe_error.append(error)

    # safe_error = torch.stack(safe_error, dim=1)

    return sl_dis


def safety(optim_vars, aux_vars):
    control = optim_vars[0].tensor.view(-1, 50, 2)
    neighbors = aux_vars[0].tensor.permute(0, 2, 1, 3)
    current_state = aux_vars[1].tensor
    ref_line = aux_vars[2].tensor

    actor_mask = torch.ne(current_state, 0)[:, 1:, -1]
    ego_current_state = current_state[:, 0]
    ego = bicycle_model(control, ego_current_state)
    ego_len, ego_width = ego_current_state[:, -3], ego_current_state[:, -2]
    neighbors_current_state = current_state[:, 1:]
    neighbors_len, neighbors_width = neighbors_current_state[..., -
                                                             3], neighbors_current_state[..., -2]

    l_eps = (ego_width.unsqueeze(1) + neighbors_width)/2 + 0.5
    frenet_neighbors = torch.stack([project_to_frenet_frame(
        neighbors[:, :, i].detach(), ref_line) for i in range(neighbors.shape[2])], dim=2)
    frenet_ego = project_to_frenet_frame(ego.detach(), ref_line)

    safe_error = []
    for t in [0, 2, 5, 9, 14, 19, 24, 29, 39, 49]:  # key frames
        # find objects of interest
        l_distance = torch.abs(frenet_ego[:, t, 1].unsqueeze(
            1) - frenet_neighbors[:, t, :, 1])
        s_distance = frenet_neighbors[:, t, :,
                                      0] - frenet_ego[:, t, 0].unsqueeze(-1)
        interactive = torch.logical_and(
            s_distance > 0, l_distance < l_eps) * actor_mask

        # find closest object
        distances = torch.norm(ego[:, t, :2].unsqueeze(
            1) - neighbors[:, t, :, :2], dim=-1).squeeze(1)
        distances = torch.masked_fill(
            distances, torch.logical_not(interactive), 100)
        distance, index = torch.min(distances, dim=1)
        s_eps = (ego_len + torch.index_select(neighbors_len, 1, index)
                 [:, 0])/2 + 5

        # calculate cost
        error = (s_eps - distance) * (distance < s_eps)
        safe_error.append(error)

    safe_error = torch.stack(safe_error, dim=1)

    return safe_error