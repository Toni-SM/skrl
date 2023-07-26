from typing import Mapping, Optional

import queue

import numpy as np
import torch

from skrl import logger


def _np_quat_mul(a, b):
    assert a.shape == b.shape
    shape = a.shape
    a = a.reshape(-1, 4)
    b = b.reshape(-1, 4)

    x1, y1, z1, w1 = a[:, 0], a[:, 1], a[:, 2], a[:, 3]
    x2, y2, z2, w2 = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    ww = (z1 + x1) * (x2 + y2)
    yy = (w1 - y1) * (w2 + z2)
    zz = (w1 + y1) * (w2 - z2)
    xx = ww + yy + zz
    qq = 0.5 * (xx + (z1 - x1) * (x2 - y2))
    w = qq - ww + (z1 - y1) * (y2 - z2)
    x = qq - xx + (x1 + w1) * (x2 + w2)
    y = qq - yy + (w1 - x1) * (y2 + z2)
    z = qq - zz + (z1 + y1) * (w2 - x2)

    return np.stack([x, y, z, w], axis=-1).reshape(shape)

def _np_quat_conjugate(a):
    shape = a.shape
    a = a.reshape(-1, 4)
    return np.concatenate((-a[:, :3], a[:, -1:]), axis=-1).reshape(shape)

def _torch_quat_mul(a, b):
    assert a.shape == b.shape
    shape = a.shape
    a = a.reshape(-1, 4)
    b = b.reshape(-1, 4)

    w1, x1, y1, z1 = a[:, 0], a[:, 1], a[:, 2], a[:, 3]
    w2, x2, y2, z2 = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    ww = (z1 + x1) * (x2 + y2)
    yy = (w1 - y1) * (w2 + z2)
    zz = (w1 + y1) * (w2 - z2)
    xx = ww + yy + zz
    qq = 0.5 * (xx + (z1 - x1) * (x2 - y2))
    w = qq - ww + (z1 - y1) * (y2 - z2)
    x = qq - xx + (x1 + w1) * (x2 + w2)
    y = qq - yy + (w1 - x1) * (y2 + z2)
    z = qq - zz + (z1 + y1) * (w2 - x2)

    return torch.stack([w, x, y, z], dim=-1).view(shape)

def _torch_quat_conjugate(a):  # wxyz
    shape = a.shape
    a = a.reshape(-1, 4)
    return torch.cat((a[:, :1], -a[:, 1:]), dim=-1).view(shape)

def ik(jacobian_end_effector: torch.Tensor,
       current_position: torch.Tensor,
       current_orientation: torch.Tensor,
       goal_position: torch.Tensor,
       goal_orientation: Optional[torch.Tensor] = None,
       method: str = "damped least-squares",
       method_cfg: Mapping[str, float] = {"scale": 1, "damping": 0.05, "min_singular_value": 1e-5},
       squeeze_output: bool = True,) -> torch.Tensor:
    """Differential inverse kinematics

    :param jacobian_end_effector: End effector's jacobian
    :type jacobian_end_effector: torch.Tensor
    :param current_position: End effector's current position
    :type current_position: torch.Tensor
    :param current_orientation: End effector's current orientation
    :type current_orientation: torch.Tensor
    :param goal_position: End effector's goal position
    :type goal_position: torch.Tensor
    :param goal_orientation: End effector's goal orientation (default: ``None``).
                             If not provided, the current orientation will be used instead.
    :type goal_orientation: torch.Tensor, optional
    :param method: Differential inverse kinematics formulation (default: ``"damped least-squares"``).
                   The supported methods are described in the following table:

                   +----------------------------------+----------------------------------+
                   |IK Method                         |Method tag                        |
                   +==================================+==================================+
                   |Damped least-squares              |``"damped least-squares"``        |
                   +----------------------------------+----------------------------------+
                   |Tanspose                          |``"transpose"``                   |
                   +----------------------------------+----------------------------------+
                   |Pseduoinverse                     |``"pseudoinverse"``               |
                   +----------------------------------+----------------------------------+
                   |Singular-vale decomposition (SVD) |``"singular-vale decomposition"`` |
                   +----------------------------------+----------------------------------+
    :type method: str, optional
    :param method_cfg: Method configurations (default: ``{"scale": 1, "damping": 0.05, "min_singular_value": 1e-5}``)
    :type method_cfg: dict, optional
    :param squeeze_output: Squeeze output (default: ``True``)
    :type squeeze_output: bool, optional

    :return: Change in joint angles
    :rtype: torch.Tensor
    """
    if goal_orientation is None:
        goal_orientation = current_orientation

    # torch
    if isinstance(jacobian_end_effector, torch.Tensor):
        # compute error
        q = _torch_quat_mul(goal_orientation, _torch_quat_conjugate(current_orientation))
        error = torch.cat([goal_position - current_position,  # position error
                           q[:, 1:] * torch.sign(q[:, 0]).unsqueeze(-1)],  # orientation error
                          dim=-1).unsqueeze(-1)

        scale = method_cfg.get("scale", 1.0)

        # adaptive Singular Value Decomposition (SVD)
        if method == "singular-vale decomposition":
            min_singular_value = method_cfg.get("min_singular_value", 1e-5)
            U, S, Vh = torch.linalg.svd(jacobian_end_effector)  # U: 6xd, S: dxd, V: d x num_dof
            inv_s = torch.where(S > min_singular_value, 1.0 / S, torch.zeros_like(S))
            pseudoinverse = torch.transpose(Vh, 1, 2)[:, :, :6] @ torch.diag_embed(inv_s) @ torch.transpose(U, 1, 2)
            if squeeze_output:
                return (scale * pseudoinverse @ error).squeeze(dim=2)
            else:
                return scale * pseudoinverse @ error
        # jacobian pseudoinverse
        elif method == "pseudoinverse":
            pseudoinverse = torch.linalg.pinv(jacobian_end_effector)
            if squeeze_output:
                return (scale * pseudoinverse @ error).squeeze(dim=2)
            else:
                return scale * pseudoinverse @ error
        # jacobian transpose
        elif method == "transpose":
            transpose = torch.transpose(jacobian_end_effector, 1, 2)
            if squeeze_output:
                return (scale * transpose @ error).squeeze(dim=2)
            else:
                return scale * transpose @ error
        # damped least-squares
        elif method == "damped least-squares":
            damping = method_cfg.get("damping", 0.05)
            transpose = torch.transpose(jacobian_end_effector, 1, 2)
            lmbda = torch.eye(jacobian_end_effector.shape[1], device=jacobian_end_effector.device) * (damping ** 2)
            if squeeze_output:
                return (scale * transpose @ torch.inverse(jacobian_end_effector @ transpose + lmbda) @ error).squeeze(dim=2)
            else:
                return scale * transpose @ torch.inverse(jacobian_end_effector @ transpose + lmbda) @ error
        else:
            raise ValueError("Invalid IK method")

    # numpy
    # TODO: test and fix this
    else:
        # compute error
        q = _np_quat_mul(goal_orientation, _np_quat_conjugate(current_orientation))
        error = np.concatenate([goal_position - current_position,  # position error
                                q[:, 0:3] * np.sign(q[:, 3])])  # orientation error

        # solve damped least squares (dO = J.T * V)
        transpose = np.transpose(jacobian_end_effector, 1, 2)
        lmbda = np.eye(6) * (method_cfg.get("damping", 0.05) ** 2)
        if squeeze_output:
            return (transpose @ np.linalg.inv(jacobian_end_effector @ transpose + lmbda) @ error)
        else:
            return transpose @ np.linalg.inv(jacobian_end_effector @ transpose + lmbda) @ error

def get_env_instance(headless: bool = True,
                     enable_livestream: bool = False,
                     enable_viewport: bool = False,
                     multi_threaded: bool = False) -> "omni.isaac.gym.vec_env.VecEnvBase":
    """
    Instantiate a VecEnvBase-based object compatible with OmniIsaacGymEnvs

    :param headless: Disable UI when running (default: ``True``)
    :type headless: bool, optional
    :param enable_livestream: Whether to enable live streaming (default: ``False``)
    :type enable_livestream: bool, optional
    :param enable_viewport: Whether to enable viewport (default: ``False``)
    :type enable_viewport: bool, optional
    :param multi_threaded: Whether to return a multi-threaded environment instance (default: ``False``)
    :type multi_threaded: bool, optional

    :return: Environment instance
    :rtype: omni.isaac.gym.vec_env.VecEnvBase

    Example::

        from skrl.envs.wrappers.torch import wrap_env
        from skrl.utils.omniverse_isaacgym_utils import get_env_instance

        # get environment instance
        env = get_env_instance(headless=True)

        # parse sim configuration
        from omniisaacgymenvs.utils.config_utils.sim_config import SimConfig
        sim_config = SimConfig({"test": False,
                                "device_id": 0,
                                "headless": True,
                                "multi_gpu": False,
                                "sim_device": "gpu",
                                "enable_livestream": False,
                                "task": {"name": "CustomTask",
                                         "physics_engine": "physx",
                                         "env": {"numEnvs": 512,
                                                 "envSpacing": 1.5,
                                                 "enableDebugVis": False,
                                                 "clipObservations": 1000.0,
                                                 "clipActions": 1.0,
                                                 "controlFrequencyInv": 4},
                                         "sim": {"dt": 0.0083,  # 1 / 120
                                                 "use_gpu_pipeline": True,
                                                 "gravity": [0.0, 0.0, -9.81],
                                                 "add_ground_plane": True,
                                                 "use_flatcache": True,
                                                 "enable_scene_query_support": False,
                                                 "enable_cameras": False,
                                                 "default_physics_material": {"static_friction": 1.0,
                                                                              "dynamic_friction": 1.0,
                                                                              "restitution": 0.0},
                                                 "physx": {"worker_thread_count": 4,
                                                           "solver_type": 1,
                                                           "use_gpu": True,
                                                           "solver_position_iteration_count": 4,
                                                           "solver_velocity_iteration_count": 1,
                                                           "contact_offset": 0.005,
                                                           "rest_offset": 0.0,
                                                           "bounce_threshold_velocity": 0.2,
                                                           "friction_offset_threshold": 0.04,
                                                           "friction_correlation_distance": 0.025,
                                                           "enable_sleeping": True,
                                                           "enable_stabilization": True,
                                                           "max_depenetration_velocity": 1000.0,
                                                           "gpu_max_rigid_contact_count": 524288,
                                                           "gpu_max_rigid_patch_count": 33554432,
                                                           "gpu_found_lost_pairs_capacity": 524288,
                                                           "gpu_found_lost_aggregate_pairs_capacity": 262144,
                                                           "gpu_total_aggregate_pairs_capacity": 1048576,
                                                           "gpu_max_soft_body_contacts": 1048576,
                                                           "gpu_max_particle_contacts": 1048576,
                                                           "gpu_heap_capacity": 33554432,
                                                           "gpu_temp_buffer_capacity": 16777216,
                                                           "gpu_max_num_partitions": 8}}}})

        # import and setup custom task
        from custom_task import CustomTask
        task = CustomTask(name="CustomTask", sim_config=sim_config, env=env)
        env.set_task(task=task, sim_params=sim_config.get_physics_params(), backend="torch", init_sim=True)

        # wrap the environment
        env = wrap_env(env, "omniverse-isaacgym")
    """
    from omni.isaac.gym.vec_env import TaskStopException, VecEnvBase, VecEnvMT
    from omni.isaac.gym.vec_env.vec_env_mt import TrainerMT

    class _OmniIsaacGymVecEnv(VecEnvBase):
        def step(self, actions):
            actions = torch.clamp(actions, -self._task.clip_actions, self._task.clip_actions).to(self._task.device).clone()
            self._task.pre_physics_step(actions)

            for _ in range(self._task.control_frequency_inv):
                self._world.step(render=self._render)
                self.sim_frame_count += 1

            observations, rewards, dones, info = self._task.post_physics_step()

            return {"obs": torch.clamp(observations, -self._task.clip_obs, self._task.clip_obs).to(self._task.rl_device).clone()}, \
                rewards.to(self._task.rl_device).clone(), dones.to(self._task.rl_device).clone(), info.copy()

        def reset(self):
            self._task.reset()
            actions = torch.zeros((self.num_envs, self._task.num_actions), device=self._task.device)
            return self.step(actions)[0]

    class _OmniIsaacGymTrainerMT(TrainerMT):
        def run(self):
            pass

        def stop(self):
            pass

    class _OmniIsaacGymVecEnvMT(VecEnvMT):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

            self.action_queue = queue.Queue(1)
            self.data_queue = queue.Queue(1)

        def run(self, trainer=None):
            super().run(_OmniIsaacGymTrainerMT() if trainer is None else trainer)

        def _parse_data(self, data):
            self._observations = torch.clamp(data["obs"], -self._task.clip_obs, self._task.clip_obs).to(self._task.rl_device).clone()
            self._rewards = data["rew"].to(self._task.rl_device).clone()
            self._dones = data["reset"].to(self._task.rl_device).clone()
            self._info = data["extras"].copy()

        def step(self, actions):
            if self._stop:
                raise TaskStopException()

            actions = torch.clamp(actions, -self._task.clip_actions, self._task.clip_actions).clone()

            self.send_actions(actions)
            data = self.get_data()

            return {"obs": self._observations}, self._rewards, self._dones, self._info

        def reset(self):
            self._task.reset()
            actions = torch.zeros((self.num_envs, self._task.num_actions), device=self._task.device)
            return self.step(actions)[0]

        def close(self):
            # end stop signal to main thread
            self.send_actions(None)
            self.stop = True

    if multi_threaded:
        try:
            return _OmniIsaacGymVecEnvMT(headless=headless, enable_livestream=enable_livestream, enable_viewport=enable_viewport)
        except TypeError:
            logger.warning("Using an older version of Isaac Sim (2022.2.0 or earlier)")
            return _OmniIsaacGymVecEnvMT(headless=headless)  # Isaac Sim 2022.2.0 and earlier
    else:
        try:
            return _OmniIsaacGymVecEnv(headless=headless, enable_livestream=enable_livestream, enable_viewport=enable_viewport)
        except TypeError:
            logger.warning("Using an older version of Isaac Sim (2022.2.0 or earlier)")
            return _OmniIsaacGymVecEnv(headless=headless)  # Isaac Sim 2022.2.0 and earlier
