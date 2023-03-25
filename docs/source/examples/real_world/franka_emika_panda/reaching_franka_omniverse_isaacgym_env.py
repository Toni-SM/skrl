import torch
import numpy as np

from omni.isaac.core.utils.extensions import enable_extension
enable_extension("omni.replicator.isaac")  # required by OIGE

from omniisaacgymenvs.tasks.base.rl_task import RLTask
from omniisaacgymenvs.robots.articulations.franka import Franka as Robot

from omni.isaac.core.prims import RigidPrimView
from omni.isaac.core.articulations import ArticulationView
from omni.isaac.core.objects import DynamicSphere
from omni.isaac.core.utils.prims import get_prim_at_path

from skrl.utils import omniverse_isaacgym_utils

# post_physics_step calls
# - get_observations()
# - get_states()
# - calculate_metrics()
# - is_done()
# - get_extras()


TASK_CFG = {"test": False,
            "device_id": 0,
            "headless": True,
            "sim_device": "gpu",
            "enable_livestream": False,
            "task": {"name": "ReachingFranka",
                     "physics_engine": "physx",
                     "env": {"numEnvs": 1024,
                             "envSpacing": 1.5,
                             "episodeLength": 100,
                             "enableDebugVis": False,
                             "clipObservations": 1000.0,
                             "clipActions": 1.0,
                             "controlFrequencyInv": 4,
                             "actionScale": 2.5,
                             "dofVelocityScale": 0.1,
                             "controlSpace": "cartesian"},
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
                                      "gpu_max_num_partitions": 8},
                             "robot": {"override_usd_defaults": False,
                                       "fixed_base": False,
                                       "enable_self_collisions": False,
                                       "enable_gyroscopic_forces": True,
                                       "solver_position_iteration_count": 4,
                                       "solver_velocity_iteration_count": 1,
                                       "sleep_threshold": 0.005,
                                       "stabilization_threshold": 0.001,
                                       "density": -1,
                                       "max_depenetration_velocity": 1000.0,
                                       "contact_offset": 0.005,
                                       "rest_offset": 0.0},
                             "target": {"override_usd_defaults": False,
                                        "fixed_base": True,
                                        "enable_self_collisions": False,
                                        "enable_gyroscopic_forces": True,
                                        "solver_position_iteration_count": 4,
                                        "solver_velocity_iteration_count": 1,
                                        "sleep_threshold": 0.005,
                                        "stabilization_threshold": 0.001,
                                        "density": -1,
                                        "max_depenetration_velocity": 1000.0,
                                        "contact_offset": 0.005,
                                        "rest_offset": 0.0}}}}


class RobotView(ArticulationView):
    def __init__(self, prim_paths_expr: str, name: str = "robot_view") -> None:
        super().__init__(prim_paths_expr=prim_paths_expr, name=name, reset_xform_properties=False)


class ReachingFrankaTask(RLTask):
    def __init__(self, name, sim_config, env, offset=None) -> None:
        self._sim_config = sim_config
        self._cfg = sim_config.config
        self._task_cfg = sim_config.task_config

        self.dt = 1 / 120.0

        self._num_envs = self._task_cfg["env"]["numEnvs"]
        self._env_spacing = self._task_cfg["env"]["envSpacing"]
        self._action_scale = self._task_cfg["env"]["actionScale"]
        self._dof_vel_scale = self._task_cfg["env"]["dofVelocityScale"]
        self._max_episode_length = self._task_cfg["env"]["episodeLength"]
        self._control_space = self._task_cfg["env"]["controlSpace"]

        # observation and action space
        self._num_observations = 18
        if self._control_space == "joint":
            self._num_actions = 7
        elif self._control_space == "cartesian":
            self._num_actions = 3
        else:
            raise ValueError("Invalid control space: {}".format(self._control_space))

        self._end_effector_link = "panda_leftfinger"

        RLTask.__init__(self, name, env)

    def set_up_scene(self, scene) -> None:
        self.get_robot()
        self.get_target()

        super().set_up_scene(scene)

        # robot view
        self._robots = RobotView(prim_paths_expr="/World/envs/.*/robot", name="robot_view")
        scene.add(self._robots)
        # end-effectors view
        self._end_effectors = RigidPrimView(prim_paths_expr="/World/envs/.*/robot/{}".format(self._end_effector_link), name="end_effector_view")
        scene.add(self._end_effectors)
        # hands view (cartesian)
        if self._control_space == "cartesian":
            self._hands = RigidPrimView(prim_paths_expr="/World/envs/.*/robot/panda_hand", name="hand_view", reset_xform_properties=False)
            scene.add(self._hands)
        # target view
        self._targets = RigidPrimView(prim_paths_expr="/World/envs/.*/target", name="target_view", reset_xform_properties=False)
        scene.add(self._targets)

        self.init_data()

    def get_robot(self):
        robot = Robot(prim_path=self.default_zero_env_path + "/robot",
                      translation=torch.tensor([0.0, 0.0, 0.0]),
                      orientation=torch.tensor([1.0, 0.0, 0.0, 0.0]),
                      name="robot")
        self._sim_config.apply_articulation_settings("robot", get_prim_at_path(robot.prim_path), self._sim_config.parse_actor_config("robot"))

    def get_target(self):
        target = DynamicSphere(prim_path=self.default_zero_env_path + "/target",
                               name="target",
                               radius=0.025,
                               color=torch.tensor([1, 0, 0]))
        self._sim_config.apply_articulation_settings("target", get_prim_at_path(target.prim_path), self._sim_config.parse_actor_config("target"))
        target.set_collision_enabled(False)

    def init_data(self) -> None:
        self.robot_default_dof_pos = torch.tensor(np.radians([0, -45, 0, -135, 0, 90, 45, 0, 0]), device=self._device, dtype=torch.float32)
        self.actions = torch.zeros((self._num_envs, self.num_actions), device=self._device)

        if self._control_space == "cartesian":
            self.jacobians = torch.zeros((self._num_envs, 10, 6, 9), device=self._device)
            self.hand_pos, self.hand_rot = torch.zeros((self._num_envs, 3), device=self._device), torch.zeros((self._num_envs, 4), device=self._device)

    def get_observations(self) -> dict:
        robot_dof_pos = self._robots.get_joint_positions(clone=False)
        robot_dof_vel = self._robots.get_joint_velocities(clone=False)
        end_effector_pos, end_effector_rot = self._end_effectors.get_world_poses(clone=False)
        target_pos, target_rot = self._targets.get_world_poses(clone=False)

        dof_pos_scaled = 2.0 * (robot_dof_pos - self.robot_dof_lower_limits) \
            / (self.robot_dof_upper_limits - self.robot_dof_lower_limits) - 1.0
        dof_vel_scaled = robot_dof_vel * self._dof_vel_scale

        generalization_noise = torch.rand((dof_vel_scaled.shape[0], 7), device=self._device) + 0.5

        self.obs_buf[:, 0] = self.progress_buf / self._max_episode_length
        self.obs_buf[:, 1:8] = dof_pos_scaled[:, :7]
        self.obs_buf[:, 8:15] = dof_vel_scaled[:, :7] * generalization_noise
        self.obs_buf[:, 15:18] = target_pos - self._env_pos

        # compute distance for calculate_metrics() and is_done()
        self._computed_distance = torch.norm(end_effector_pos - target_pos, dim=-1)

        if self._control_space == "cartesian":
            self.jacobians = self._robots.get_jacobians(clone=False)
            self.hand_pos, self.hand_rot = self._hands.get_world_poses(clone=False)
            self.hand_pos -= self._env_pos

        return {self._robots.name: {"obs_buf": self.obs_buf}}

    def pre_physics_step(self, actions) -> None:
        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_env_ids) > 0:
            self.reset_idx(reset_env_ids)

        self.actions = actions.clone().to(self._device)
        env_ids_int32 = torch.arange(self._robots.count, dtype=torch.int32, device=self._device)

        if self._control_space == "joint":
            targets = self.robot_dof_targets[:, :7] + self.robot_dof_speed_scales[:7] * self.dt * self.actions * self._action_scale

        elif self._control_space == "cartesian":
            goal_position = self.hand_pos + actions / 100.0
            delta_dof_pos = omniverse_isaacgym_utils.ik(jacobian_end_effector=self.jacobians[:, 8 - 1, :, :7],  # franka hand index: 8
                                                        current_position=self.hand_pos,
                                                        current_orientation=self.hand_rot,
                                                        goal_position=goal_position,
                                                        goal_orientation=None)
            targets = self.robot_dof_targets[:, :7] + delta_dof_pos

        self.robot_dof_targets[:, :7] = torch.clamp(targets, self.robot_dof_lower_limits[:7], self.robot_dof_upper_limits[:7])
        self.robot_dof_targets[:, 7:] = 0
        self._robots.set_joint_position_targets(self.robot_dof_targets, indices=env_ids_int32)

    def reset_idx(self, env_ids) -> None:
        indices = env_ids.to(dtype=torch.int32)

        # reset robot
        pos = torch.clamp(self.robot_default_dof_pos.unsqueeze(0) + 0.25 * (torch.rand((len(env_ids), self.num_robot_dofs), device=self._device) - 0.5),
                          self.robot_dof_lower_limits, self.robot_dof_upper_limits)
        dof_pos = torch.zeros((len(indices), self._robots.num_dof), device=self._device)
        dof_pos[:, :] = pos
        dof_pos[:, 7:] = 0
        dof_vel = torch.zeros((len(indices), self._robots.num_dof), device=self._device)
        self.robot_dof_targets[env_ids, :] = pos
        self.robot_dof_pos[env_ids, :] = pos

        self._robots.set_joint_position_targets(self.robot_dof_targets[env_ids], indices=indices)
        self._robots.set_joint_positions(dof_pos, indices=indices)
        self._robots.set_joint_velocities(dof_vel, indices=indices)

        # reset target
        pos = (torch.rand((len(env_ids), 3), device=self._device) - 0.5) * 2 \
            * torch.tensor([0.25, 0.25, 0.10], device=self._device) \
            + torch.tensor([0.50, 0.00, 0.20], device=self._device)

        self._targets.set_world_poses(pos + self._env_pos[env_ids], indices=indices)

        # bookkeeping
        self.reset_buf[env_ids] = 0
        self.progress_buf[env_ids] = 0

    def post_reset(self):
        self.num_robot_dofs = self._robots.num_dof
        self.robot_dof_pos = torch.zeros((self.num_envs, self.num_robot_dofs), device=self._device)
        dof_limits = self._robots.get_dof_limits()
        self.robot_dof_lower_limits = dof_limits[0, :, 0].to(device=self._device)
        self.robot_dof_upper_limits = dof_limits[0, :, 1].to(device=self._device)
        self.robot_dof_speed_scales = torch.ones_like(self.robot_dof_lower_limits)
        self.robot_dof_targets = torch.zeros((self._num_envs, self.num_robot_dofs), dtype=torch.float, device=self._device)

        # randomize all envs
        indices = torch.arange(self._num_envs, dtype=torch.int64, device=self._device)
        self.reset_idx(indices)

    def calculate_metrics(self) -> None:
        self.rew_buf[:] = -self._computed_distance

    def is_done(self) -> None:
        self.reset_buf.fill_(0)
        # target reached
        self.reset_buf = torch.where(self._computed_distance <= 0.035, torch.ones_like(self.reset_buf), self.reset_buf)
        # max episode length
        self.reset_buf = torch.where(self.progress_buf >= self._max_episode_length - 1, torch.ones_like(self.reset_buf), self.reset_buf)
