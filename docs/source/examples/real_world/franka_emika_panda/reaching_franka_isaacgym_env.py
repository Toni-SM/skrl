import os
import numpy as np
import torch

from isaacgym import gymtorch, gymapi

# isaacgymenvs (VecTask class)
import sys
import isaacgymenvs
sys.path.append(list(isaacgymenvs.__path__)[0])
from tasks.base.vec_task import VecTask

from skrl.utils import isaacgym_utils


TASK_CFG = {"name": "ReachingFranka",
            "physics_engine": "physx",
            "rl_device": "cuda:0",
            "sim_device": "cuda:0",
            "graphics_device_id": 0,
            "headless": False,
            "virtual_screen_capture": False,
            "force_render": True,
            "env": {"numEnvs": 1024,
                    "envSpacing": 1.5,
                    "episodeLength": 100,
                    "enableDebugVis": False,
                    "clipObservations": 1000.0,
                    "clipActions": 1.0,
                    "controlFrequencyInv": 4,
                    "actionScale": 2.5,
                    "dofVelocityScale": 0.1,
                    "controlSpace": "cartesian",
                    "enableCameraSensors": False},
            "sim": {"dt": 0.0083,  # 1 / 120
                    "substeps": 1,
                    "up_axis": "z",
                    "use_gpu_pipeline": True,
                    "gravity": [0.0, 0.0, -9.81],
                    "physx": {"num_threads": 4,
                              "solver_type": 1,
                              "use_gpu": True,
                              "num_position_iterations": 4,
                              "num_velocity_iterations": 1,
                              "contact_offset": 0.005,
                              "rest_offset": 0.0,
                              "bounce_threshold_velocity": 0.2,
                              "max_depenetration_velocity": 1000.0,
                              "default_buffer_size_multiplier": 5.0,
                              "max_gpu_contact_pairs": 1048576,
                              "num_subscenes": 4,
                              "contact_collection": 0}},
            "task": {"randomize": False}}


class ReachingFrankaTask(VecTask):
    def __init__(self, cfg):
        self.cfg = cfg
        rl_device = cfg["rl_device"]
        sim_device = cfg["sim_device"]
        graphics_device_id = cfg["graphics_device_id"]
        headless = cfg["headless"]
        virtual_screen_capture = cfg["virtual_screen_capture"]
        force_render = cfg["force_render"]

        self.dt = 1 / 120.0

        self._action_scale = self.cfg["env"]["actionScale"]
        self._dof_vel_scale = self.cfg["env"]["dofVelocityScale"]
        self._control_space = self.cfg["env"]["controlSpace"]
        self.max_episode_length = self.cfg["env"]["episodeLength"]  # name required for VecTask

        self.debug_viz = self.cfg["env"]["enableDebugVis"]

        # observation and action space
        self.cfg["env"]["numObservations"] = 18
        if self._control_space == "joint":
            self.cfg["env"]["numActions"] = 7
        elif self._control_space == "cartesian":
            self.cfg["env"]["numActions"] = 3
        else:
            raise ValueError("Invalid control space: {}".format(self._control_space))

        self._end_effector_link = "panda_leftfinger"

        # setup VecTask
        super().__init__(config=self.cfg,
                         rl_device=rl_device,
                         sim_device=sim_device,
                         graphics_device_id=graphics_device_id,
                         headless=headless,
                         virtual_screen_capture=virtual_screen_capture,
                         force_render=force_render)

        # tensors and views: DOFs, roots, rigid bodies
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        root_state_tensor = self.gym.acquire_actor_root_state_tensor(self.sim)
        rigid_body_state_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)

        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.root_state = gymtorch.wrap_tensor(root_state_tensor)
        self.rigid_body_state = gymtorch.wrap_tensor(rigid_body_state_tensor)

        self.dof_pos = self.dof_state.view(self.num_envs, -1, 2)[..., 0]
        self.dof_vel = self.dof_state.view(self.num_envs, -1, 2)[..., 1]

        self.root_pos = self.root_state[:, 0:3].view(self.num_envs, -1, 3)
        self.root_rot = self.root_state[:, 3:7].view(self.num_envs, -1, 4)
        self.root_vel_lin = self.root_state[:, 7:10].view(self.num_envs, -1, 3)
        self.root_vel_ang = self.root_state[:, 10:13].view(self.num_envs, -1, 3)

        self.rigid_body_pos = self.rigid_body_state[:, 0:3].view(self.num_envs, -1, 3)
        self.rigid_body_rot = self.rigid_body_state[:, 3:7].view(self.num_envs, -1, 4)
        self.rigid_body_vel_lin = self.rigid_body_state[:, 7:10].view(self.num_envs, -1, 3)
        self.rigid_body_vel_ang = self.rigid_body_state[:, 10:13].view(self.num_envs, -1, 3)

        # tensors and views: jacobian
        if self._control_space == "cartesian":
            jacobian_tensor = self.gym.acquire_jacobian_tensor(self.sim, "robot")
            self.jacobian = gymtorch.wrap_tensor(jacobian_tensor)
            self.jacobian_end_effector = self.jacobian[:, self.rigid_body_dict_robot[self._end_effector_link] - 1, :, :7]

        self.reset_idx(torch.arange(self.num_envs, device=self.device))

    def create_sim(self):
        self.sim_params.up_axis = gymapi.UP_AXIS_Z
        self.sim_params.gravity.x = 0
        self.sim_params.gravity.y = 0
        self.sim_params.gravity.z = -9.81
        self.sim = super().create_sim(self.device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        self._create_ground_plane()
        self._create_envs(self.num_envs, self.cfg["env"]["envSpacing"], int(np.sqrt(self.num_envs)))

    def _create_ground_plane(self):
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        self.gym.add_ground(self.sim, plane_params)

    def _create_envs(self, num_envs, spacing, num_per_row):
        lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        upper = gymapi.Vec3(spacing, spacing, spacing)

        asset_root = os.path.join(os.path.dirname(os.path.abspath(isaacgymenvs.__file__)), "../assets")
        robot_asset_file = "urdf/franka_description/robots/franka_panda.urdf"

        # robot asset
        asset_options = gymapi.AssetOptions()
        asset_options.flip_visual_attachments = True
        asset_options.fix_base_link = True
        asset_options.collapse_fixed_joints = True
        asset_options.disable_gravity = True
        asset_options.thickness = 0.001
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_POS
        asset_options.use_mesh_materials = True
        robot_asset = self.gym.load_asset(self.sim, asset_root, robot_asset_file, asset_options)

        # target asset
        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = True
        asset_options.collapse_fixed_joints = False
        asset_options.disable_gravity = True
        asset_options.thickness = 0.001
        asset_options.use_mesh_materials = True
        target_asset = self.gym.create_sphere(self.sim, 0.025, asset_options)

        robot_dof_stiffness = torch.tensor([400, 400, 400, 400, 400, 400, 400, 1.0e6, 1.0e6], dtype=torch.float32, device=self.device)
        robot_dof_damping = torch.tensor([80, 80, 80, 80, 80, 80, 80, 1.0e2, 1.0e2], dtype=torch.float, device=self.device)

        # set robot dof properties
        robot_dof_props = self.gym.get_asset_dof_properties(robot_asset)
        self.robot_dof_lower_limits = []
        self.robot_dof_upper_limits = []
        for i in range(9):
            robot_dof_props["driveMode"][i] = gymapi.DOF_MODE_POS
            if self.physics_engine == gymapi.SIM_PHYSX:
                robot_dof_props["stiffness"][i] = robot_dof_stiffness[i]
                robot_dof_props["damping"][i] = robot_dof_damping[i]
            else:
                robot_dof_props["stiffness"][i] = 7000.0
                robot_dof_props["damping"][i] = 50.0

            self.robot_dof_lower_limits.append(robot_dof_props["lower"][i])
            self.robot_dof_upper_limits.append(robot_dof_props["upper"][i])

        self.robot_dof_lower_limits = torch.tensor(self.robot_dof_lower_limits, device=self.device)
        self.robot_dof_upper_limits = torch.tensor(self.robot_dof_upper_limits, device=self.device)
        self.robot_dof_speed_scales = torch.ones_like(self.robot_dof_lower_limits)
        robot_dof_props["effort"][7] = 200
        robot_dof_props["effort"][8] = 200

        self.handle_targets = []
        self.handle_robots = []
        self.handle_envs = []

        indexes_sim_robot = []
        indexes_sim_target = []

        for i in range(self.num_envs):
            # create env instance
            env_ptr = self.gym.create_env(self.sim, lower, upper, num_per_row)

            # create robot instance
            pose = gymapi.Transform()
            pose.p = gymapi.Vec3(0.0, 0.0, 0.0)
            pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1)

            robot_actor = self.gym.create_actor(env=env_ptr,
                                                asset=robot_asset,
                                                pose=pose,
                                                name="robot",
                                                group=i, # collision group
                                                filter=1, # mask off collision
                                                segmentationId=0)
            self.gym.set_actor_dof_properties(env_ptr, robot_actor, robot_dof_props)
            indexes_sim_robot.append(self.gym.get_actor_index(env_ptr, robot_actor, gymapi.DOMAIN_SIM))

            # create target instance
            pose = gymapi.Transform()
            pose.p = gymapi.Vec3(0.5, 0.0, 0.2)
            pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1)

            target_actor = self.gym.create_actor(env=env_ptr,
                                                 asset=target_asset,
                                                 pose=pose,
                                                 name="target",
                                                 group=i + 1, # collision group
                                                 filter=1, # mask off collision
                                                 segmentationId=1)
            indexes_sim_target.append(self.gym.get_actor_index(env_ptr, target_actor, gymapi.DOMAIN_SIM))

            self.gym.set_rigid_body_color(env_ptr, target_actor, 0, gymapi.MESH_VISUAL, gymapi.Vec3(1., 0., 0.))

            self.handle_envs.append(env_ptr)
            self.handle_robots.append(robot_actor)
            self.handle_targets.append(target_actor)

        self.indexes_sim_robot = torch.tensor(indexes_sim_robot, dtype=torch.int32, device=self.device)
        self.indexes_sim_target = torch.tensor(indexes_sim_target, dtype=torch.int32, device=self.device)

        self.num_robot_dofs = self.gym.get_asset_dof_count(robot_asset)
        self.rigid_body_dict_robot = self.gym.get_asset_rigid_body_dict(robot_asset)

        self.init_data()

    def init_data(self):
        self.robot_default_dof_pos = torch.tensor(np.radians([0, -45, 0, -135, 0, 90, 45, 0, 0]), device=self.device, dtype=torch.float32)
        self.robot_dof_targets = torch.zeros((self.num_envs, self.num_robot_dofs), device=self.device, dtype=torch.float32)

        if self._control_space == "cartesian":
            self.end_effector_pos = torch.zeros((self.num_envs, 3), device=self.device)
            self.end_effector_rot = torch.zeros((self.num_envs, 4), device=self.device)

    def compute_reward(self):
        self.rew_buf[:] = -self._computed_distance

        self.reset_buf.fill_(0)
        # target reached
        self.reset_buf = torch.where(self._computed_distance <= 0.035, torch.ones_like(self.reset_buf), self.reset_buf)
        # max episode length
        self.reset_buf = torch.where(self.progress_buf >= self.max_episode_length - 1, torch.ones_like(self.reset_buf), self.reset_buf)

        # double restart correction (why?, is it necessary?)
        self.rew_buf = torch.where(self.progress_buf == 0, -0.75 * torch.ones_like(self.reset_buf), self.rew_buf)
        self.reset_buf = torch.where(self.progress_buf == 0, torch.zeros_like(self.reset_buf), self.reset_buf)

    def compute_observations(self):
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        if self._control_space == "cartesian":
            self.gym.refresh_jacobian_tensors(self.sim)

        robot_dof_pos = self.dof_pos
        robot_dof_vel = self.dof_vel
        self.end_effector_pos = self.rigid_body_pos[:, self.rigid_body_dict_robot[self._end_effector_link]]
        self.end_effector_rot = self.rigid_body_rot[:, self.rigid_body_dict_robot[self._end_effector_link]]
        target_pos = self.root_pos[:, 1]
        target_rot = self.root_rot[:, 1]

        dof_pos_scaled = 2.0 * (robot_dof_pos - self.robot_dof_lower_limits) \
            / (self.robot_dof_upper_limits - self.robot_dof_lower_limits) - 1.0
        dof_vel_scaled = robot_dof_vel * self._dof_vel_scale

        generalization_noise = torch.rand((dof_vel_scaled.shape[0], 7), device=self.device) + 0.5

        self.obs_buf[:, 0] = self.progress_buf / self.max_episode_length
        self.obs_buf[:, 1:8] = dof_pos_scaled[:, :7]
        self.obs_buf[:, 8:15] = dof_vel_scaled[:, :7] * generalization_noise
        self.obs_buf[:, 15:18] = target_pos

        # compute distance for compute_reward()
        self._computed_distance = torch.norm(self.end_effector_pos - target_pos, dim=-1)

    def reset_idx(self, env_ids):
        # reset robot
        pos = torch.clamp(self.robot_default_dof_pos.unsqueeze(0) + 0.25 * (torch.rand((len(env_ids), self.num_robot_dofs), device=self.device) - 0.5),
                          self.robot_dof_lower_limits, self.robot_dof_upper_limits)
        pos[:, 7:] = 0

        self.robot_dof_targets[env_ids, :] = pos[:]
        self.dof_pos[env_ids, :] = pos[:]
        self.dof_vel[env_ids, :] = 0

        indexes = self.indexes_sim_robot[env_ids]
        self.gym.set_dof_position_target_tensor_indexed(self.sim,
                                                        gymtorch.unwrap_tensor(self.robot_dof_targets),
                                                        gymtorch.unwrap_tensor(indexes),
                                                        len(env_ids))

        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.dof_state),
                                              gymtorch.unwrap_tensor(indexes),
                                              len(env_ids))

        # reset targets
        pos = (torch.rand((len(env_ids), 3), device=self.device) - 0.5) * 2
        pos[:, 0] = 0.50 + pos[:, 0] * 0.25
        pos[:, 1] = 0.00 + pos[:, 1] * 0.25
        pos[:, 2] = 0.20 + pos[:, 2] * 0.10

        self.root_pos[env_ids, 1, :] = pos[:]

        indexes = self.indexes_sim_target[env_ids]
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self.root_state),
                                                     gymtorch.unwrap_tensor(indexes),
                                                     len(env_ids))

        # bookkeeping
        self.reset_buf[env_ids] = 0
        self.progress_buf[env_ids] = 0

    def pre_physics_step(self, actions):
        actions = actions.clone().to(self.device)

        if self._control_space == "joint":
            targets = self.robot_dof_targets[:, :7] + self.robot_dof_speed_scales[:7] * self.dt * actions * self._action_scale

        elif self._control_space == "cartesian":
            goal_position = self.end_effector_pos + actions / 100.0
            delta_dof_pos = isaacgym_utils.ik(jacobian_end_effector=self.jacobian_end_effector,
                                              current_position=self.end_effector_pos,
                                              current_orientation=self.end_effector_rot,
                                              goal_position=goal_position,
                                              goal_orientation=None)
            targets = self.robot_dof_targets[:, :7] + delta_dof_pos

        self.robot_dof_targets[:, :7] = torch.clamp(targets, self.robot_dof_lower_limits[:7], self.robot_dof_upper_limits[:7])
        self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(self.robot_dof_targets))

    def post_physics_step(self):
        self.progress_buf += 1

        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(env_ids) > 0:
            self.reset_idx(env_ids)

        self.compute_observations()
        self.compute_reward()
