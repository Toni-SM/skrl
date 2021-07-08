import sys
from typing import Union

from isaacgym import gymapi
import torch


class BaseEnv:
    def __init__(self):

        # initialize gym
        self.gym = gymapi.acquire_gym()

        self.sim_params = gymapi.SimParams()

        self.viewer = None
        self._sync_viewer = True

        # optimization flags for pytorch JIT
        torch._C._jit_set_profiling_mode(False)
        torch._C._jit_set_profiling_executor(False)

        self.num_envs = 0

    def cleanup(self):
        # TODO: cleanup
        if self.viewer is not None:
            self.gym.destroy_viewer(self.viewer)
        if self.sim is not None:
            self.gym.destroy_sim(self.sim)

    def setup_up_axis(self, axis:str = "z", gravity: Union[gymapi.Vec3, list, tuple] = (0.0, 0.0, -9.8)) -> int:
        """
        Setup up axis and gravity
        """
        # gravity
        self.sim_params.gravity = gravity if type(gravity) is gymapi.Vec3 else gymapi.Vec3(*gravity)
        # axis
        if axis.lower() == 'z':
            self.sim_params.up_axis = gymapi.UP_AXIS_Z
            return 2
        self.sim_params.up_axis = gymapi.UP_AXIS_Y
        return 1
    
    def setup_env_grid(self, num_envs: int, spacing: float = 1.0) -> None:
        self.num_envs = num_envs
        self.env_lower = gymapi.Vec3(-spacing, 0.0, -spacing)
        self.env_upper = gymapi.Vec3(spacing, spacing, spacing)

    def setup_sim(self, cfg: dict) -> None:
        self.headless = cfg.get("headless", False)

        # device
        self.device_id = cfg.get("device_id", 0)
        self.device_type = cfg.get("device_type", "cuda").lower()
        self.device = "cuda:{}".format(self.device_id) if self.device_type in ["cuda", "gpu"] else "cpu"
        self.graphics_device_id = -1 if self.headless else self.device_id

    def create_sim(self, physics_engine):
        # create the simulation
        self.sim = self.gym.create_sim(compute_device=self.device_id, 
                                       graphics_device=self.graphics_device_id, 
                                       type=physics_engine, 
                                       params=self.sim_params)
        if self.sim is None:
            print("[ERROR] Failed to create sim")
            quit()

        # create the viewer
        if not self.headless:
            self.viewer = self.gym.create_viewer(self.sim, gymapi.CameraProperties())
            # TODO: set camera position based on up axis

            # subscribe to input events
            self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_ESCAPE, "quit")
            self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_Q, "quit")
            self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_V, "sync_viewer")

    def add_ground_plane(self, normal: Union[gymapi.Vec3, list, tuple] = [0, 0, 1], distance: float = 0, static_friction: float = 1, dynamic_friction: float = 1, restitution: float = 0) -> None:
        """
        Configure and add a ground plane
        """
        plane_params = gymapi.PlaneParams()
        plane_params.normal = normal if type(normal) is gymapi.Vec3 else gymapi.Vec3(*normal)
        plane_params.distance = distance
        plane_params.static_friction = static_friction
        plane_params.dynamic_friction = dynamic_friction
        plane_params.restitution = restitution

        # create the ground plane
        self.gym.add_ground(self.sim, plane_params)

    def pre_physics_step(self, actions) -> None:
        """
        Simulation code before step physics (pre-physics)

        E.g. apply actions
        """
        raise NotImplementedError

    def post_physics_step(self) -> None:
        """
        Simulation code after step physics (post-physics)

        E.g. compute reward and observations
        """
        raise NotImplementedError

    def reset(self, env_ids):
        """
        Reset the environment's state
        """
        raise NotImplementedError

    def step(self, actions) -> None:
        """
        Step the environment by one timestep
        """
        # TODO: domain randomization on actions

        self.pre_physics_step(actions)

        # TODO: control number of simulation steps between actions and observations
        # for i in range(self.frequency):
        #     pass

        # step physics
        self.gym.simulate(self.sim)
        self.gym.fetch_results(self.sim, True)

        # render each frame
        self.render()

        self.post_physics_step()

        # TODO: domain randomization on observations

    def render(self, mode='human') -> None:
        """
        Render the stage and handle viewer events
        """
        if self.viewer:
            # handle closed window
            if self.gym.query_viewer_has_closed(self.viewer):
                sys.exit()

            # handle input actions from the viewer
            for event in self.gym.query_viewer_action_events(self.viewer):
                # exit
                if event.action == "quit" and event.value > 0:
                    sys.exit()
                # toggle viewer sync  
                elif event.action == "sync_viewer" and event.value > 0:
                    self._sync_viewer = not self._sync_viewer

            # step graphics
            if self._sync_viewer:
                self.gym.step_graphics(self.sim)
                self.gym.draw_viewer(self.viewer, self.sim, True)
            else:
                self.gym.poll_viewer_events(self.viewer)
    