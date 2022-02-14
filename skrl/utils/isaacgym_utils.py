from typing import Union, List

import threading
import imageio
import isaacgym
try:
    import flask
except ImportError:
    flask = None

import torch

import isaacgym.torch_utils as torch_utils
from isaacgym import gymutil, gymtorch, gymapi

# Make sure graphics_device_id is not -1
# pip install flask-socketio

class WebViewer:
    def __init__(self, host: str = "127.0.0.1", port: int = 5000) -> None:
        """
        Web viewer for Isaac Gym

        :param host: Host address (default: "127.0.0.1")
        :type host: str
        :param port: Port number (default: 5000)
        :type port: int
        :param use_socket: Stream using socket.io (default: False)
        :type use_socket: bool
        """
        self._app = flask.Flask(__name__)
        self._app.add_url_rule("/", view_func=self._route_index)
        self._app.add_url_rule("/_route_stream", view_func=self._route_stream)

        self._image = None
        self._camera_id = 0
        self._wait_for_page = True
        self._event_load = threading.Event()
        self._event_stream = threading.Event()

        # start server
        self._thread = threading.Thread(target=lambda: self._app.run(host=host, port=port, debug=False, use_reloader=False), 
                                        daemon=True)
        self._thread.start()
        print("\nStarting web viewer on http://{}:{}/\n".format(host, port))

    def _route_index(self) -> 'flask.Response':
        """Render the web page

        :return: Flask response
        :rtype: flask.Response
        """
        template = """<!doctype html>
        <html lang="en">
        <head>
            <meta charset="utf-8">
            <meta name="viewport" content="width=device-width, initial-scale=1, shrink-to-fit=no">
        </head>
        <body>
            <div>
                <img src="{{ url_for('_route_stream') }}" width="100%">
            </div>
        </body>
        </html>
        """
        self._event_load.set()
        return flask.render_template_string(template)

    def _route_stream(self) -> 'flask.Response':
        """Stream the image to the web page
        
        :return: Flask response
        :rtype: flask.Response
        """
        return flask.Response(self._stream(), mimetype='multipart/x-mixed-replace; boundary=frame')
        
    def _stream(self) -> bytes:
        """Format the image to be streamed

        :return: Image encoded as Content-Type
        :rtype: bytes
        """
        while True:
            # prepare image
            self._event_stream.wait()
            image = imageio.imwrite("<bytes>", self._image, format="JPEG")
            self._event_stream.clear()

            # strem image
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + image + b'\r\n')

    def setup(self, gym: 'isaacgym.gymapi.Gym', sim: 'Sim', envs: List[int], cameras: List[int]) -> None:
        """Setup the web viewer

        :param gym: The gym
        :type gym: isaacgym.gymapi.Gym
        :param sim: Simulation handle
        :type sim: Sim
        :param envs: Environment handles
        :type envs: list of ints
        :param cameras: Camera handles
        :type cameras: list of ints
        """
        self._gym = gym
        self._sim = sim
        self._envs = envs
        self._cameras = cameras

    def render(self, 
               fetch_results: bool = True, 
               step_graphics: bool = True, 
               render_all_camera_sensors: bool = True,
               wait_for_page_load: bool = True) -> None:
        """Render and get the image from the current camera

        This function must be called after the simulation is stepped (post_physics_step).
        The following Isaac Gym functions are called before get the image.
        Their calling can be skipped by setting the corresponding argument to False
        
        - fetch_results
        - step_graphics
        - render_all_camera_sensors
        
        :param fetch_results: Call Gym.fetch_results method (default: True)
        :type fetch_results: bool
        :param step_graphics: Call Gym.step_graphics method (default: True)
        :type step_graphics: bool
        :param render_all_camera_sensors: Call Gym.render_all_camera_sensors method (default: True)
        :type render_all_camera_sensors: bool
        :param wait_for_page_load: Wait for the page to load (default: True)
        :type wait_for_page_load: bool
        """
        # wait for page to load
        if self._wait_for_page:
            if wait_for_page_load:
                if not self._event_load.is_set():
                    print("Waiting for web page to begin loading...")
                self._event_load.wait()
                self._event_load.clear()
            self._wait_for_page = False

        # isaac gym API
        if fetch_results:
            self._gym.fetch_results(self._sim, True)
        if step_graphics:
            self._gym.step_graphics(self._sim)
        if render_all_camera_sensors:
            self._gym.render_all_camera_sensors(self._sim)
        
        # get image
        image = self._gym.get_camera_image(self._sim, 
                                           self._envs[self._camera_id],
                                           self._cameras[self._camera_id], 
                                           gymapi.IMAGE_COLOR)
        self._image = image.reshape(image.shape[0], -1, 4)[..., :3]
        
        # notify stream thread
        self._event_stream.set()


def ik(jacobian_end_effector: torch.Tensor, 
       current_position: torch.Tensor,
       current_orientation: torch.Tensor,
       goal_position: torch.Tensor,
       goal_orientation: Union[torch.Tensor, None] = None,
       damping_factor: float = 0.05) -> torch.Tensor:
    """
    Inverse kinematics using damped least squares method

    :param jacobian_end_effector: Jacobian of end effector
    :type jacobian_end_effector: torch.Tensor
    :param current_position: Current position of end effector
    :type current_position: torch.Tensor
    :param current_orientation: Current orientation of end effector
    :type current_orientation: torch.Tensor
    :param goal_position: Goal position of end effector
    :type goal_position: torch.Tensor
    :param goal_orientation: Goal orientation of end effector (default: None)
    :type goal_orientation: torch.Tensor or None
    :param damping_factor: Damping factor (default: 0.05)
    :type damping_factor: float

    :return: Joint angles delta
    :rtype: torch.Tensor
    """
    # compute errors
    if goal_orientation is None:
        goal_orientation = current_orientation
    q = torch_utils.quat_mul(goal_orientation, torch_utils.quat_conjugate(current_orientation))
    error = torch.cat([goal_position - current_position,  # position error
                       q[:, 0:3] * torch.sign(q[:, 3]).unsqueeze(-1)],  # orientation error
                      dim=-1).unsqueeze(-1)

    # solve damped least squares (dO = J.T * V)
    transpose = torch.transpose(jacobian_end_effector, 1, 2)
    lmbda = torch.eye(6, device=jacobian_end_effector.device) * (damping_factor ** 2)
    return (transpose @ torch.inverse(jacobian_end_effector @ transpose + lmbda) @ error)


def print_arguments(args):
    print("")
    print("Arguments")
    for a in args.__dict__:
        print("  |-- {}: {}".format(a, args.__getattribute__(a)))

def print_asset_option(option):
    print("")
    print("Asset option")
    print("  |-- angular_damping:", option.angular_damping)
    print("  |-- armature:", option.armature)
    print("  |-- collapse_fixed_joints:", option.collapse_fixed_joints)
    print("  |-- convex_decomposition_from_submeshes:", option.convex_decomposition_from_submeshes)
    print("  |-- default_dof_drive_mode:", option.default_dof_drive_mode)
    print("  |-- density:", option.density)
    print("  |-- disable_gravity:", option.disable_gravity)
    print("  |-- fix_base_link:", option.fix_base_link)
    print("  |-- flip_visual_attachments:", option.flip_visual_attachments)
    print("  |-- linear_damping:", option.linear_damping)
    print("  |-- max_angular_velocity:", option.max_angular_velocity)
    print("  |-- max_linear_velocity:", option.max_linear_velocity)
    print("  |-- mesh_normal_mode:", option.mesh_normal_mode)
    print("  |-- min_particle_mass:", option.min_particle_mass)
    print("  |-- override_com:", option.override_com)
    print("  |-- override_inertia:", option.override_inertia)
    print("  |-- replace_cylinder_with_capsule:", option.replace_cylinder_with_capsule)
    print("  |-- slices_per_cylinder:", option.slices_per_cylinder)
    print("  |-- tendon_limit_stiffness:", option.tendon_limit_stiffness)
    print("  |-- thickness:", option.thickness)
    print("  |-- use_mesh_materials:", option.use_mesh_materials)
    print("  |-- use_physx_armature:", option.use_physx_armature)
    print("  |-- vhacd_enabled:", option.vhacd_enabled)
    # print("  |-- vhacd_param:", option.vhacd_param)   # AttributeError: 'isaacgym._bindings.linux-x86_64.gym_36.AssetOption' object has no attribute 'vhacd_param'

def print_sim_components(gym, sim):
    print("")
    print("Sim components")
    print("  |--  env count:", gym.get_env_count(sim))
    print("  |--  actor count:", gym.get_sim_actor_count(sim))
    print("  |--  rigid body count:", gym.get_sim_rigid_body_count(sim))
    print("  |--  joint count:", gym.get_sim_joint_count(sim))
    print("  |--  dof count:", gym.get_sim_dof_count(sim))
    print("  |--  force sensor count:", gym.get_sim_force_sensor_count(sim))

def print_env_components(gym, env):
    print("")
    print("Env components")
    print("  |--  actor count:", gym.get_actor_count(env))
    print("  |--  rigid body count:", gym.get_env_rigid_body_count(env))
    print("  |--  joint count:", gym.get_env_joint_count(env))
    print("  |--  dof count:", gym.get_env_dof_count(env))

def print_actor_components(gym, env, actor):
    print("")
    print("Actor components")
    print("  |--  rigid body count:", gym.get_actor_rigid_body_count(env, actor))
    print("  |--  joint count:", gym.get_actor_joint_count(env, actor))
    print("  |--  dof count:", gym.get_actor_dof_count(env, actor))
    print("  |--  actuator count:", gym.get_actor_actuator_count(env, actor))
    print("  |--  rigid shape count:", gym.get_actor_rigid_shape_count(env, actor))
    print("  |--  soft body count:", gym.get_actor_soft_body_count(env, actor))
    print("  |--  tendon count:", gym.get_actor_tendon_count(env, actor))

def print_dof_properties(gymapi, props):
    print("")
    print("DOF properties")
    print("  |--  hasLimits:", props["hasLimits"])
    print("  |--  lower:", props["lower"])
    print("  |--  upper:", props["upper"])
    print("  |--  driveMode:", props["driveMode"])
    print("  |      |-- {}: gymapi.DOF_MODE_NONE".format(int(gymapi.DOF_MODE_NONE)))
    print("  |      |-- {}: gymapi.DOF_MODE_POS".format(int(gymapi.DOF_MODE_POS)))
    print("  |      |-- {}: gymapi.DOF_MODE_VEL".format(int(gymapi.DOF_MODE_VEL)))
    print("  |      |-- {}: gymapi.DOF_MODE_EFFORT".format(int(gymapi.DOF_MODE_EFFORT)))
    print("  |--  stiffness:", props["stiffness"])
    print("  |--  damping:", props["damping"])
    print("  |--  velocity (max):", props["velocity"])
    print("  |--  effort (max):", props["effort"])
    print("  |--  friction:", props["friction"])
    print("  |--  armature:", props["armature"])

def print_links_and_dofs(gym, asset):
    link_dict = gym.get_asset_rigid_body_dict(asset)
    dof_dict = gym.get_asset_dof_dict(asset)

    print("")
    print("Links")
    for k in link_dict:
        print("  |-- {}: {}".format(k, link_dict[k]))
    print("DOFs")
    for k in dof_dict:
        print("  |-- {}: {}".format(k, dof_dict[k]))
