from typing import List, Optional

import logging
import math
import threading

import numpy as np
import torch


try:
    import flask
except ImportError:
    flask = None

try:
    import imageio
    import isaacgym
    import isaacgym.torch_utils as torch_utils
    from isaacgym import gymapi
except ImportError:
    imageio = None
    isaacgym = None
    torch_utils = None
    gymapi = None


class WebViewer:
    def __init__(self, host: str = "127.0.0.1", port: int = 5000) -> None:
        """
        Web viewer for Isaac Gym

        :param host: Host address (default: "127.0.0.1")
        :type host: str
        :param port: Port number (default: 5000)
        :type port: int
        """
        self._app = flask.Flask(__name__)
        self._app.add_url_rule("/", view_func=self._route_index)
        self._app.add_url_rule("/_route_stream", view_func=self._route_stream)
        self._app.add_url_rule("/_route_input_event", view_func=self._route_input_event, methods=["POST"])

        self._log = logging.getLogger('werkzeug')
        self._log.disabled = True
        self._app.logger.disabled = True

        self._image = None
        self._camera_id = 0
        self._camera_type = gymapi.IMAGE_COLOR
        self._notified = False
        self._wait_for_page = True
        self._pause_stream = False
        self._event_load = threading.Event()
        self._event_stream = threading.Event()

        # start server
        self._thread = threading.Thread(target=lambda: \
            self._app.run(host=host, port=port, debug=False, use_reloader=False), daemon=True)
        self._thread.start()
        print(f"\nStarting web viewer on http://{host}:{port}/\n")

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
            <style>
                html, body {
                    width: 100%; height: 100%;
                    margin: 0; overflow: hidden; display: block;
                    background-color: #000;
                }
            </style>
        </head>
        <body>
            <div>
                <canvas id="canvas" tabindex='1'></canvas>
            </div>

            <script>
                var canvas, context, image;

                function sendInputRequest(data){
                    let xmlRequest = new XMLHttpRequest();
                    xmlRequest.open("POST", "{{ url_for('_route_input_event') }}", true);
                    xmlRequest.setRequestHeader("Content-Type", "application/json");
                    xmlRequest.send(JSON.stringify(data));
                }

                window.onload = function(){
                    canvas = document.getElementById("canvas");
                    context = canvas.getContext('2d');
                    image = new Image();
                    image.src = "{{ url_for('_route_stream') }}";

                    canvas.width = window.innerWidth;
                    canvas.height = window.innerHeight;

                    window.addEventListener('resize', function(){
                        canvas.width = window.innerWidth;
                        canvas.height = window.innerHeight;
                    }, false);

                    window.setInterval(function(){
                        let ratio = image.naturalWidth / image.naturalHeight;
                        context.drawImage(image, 0, 0, canvas.width, canvas.width / ratio);
                    }, 50);

                    canvas.addEventListener('keydown', function(event){
                        if(event.keyCode != 18)
                            sendInputRequest({key: event.keyCode});
                    }, false);

                    canvas.addEventListener('mousemove', function(event){
                        if(event.buttons){
                            let data = {dx: event.movementX, dy: event.movementY};
                            if(event.altKey && event.buttons == 1){
                                data.key = 18;
                                data.mouse = "left";
                            }
                            else if(event.buttons == 2)
                                data.mouse = "right";
                            else if(event.buttons == 4)
                                data.mouse = "middle";
                            else
                                return;
                            sendInputRequest(data);
                        }
                    }, false);

                    canvas.addEventListener('wheel', function(event){
                        sendInputRequest({mouse: "wheel", dz: Math.sign(event.deltaY)});
                    }, false);
                }
            </script>
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

    def _route_input_event(self) -> 'flask.Response':
        """Handle keyboard and mouse input

        :return: Flask response
        :rtype: flask.Response
        """
        def q_mult(q1, q2):
            return [q1[0] * q2[0] - q1[1] * q2[1] - q1[2] * q2[2] - q1[3] * q2[3],
                    q1[0] * q2[1] + q1[1] * q2[0] + q1[2] * q2[3] - q1[3] * q2[2],
                    q1[0] * q2[2] + q1[2] * q2[0] + q1[3] * q2[1] - q1[1] * q2[3],
                    q1[0] * q2[3] + q1[3] * q2[0] + q1[1] * q2[2] - q1[2] * q2[1]]

        def q_conj(q):
            return [q[0], -q[1], -q[2], -q[3]]

        def qv_mult(q, v):
            q2 = [0] + v
            return q_mult(q_mult(q, q2), q_conj(q))[1:]

        def q_from_angle_axis(angle, axis):
            s = math.sin(angle / 2.0)
            return [math.cos(angle / 2.0), axis[0] * s, axis[1] * s, axis[2] * s]

        def p_target(p, q, a=0, b=0, c=1, d=0):
            v = qv_mult(q, [1, 0, 0])
            p1 = [c0 + c1 for c0, c1 in zip(p, v)]
            denominator = a * (p1[0] - p[0]) + b * (p1[1] - p[1]) + c * (p1[2] - p[2])
            if denominator:
                t = -(a * p[0] + b * p[1] + c * p[2] + d) / denominator
                return [p[0] + t * (p1[0] - p[0]), p[1] + t * (p1[1] - p[1]), p[2] + t * (p1[2] - p[2])]
            return v

        # get keyboard and mouse inputs
        data = flask.request.get_json()
        key, mouse = data.get("key", None), data.get("mouse", None)
        dx, dy, dz = data.get("dx", None), data.get("dy", None), data.get("dz", None)

        transform = self._gym.get_camera_transform(self._sim,
                                                   self._envs[self._camera_id],
                                                   self._cameras[self._camera_id])

        # zoom in/out
        if mouse == "wheel":
            # compute zoom vector
            vector = qv_mult([transform.r.w, transform.r.x, transform.r.y, transform.r.z],
                                [-0.025 * dz, 0, 0])

            # update transform
            transform.p.x += vector[0]
            transform.p.y += vector[1]
            transform.p.z += vector[2]

        # orbit camera
        elif mouse == "left":
            # convert mouse movement to angle
            dx *= 0.1 * math.pi / 180
            dy *= 0.1 * math.pi / 180

            # compute rotation (Z-up)
            q = q_from_angle_axis(dx, [0, 0, -1])
            q = q_mult(q, q_from_angle_axis(dy, [1, 0, 0]))

            # apply rotation
            t = p_target([transform.p.x, transform.p.y, transform.p.z],
                        [transform.r.w, transform.r.x, transform.r.y, transform.r.z])
            p = qv_mult(q, [transform.p.x - t[0], transform.p.y - t[1], transform.p.z - t[2]])
            q = q_mult(q, [transform.r.w, transform.r.x, transform.r.y, transform.r.z])

            # update transform
            transform.p.x = p[0] + t[0]
            transform.p.y = p[1] + t[1]
            transform.p.z = p[2] + t[2]
            transform.r.w, transform.r.x, transform.r.y, transform.r.z = q

        # pan camera
        elif mouse == "right":
            # convert mouse movement to angle
            dx *= 0.1 * math.pi / 180
            dy *= 0.1 * math.pi / 180

            # compute rotation (Z-up)
            q = q_from_angle_axis(dx, [0, 0, -1])
            q = q_mult(q, q_from_angle_axis(dy, [1, 0, 0]))

            # apply rotation
            q = q_mult(q, [transform.r.w, transform.r.x, transform.r.y, transform.r.z])

            # update transform
            transform.r.w, transform.r.x, transform.r.y, transform.r.z = q

        # walk camera
        elif mouse == "middle":
            # compute displacement
            vector = qv_mult([transform.r.w, transform.r.x, transform.r.y, transform.r.z],
                             [0, 0.001 * dx, 0.001 * dy])

            # update transform
            transform.p.x += vector[0]
            transform.p.y += vector[1]
            transform.p.z += vector[2]

        # pause stream (V: 86)
        elif key == 86:
            self._pause_stream = not self._pause_stream
            return flask.Response(status=200)

        # change image type (T: 84)
        elif key == 84:
            if self._camera_type == gymapi.IMAGE_COLOR:
                self._camera_type = gymapi.IMAGE_DEPTH
            elif self._camera_type == gymapi.IMAGE_DEPTH:
                self._camera_type = gymapi.IMAGE_COLOR
            return flask.Response(status=200)

        else:
            return flask.Response(status=200)

        self._gym.set_camera_transform(self._cameras[self._camera_id],
                                       self._envs[self._camera_id],
                                       transform)

        return flask.Response(status=200)

    def _stream(self) -> bytes:
        """Format the image to be streamed

        :return: Image encoded as Content-Type
        :rtype: bytes
        """
        while True:
            self._event_stream.wait()

            # prepare image
            image = imageio.imwrite("<bytes>", self._image, format="JPEG")

            # stream image
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + image + b'\r\n')

            self._event_stream.clear()
            self._notified = False

    def setup(self, gym: 'isaacgym.gymapi.Gym', sim: 'isaacgym.gymapi.Sim', envs: List[int], cameras: List[int]) -> None:
        """Setup the web viewer

        :param gym: The gym
        :type gym: isaacgym.gymapi.Gym
        :param sim: Simulation handle
        :type sim: isaacgym.gymapi.Sim
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

        # pause stream
        if self._pause_stream:
            return

        if self._notified:
            return

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
                                           self._camera_type)
        if self._camera_type == gymapi.IMAGE_COLOR:
            self._image = image.reshape(image.shape[0], -1, 4)[..., :3]
        elif self._camera_type == gymapi.IMAGE_DEPTH:
            self._image = -image.reshape(image.shape[0], -1)
            minimum = 0 if np.isinf(np.min(self._image)) else np.min(self._image)
            maximum = 5 if np.isinf(np.max(self._image)) else np.max(self._image)
            self._image = np.clip(1 - (self._image - minimum) / (maximum - minimum), 0, 1)
            self._image = np.uint8(255 * self._image)
        else:
            raise ValueError("Unsupported camera type")

        # notify stream thread
        self._event_stream.set()
        self._notified = True


def ik(jacobian_end_effector: torch.Tensor,
       current_position: torch.Tensor,
       current_orientation: torch.Tensor,
       goal_position: torch.Tensor,
       goal_orientation: Optional[torch.Tensor] = None,
       damping_factor: float = 0.05,
       squeeze_output: bool = True) -> torch.Tensor:
    """
    Inverse kinematics using damped least squares method

    :param jacobian_end_effector: End effector's jacobian
    :type jacobian_end_effector: torch.Tensor
    :param current_position: End effector's current position
    :type current_position: torch.Tensor
    :param current_orientation: End effector's current orientation
    :type current_orientation: torch.Tensor
    :param goal_position: End effector's goal position
    :type goal_position: torch.Tensor
    :param goal_orientation: End effector's goal orientation (default: None)
    :type goal_orientation: torch.Tensor or None
    :param damping_factor: Damping factor (default: 0.05)
    :type damping_factor: float
    :param squeeze_output: Squeeze output (default: True)
    :type squeeze_output: bool

    :return: Change in joint angles
    :rtype: torch.Tensor
    """
    if goal_orientation is None:
        goal_orientation = current_orientation

    # compute error
    q = torch_utils.quat_mul(goal_orientation, torch_utils.quat_conjugate(current_orientation))
    error = torch.cat([goal_position - current_position,  # position error
                       q[:, 0:3] * torch.sign(q[:, 3]).unsqueeze(-1)],  # orientation error
                      dim=-1).unsqueeze(-1)

    # solve damped least squares (dO = J.T * V)
    transpose = torch.transpose(jacobian_end_effector, 1, 2)
    lmbda = torch.eye(6, device=jacobian_end_effector.device) * (damping_factor ** 2)
    if squeeze_output:
        return (transpose @ torch.inverse(jacobian_end_effector @ transpose + lmbda) @ error).squeeze(dim=2)
    else:
        return transpose @ torch.inverse(jacobian_end_effector @ transpose + lmbda) @ error

def print_arguments(args):
    print("")
    print("Arguments")
    for a in args.__dict__:
        print(f"  |-- {a}: {args.__getattribute__(a)}")

def print_asset_options(asset_options: 'isaacgym.gymapi.AssetOptions', asset_name: str = ""):
    attrs = ["angular_damping", "armature", "collapse_fixed_joints", "convex_decomposition_from_submeshes",
             "default_dof_drive_mode", "density", "disable_gravity", "fix_base_link", "flip_visual_attachments",
             "linear_damping", "max_angular_velocity", "max_linear_velocity", "mesh_normal_mode", "min_particle_mass",
             "override_com", "override_inertia", "replace_cylinder_with_capsule", "tendon_limit_stiffness", "thickness",
             "use_mesh_materials", "use_physx_armature", "vhacd_enabled"]  # vhacd_params
    print("\nAsset options{}".format(f" ({asset_name})" if asset_name else ""))
    for attr in attrs:
        print("  |-- {}: {}".format(attr, getattr(asset_options, attr) if hasattr(asset_options, attr) else "--"))
        # vhacd attributes
        if attr == "vhacd_enabled" and hasattr(asset_options, attr) and getattr(asset_options, attr):
            vhacd_attrs = ["alpha", "beta", "concavity", "convex_hull_approximation", "convex_hull_downsampling",
                           "max_convex_hulls", "max_num_vertices_per_ch", "min_volume_per_ch", "mode", "ocl_acceleration",
                           "pca", "plane_downsampling", "project_hull_vertices", "resolution"]
            print("  |-- vhacd_params:")
            for vhacd_attr in vhacd_attrs:
                print("  |   |-- {}: {}".format(vhacd_attr, getattr(asset_options.vhacd_params, vhacd_attr) \
                    if hasattr(asset_options.vhacd_params, vhacd_attr) else "--"))

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
        print(f"  |-- {k}: {link_dict[k]}")
    print("DOFs")
    for k in dof_dict:
        print(f"  |-- {k}: {dof_dict[k]}")
