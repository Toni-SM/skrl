import math
from isaacgym import gymapi

from skrl.utils import isaacgym_utils


# create a web viewer instance
web_viewer = isaacgym_utils.WebViewer()

# configure and create simulation
sim_params = gymapi.SimParams()
sim_params.up_axis = gymapi.UP_AXIS_Z
sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.8)
sim_params.physx.solver_type = 1
sim_params.physx.num_position_iterations = 4
sim_params.physx.num_velocity_iterations = 1
sim_params.physx.use_gpu = True
sim_params.use_gpu_pipeline = True

gym = gymapi.acquire_gym()
sim = gym.create_sim(compute_device=0, graphics_device=0, type=gymapi.SIM_PHYSX, params=sim_params)

# setup num_envs and env's grid
num_envs = 1
spacing = 2.0
env_lower = gymapi.Vec3(-spacing, -spacing, 0.0)
env_upper = gymapi.Vec3(spacing, 0.0, spacing)

# add ground plane
plane_params = gymapi.PlaneParams()
plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
gym.add_ground(sim, plane_params)

envs = []
cameras = []

for i in range(num_envs):
    # create env
    env = gym.create_env(sim, env_lower, env_upper, int(math.sqrt(num_envs)))

    # add sphere
    pose = gymapi.Transform()
    pose.p, pose.r = gymapi.Vec3(0.0, 0.0, 1.0), gymapi.Quat(0.0, 0.0, 0.0, 1.0)
    gym.create_actor(env, gym.create_sphere(sim, 0.2, None), pose, "sphere", i, 0)

    # add camera
    cam_props = gymapi.CameraProperties()
    cam_props.width, cam_props.height = 300, 300
    cam_handle = gym.create_camera_sensor(env, cam_props)
    gym.set_camera_location(cam_handle, env, gymapi.Vec3(1, 1, 1), gymapi.Vec3(0, 0, 0))

    envs.append(env)
    cameras.append(cam_handle)

# setup web viewer
web_viewer.setup(gym, sim, envs, cameras)

gym.prepare_sim(sim)


for i in range(100000):
    gym.simulate(sim)

    # render the scene
    web_viewer.render(fetch_results=True,
                      step_graphics=True,
                      render_all_camera_sensors=True,
                      wait_for_page_load=True)
