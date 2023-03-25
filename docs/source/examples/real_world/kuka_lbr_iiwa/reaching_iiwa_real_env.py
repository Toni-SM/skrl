import time
import numpy as np
import gymnasium as gym

import libiiwa


class ReachingIiwa(gym.Env):
    def __init__(self, control_space="joint"):

        self.control_space = control_space  # joint or cartesian

        # spaces
        self.observation_space = gym.spaces.Box(low=-1000, high=1000, shape=(18,), dtype=np.float32)
        if self.control_space == "joint":
            self.action_space = gym.spaces.Box(low=-1, high=1, shape=(7,), dtype=np.float32)
        elif self.control_space == "cartesian":
            self.action_space = gym.spaces.Box(low=-1, high=1, shape=(3,), dtype=np.float32)
        else:
            raise ValueError("Invalid control space:", self.control_space)

        # init iiwa
        print("Connecting to robot...")

        self.robot = libiiwa.LibIiwa()
        self.robot.set_control_interface(libiiwa.ControlInterface.CONTROL_INTERFACE_SERVO)

        self.robot.set_desired_joint_velocity_rel(0.5)
        self.robot.set_desired_joint_acceleration_rel(0.5)
        self.robot.set_desired_joint_jerk_rel(0.5)

        self.robot.set_desired_cartesian_velocity(10)
        self.robot.set_desired_cartesian_acceleration(10)
        self.robot.set_desired_cartesian_jerk(10)

        print("Robot connected")

        self.motion = None
        self.motion_thread = None

        self.dt = 1 / 120.0
        self.action_scale = 2.5
        self.dof_vel_scale = 0.1
        self.max_episode_length = 100
        self.robot_dof_speed_scales = 1
        self.target_pos = np.array([0.65, 0.2, 0.2])
        self.robot_default_dof_pos = np.radians([0, 0, 0, -90, 0, 90, 0])
        self.robot_dof_lower_limits = np.array([-2.9671, -2.0944, -2.9671, -2.0944, -2.9671, -2.0944, -3.0543])
        self.robot_dof_upper_limits = np.array([ 2.9671,  2.0944,  2.9671,  2.0944,  2.9671,  2.0944,  3.0543])

        self.progress_buf = 1
        self.obs_buf = np.zeros((18,), dtype=np.float32)

    def _get_observation_reward_done(self):
        # get robot state
        robot_state = self.robot.get_state(refresh=True)

        # observation
        robot_dof_pos = robot_state["joint_position"]
        robot_dof_vel = robot_state["joint_velocity"]
        end_effector_pos = robot_state["cartesian_position"]

        dof_pos_scaled = 2.0 * (robot_dof_pos - self.robot_dof_lower_limits) / (self.robot_dof_upper_limits - self.robot_dof_lower_limits) - 1.0
        dof_vel_scaled = robot_dof_vel * self.dof_vel_scale

        self.obs_buf[0] = self.progress_buf / float(self.max_episode_length)
        self.obs_buf[1:8] = dof_pos_scaled
        self.obs_buf[8:15] = dof_vel_scaled
        self.obs_buf[15:18] = self.target_pos

        # reward
        distance = np.linalg.norm(end_effector_pos - self.target_pos)
        reward = -distance

        # done
        done = self.progress_buf >= self.max_episode_length - 1
        done = done or distance <= 0.075

        print("Distance:", distance)
        if done:
            print("Target or Maximum episode length reached")
            time.sleep(1)

        return self.obs_buf, reward, done

    def reset(self):
        print("Reseting...")

        # go to 1) safe position, 2) random position
        self.robot.command_joint_position(self.robot_default_dof_pos)
        time.sleep(3)
        dof_pos = self.robot_default_dof_pos + 0.25 * (np.random.rand(7) - 0.5)
        self.robot.command_joint_position(dof_pos)
        time.sleep(1)

        # get target position from prompt
        while True:
            try:
                print("Enter target position (X, Y, Z) in meters")
                raw = input("or press [Enter] key for a random target position: ")
                if raw:
                    self.target_pos = np.array([float(p) for p in raw.replace(' ', '').split(',')])
                else:
                    noise = (2 * np.random.rand(3) - 1) * np.array([0.1, 0.2, 0.2])
                    self.target_pos = np.array([0.6, 0.0, 0.4]) + noise
                print("Target position:", self.target_pos)
                break
            except ValueError:
                print("Invalid input. Try something like: 0.65, 0.0, 0.4")

        input("Press [Enter] to continue")

        self.progress_buf = 0
        observation, reward, done = self._get_observation_reward_done()

        return observation, {}

    def step(self, action):
        self.progress_buf += 1

        # get robot state
        robot_state = self.robot.get_state(refresh=True)

        # control space
        # joint
        if self.control_space == "joint":
            dof_pos = robot_state["joint_position"] + (self.robot_dof_speed_scales * self.dt * action * self.action_scale)
            self.robot.command_joint_position(dof_pos)
        # cartesian
        elif self.control_space == "cartesian":
            end_effector_pos = robot_state["cartesian_position"] + action / 100.0
            self.robot.command_cartesian_pose(end_effector_pos)

        # the use of time.sleep is for simplicity. It does not guarantee control at a specific frequency
        time.sleep(1 / 30.0)

        observation, reward, terminated = self._get_observation_reward_done()

        return observation, reward, terminated, False, {}

    def render(self, *args, **kwargs):
        pass

    def close(self):
        pass
