import time
import numpy as np
import gymnasium as gym

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSPresetProfiles
import sensor_msgs.msg
import geometry_msgs.msg

import libiiwa_msgs.srv


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

        # initialize the ROS node
        rclpy.init()
        self.node = Node(self.__class__.__name__)

        import threading
        threading.Thread(target=self._spin).start()

        # create publishers

        self.pub_command_joint = self.node.create_publisher(sensor_msgs.msg.JointState, '/iiwa/command/joint', QoSPresetProfiles.SYSTEM_DEFAULT.value)
        self.pub_command_cartesian = self.node.create_publisher(geometry_msgs.msg.Pose, '/iiwa/command/cartesian', QoSPresetProfiles.SYSTEM_DEFAULT.value)

        # keep compatibility with libiiwa Python API
        self.robot_state = {"joint_position": np.zeros((7,)),
                            "joint_velocity": np.zeros((7,)),
                            "cartesian_position": np.zeros((3,))}

        # create subscribers
        self.node.create_subscription(msg_type=sensor_msgs.msg.JointState,
                                      topic='/iiwa/state/joint_states',
                                      callback=self._callback_joint_states,
                                      qos_profile=QoSPresetProfiles.SYSTEM_DEFAULT.value)
        self.node.create_subscription(msg_type=geometry_msgs.msg.Pose,
                                      topic='/iiwa/state/end_effector_pose',
                                      callback=self._callback_end_effector_pose,
                                      qos_profile=QoSPresetProfiles.SYSTEM_DEFAULT.value)

        # service clients
        client_control_interface = self.node.create_client(libiiwa_msgs.srv.SetString, '/iiwa/set_control_interface')
        client_control_interface.wait_for_service()
        request = libiiwa_msgs.srv.SetString.Request()
        request.data = "SERVO"  # or "servo"
        client_control_interface.call(request)

        client_joint_velocity_rel = self.node.create_client(libiiwa_msgs.srv.SetNumber, '/iiwa/set_desired_joint_velocity_rel')
        client_joint_acceleration_rel = self.node.create_client(libiiwa_msgs.srv.SetNumber, '/iiwa/set_desired_joint_acceleration_rel')
        client_joint_jerk_rel = self.node.create_client(libiiwa_msgs.srv.SetNumber, '/iiwa/set_desired_joint_jerk_rel')

        client_cartesian_velocity = self.node.create_client(libiiwa_msgs.srv.SetNumber, '/iiwa/set_desired_cartesian_velocity')
        client_cartesian_acceleration = self.node.create_client(libiiwa_msgs.srv.SetNumber, '/iiwa/set_desired_cartesian_acceleration')
        client_cartesian_jerk = self.node.create_client(libiiwa_msgs.srv.SetNumber, '/iiwa/set_desired_cartesian_jerk')

        client_joint_velocity_rel.wait_for_service()
        client_joint_acceleration_rel.wait_for_service()
        client_joint_jerk_rel.wait_for_service()

        client_cartesian_velocity.wait_for_service()
        client_cartesian_acceleration.wait_for_service()
        client_cartesian_jerk.wait_for_service()

        request = libiiwa_msgs.srv.SetNumber.Request()

        request.data = 0.5
        client_joint_velocity_rel.call(request)
        client_joint_acceleration_rel.call(request)
        client_joint_jerk_rel.call(request)

        request.data = 10.0
        client_cartesian_velocity.call(request)
        client_cartesian_acceleration.call(request)
        client_cartesian_jerk.call(request)

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

    def _spin(self):
        rclpy.spin(self.node)

    def _callback_joint_states(self, msg):
        self.robot_state["joint_position"] = np.array(msg.position)
        self.robot_state["joint_velocity"] = np.array(msg.velocity)

    def _callback_end_effector_pose(self, msg):
        positon = msg.position
        self.robot_state["cartesian_position"] = np.array([positon.x, positon.y, positon.z])

    def _get_observation_reward_done(self):
        # observation
        robot_dof_pos = self.robot_state["joint_position"]
        robot_dof_vel = self.robot_state["joint_velocity"]
        end_effector_pos = self.robot_state["cartesian_position"]

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
        msg = sensor_msgs.msg.JointState()
        msg.position = self.robot_default_dof_pos.tolist()
        self.pub_command_joint.publish(msg)
        time.sleep(3)
        msg.position = (self.robot_default_dof_pos + 0.25 * (np.random.rand(7) - 0.5)).tolist()
        self.pub_command_joint.publish(msg)
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

        # control space
        # joint
        if self.control_space == "joint":
            joint_positions = self.robot_state["joint_position"] + (self.robot_dof_speed_scales * self.dt * action * self.action_scale)
            msg = sensor_msgs.msg.JointState()
            msg.position = joint_positions.tolist()
            self.pub_command_joint.publish(msg)
        # cartesian
        elif self.control_space == "cartesian":
            end_effector_pos = self.robot_state["cartesian_position"] + action / 100.0
            msg = geometry_msgs.msg.Pose()
            msg.position.x = end_effector_pos[0]
            msg.position.y = end_effector_pos[1]
            msg.position.z = end_effector_pos[2]
            msg.orientation.x = np.nan
            msg.orientation.y = np.nan
            msg.orientation.z = np.nan
            msg.orientation.w = np.nan
            self.pub_command_cartesian.publish(msg)

        # the use of time.sleep is for simplicity. It does not guarantee control at a specific frequency
        time.sleep(1 / 30.0)

        observation, reward, terminated = self._get_observation_reward_done()

        return observation, reward, terminated, False, {}

    def render(self, *args, **kwargs):
        pass

    def close(self):
        # shutdown the node
        self.node.destroy_node()
        rclpy.shutdown()
