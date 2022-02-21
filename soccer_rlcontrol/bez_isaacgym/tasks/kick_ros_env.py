import os

import numpy as np

if "ROS_NAMESPACE" not in os.environ:
    os.environ["ROS_NAMESPACE"] = "/robot1"
import time
import math
import sys
import rospy
from sensor_msgs.msg import JointState, Imu

from std_msgs.msg import Int32, Int8MultiArray

from matplotlib import pyplot as plt

from utils.torch_jit_utils import *
from .base.ros_task import RosTask

import torch
from torch._tensor import Tensor
from typing import Tuple, Dict, Any
import enum

import matplotlib

matplotlib.use("TkAgg")


class Joints(enum.IntEnum):
    HEAD_1 = 0
    HEAD_2 = 1
    LEFT_ARM_1 = 2
    LEFT_ARM_2 = 3
    LEFT_LEG_1 = 4
    LEFT_LEG_2 = 5
    LEFT_LEG_3 = 6
    LEFT_LEG_4 = 7
    LEFT_LEG_5 = 8
    LEFT_LEG_6 = 9
    RIGHT_ARM_1 = 10
    RIGHT_ARM_2 = 11
    RIGHT_LEG_1 = 12
    RIGHT_LEG_2 = 13
    RIGHT_LEG_3 = 14
    RIGHT_LEG_4 = 15
    RIGHT_LEG_5 = 16
    RIGHT_LEG_6 = 17


class KickRosEnv(RosTask):

    def __init__(self, cfg, sim_device, graphics_device_id, headless):
        # Setup
        self.cfg = cfg

        # bez base init state
        pos = self.cfg["env"]["bezInitState"]["pos"]
        rot = self.cfg["env"]["bezInitState"]["rot"]
        v_lin = self.cfg["env"]["bezInitState"]["vLinear"]
        v_ang = self.cfg["env"]["bezInitState"]["vAngular"]
        self.bez_init_state = pos + rot + v_lin + v_ang

        # ball base init state
        pos = self.cfg["env"]["ballInitState"]["pos"]
        rot = self.cfg["env"]["ballInitState"]["rot"]
        v_lin = self.cfg["env"]["ballInitState"]["vLinear"]
        v_ang = self.cfg["env"]["ballInitState"]["vAngular"]
        self.ball_init_state = pos + rot + v_lin + v_ang

        # goal state
        goal = self.cfg["env"]["goalState"]["goal"]

        # max episode length
        self.max_episode_length_s = self.cfg["env"]["learn"]["episodeLength_s"]

        # debug
        self.debug_rewards = self.cfg["env"]["debug"]["rewards"]

        # joint positions
        self.named_default_joint_angles = self.cfg["env"]["readyJointAngles"]  # defaultJointAngles  readyJointAngles
        self.named_joint_limit_high = self.cfg["env"]["JointLimitHigh"]
        self.named_joint_limit_low = self.cfg["env"]["JointLimitLow"]

        # Observation dimension
        self.orn_dim = 2
        self.imu_dim = 6
        self.feet_dim = 8
        self.dof_dim = 18  # or 16 if we remove head
        self.ball_dim = 2  # ball position

        # Limits
        self.imu_max_ang_vel = 8.7266
        self.imu_max_lin_acc = 2. * 9.81
        self.AX_12_velocity = (59 / 60) * 2 * np.pi  # Ask sharyar for later - 11.9 rad/s
        self.MX_28_velocity = 2 * np.pi  # 24.5 rad/s

        # IMU NOISE
        self._IMU_LIN_STDDEV_BIAS = 0.  # 0.02 * _MAX_LIN_ACC
        self._IMU_ANG_STDDEV_BIAS = 0.  # 0.02 * _MAX_ANG_VEL
        self._IMU_LIN_STDDEV = 0.00203 * self.imu_max_lin_acc
        self._IMU_ANG_STDDEV = 0.00804 * self.imu_max_ang_vel

        # FEET NOISE
        self._FEET_FALSE_CHANCE = 0.01

        # Joint angle noise
        self._JOIN_ANGLE_STDDEV = np.pi / 2048
        self._JOIN_VELOCITY_STDDEV = self._JOIN_ANGLE_STDDEV / 120

        # Number of observation and actions
        self.cfg["env"][
            "numObservations"] = self.dof_dim + self.dof_dim + self.imu_dim + self.orn_dim + self.feet_dim + self.ball_dim  # 54
        self.cfg["env"]["numActions"] = self.dof_dim

        super().__init__(config=self.cfg, sim_device=sim_device, graphics_device_id=graphics_device_id,
                         headless=headless)

        # simulation parameters
        self.dt = self.cfg["sim"]["dt"]  # / self.cfg["sim"]["substeps"]

        # TODO Implement
        self.orn_limit = torch.tensor([1.] * self.orn_dim, device=self.device)
        self.feet_limit = torch.tensor([1.6] * self.feet_dim, device=self.device)
        self.ball_start_limit = torch.tensor([0.3] * self.ball_dim, device=self.device)

        # Setting default positions
        self.default_dof_pos = torch.zeros(self.num_envs, self.dof_dim, dtype=torch.float, device=self.device,
                                           requires_grad=False)
        self.default_dof_pos_ext = torch.zeros(self.num_envs, self.dof_dim, dtype=torch.float, device=self.device,
                                           requires_grad=False)
        self.dof_pos_limits_upper = torch.zeros_like(self.default_dof_pos)
        self.dof_pos_limits_lower = torch.zeros_like(self.default_dof_pos)
        self.dof_names = [
            "left_arm_motor_0",
            "left_arm_motor_1",
            "right_arm_motor_0",
            "right_arm_motor_1",
            "left_leg_motor_0",
            "left_leg_motor_1",
            "left_leg_motor_2",
            "left_leg_motor_3",
            "left_leg_motor_4",
            "left_leg_motor_5",
            "right_leg_motor_0",
            "right_leg_motor_1",
            "right_leg_motor_2",
            "right_leg_motor_3",
            "right_leg_motor_4",
            "right_leg_motor_5",
            "head_motor_0",
            "head_motor_1"
        ]
        self.dof_names_ext = [
            "head_motor_0",
            "head_motor_1",
            "left_arm_motor_0",
            "left_arm_motor_1",
            "left_leg_motor_0",
            "left_leg_motor_1",
            "left_leg_motor_2",
            "left_leg_motor_3",
            "left_leg_motor_4",
            "left_leg_motor_5",
            "right_arm_motor_0",
            "right_arm_motor_1",
            "right_leg_motor_0",
            "right_leg_motor_1",
            "right_leg_motor_2",
            "right_leg_motor_3",
            "right_leg_motor_4",
            "right_leg_motor_5"

        ]

        for i in range(self.cfg["env"]["numActions"]):
            name = self.dof_names[i]
            angle = self.named_default_joint_angles[name]
            self.default_dof_pos[:, i] = angle
            self.dof_pos_limits_upper[:, i] = self.named_joint_limit_high[name]
            self.dof_pos_limits_lower[:, i] = self.named_joint_limit_low[name]

        for i, name in enumerate(self.dof_names_ext):
            dof_index = self.dof_names.index(name)
            self.default_dof_pos_ext[:, i] = self.default_dof_pos[..., dof_index]

        # initialize some data used later on
        self.actions = torch.zeros_like(self.default_dof_pos)
        self.max_episode_length = int(self.max_episode_length_s / self.dt + 0.5)

        self.goal = torch.tensor([goal], device=self.device)
        self.bez_init_xy = torch.tensor(self.bez_init_state[0:2], device=self.device)
        self.ball_init = torch.tensor([self.ball_init_state[0:2]], device=self.device)

        # Dof
        self.dof_pos_bez = torch.zeros_like(self.actions)
        self.dof_vel_bez = torch.zeros_like(self.actions)
        self.prev_dof_pos_bez = torch.zeros_like(self.actions)

        # Bez
        self.root_pos_bez = torch.zeros(self.num_envs, 3,
                                        device=self.device)  # could use amcl update but not really needed
        self.root_orient_bez = torch.zeros(self.num_envs, 4, device=self.device)

        # Ball
        self.root_pos_ball = torch.zeros_like(self.root_pos_bez)
        self.root_vel_ball = torch.zeros_like(self.root_pos_bez)  # TODO

        # Sensors
        self.imu = torch.zeros(self.num_envs, self.imu_dim, device=self.device)
        self.feet = -1.0 * torch.ones(self.num_envs, self.feet_dim, device=self.device)

        self.dof_publisher = rospy.Publisher(os.environ["ROS_NAMESPACE"] + "/joint_command", JointState, queue_size=10)
        self.dof_subscriber = rospy.Subscriber(os.environ["ROS_NAMESPACE"] + "/joint_states", JointState,
                                               self.dofStateCallback)
        self.imu_subscriber = rospy.Subscriber(os.environ["ROS_NAMESPACE"] + "/imu_filtered", Imu, self.imu_callback,
                                               queue_size=1)
        self.feet_subscriber = rospy.Subscriber(os.environ["ROS_NAMESPACE"] + "/foot_pressure", Int8MultiArray,
                                                self.feet_callback,
                                                queue_size=1)

        if self.debug_rewards:
            self.time = []
            self.kick_vel = []
            self.up_proj = []
            self.goal_angle_diff = []
            self.distance_kicked_norm = []
            self.vel_reward_scaled = []
            self.pos_reward_scaled = []
            self.max_kick_velocity = 0.0

            self.fig, self.ax = plt.subplots(2, 3)

            self.fig.show()

            # We need to draw the canvas before we start animating...
            self.fig.canvas.draw()

        self.reset_idx()

    def imu_callback(self, imu: Imu):
        lin_accel = torch.tensor(
            np.array([imu.linear_acceleration.x, imu.linear_acceleration.y, imu.linear_acceleration.z]),
            device=self.device)
        ang_vel = torch.tensor(np.array([imu.angular_velocity.x, imu.angular_velocity.y, imu.angular_velocity.z]),
                               device=self.device)
        self.imu[0] = torch.cat((lin_accel, ang_vel))
        self.root_orient_bez[0] = torch.tensor(
            np.array([imu.orientation.x, imu.orientation.y, imu.orientation.z, imu.orientation.w]), device=self.device)

    def feet_callback(self, pressure_sensors: Int8MultiArray):
        self.feet[0] = torch.tensor(pressure_sensors.data, device=self.device)

    def dofStateCallback(self, dofState: JointState):
        self.dof_pos_bez[0] = torch.tensor(dofState.position, device=self.device)
        self.dof_vel_bez[0] = (self.dof_pos_bez[0] - self.prev_dof_pos_bez[0]) / 0.00833  # self.dt
        self.prev_dof_pos_bez[0] = self.dof_pos_bez[0]

    def pre_physics_step(self, actions):
        # implement pre-physics simulation code here
        #    - e.g. apply actions
        actions[..., 0:2] = 0.0  # Remove head action
        self.actions = actions.clone().to(self.device) + self.default_dof_pos_ext
        temp_actions = actions.clone().to(self.device) + self.default_dof_pos_ext
        for i, name in enumerate(self.dof_names):
            dof_index = self.dof_names_ext.index(name)
            self.actions[..., i] = temp_actions[..., dof_index]

        print("self.action: ", self.actions.cpu().numpy())
        # Position Control
        targets = tensor_clamp(self.actions, self.dof_pos_limits_lower,
                               self.dof_pos_limits_upper)
        # targets = tensor_clamp(self.default_dof_pos, self.dof_pos_limits_lower,
        #                        self.dof_pos_limits_upper)
        targets = targets.cpu().numpy()[0]
        js = JointState()
        js.name = []
        js.header.stamp = rospy.Time.now()
        js.position = []
        js.effort = []
        for i, n in enumerate(self.dof_names):
            js.name.append(n)
            js.position.append(targets[i])
        try:
            self.dof_publisher.publish(js)
        except rospy.exceptions.ROSException as ex:
            print(ex)
            exit(0)

    def post_physics_step(self):
        # implement post-physics simulation code here
        #    - e.g. compute reward, compute observations
        self.progress_buf += 1

        # TODO Unsure how to use reset
        # env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        # if len(env_ids) > 0:
        #     self.reset_idx(env_ids)

        self.compute_observations()
        self.compute_reward(self.actions)

    def off_orn(self):
        vec = compute_off_orn(
            # tensors
            self.root_pos_bez,
            self.root_orient_bez,
            self.goal
        )
        return vec

    def compute_reward(self, actions):

        self.rew_buf[:], self.reset_buf[:] = compute_bez_reward(
            # tensors
            self.dof_pos_bez,
            self.dof_vel_bez,
            self.default_dof_pos,
            self.imu,
            self.root_pos_bez,
            self.root_orient_bez,
            self.root_pos_ball,
            self.root_vel_ball,
            self.goal,
            self.ball_init,
            self.bez_init_xy,
            self.reset_buf,
            self.progress_buf,
            self.feet,
            # self.left_arm_contact_forces,
            # self.right_arm_contact_forces,
            # int
            self.max_episode_length,
            self.num_envs,
        )

    def compute_observations(self):

        off_orn = self.off_orn()
        print("self.imu: ", self.imu[0].cpu().numpy())
        print("self.feet: ", self.feet[0].cpu().numpy())
        print("self.dof_pos_bez: ", self.dof_pos_bez[0].cpu().numpy())
        print("self.dof_vel_bez: ", self.dof_vel_bez[0].cpu().numpy())
        print("self.orient: ", self.root_orient_bez[0].cpu().numpy())
        self.obs_buf[:] = compute_bez_observations(
            # tensors
            self.dof_pos_bez,  # 18
            self.dof_vel_bez,  # 18
            self.imu,  # 6
            off_orn,  # 2
            self.feet,  # 8
            self.ball_init  # 2
        )

    def reset_idx(self):
        # Randomization can happen only at reset time, since it can reset actor positions on GPU

        self.dof_pos_bez = tensor_clamp(self.default_dof_pos,
                                        self.dof_pos_limits_lower, self.dof_pos_limits_upper)

        if self.debug_rewards:
            self.kick_vel = []
            self.up_proj = []
            self.goal_angle_diff = []
            self.distance_kicked_norm = []
            self.vel_reward_scaled = []
            self.pos_reward_scaled = []
            self.time = []

            self.ax[0, 0].cla()
            self.ax[0, 1].cla()
            self.ax[0, 2].cla()
            self.ax[1, 0].cla()
            self.ax[1, 1].cla()
            self.ax[1, 2].cla()


#####################################################################
###=========================jit functions=========================###
#####################################################################

@torch.jit.script
def quaternion_to_matrix(quaternions: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as quaternions to rotation matrices.
    Args:
        quaternions: quaternions with real part first,
            as tensor of shape (..., 4).
    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    r, i, j, k = torch.unbind(quaternions, -1)
    two_s = 2.0 / (quaternions * quaternions).sum(-1)

    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )

    return o.reshape(quaternions.shape[:-1] + (3, 3))


@torch.jit.script
def compute_off_orn(
        # tensors
        root_pos_bez: Tensor,
        root_orient_bez: Tensor,
        goal: Tensor

) -> Tensor:
    distance_to_goal = torch.sub(goal, root_pos_bez[..., 0:2])
    distance_to_goal_norm = torch.reshape(torch.linalg.norm(distance_to_goal, dim=1), (-1, 1))
    distance_unit_vec = torch.div(distance_to_goal, distance_to_goal_norm)

    x = root_orient_bez[..., 0]
    y = root_orient_bez[..., 1]
    z = root_orient_bez[..., 2]
    w = root_orient_bez[..., 3]

    roll, pitch, yaw = get_euler_xyz(root_orient_bez[..., 0:4])
    cos = torch.cos(yaw)
    sin = torch.sin(yaw)
    d2_vect = torch.cat((cos.reshape((-1, 1)), sin.reshape((-1, 1))), dim=-1)
    cos = torch.sum(d2_vect * distance_unit_vec, dim=-1)
    distance_unit_vec_3d = torch.nn.functional.pad(input=distance_unit_vec, pad=(0, 1, 0, 0), mode='constant',
                                                   value=0.0)
    d2_vect_3d = torch.nn.functional.pad(input=d2_vect, pad=(0, 1, 0, 0), mode='constant',
                                         value=0.0)
    sin = torch.linalg.norm(torch.cross(distance_unit_vec_3d, d2_vect_3d, dim=1), dim=1)
    vec = torch.cat((sin.reshape((-1, 1)), -cos.reshape((-1, 1))), dim=1)

    return vec


@torch.jit.script
def compute_bez_reward(
        # tensors
        dof_pos_bez: Tensor,
        dof_vel_bez: Tensor,
        default_dof_pos: Tensor,
        imu: Tensor,
        root_pos_bez: Tensor,
        root_orient_bez: Tensor,
        root_pos_ball: Tensor,
        root_vel_ball: Tensor,
        goal: Tensor,
        ball_init: Tensor,
        bez_init_xy: Tensor,
        reset_buf: Tensor,
        progress_buf: Tensor,
        feet: Tensor,
        # left_contact_forces: Tensor,
        # right_contact_forces: Tensor,
        max_episode_length: int,
        num_envs: int

) -> Tuple[Tensor, Tensor]:  # (reward, reset)
    # # Calculate Velocity direction field
    # distance_to_ball = torch.sub(root_pos_ball[..., 0:2], root_pos_bez[..., 0:2])
    # distance_to_ball_norm = torch.reshape(torch.linalg.norm(distance_to_ball, dim=1), (-1, 1))
    # bez_to_ball_unit_vec = torch.div(distance_to_ball, distance_to_ball_norm)
    # velocity_forward_reward = torch.sum(torch.mul(bez_to_ball_unit_vec, imu[..., 0:2]), dim=1)
    #
    # distance_to_goal = torch.sub(goal, root_pos_ball[..., 0:2])
    # distance_to_goal_norm = torch.reshape(torch.linalg.norm(distance_to_goal, dim=1), (-1, 1))
    # ball_to_goal_unit_vec = torch.div(distance_to_goal, distance_to_goal_norm)
    # ball_velocity_forward_reward = torch.sum(torch.mul(ball_to_goal_unit_vec, root_vel_ball[..., 0:2]), dim=1)
    #
    # # Testing
    # # ball_init[..., 0] = 0.175
    # # ball_init[..., 0] = 0.0
    #
    # distance_to_goal_init = torch.sub(goal, ball_init)  # nx2
    # distance_to_goal_init_norm = torch.reshape(torch.linalg.norm(distance_to_goal_init, dim=1), (-1, 1))
    # ball_init_to_goal_unit_vec = torch.div(distance_to_goal_init, distance_to_goal_init_norm)  # 2xn / nx1 = nx2
    #
    # ball_to_goal_angle = torch.atan2(ball_to_goal_unit_vec[..., 1], ball_to_goal_unit_vec[..., 0])
    # ball_init_to_goal_angle = torch.atan2(ball_init_to_goal_unit_vec[..., 1], ball_init_to_goal_unit_vec[..., 0])
    # goal_angle_diff = torch.abs((ball_init_to_goal_angle - ball_to_goal_angle))
    # goal_angle_diff = torch.reshape(goal_angle_diff, (-1))
    #
    # up_vec = get_basis_vector(root_orient_bez[..., 0:4], up_vec).view(num_envs, 3)
    # up_proj = up_vec[:, 2]
    # roll, pitch, yaw = get_euler_xyz(root_orient_bez[..., 0:4])
    #
    # DESIRED_HEIGHT = 0.325  # Height of ready position
    #
    # # reward
    #
    # # old
    # vel_bez = torch.cat((imu_lin_bez, imu_ang_bez), dim=1)
    # vel_reward = torch.linalg.norm(vel_bez, dim=1)
    # vel_reward_scaled = torch.mul(vel_reward, 0.05)
    # # new
    # # vel_lin_reward = torch.mul(torch.linalg.norm(imu_lin_bez, dim=1), 0.05)
    # # vel_ang_reward = torch.mul(torch.linalg.norm(imu_ang_bez, dim=1), 0.05)
    # # vel_reward = torch.add(vel_lin_reward, vel_ang_reward)
    #
    # pos_reward = torch.linalg.norm(default_dof_pos - dof_pos_bez, dim=1)
    # pos_reward_scaled = torch.mul(pos_reward, 0.05)
    # distance_to_height_orig = torch.abs(0.325 - root_pos_bez[..., 2])
    # distance_to_height = torch.abs(DESIRED_HEIGHT - root_pos_bez[..., 2])
    # # distance_to_height = torch.abs(DESIRED_HEIGHT - up_proj)
    # distance_to_height_scaled = torch.mul(distance_to_height, 1)
    # distance_kicked_norm = torch.linalg.norm(torch.sub(root_pos_ball[..., 0:2], ball_init), dim=1)
    # goal_angle_diff_scaled = torch.mul(goal_angle_diff, 0.1)
    #
    # # Feet reward
    # ground_feet = torch.sum(feet, dim=1)
    # ground_feet_scaled = torch.mul(ground_feet, 0.01)  # not necessary
    # # ground_feet_scaled = torch.sub(ground_feet_scaled, 0.004)
    #
    # # DOF Velocity reward
    # dof_vel_reward = torch.linalg.norm(dof_vel_bez, dim=1)
    # dof_vel_reward_scaled = torch.mul(dof_vel_reward, 0.001)  # doesnt work
    #
    # #  0.1 * ball_velocity_forward_reward - ((distance_to_height + (0.05 * vel_reward + 0.05 * pos_reward)) - 0.01 * ground_feet)
    # #  0.1 * ball_velocity_forward_reward - distance_to_height - 0.05 * vel_reward - 0.05 * pos_reward
    # #  0.1 * ball_velocity_forward_reward - distance_to_height - 0.05 * vel_reward - 0.05 * pos_reward + 0.01 * ground_feet
    # #  0.1 * ball_velocity_forward_reward - distance_to_height - 0.05 * vel_reward - 0.05 * pos_reward - 0.01 * goal_angle_diff
    # vel_pos_reward = torch.add(vel_reward_scaled, pos_reward_scaled)
    # height_vel_pos_reward = torch.add(distance_to_height_scaled, vel_pos_reward)
    # # height_vel_pos_reward = torch.sub(height_vel_pos_reward, ground_feet_scaled)
    # # height_vel_pos_reward = torch.add(height_vel_pos_reward, goal_angle_diff_scaled)
    # # height_vel_pos_reward = torch.add(height_vel_pos_reward, dof_vel_reward_scaled)
    # # height_pos_reward = torch.add(distance_to_height, pos_reward)
    # # height_pos_reward_scaled = torch.mul(height_pos_reward, 1)
    # ball_velocity_forward_reward_scaled_after = torch.mul(ball_velocity_forward_reward, 0.1)
    # ball_velocity_forward_reward_scaled_before = torch.mul(ball_velocity_forward_reward, 0.1)
    # ball_height_vel_pos_reward = torch.sub(ball_velocity_forward_reward_scaled_after, height_vel_pos_reward)
    #
    # # 0.1 * ball_velocity_forward_reward + 0.05 * velocity_forward_reward - distance_to_height
    # velocity_forward_reward_scaled = torch.mul(velocity_forward_reward, 0.05)
    # vel_height_reward = torch.sub(velocity_forward_reward_scaled, distance_to_height_scaled)
    # # height_goal_reward = torch.add(distance_to_height_scaled, goal_angle_diff_scaled)
    # # vel_height_reward = torch.sub(velocity_forward_reward_scaled, height_goal_reward)
    # ball_vel_height_reward = torch.add(ball_velocity_forward_reward_scaled_before, vel_height_reward)
    #
    # reward = torch.where(distance_kicked_norm > 0.3,
    #                      ball_height_vel_pos_reward,
    #                      ball_vel_height_reward
    #                      )
    #
    # # Reset
    ones = torch.ones_like(reset_buf)
    # termination_scale = -1.0
    #
    # # Fall
    # # state = torch.where(distance_kicked_norm > 0.75, ones,
    # #                     torch.zeros_like(reward))
    # #
    # # reset_dist = torch.where(root_pos_bez[..., 2] < 0.275, ones, reset_buf)
    # # reward_dist = torch.where(root_pos_bez[..., 2] < 0.275, torch.ones_like(reward) * termination_scale, reward)
    # # reset_proj = torch.where(up_proj < 0.7, ones, reset_buf)
    # # reward_proj = torch.where(up_proj < 0.7, torch.ones_like(reward) * termination_scale, reward)
    # #
    # # reset = torch.where(
    # #     state == 1.0,
    # #     reset_proj,
    # #     reset_dist)
    # # reward = torch.where(
    # #     state == 1.0,
    # #     reward_proj,
    # #     reward_dist)
    #
    # reset = torch.where(root_pos_bez[..., 2] < 0.275, ones, reset_buf)
    # reward = torch.where(root_pos_bez[..., 2] < 0.275, torch.ones_like(reward) * termination_scale, reward)
    # # pitch_angle = torch.abs(normalize_angle(pitch))
    # # reset = torch.where(pitch_angle > 1, ones, reset_buf)
    # # reward = torch.where(pitch_angle > 1, torch.ones_like(reward) * termination_scale, reward)
    # # reset = torch.where(up_proj < 0.7, ones, reset_buf)
    # # reward = torch.where(up_proj < 0.7, torch.ones_like(reward) * termination_scale, reward)
    #
    # # Bez: Out of Bound
    # distance_traveled_norm = torch.reshape(torch.linalg.norm(torch.sub(root_pos_bez[..., 0:2], bez_init_xy), dim=1),
    #                                        (-1))
    # reset = torch.where(
    #     distance_traveled_norm > 0.5,
    #     ones,
    #     reset)
    # reward = torch.where(
    #     distance_traveled_norm > 0.5,
    #     torch.ones_like(reward) * termination_scale,
    #     reward)
    #
    # # Ball: Out of Bound
    #
    # distance_to_goal_norm = torch.reshape(distance_to_goal_norm, (-1))
    # # maybe needed for training after does not work better as a reward
    # # state = torch.where(distance_kicked_norm < 0.3, ones,
    # #                     torch.zeros_like(reward))
    # #
    # # state = torch.where(goal_angle_diff > 0.01, state + torch.ones_like(reward),
    # #                     state)
    # #
    # # reset = torch.where(
    # #     state == 2,
    # #     ones,
    # #     reset)
    # # reward = torch.where(
    # #     state == 2,
    # #     torch.ones_like(reward) * termination_scale,
    # #     reward)
    #
    # reset = torch.where(
    #     goal_angle_diff > 1.5708,  # 1.5708  0.01 0.3
    #     ones,
    #     reset)
    # reward = torch.where(
    #     goal_angle_diff > 1.5708,
    #     torch.ones_like(reward) * termination_scale,
    #     reward)
    #
    # # Close to the Goal
    # reset = torch.where(distance_to_goal_norm < 0.05, ones,
    #                     reset)
    # reward = torch.where(distance_to_goal_norm < 0.05,
    #                      torch.ones_like(reward) * (100.0 - 100.0 * (progress_buf / max_episode_length)),
    #                      # roughly 3 seconds
    #                      reward)
    #
    # Horizon Ended
    reset = torch.zeros_like(reset_buf)
    reward = torch.zeros_like(reset_buf)
    reset = torch.where(progress_buf >= max_episode_length, ones, reset)

    reward = torch.where(progress_buf >= max_episode_length, torch.zeros_like(reward),
                         reward)

    return reward, reset


@torch.jit.script
def compute_bez_observations(
        # tensors
        dof_pos_bez: Tensor,
        dof_vel_bez: Tensor,  # 18
        imu: Tensor,  # 6
        off_orn: Tensor,  # 2
        feet: Tensor,  # 8
        ball_init: Tensor  # 2

) -> Tensor:
    obs = torch.cat((dof_pos_bez,
                     dof_vel_bez,  # 18
                     imu,  # 6
                     off_orn,  # 2
                     feet,  # 8
                     ball_init  # 2
                     ), dim=-1)

    return obs
