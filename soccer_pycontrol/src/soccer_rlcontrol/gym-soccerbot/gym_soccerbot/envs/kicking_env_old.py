import gym
from gym import spaces
from gym.utils import seeding

import os
import pybullet as pb
import pybullet_data
import math
import numpy as np
import enum
from pybullet_utils import bullet_client as bc
import logging

import gym_soccerbot

logger = logging.getLogger(__name__)

class Links(enum.IntEnum):
    TORSO = -1
    LEFT_ARM_1 = 0
    LEFT_ARM_2 = 1
    RIGHT_ARM_1 = 2
    RIGHT_ARM_2 = 3
    LEFT_LEG_1 = 4
    LEFT_LEG_2 = 5
    LEFT_LEG_3 = 6
    LEFT_LEG_4 = 7
    LEFT_LEG_5 = 8
    LEFT_LEG_6 = 9
    RIGHT_LEG_1 = 10
    RIGHT_LEG_2 = 11
    RIGHT_LEG_3 = 12
    RIGHT_LEG_4 = 13
    RIGHT_LEG_5 = 14
    RIGHT_LEG_6 = 15
    HEAD_1 = 16
    HEAD_2 = 17
    HEAD_CAMERA = 18
    IMU = 19


class Joints(enum.IntEnum):
    LEFT_ARM_1 = 0
    LEFT_ARM_2 = 1
    RIGHT_ARM_1 = 2
    RIGHT_ARM_2 = 3
    LEFT_LEG_1 = 4
    LEFT_LEG_2 = 5
    LEFT_LEG_3 = 6
    LEFT_LEG_4 = 7
    LEFT_LEG_5 = 8
    LEFT_LEG_6 = 9
    RIGHT_LEG_1 = 10
    RIGHT_LEG_2 = 11
    RIGHT_LEG_3 = 12
    RIGHT_LEG_4 = 13
    RIGHT_LEG_5 = 14
    RIGHT_LEG_6 = 15
    HEAD_1 = 16
    HEAD_2 = 17
    HEAD_CAMERA = 18
    IMU = 19


class Control_Mode(enum.IntEnum):
    POSITION = 1
    VELOCITY = 2


class Kick(gym.Env):
    metadata = {'render.modes': ['human', 'rgb_array'], 'video.frames_per_second': 50}

    DTYPE = np.float32

    _ORN_DIM = 2
    _IMU_DIM = 6
    _FEET_DIM = 8
    _JOINT_DIM = 16
    _HANDS_DIM = 4
    _BALL_LIMIT_DIM = 2
    _RNN_DIM = 1

    #### Joint Limits HARD CODE
    _joint_limit_high = np.zeros(_JOINT_DIM)

    _joint_limit_high[Joints.RIGHT_LEG_1] = 0.2
    _joint_limit_high[Joints.RIGHT_LEG_2] = 0.2
    _joint_limit_high[Joints.RIGHT_LEG_3] = 0.67
    _joint_limit_high[Joints.RIGHT_LEG_4] = 0.05
    _joint_limit_high[Joints.RIGHT_LEG_5] = 0.2  # 0.5
    _joint_limit_high[Joints.RIGHT_LEG_6] = 0.15
    _joint_limit_high[Joints.LEFT_LEG_1] = 0.2
    _joint_limit_high[Joints.LEFT_LEG_2] = 0.2
    _joint_limit_high[Joints.LEFT_LEG_3] = 0.67
    _joint_limit_high[Joints.LEFT_LEG_4] = 0.05
    _joint_limit_high[Joints.LEFT_LEG_5] = 0.2  # 0.5
    _joint_limit_high[Joints.LEFT_LEG_6] = 0.15
    _joint_limit_high[Joints.RIGHT_ARM_1] = 0.95
    _joint_limit_high[Joints.RIGHT_ARM_2] = 0.8
    _joint_limit_high[Joints.LEFT_ARM_1] = 0.95
    _joint_limit_high[Joints.LEFT_ARM_2] = 0.8

    _joint_limit_high *= (np.pi)

    _joint_limit_low = np.zeros(_JOINT_DIM)

    _joint_limit_low[Joints.RIGHT_LEG_1] = 0.3
    _joint_limit_low[Joints.RIGHT_LEG_2] = 0.1
    _joint_limit_low[Joints.RIGHT_LEG_3] = 0.2
    _joint_limit_low[Joints.RIGHT_LEG_4] = 0.45
    _joint_limit_low[Joints.RIGHT_LEG_5] = 0.12
    _joint_limit_low[Joints.RIGHT_LEG_6] = 0.1
    _joint_limit_low[Joints.LEFT_LEG_1] = 0.3
    _joint_limit_low[Joints.LEFT_LEG_2] = 0.1
    _joint_limit_low[Joints.LEFT_LEG_3] = 0.2
    _joint_limit_low[Joints.LEFT_LEG_4] = 0.45
    _joint_limit_low[Joints.LEFT_LEG_5] = 0.12
    _joint_limit_low[Joints.LEFT_LEG_6] = 0.1
    _joint_limit_low[Joints.RIGHT_ARM_1] = 0.4
    _joint_limit_low[Joints.RIGHT_ARM_2] = 0.0
    _joint_limit_low[Joints.LEFT_ARM_1] = 0.4
    _joint_limit_low[Joints.LEFT_ARM_2] = 0.0

    _joint_limit_low *= (-np.pi)

    _AX_12_force = 1.5
    _MX_28_force = 2.5
    _AX_12_velocity = (59 / 60) * 2 * np.pi
    _MX_28_velocity = 2 * np.pi
    _ENABLE_HANDS = True

    #### End of Joint Limits HARD CODE
    @classmethod
    def joint_limit_high_val(cls):
        return cls._joint_limit_high.copy()

    @classmethod
    def joint_limit_low_val(cls):
        return cls._joint_limit_low.copy()

    _STANDING_HEIGHT = 0.29  # 0.32
    _GRAVITY_VECTOR = [0, 0, -9.81]
    _CLOSENESS = 0.05  # in meters presumably
    _MAX_ANG_VEL = 8.7266  # LSM6DSOX
    _MAX_LIN_ACC = 2. * 9.81  # LSM6DSOX
    # Action Space
    rnn_limit = np.array([1] * _RNN_DIM)
    if _ENABLE_HANDS:
        _DIM_SUB_HANDS = 0
        vel_limit = [_MX_28_velocity] * (Joints.HEAD_1 - Joints.LEFT_LEG_1)
        vel_limit.extend([_AX_12_velocity] * (Joints.LEFT_LEG_1 - Joints.LEFT_ARM_1))
        vel_limit = np.array(vel_limit, dtype=DTYPE)
        action_space = spaces.Box(low=-np.concatenate((vel_limit, rnn_limit)),
                                  high=np.concatenate((vel_limit, rnn_limit)), dtype=DTYPE)
    else:
        _DIM_SUB_HANDS = 4
        vel_size = Joints.HEAD_1 - Joints.LEFT_LEG_1
        vel_limit = np.array([_MX_28_velocity] * vel_size, dtype=DTYPE)
        action_space = spaces.Box(low=-np.concatenate((vel_limit, rnn_limit)),
                                  high=np.concatenate((vel_limit, rnn_limit)), dtype=DTYPE)

    # Observation Space
    _OBSERVATION_DIM = _JOINT_DIM + _JOINT_DIM + _IMU_DIM + _ORN_DIM + _FEET_DIM - (
                _DIM_SUB_HANDS * 2) + _BALL_LIMIT_DIM + _RNN_DIM
    imu_limit = np.concatenate((np.array([_MAX_LIN_ACC] * int((_IMU_DIM) / 2)),
                                np.array([_MAX_ANG_VEL] * int((_IMU_DIM) / 2))))
    # pose_limit = np.array([3.] * _POSE_DIM)
    orn_limit = np.array([1.] * _ORN_DIM)

    feet_limit = np.array([1.6] * _FEET_DIM)

    ball_start_limit = np.array([0.3] * _BALL_LIMIT_DIM)

    # TODO speed
    # joint_limit = np.array([np.pi] * (_JOINT_DIM - _DIM_SUB_HANDS))
    joint_limit = np.array([np.pi] * (_JOINT_DIM))

    observation_limit_high = np.concatenate((joint_limit[_DIM_SUB_HANDS:Joints.HEAD_1], vel_limit * 2,
                                             imu_limit, orn_limit, feet_limit, ball_start_limit, rnn_limit))
    observation_limit_low = np.concatenate((-joint_limit[_DIM_SUB_HANDS:Joints.HEAD_1], -vel_limit * 2,
                                            -imu_limit, -orn_limit, -feet_limit, -ball_start_limit, -rnn_limit))
    observation_space = spaces.Box(low=observation_limit_low, high=observation_limit_high, dtype=DTYPE)

    # Reward
    reward_limit_low = -1
    reward_limit_high = 1
    reward_range = [float(reward_limit_low), float(reward_limit_high)]

    # MISC
    _render_height = 200
    _render_width = 320

    # IMU NOISE
    _IMU_LIN_STDDEV_BIAS = 0.  # 0.02 * _MAX_LIN_ACC
    _IMU_ANG_STDDEV_BIAS = 0.  # 0.02 * _MAX_ANG_VEL
    _IMU_LIN_STDDEV = 0.00203 * _MAX_LIN_ACC
    _IMU_ANG_STDDEV = 0.00804 * _MAX_ANG_VEL

    # FEET
    _FEET_FALSE_CHANCE = 0.01

    # Joint angle noise
    _JOIN_ANGLE_STDDEV = np.pi / 2048
    _JOIN_VELOCITY_STDDEV = _JOIN_ANGLE_STDDEV / 120

    def __init__(self, renders=False, warm_up=False, goal=(1, 0), ball_start=(0.15, 0),
                 start_ang=None, random_tilt=None, random_start_vel=None,
                 random_joint_vel=None, sensor_noise=False,
                 random_joint_angles=False, horizon=1024, seed=42):
        # start the bullet physics server
        self._renders = renders
        self._physics_client_id = -1

        self.goal_xy = goal
        self.init_ball_pose = np.array(ball_start)
        self.start_ang = start_ang
        if random_start_vel is not None:
            self.random_start_vel = np.array(random_start_vel)
        else:
            self.random_start_vel = None
        self.random_tilt = random_tilt
        self.random_joint_vel = random_joint_vel
        self.WARM_UP = warm_up
        self.sensor_noise = sensor_noise
        self.random_joint_angles = random_joint_angles
        self.horizon = horizon
        self.seed(seed=seed)
        # self.reset()
        # self.st = RollingAvg(256, 0.01, 0.01)

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _ball_pos(self):
        p = self._p
        pos, _ = p.getBasePositionAndOrientation(self.ballUid)
        return np.array(pos, dtype=self.DTYPE)[0:2]

    def _joints_pos(self):
        p = self._p
        joint_states = p.getJointStates(self.soccerbotUid, list(range(self._DIM_SUB_HANDS, self._JOINT_DIM, 1)))
        joints_pos = np.array([state[0] for state in joint_states], dtype=self.DTYPE)
        # joints_pos = np.unwrap(joints_pos + np.pi) - np.pi
        if self.sensor_noise:
            joints_pos = joints_pos + self.np_random.normal(0, self._JOIN_ANGLE_STDDEV, joints_pos.size)
        joints_pos = np.clip(joints_pos, -self.joint_limit[self._DIM_SUB_HANDS:Joints.HEAD_1],
                             self.joint_limit[self._DIM_SUB_HANDS:Joints.HEAD_1])
        return joints_pos

    def _joints_vel(self):
        p = self._p
        joint_states = p.getJointStates(self.soccerbotUid, list(range(self._DIM_SUB_HANDS, self._JOINT_DIM, 1)))
        joints_vel = np.array([state[1] for state in joint_states], dtype=self.DTYPE)
        # joints_pos = np.unwrap(joints_pos + np.pi) - np.pi
        if self.sensor_noise:
            joints_vel = joints_vel + self.np_random.normal(0, self._JOIN_VELOCITY_STDDEV, joints_vel.size)
        joints_vel = np.clip(joints_vel, -self.vel_limit, self.vel_limit)
        return joints_vel

    def _imu(self):
        p = self._p
        [quart_link, lin_vel, ang_vel] = p.getLinkState(bodyUniqueId=self.soccerbotUid, linkIndex=Links.IMU,
                                                        computeLinkVelocity=1)[5:8]
        lin_vel = np.array(lin_vel, dtype=self.DTYPE)
        lin_acc = (lin_vel - self.prev_lin_vel) / (self.timeStep)
        lin_acc -= self._GRAVITY_VECTOR
        rot_mat = np.array(pb.getMatrixFromQuaternion(quart_link), dtype=self.DTYPE).reshape((3, 3))
        lin_acc = np.matmul(rot_mat, lin_acc)
        ang_vel = np.array(ang_vel, dtype=self.DTYPE)
        self.prev_lin_vel = lin_vel
        imu_val = np.concatenate((lin_acc, ang_vel))
        if self.sensor_noise:
            imu_noise = np.concatenate((self.np_random.normal(0, self._IMU_LIN_STDDEV, int(self._IMU_DIM / 2)),
                                        self.np_random.normal(0, self._IMU_ANG_STDDEV, int(self._IMU_DIM / 2))))
            imu_val = imu_val + imu_noise + self.imu_bias
        imu_val = np.clip(imu_val, -self.imu_limit, self.imu_limit)
        return imu_val

    def _global_pos(self):
        p = self._p
        pos, _ = p.getBasePositionAndOrientation(self.soccerbotUid)
        return np.array(pos, dtype=self.DTYPE)

    def _global_orn(self):
        p = self._p
        _, orn = p.getBasePositionAndOrientation(self.soccerbotUid)
        return np.array(orn, dtype=self.DTYPE)

    def _off_orn(self):
        p = self._p
        distance_unit_vec = (self.goal_xy - self._global_pos()[0:2])
        distance_unit_vec /= np.linalg.norm(distance_unit_vec)
        mat = p.getMatrixFromQuaternion(p.getBasePositionAndOrientation(self.soccerbotUid)[1])
        d2_vect = np.array([mat[0], mat[3]], dtype=self.DTYPE)
        d2_vect /= np.linalg.norm(d2_vect)
        cos = np.dot(d2_vect, distance_unit_vec)
        sin = np.linalg.norm(np.cross(distance_unit_vec, d2_vect))
        vec = np.array([cos, sin], dtype=self.DTYPE)
        # print(f'Orn: {vec}')
        vec = np.matmul([[0, 1], [-1, 0]], vec)
        return vec

    def _feet(self):
        """
        Checks if 4 corners of the each feet are in contact with ground
        Indicies for looking from above on the feet plates:
          Left         Right
        4-------5    0-------1
        |   ^   |    |   ^   |      ^
        |   |   |    |   |   |      | : forward direction
        |       |    |       |
        6-------7    2-------3
        :return: int array of 8 contact points on both feet, 1: that point is touching the ground -1: otherwise
        """
        locations = [-1.] * self._FEET_DIM
        right_pts = pb.getContactPoints(bodyA=self.soccerbotUid, bodyB=self.planeUid, linkIndexA=Links.RIGHT_LEG_6)
        left_pts = pb.getContactPoints(bodyA=self.soccerbotUid, bodyB=self.planeUid, linkIndexA=Links.LEFT_LEG_6)
        right_center = np.array(pb.getLinkState(bodyUniqueId=self.soccerbotUid, linkIndex=Links.RIGHT_LEG_6)[4])
        left_center = np.array(pb.getLinkState(bodyUniqueId=self.soccerbotUid, linkIndex=Links.LEFT_LEG_6)[4])
        right_tr = np.array(pb.getMatrixFromQuaternion(
            pb.getLinkState(bodyUniqueId=self.soccerbotUid, linkIndex=Links.RIGHT_LEG_6)[5])
            , dtype=self.DTYPE).reshape((3, 3))
        left_tr = np.array(pb.getMatrixFromQuaternion(
            pb.getLinkState(bodyUniqueId=self.soccerbotUid, linkIndex=Links.LEFT_LEG_6)[5])
            , dtype=self.DTYPE).reshape((3, 3))
        for point in right_pts:
            index = np.signbit(np.matmul(right_tr, point[5] - right_center))[0:2]
            locations[index[1] + index[0] * 2] = 1.
        for point in left_pts:
            index = np.signbit(np.matmul(left_tr, point[5] - left_center))[0:2]
            locations[index[1] + (index[0] * 2) + 4] = 1.
        ground_truth_feet = np.array(locations)
        if self.sensor_noise:
            for i in range(len(locations)):  # 5% chance of incorrect reading
                locations[i] *= np.sign(self.np_random.uniform(low=- self._FEET_FALSE_CHANCE,
                                                               high=1 - (self._FEET_FALSE_CHANCE)),
                                        dtype=self.DTYPE)
        return ground_truth_feet, np.array(locations)

    @classmethod
    def _standing_poses(cls, np_random=None, scale=0.1):
        standing_poses = [0.] * (cls._JOINT_DIM + 2)
        standing_poses[Joints.RIGHT_LEG_1] = 0.0
        standing_poses[Joints.RIGHT_LEG_2] = 0.05
        standing_poses[Joints.RIGHT_LEG_3] = 0.4
        standing_poses[Joints.RIGHT_LEG_4] = -0.8
        standing_poses[Joints.RIGHT_LEG_5] = 0.4
        standing_poses[Joints.RIGHT_LEG_6] = -0.05

        standing_poses[Joints.LEFT_LEG_1] = 0.0
        standing_poses[Joints.LEFT_LEG_2] = 0.05
        standing_poses[Joints.LEFT_LEG_3] = 0.4
        standing_poses[Joints.LEFT_LEG_4] = -0.8
        standing_poses[Joints.LEFT_LEG_5] = 0.4
        standing_poses[Joints.LEFT_LEG_6] = -0.05

        standing_poses[Joints.HEAD_1] = 0.0
        standing_poses[Joints.HEAD_2] = 0.0

        standing_poses[Joints.LEFT_ARM_1] = 0.0
        standing_poses[Joints.LEFT_ARM_2] = 2.8
        standing_poses[Joints.RIGHT_ARM_1] = 0.0
        standing_poses[Joints.RIGHT_ARM_2] = 2.8

        if np_random is not None:
            standing_poses[cls._DIM_SUB_HANDS:cls._JOINT_DIM] = standing_poses[cls._DIM_SUB_HANDS:cls._JOINT_DIM] + \
                                                                scale * np_random.uniform(-0., 1., np.size(
                standing_poses[cls._DIM_SUB_HANDS:cls._JOINT_DIM])) * \
                                                                (cls._joint_limit_high[
                                                                 cls._DIM_SUB_HANDS:cls._JOINT_DIM] - standing_poses[
                                                                                                      cls._DIM_SUB_HANDS:cls._JOINT_DIM]) + \
                                                                scale * np_random.uniform(-0., 1., np.size(
                standing_poses[cls._DIM_SUB_HANDS:cls._JOINT_DIM])) * \
                                                                (cls._joint_limit_low[
                                                                 cls._DIM_SUB_HANDS:cls._JOINT_DIM] - standing_poses[
                                                                                                      cls._DIM_SUB_HANDS:cls._JOINT_DIM])
            standing_poses[cls._DIM_SUB_HANDS:cls._JOINT_DIM] = np.clip(
                standing_poses[cls._DIM_SUB_HANDS:cls._JOINT_DIM],
                cls._joint_limit_low[cls._DIM_SUB_HANDS:cls._JOINT_DIM],
                cls._joint_limit_high[cls._DIM_SUB_HANDS:cls._JOINT_DIM])
            return standing_poses
        return np.array(standing_poses)

    def motor_control(self, action):
        p = self._p
        # CLIP ACTIONS
        # action = np.clip(action, self._joint_limit_low, self._joint_limit_high)
        # MX-28s
        gain = 0.78
        for i in range(Joints.LEFT_ARM_1, Joints.HEAD_1, 1):
            joint_cur_pos = p.getJointState(self.soccerbotUid, i)[0]
            velocity = action[i]
            velocity = velocity if joint_cur_pos < self._joint_limit_high[i] else -self._MX_28_velocity
            velocity = velocity if joint_cur_pos > self._joint_limit_low[i] else self._MX_28_velocity
            p.setJointMotorControl2(bodyIndex=self.soccerbotUid,
                                    controlMode=pb.VELOCITY_CONTROL,
                                    jointIndex=i,
                                    targetVelocity=velocity,
                                    velocityGain=gain,
                                    maxVelocity=self._MX_28_velocity,
                                    force=self._MX_28_force,
                                    )

    def motor_control_array(self, action):
        p = self._p
        # CLIP ACTIONS
        # action = np.clip(action, self._joint_limit_low, self._joint_limit_high)
        # MX-28s
        gain = 0.78
        for i in range(Joints.LEFT_ARM_1, Joints.HEAD_1, 1):
            joint_cur_pos = p.getJointState(self.soccerbotUid, i)[0]
            velocity = action[i]
            velocity = velocity if joint_cur_pos < self._joint_limit_high[i] else -self._MX_28_velocity
            velocity = velocity if joint_cur_pos > self._joint_limit_low[i] else self._MX_28_velocity
            action[i] = velocity
        p.setJointMotorControlArray(bodyIndex=self.soccerbotUid,
                                    controlMode=pb.VELOCITY_CONTROL,
                                    jointIndices=list(range(Joints.HEAD_1)),
                                    targetVelocities=action,
                                    velocityGains=[gain] * Joints.HEAD_1,
                                    forces=[self._MX_28_force] * Joints.HEAD_1,
                                    )

    def step(self, action):
        p = self._p

        # Construct Observation - Pipeline Structure
        ground_truth_feet, feet = self._feet()
        imu = self._imu()
        joints_pos = self._joints_pos()
        joints_vel = self._joints_vel()
        height = np.array([self._global_pos()[2]], dtype=self.DTYPE)
        # orn = self._global_orn()
        orn = self._off_orn()
        observation = np.concatenate(
            (joints_pos, joints_vel, imu, orn, feet, self.init_ball_pose, action[self._JOINT_DIM:]))

        self.motor_control_array(action[:self._JOINT_DIM])

        # 120Hz - Step Simulation
        p.stepSimulation()
        # p.stepSimulation()

        self.horizon_counter = self.horizon_counter + 1

        ## Calculate Reward, Done, Info
        # Calculate Velocity direction field
        [lin_vel, ang_vel] = p.getBaseVelocity(self.soccerbotUid)
        lin_vel_xy = np.array(lin_vel, dtype=self.DTYPE)[0:2]
        distance_unit_vec = (self._ball_pos() - self._global_pos()[0:2]) \
                            / np.linalg.norm(self._ball_pos() - self._global_pos()[0:2])
        velocity_forward_reward = np.dot(distance_unit_vec, lin_vel_xy)
        # velocity_downward_penalty = np.min(lin_vel[2], 0) # Only consider the negative component
        [lin_vel, _] = p.getBaseVelocity(self.ballUid)
        lin_vel_xy = np.array(lin_vel, dtype=self.DTYPE)[0:2]
        distance_unit_vec = (self.goal_xy - self._ball_pos()[0:2]) \
                            / np.linalg.norm(self.goal_xy - self._ball_pos()[0:2])
        ball_velocity_forward_reward = np.dot(distance_unit_vec, lin_vel_xy)
        DESIRED_HEIGHT = 0.27
        info = dict(end_cond="None")
        # Fall
        if self._global_pos()[2] < 0.22:  # HARDCODE (self._STANDING_HEIGHT / 2): # check z component
            done = True
            reward = -1
            info['end_cond'] = "Robot Fell"
        # Close to the Goal
        elif np.linalg.norm(self._ball_pos()[0:2] - self.goal_xy) < self._CLOSENESS:
            done = True
            reward = 1e-1
            info['end_cond'] = "Ball Goal Reached"
        # Out of Bound
        elif np.linalg.norm(self._ball_pos()[0:2] - self.goal_xy) > (
                2 * np.linalg.norm(self.goal_xy)):  # out of bound
            done = True
            reward = -1
            info['end_cond'] = "Ball Out"
        # Horizon Ended
        elif self.horizon_counter >= self.horizon:
            done = True
            reward = 0
            info['end_cond'] = "Horizon Ended"
        # Normal case
        else:
            done = False
            if np.linalg.norm(self._ball_pos()[0:2] - self.init_ball_pose) > 0.3:
                vel_reward = 0.05 * np.linalg.norm(ang_vel)
                pos_reward = 0.05 * np.linalg.norm(self._standing_poses()[:Joints.HEAD_1] - joints_pos)
                reward = 0.1 * ball_velocity_forward_reward - (
                            DESIRED_HEIGHT - self._global_pos()[2]) - vel_reward - pos_reward
            else:
                reward = 0.1 * ball_velocity_forward_reward + 0.05 * velocity_forward_reward - (
                        DESIRED_HEIGHT - self._global_pos()[2])
        '''
        elif (np.sum(ground_truth_feet[0:4]) < 2) and (np.sum(ground_truth_feet[4:8]) < 2):
            done = True
            reward = -1
            info['end_cond'] = "Feet Separation"
        '''
        return observation, reward, done, info

    def reset(self):
        if self._physics_client_id < 0:
            if self._renders:
                self._p = bc.BulletClient(connection_mode=pb.GUI)
                self._p.configureDebugVisualizer(pb.COV_ENABLE_RENDERING, 0)
            else:
                self._p = bc.BulletClient()
                # self._p.configureDebugVisualizer(pb.COV_ENABLE_RENDERING, 0)
            self._physics_client_id = self._p._client

            p = self._p
            p.resetSimulation()

            urdfBotPath = gym_soccerbot.getModelPath(self._renders)
            self.soccerbotUid = p.loadURDF(urdfBotPath,
                                           useFixedBase=False,
                                           useMaximalCoordinates=False,
                                           basePosition=[0, 0, self._STANDING_HEIGHT],
                                           baseOrientation=[0., 0., 0., 1.],
                                           flags=pb.URDF_USE_INERTIA_FROM_FILE
                                                 | pb.URDF_USE_SELF_COLLISION
                                                 | pb.URDF_USE_SELF_COLLISION_EXCLUDE_ALL_PARENTS
                                           )
            # load ramp
            urdfRootPath = pybullet_data.getDataPath()
            self.planeUid = p.loadURDF(os.path.join(urdfRootPath, "plane_implicit.urdf"),
                                       useMaximalCoordinates=True,
                                       basePosition=[0, 0, 0])
            # Ball
            urdfBallPath = gym_soccerbot.getBallPath()
            self.ballUid = p.loadURDF(urdfBallPath,
                                      useMaximalCoordinates=False,
                                      basePosition=[0.5, 0, 0.1],
                                      baseOrientation=[0., 0., 0., 1.])
            p.changeDynamics(self.ballUid, linkIndex=-1, restitution=0.99)

            # TODO change dynamics ...
            # for i in range(p.getNumJoints(bodyUniqueId=self.soccerbotUid)):
            # print(p.getJointInfo(bodyUniqueId=self.soccerbotUid, jointIndex=i)[1])
            p.changeDynamics(self.planeUid, linkIndex=-1, lateralFriction=0.9, spinningFriction=0.9, rollingFriction=0)
            # p.changeDynamics(self.soccerbotUid, linkIndex=Links.IMU, mass=0.01, localInertiaDiagonal=[7e-7, 7e-7, 7e-7])
            # p.changeDynamics(self.soccerbotUid, linkIndex=Links.IMU, mass=0., localInertiaDiagonal=[0., 0., 0.])
            '''
            p.changeDynamics(bodyUniqueId=self.soccerbotUid, linkIndex=Links.RIGHT_LEG_6,
                             frictionAnchor=1, lateralFriction=1,
                             rollingFriction=1, spinningFriction=1)
            p.changeDynamics(bodyUniqueId=self.soccerbotUid, linkIndex=Links.RIGHT_LEG_6,
                             frictionAnchor=1, lateralFriction=1,
                             rollingFriction=1, spinningFriction=1)
            '''
            # Simulation Physics General Settings
            self.timeStep = 1. / 120
            p.setTimeStep(self.timeStep)
            p.setGravity(*self._GRAVITY_VECTOR)
            p.setPhysicsEngineParameter(numSolverIterations=10)
            p.setPhysicsEngineParameter(numSubSteps=1)
            p.setRealTimeSimulation(0)  # To manually step simulation later

        p = self._p

        # Bring robot back to origin
        starting_ang = 0
        if self.start_ang is not None:
            starting_ang = self.np_random.uniform(-self.start_ang, self.start_ang)
        theta = np.pi * (starting_ang) / 360  # start_ang in degrees
        tilt_x = 0.
        tilt_y = 0.
        tilt_z = 1.
        if self.random_tilt is not None:
            tilt_x = self.np_random.uniform(-self.random_tilt, self.random_tilt)
            tilt_y = self.np_random.uniform(-self.random_tilt, self.random_tilt)
            tilt_z = np.sqrt(1. - ((tilt_x ** 2) + (tilt_y ** 2)))
        p.resetBasePositionAndOrientation(self.soccerbotUid, [0, 0, self._STANDING_HEIGHT],
                                          [tilt_x * np.sin(theta), tilt_y * np.sin(theta), tilt_z * np.sin(theta),
                                           np.cos(theta)])
        if self.random_start_vel is not None:
            start_vel = self.np_random.uniform(-self.random_start_vel, self.random_start_vel)
            p.resetBaseVelocity(self.soccerbotUid, start_vel[0:3], start_vel[3:6])
            self.prev_lin_vel = np.array(start_vel[0:3])  # not accurate? - same rigid frame translationaly
        else:
            p.resetBaseVelocity(self.soccerbotUid, [0, 0, 0], [0, 0, 0])
            self.prev_lin_vel = np.array([0, 0, 0])

        if self.random_joint_angles:
            standing_poses = self._standing_poses(self.np_random)
        else:
            standing_poses = self._standing_poses()

        # MX-28s:
        for i in range(Joints.LEFT_LEG_1, Joints.HEAD_1):
            if self.random_joint_vel is not None:
                start_vel_joint = self.np_random.uniform(-self.random_joint_vel,
                                                         self.random_joint_vel) * self._MX_28_velocity
            else:
                start_vel_joint = 0.
            p.resetJointState(self.soccerbotUid, i, standing_poses[i], start_vel_joint)
            p.changeDynamics(self.soccerbotUid, i,
                             jointLowerLimit=-self._joint_limit_low[i],
                             jointUpperLimit=self._joint_limit_high[i],
                             jointLimitForce=self._MX_28_force)
        # AX-12s:
        for i in range(Joints.LEFT_ARM_1, Joints.LEFT_LEG_1):
            if self.random_joint_vel is not None:
                start_vel_joint = self.np_random.uniform(-self.random_joint_vel,
                                                         self.random_joint_vel) * self._AX_12_velocity
            else:
                start_vel_joint = 0.
            p.resetJointState(self.soccerbotUid, i, standing_poses[i])
            p.changeDynamics(self.soccerbotUid, i,
                             jointLowerLimit=-self._joint_limit_low[i],
                             jointUpperLimit=self._joint_limit_high[i],
                             jointLimitForce=self._AX_12_force)
        p.resetJointState(self.soccerbotUid, Joints.HEAD_1, standing_poses[Joints.HEAD_1])
        p.changeDynamics(self.soccerbotUid, Joints.HEAD_1,
                         jointLowerLimit=-np.pi, jointUpperLimit=np.pi,
                         jointLimitForce=self._AX_12_force)
        p.resetJointState(self.soccerbotUid, Joints.HEAD_2, standing_poses[Joints.HEAD_2])
        p.changeDynamics(self.soccerbotUid, Joints.HEAD_2,
                         jointLowerLimit=-np.pi, jointUpperLimit=np.pi,
                         jointLimitForce=self._AX_12_force)

        # Ball reset
        p.resetBasePositionAndOrientation(self.ballUid, [self.init_ball_pose[0], self.init_ball_pose[1], 0.1],
                                          [0, 0, 0, 1])
        p.resetBaseVelocity(self.ballUid, [0, 0, 0], [0, 0, 0])
        self.prev_lin_vel = np.array([0, 0, 0])
        self.imu_bias = np.concatenate((self.np_random.normal(0, self._IMU_LIN_STDDEV_BIAS, int(self._IMU_DIM / 2)),
                                        self.np_random.normal(0, self._IMU_ANG_STDDEV_BIAS, int(self._IMU_DIM / 2))))
        # WARM UP SIMULATION
        p.stepSimulation()
        # p.stepSimulation()
        if self.WARM_UP:
            warm_up = self.np_random.randint(1, 11)  # warm_up = 100
            for _ in range(warm_up):
                imu = self._imu()
                ground_truth_feet, feet = self._feet()
                # if (np.sum(ground_truth_feet[0:4]) >= 2) or (np.sum(ground_truth_feet[4:8]) >= 2):
                #     break
                p.stepSimulation()
                # p.stepSimulation()

        # Get Observation
        # self.st = RollingAvg(256, 0.01, 0.01)

        # Construct Observation
        # imu = self._imu()
        if self.WARM_UP == False:
            imu = self._imu()
            _, feet = self._feet()
        joints_pos = self._joints_pos()
        joints_vel = self._joints_vel()
        orn = self._off_orn()
        observation = np.concatenate((joints_pos, joints_vel, imu, orn, feet, self.init_ball_pose, [0] * self._RNN_DIM))

        # To keep up with the pipeline - 120Hz
        p.stepSimulation()
        # p.stepSimulation()

        # Horizon Counter
        self.horizon_counter = 0

        if self._renders:
            pb.configureDebugVisualizer(pb.COV_ENABLE_RENDERING, 1)
        return observation

    def render(self, mode='human'):
        if mode == "human":
            self._renders = True
        if mode != "rgb_array":
            return np.array([])
        base_pos, orn = self._p.getBasePositionAndOrientation(self.soccerbotUid)
        base_pos = np.asarray(base_pos)
        # TODO tune parameters
        # track the position
        base_pos[1] += 0.3
        rpy = self._p.getEulerFromQuaternion(orn)  # rpy, in radians
        rpy = 180 / np.pi * np.asarray(rpy)  # convert rpy in degrees

        self._cam_dist = 3
        self._cam_pitch = 0.3
        self._cam_yaw = 0
        if not (self._p is None):
            view_matrix = self._p.computeViewMatrixFromYawPitchRoll(
                cameraTargetPosition=base_pos,
                distance=self._cam_dist,
                yaw=self._cam_yaw,
                pitch=self._cam_pitch,
                roll=0,
                upAxisIndex=1)
            proj_matrix = self._p.computeProjectionMatrixFOV(fov=60,
                                                             aspect=float(self._render_width) / self._render_height,
                                                             nearVal=0.1,
                                                             farVal=100.0)
            (_, _, px, _, _) = self._p.getCameraImage(
                width=self._render_width,
                height=self._render_height,
                renderer=self._p.ER_BULLET_HARDWARE_OPENGL,
                viewMatrix=view_matrix,
                projectionMatrix=proj_matrix)
            '''
            self._p.resetDebugVisualizerCamera(
              cameraDistance=2 * self._cam_dist,
              cameraYaw=self._cam_yaw,
              cameraPitch=self._cam_pitch,
              cameraTargetPosition=base_pos
            )
            '''
            pb.configureDebugVisualizer(pb.COV_ENABLE_RENDERING, 1)
        else:
            px = np.array([[[255, 255, 255, 255]] * self._render_width] * self._render_height, dtype=np.uint8)
        rgb_array = np.array(px, dtype=np.uint8)
        rgb_array = np.reshape(np.array(px), (self._render_height, self._render_width, -1))
        rgb_array = rgb_array[:, :, :3]
        return rgb_array

    def close(self):
        if self._physics_client_id >= 0:
            self._p.disconnect()
        self._physics_client_id = -1