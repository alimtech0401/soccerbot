import pybullet as pb
import numpy as np
import os
from soccerbot import Soccerbot, Joints


class SoccerbotRl(Soccerbot):

    def ready_kick(self):
        """
        Sets the robot's joint angles for the robot to standing pose.
        :return: None
        """

        # head
        self.configuration[Joints.HEAD_1] = 0
        self.configuration[Joints.HEAD_2] = 0
        standing_poses = [0.] * (20)
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

        self.configuration_kick = np.array(standing_poses)
        if os.getenv('ENABLE_PYBULLET', True):
            pb.setJointMotorControlArray(bodyIndex=self.body,
                                         controlMode=pb.POSITION_CONTROL,
                                         jointIndices=list(range(0, 20, 1)),
                                         targetPositions=self.configuration_kick,
                                         forces=self.max_forces)

    def joints_pos(self):

        joint_states = pb.getJointStates(self.body, list(range(0, 16, 1)))
        joints_pos = np.array([state[0] for state in joint_states], dtype=np.float32)
        # joints_pos = np.unwrap(joints_pos + np.pi) - np.pi
        joints_pos = np.clip(joints_pos, -self.joint_limit[0:16],self.joint_limit[0:16])
        return joints_pos

    def joints_vel(self):
        joint_states = pb.getJointStates(self.body, list(range(0, 16, 1)))
        joints_vel = np.array([state[1] for state in joint_states], dtype=np.float32)
        # joints_pos = np.unwrap(joints_pos + np.pi) - np.pi
        joints_vel = np.clip(joints_vel, -self.vel_limit, self.vel_limit)
        return joints_vel

    def global_orn(self):
        _, orn = pb.getBasePositionAndOrientation(self.body)
        return np.array(orn, dtype=np.float32)

    def off_orn(self):
        distance_unit_vec = ((0, 0.3) - self.global_pos()[0:2])
        distance_unit_vec /= np.linalg.norm(distance_unit_vec)
        mat = pb.getMatrixFromQuaternion(pb.getBasePositionAndOrientation(self.body)[1])
        d2_vect = np.array([mat[0], mat[3]], dtype=np.float32)
        d2_vect /= np.linalg.norm(d2_vect)
        cos = np.dot(d2_vect, distance_unit_vec)
        sin = np.linalg.norm(np.cross(distance_unit_vec, d2_vect))
        vec = np.array([cos, sin], dtype=np.float32)
        # print(f'Orn: {vec}')
        vec = np.matmul([[0, 1], [-1, 0]], vec)
        return vec

    def global_pos(self):

        pos, _ = pb.getBasePositionAndOrientation(self.body)
        return np.array(pos, dtype=np.float32)

    def motor_control(self, action, joint_angles, env):
        _MX_28_velocity = 2 * np.pi
        # CLIP ACTIONS
        # action = np.clip(action, self._joint_limit_low, self._joint_limit_high)
        # MX-28s
        gain = 0.78
        for i in range(Joints.LEFT_ARM_1, Joints.HEAD_1, 1):
            joint_cur_pos = pb.getJointState(self.body, i)[0]
            velocity = action[i]
            velocity = velocity if joint_cur_pos < env._joint_limit_high[i] else -_MX_28_velocity
            velocity = velocity if joint_cur_pos > env._joint_limit_low[i] else _MX_28_velocity
            if velocity != action[i]:
                print(f'***** Joint {i} capped')
            action[i] = velocity
            # old way
            # pb.setJointMotorControl2(bodyIndex=self.body,
            #                          controlMode=pb.VELOCITY_CONTROL,
            #                          jointIndex=i,
            #                          targetVelocity=velocity,
            #                          velocityGain=gain,
            #                          maxVelocity=_MX_28_velocity,
            #                          force=2.5,
            #                          )
        pb.setJointMotorControlArray(bodyIndex=self.body,
                                     controlMode=pb.VELOCITY_CONTROL,
                                     jointIndices=list(range(16)),
                                     targetVelocities=action,
                                     velocityGains=[gain] * 16,
                                     forces=[2.5] * Joints.HEAD_1,
                                     )
