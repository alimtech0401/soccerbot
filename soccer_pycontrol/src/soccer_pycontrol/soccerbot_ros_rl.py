import rospy
import numpy as np
import pybullet as pb
from sensor_msgs.msg import JointState, Imu
from nav_msgs.msg import Odometry, Path
from soccerbot import Joints
from soccerbot_ros import SoccerbotRos
from geometry_msgs.msg import PoseStamped
from transformation import Transformation
from std_msgs.msg import Int32
from soccerbot_rl import SoccerbotRl
LOOP_FREQUENCY = 120

class SoccerbotRosRl(SoccerbotRos, SoccerbotRl):

    def __init__(self, pose, usePybullet=False):
        super().__init__(pose, usePybullet)
        self.velocity_configuration = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        self.jointState = JointState()
        self.pub_all_motor = rospy.Publisher("joint_command", JointState, queue_size=10)
        self.sub_all_motor = rospy.Subscriber("joint_states", JointState, self.jointStateCallback)
        self.odom_publisher = rospy.Publisher("odom", Odometry, queue_size=1)
        self.path_publisher = rospy.Publisher("path", Path, queue_size=1)
        self.imu_subscriber = rospy.Subscriber("imu_filtered", Imu, self.imu_callback)

        self.foot_pressure_sensor_subscriber_list = []
        self.foot_pressure_values = [-1, -1, -1, -1, -1, -1, -1, -1]
        for i in range(8):  # TODO get correct topic
            temp_string = "num_foot_pressure_" + str(i)
            foot_pressure_sensor = rospy.Subscriber(temp_string, Int32, self.foot_pressure_sensor_callback,
                                                    i)
            self.foot_pressure_sensor_subscriber_list.append(foot_pressure_sensor)
        self.lastjointState = JointState()
        self.lastjointState.position = [0] * 18

    def ready(self):
        """
        Sets the robot's joint angles for the robot to standing pose.
        :return: None
        """

        # Set initial posit
        self.configuration[Joints.RIGHT_LEG_1] = 0.0
        self.configuration[Joints.RIGHT_LEG_2] = 0.05
        self.configuration[Joints.RIGHT_LEG_3] = 0.4
        self.configuration[Joints.RIGHT_LEG_4] = -0.8
        self.configuration[Joints.RIGHT_LEG_5] = 0.4
        self.configuration[Joints.RIGHT_LEG_6] = -0.05

        self.configuration[Joints.LEFT_LEG_1] = 0.0
        self.configuration[Joints.LEFT_LEG_2] = 0.05
        self.configuration[Joints.LEFT_LEG_3] = 0.4
        self.configuration[Joints.LEFT_LEG_4] = -0.8
        self.configuration[Joints.LEFT_LEG_5] = 0.4
        self.configuration[Joints.LEFT_LEG_6] = -0.05

        self.configuration[Joints.LEFT_ARM_1] = 0.0 # -0.5
        self.configuration[Joints.LEFT_ARM_2] = 2.8
        self.configuration[Joints.RIGHT_ARM_1] = 0.0 # -0.5
        self.configuration[Joints.RIGHT_ARM_2] = 2.8
        # Publish angles using position control
        super().publishAngles()

    def jointStateCallback(self, jointState: JointState):
        self.jointState = jointState
        self.jointState.velocity = tuple((np.array(self.jointState.position) - np.array(self.lastjointState.position)) / 0.008)
        self.lastjointState = jointState
        pass

    def imu_callback(self, imu: Imu):
        self.imu_msg = imu

    def foot_pressure_sensor_callback(self, sensor_msg: Int32, footnum):
        if sensor_msg.data == 0:
            self.foot_pressure_values[footnum] = -1
        else:
            self.foot_pressure_values[footnum] = sensor_msg.data

    def getObservationVector(self):
        positions = self.jointState.position[:16]  # Array of floats
        print(f'positions: {positions}')
        velocities = self.jointState.velocity[:16]  # Array of floats
        print(f'velocities: {velocities}')
        imu_angvel = [self.imu_msg.angular_velocity.x, self.imu_msg.angular_velocity.y,
                      self.imu_msg.angular_velocity.z]  # IMU angular velocity
        print(f'imu_angvel: {imu_angvel}')
        imu_linacc = [self.imu_msg.linear_acceleration.x, self.imu_msg.linear_acceleration.y,
                      self.imu_msg.linear_acceleration.z]  # IMU linear acceleration
        print(f'imu_licacc: {imu_linacc}')
        imu_orientation = [self.imu_msg.orientation.x, self.imu_msg.orientation.y, self.imu_msg.orientation.z,
                           self.imu_msg.orientation.w]  # IMU orientation
        # print(f'imu_orientation: {imu_orientation}')
        feet_pressure_sensors = self.foot_pressure_values
        feet_pressure_sensors = np.array([feet_pressure_sensors[0],
                                          feet_pressure_sensors[1],
                                          feet_pressure_sensors[3],
                                          feet_pressure_sensors[2],
                                          feet_pressure_sensors[4],
                                          feet_pressure_sensors[5],
                                          feet_pressure_sensors[7],
                                          feet_pressure_sensors[6],
                                          ])
        print(f'feet_pressure_sensors: {feet_pressure_sensors}')
        orientation_vector = self.getDirectionVector()
        print(f'orientation_vector: {orientation_vector}')
        # Concatenate vector
        # return np.concatenate(
        #     (positions, velocities, imu_angvel, imu_linacc, imu_orientation, orientation_vector, feet_pressure_sensors))
        return np.concatenate(
            (positions, velocities, imu_linacc, imu_angvel, orientation_vector, feet_pressure_sensors))

    def getDirectionVector(self):
        current = self.pose
        end = self.goal_position
        print(f'current: {self.pose}')
        print(f'goal: {self.goal_position}')

        distance_unit_vec = end.get_position()[0:2] - current.get_position()[0:2]
        distance_unit_vec /= np.linalg.norm(distance_unit_vec)
        mat = current[0:3, 0:3]

        d2_vect = np.array([mat[0, 0], mat[1, 0]], dtype=np.float32)
        d2_vect /= np.linalg.norm(d2_vect)
        cos = np.dot(d2_vect, distance_unit_vec)
        sin = np.linalg.norm(np.cross(distance_unit_vec, d2_vect))
        vec = np.array([cos, sin], dtype=np.float32)
        return vec

    def setGoal(self, finishPosition):
        self.goal_position = finishPosition

    def publishAngularVelocities(self):
        js = JointState()
        js.name = []
        js.header.stamp = rospy.Time.now()
        js.velocity = []
        js.effort = []
        for i, n in enumerate(self.motor_names):
            js.name.append(n)
            js.velocity.append(self.velocity_configuration[i])  # Velocity control only
        self.pub_all_motor.publish(js)

    def setPose(self, pose):
        self.pose = pose

    def motor_control(self, action, joint_angles, env):
        _MX_28_velocity = 2 * np.pi
        # CLIP ACTIONS
        # action = np.clip(action, self._joint_limit_low, self._joint_limit_high)
        for i in range(Joints.LEFT_ARM_1, Joints.HEAD_1, 1):
            joint_cur_pos = joint_angles[i]
            velocity = action[i]
            velocity = velocity if joint_cur_pos < env._joint_limit_high[i] else -_MX_28_velocity
            velocity = velocity if joint_cur_pos > env._joint_limit_low[i] else _MX_28_velocity
            if velocity != action[i]:
                print(f'***** Joint {i} capped')
            action[i] = velocity
        return action