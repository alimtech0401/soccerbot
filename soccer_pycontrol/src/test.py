import math
from time import sleep
from unittest import TestCase

import numpy as np
import matplotlib.pyplot as plt
import os
if "ROS_NAMESPACE" not in os.environ:
    os.environ["ROS_NAMESPACE"] = "/robot1"
import rospy

import soccerbot_controller

from soccer_pycontrol.src.transformation import Transformation
RUN_RL = True
RUN_IN_ROS = True

class Test(TestCase):

    def setUp(self): # -> None:
        if RUN_IN_ROS:
            from std_msgs.msg import String
            rospy.init_node("soccer_control")
            resetPublisher = rospy.Publisher("/reset", String, latch=True)
            b = String()
            b.data = ""
            resetPublisher.publish(b)
            if RUN_RL:
                import soccerbot_controller_ros_rl
                self.walker = soccerbot_controller_ros_rl.SoccerbotControllerRosRl()
            else:
                import soccerbot_controller_ros
                self.walker = soccerbot_controller_ros.SoccerbotControllerRos()
        else:
            self.walker = soccerbot_controller.SoccerbotController()


    def test_walk_1(self):
        self.walker.soccerbot.setPose(Transformation([0, 0, 0], [0, 0, 0, 1]))
        self.walker.wait(100)
        self.walker.soccerbot.ready()
        self.walker.wait(200)
        self.walker.soccerbot.setGoal(Transformation([5, 0, 0], [0, 0, 0, 1]))
        # self.walker.soccerbot.robot_path.show()
        self.walker.run()

    def test_walk_1_rl(self):
        from geometry_msgs.msg import PoseStamped
        self.walker.soccerbot.setPose(Transformation([0, 0, 0], [0, 0, 0, 1]))
        self.walker.wait(100)
        self.walker.soccerbot.ready()
        self.walker.wait(200)
        self.walker.new_goal = PoseStamped()
        self.walker.new_goal.pose.position.x = 5
        self.walker.new_goal.pose.position.y = 0
        self.walker.new_goal.pose.position.z = 0
        self.walker.new_goal.pose.orientation.x = 0
        self.walker.new_goal.pose.orientation.y = 0
        self.walker.new_goal.pose.orientation.z = 0
        self.walker.new_goal.pose.orientation.w = 1
        self.walker.run()

    def test_walk_2(self):
        self.walker.soccerbot.setPose(Transformation([-0.7384, -0.008, 0], [0.00000, 0, 0, 1]))
        self.walker.wait(100)
        self.walker.soccerbot.ready()
        self.walker.wait(200)
        self.walker.soccerbot.setGoal(Transformation([0.0198, -0.0199, 0], [0.00000, 0, 0, 1]))
        # self.walker.soccerbot.robot_path.show()
        self.walker.run()

    def test_walk_side(self):
        self.walker.soccerbot.setPose(Transformation([0, 0, 0], [0.00000, 0, 0, 1]))
        self.walker.wait(100)
        self.walker.soccerbot.ready()
        self.walker.wait(200)
        self.walker.soccerbot.setGoal(Transformation([0, -1, 0], [0.00000, 0, 0, 1]))
        # self.walker.soccerbot.robot_path.show()
        self.walker.run()

    def test_walk_backward(self):
        self.walker.soccerbot.setPose(Transformation([0, 0, 0], [0.00000, 0, 0, 1]))
        self.walker.wait(100)
        self.walker.soccerbot.ready()
        self.walker.wait(200)
        self.walker.soccerbot.setGoal(Transformation([-1, 0, 0], [0.00000, 0, 0, 1]))
        self.walker.run()

    def test_turn_in_place(self):
        self.walker.soccerbot.setPose(Transformation([0, 0, 0], [0.00000, 0, 0, 1]))
        self.walker.wait(100)
        self.walker.soccerbot.ready()
        self.walker.wait(200)
        goal = Transformation.get_transform_from_euler([np.pi, 0, 0])
        self.walker.soccerbot.setGoal(goal)
        # self.walker.soccerbot.robot_path.show()
        self.walker.run()

    def test_small_movement_1(self):
        self.walker.soccerbot.setPose(Transformation([0, 0, 0], [0.00000, 0, 0, 1]))
        self.walker.wait(100)
        self.walker.soccerbot.ready()
        self.walker.wait(200)
        goal = Transformation.get_transform_from_euler([np.pi, 0, 0])
        goal.set_position([0.3, 0, 0])
        self.walker.soccerbot.setGoal(goal)
        self.walker.soccerbot.robot_path.show()
        self.walker.run()

    def test_small_movement_2(self):
        self.walker.soccerbot.setPose(Transformation([0, 0, 0], [0.00000, 0, 0, 1]))
        self.walker.wait(100)
        self.walker.soccerbot.ready()
        self.walker.wait(200)
        goal = Transformation.get_transform_from_euler([np.pi, 0, 0])
        goal.set_position([-0.3, 0, 0])
        self.walker.soccerbot.setGoal(goal)
        self.walker.soccerbot.robot_path.show()
        self.walker.run()

    def test_small_movement_3(self):
        self.walker.soccerbot.setPose(Transformation([0, 0, 0], [0.00000, 0, 0, 1]))
        self.walker.wait(100)
        self.walker.soccerbot.ready()
        self.walker.wait(200)
        goal = Transformation.get_transform_from_euler([-np.pi/2, 0, 0])
        goal.set_position([-0.3, -0.3, 0])
        self.walker.soccerbot.setGoal(goal)
        self.walker.soccerbot.robot_path.show()
        self.walker.run()

    def test_do_nothing(self):
        self.walker.soccerbot.setPose(Transformation([0, 0, 0], [0.00000, 0, 0, 1]))
        self.walker.wait(100)
        self.walker.soccerbot.ready()
        self.walker.wait(200)
        goal = Transformation.get_transform_from_euler([0, 0, 0])
        self.walker.soccerbot.setGoal(goal)
        # self.walker.soccerbot.robot_path.show()
        self.walker.run()

    def test_foot_pressure_synchronization(self):
        import pybullet as pb
        fig, axs = plt.subplots(2)

        self.walker.soccerbot.setPose(Transformation([0, 0, 0], [0, 0, 0, 1]))
        self.walker.wait(100)
        self.walker.soccerbot.ready()
        self.walker.wait(200)
        self.walker.soccerbot.setGoal(Transformation([1, 0, 0], [0, 0, 0, 1]))

        times = np.linspace(0, self.walker.soccerbot.robot_path.duration(), num=math.ceil(self.walker.soccerbot.robot_path.duration() / self.walker.soccerbot.robot_path.step_size) + 1)
        lfp = np.zeros((4, 4, len(times)))
        rfp = np.zeros((4, 4, len(times)))
        i = 0
        for t in times:
            [lfp[:,:, i], rfp[:,:, i]] = self.walker.soccerbot.robot_path.footPosition(t)
            i = i + 1

        right = rfp[2, 3, :].ravel()
        left = lfp[2, 3, :].ravel()
        right = 0.1 - right
        left = left - 0.1
        axs[1].plot(times, left, label='Left')
        axs[1].plot(times, right, label='Right')
        axs[1].grid(b=True, which='both', axis='both')
        axs[1].legend()

        t = 0
        scatter_pts_x = []
        scatter_pts_y = []
        scatter_pts_x_1 = []
        scatter_pts_y_1 = []

        while t <= self.walker.soccerbot.robot_path.duration():
            if self.walker.soccerbot.current_step_time <= t <= self.walker.soccerbot.robot_path.duration():
                self.walker.soccerbot.stepPath(t, verbose=True)
                self.walker.soccerbot.apply_imu_feedback(self.walker.soccerbot.get_imu())
                sensors = self.walker.soccerbot.get_foot_pressure_sensors(self.walker.ramp.plane)
                for i in range(len(sensors)):
                    if sensors[i] == True:
                        scatter_pts_x.append(t)
                        scatter_pts_y.append(i)
                if np.sum(sensors[0:4]) >= 2:
                    scatter_pts_x_1.append(t)
                    scatter_pts_y_1.append(-0.1)
                if np.sum(sensors[4:8]) >= 2:
                    scatter_pts_x_1.append(t)
                    scatter_pts_y_1.append(0.1)

                forces = self.walker.soccerbot.apply_foot_pressure_sensor_feedback(self.walker.ramp.plane)
                pb.setJointMotorControlArray(bodyIndex=self.walker.soccerbot.body, controlMode=pb.POSITION_CONTROL,
                                             jointIndices=list(range(0, 20, 1)),
                                             targetPositions=self.walker.soccerbot.get_angles(),
                                             forces=forces
                                             )
                self.walker.soccerbot.current_step_time = self.walker.soccerbot.current_step_time + self.walker.soccerbot.robot_path.step_size

            pb.stepSimulation()
            t = t + self.walker.PYBULLET_STEP

        axs[0].scatter(scatter_pts_x, scatter_pts_y, s=3)
        axs[1].scatter(scatter_pts_x_1, scatter_pts_y_1, s=3)
        plt.show()
