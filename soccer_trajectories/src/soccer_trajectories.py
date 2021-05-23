#!/usr/bin/env python3
import rospy
import soccer_trajectories
import os
from std_msgs.msg import String
import std_msgs.msg._Bool

trajectory_path = ""
simulation = False


def run_trajectory(command):
    try:
        get_up = rospy.Publisher('get_up', std_msgs.msg.Bool, queue_size=1)
        finish_walking = rospy.Publisher('walking', std_msgs.msg.Bool, queue_size=1)

        if simulation:
            path = trajectory_path + "/" + "simulation_" + command.data + ".csv"
        else:
            path = trajectory_path + "/" + command.data + ".csv"

        if not os.path.exists(path):
            return

        print("Now publishing: ", command.data)
        trajectory = soccer_trajectories.Trajectory(path)
        trajectory.publish()
        print("Finished publishing:", command.data)

        if command.data == "getupfront":
            msg = std_msgs.msg.Bool()
            msg.data = True
            get_up.publish(msg)
        elif command.data == "getupback":
            msg = std_msgs.msg.Bool()
            msg.data = True
            get_up.publish(msg)
        else:
            msg = std_msgs.msg.Bool()
            msg.data = True
            finish_walking.publish(msg)




    except rospy.ROSInterruptException:
        pass


if __name__ == '__main__':
    rospy.init_node("soccer_trajectories")
    trajectory_path = rospy.get_param("~trajectory_path")
    simulation = rospy.get_param("~simulation")
    rospy.Subscriber("command", String, run_trajectory, queue_size=1)
    rospy.spin()
