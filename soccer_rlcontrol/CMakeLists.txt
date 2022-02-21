cmake_minimum_required(VERSION 3.0.2)
project(soccer_rlcontrol)

find_package(catkin REQUIRED COMPONENTS
        rospy
        sensor_msgs
        )

catkin_package()
catkin_python_setup()

include_directories(
        ${catkin_INCLUDE_DIRS}
)