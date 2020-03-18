cmake_minimum_required(VERSION 2.8.3)
project(interiornet_to_rosbag)

find_package(catkin_simple REQUIRED)
catkin_simple(ALL_DEPS_REQUIRED)

catkin_install_python(PROGRAMS nodes/interiornet_to_rosbag.py
                      DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION})

cs_install()
cs_export()
