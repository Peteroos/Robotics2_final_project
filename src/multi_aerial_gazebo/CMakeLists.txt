cmake_minimum_required(VERSION 3.8)
project(multi_aerial_gazebo)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)

install(
  PROGRAMS
    scripts/multi_robot_state_interface.py
    scripts/multi_robot_pid_controller.py
  DESTINATION lib/${PROJECT_NAME}
)

install(
  DIRECTORY launch urdf worlds
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
