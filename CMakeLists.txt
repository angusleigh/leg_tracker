cmake_minimum_required(VERSION 2.8.3)
project(leg_tracker)

set(CMAKE_CXX_FLAGS "-std=c++11")
find_package(Boost REQUIRED system thread)
include_directories(${Boost_INCLUDE_DIRS})
link_directories(${Boost_LIBRARY_DIRS})
add_definitions(${Boost_DEFINITIONS})
if(NOT Boost_FOUND)
    message(ERROR " Boost not found!")
endif(NOT Boost_FOUND)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  std_msgs
  tf
  interactive_markers
  rosbag
  image_geometry
  message_generation
)

add_message_files(
  FILES
  Person.msg
  PersonArray.msg
  Leg.msg
  LegArray.msg
  Cluster.msg
  ClusterArray.msg
)

generate_messages(   
  DEPENDENCIES  
  std_msgs  
  geometry_msgs
)  


catkin_package(
  CATKIN_DEPENDS message_runtime
)


include_directories(
	INCLUDE include
	${catkin_INCLUDE_DIRS}
)



add_executable(
  detect_leg_clusters 
  src/detect_leg_clusters.cpp
  src/laser_processor.cpp
  src/cluster_features.cpp
)
target_link_libraries(
  detect_leg_clusters 
  ${catkin_LIBRARIES}
)


add_executable(
  local_occupancy_grid_mapping
  src/local_occupancy_grid_mapping.cpp
  src/laser_processor.cpp
)
target_link_libraries(
  local_occupancy_grid_mapping 
  ${catkin_LIBRARIES}
)


add_executable(
  extract_positive_training_clusters
  src/extract_positive_training_clusters.cpp
  src/laser_processor.cpp
)
target_link_libraries(
  extract_positive_training_clusters
  ${catkin_LIBRARIES}
)


add_executable(
  train_leg_detector
  src/train_leg_detector.cpp
  src/laser_processor.cpp
  src/cluster_features.cpp  
)
target_link_libraries(
  train_leg_detector
  ${catkin_LIBRARIES}
)

add_dependencies(local_occupancy_grid_mapping ${PROJECT_NAME}_generate_messages_cpp)
add_dependencies(local_occupancy_grid_mapping ${PROJECT_NAME}_gencpp)
add_dependencies(local_occupancy_grid_mapping ${PROJECT_NAME}_gencfg)
add_dependencies(detect_leg_clusters ${PROJECT_NAME}_generate_messages_cpp)
add_dependencies(detect_leg_clusters ${PROJECT_NAME}_gencpp)
add_dependencies(detect_leg_clusters ${PROJECT_NAME}_gencfg)
add_dependencies(extract_positive_training_clusters ${PROJECT_NAME}_generate_messages_cpp)
add_dependencies(extract_positive_training_clusters ${PROJECT_NAME}_gencpp)
add_dependencies(extract_positive_training_clusters ${PROJECT_NAME}_gencfg)
add_dependencies(train_leg_detector ${PROJECT_NAME}_generate_messages_cpp)
add_dependencies(train_leg_detector ${PROJECT_NAME}_gencpp)
add_dependencies(train_leg_detector ${PROJECT_NAME}_gencfg)


install(
  PROGRAMS scripts/joint_leg_tracker.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)


install(
  PROGRAMS scripts/individual_leg_tracker.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)






