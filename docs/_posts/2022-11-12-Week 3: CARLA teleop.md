# Week 3: CARLA teleop

For this week our task will be to programm a simple teleop on ROS2 that can control a vehicle on CARLA. 

## vehicle launching
Before we start developing the teleop, we need to spawn a vehicle to controlled. For this task we will use the launchers that carla_ros_bridge offers us in it's code, more specifically we will use the carla_generate_vehicle.launch.py this launcher spawns a vehicle in a random place on the map that we choose. This vehicle thurtle, direction, brake etc. can be controlled through ros2 topics.

## Teleop
The teleop HRI1 is made with pygame, we have a window where we can se two images that carla pulishes on different topics, a first person view of the vehicle and a third person view of the vehicle, besides we show the fps of the first person image on the right upper corner of the image.

![image](https://user-images.githubusercontent.com/78978326/216432598-d1e4df06-263d-45c1-b9a6-43b811c0a596.png)




