# Week 3: CARLA teleop

For this week our task will be to programm a simple teleop on ROS2 that can control a vehicle on CARLA. 

## vehicle launching
Before proceeding with the teleop development, we need to spawn a vehicle to control. We will utilize the launchers provided by carla_ros_bridge, particularly the "carla_generate_vehicle.launch.py." laucnher. This launcher will randomly place a vehicle on a selected map. The vehicle's throttle, direction, brake, and other functions can be manipulated through ROS2 topics."

## Interface
The teleop HRI is made with pygame, we have a window where we can se two images that carla pulishes on different topics, a first person view of the vehicle and a third person view of the vehicle, besides we show the fps of the first person i topic,mage on the right upper corner of the image.

![image](https://user-images.githubusercontent.com/78978326/216432598-d1e4df06-263d-45c1-b9a6-43b811c0a596.png)

## Teleop
As previously mentioned, the teleop is developed using Pygame. Pygame events will be utilized to capture key press events. Upon detection of a key press, a Carla vehicle command controller message will be modified and published on the Carla command controller topic, allowing the vehicle to respond to the key inputs, creating this way a simple  teleop.



