For this following weeks our task will be to program a simple teleop on ROS2 that can control a vehicle on CARLA. 

## vehicle launching
Before proceeding with the teleop development, we need to spawn a vehicle to control. We will utilize the launchers provided by carla_ros_bridge, particularly the "carla_generate_vehicle.launch.py." laucnher. This launcher will randomly place a vehicle on a selected map. The vehicle's throttle, direction, brake, and other functions can be manipulated through ROS2 topics."

## Interface
The teleop HRI is made with pygame, we have a window where we can se two images that carla pulishes on different topics, a first person view of the vehicle and a third person view of the vehicle, besides we show the fps of the first person i topic,mage on the right upper corner of the image.

![image](https://user-images.githubusercontent.com/78978326/216432598-d1e4df06-263d-45c1-b9a6-43b811c0a596.png)

## Teleop
As previously mentioned, the teleop is developed using Pygame. Pygame events will be utilized to capture key press events. Upon detection of a key press, a Carla vehicle command controller message will be modified and published on the Carla command controller topic, allowing the vehicle to respond to the key inputs, creating this way a simple teleop. On the following video https://youtu.be/mJZwup0JafE we can se a short demostration of how the teleop works.

## Problems and solutions

During the course of the project, several issues arose. 

The initial problem was becoming familiar with utilizing Carla and ROS through the carla_to_ros_bridge. This required research into the various topics and their respective controls, as well as a deeper understanding of how the bridge operates

One of the biggest challenges was simultaneously displaying images and controlling the vehicle. The image refresh was implemented using subscribers callbacks, which resulted in the vehicle control being blocked by a ROS spin on the node. To resolve this issue, we employed the use of threads. A separate thread was dedicated to executing the vehicle control function while the main program performed the ROS spin to continuously refresh the images.

Another problem encountered was the low frame rate of the images, which had an average of 8 to 12 FPS, as observed in the video. This issue was caused by the low frequency of the image publishers publishing rate and could not be easily resolved. The only feasible solution was to modify the source code of carla_to_ros_bridge, which will likely be pursued in the future as the main project progresses." 
