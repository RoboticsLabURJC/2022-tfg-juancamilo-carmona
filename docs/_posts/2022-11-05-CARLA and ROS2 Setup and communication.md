This first week job has been focused mainly on setting up our hole working enviroment. The main tasks where to install and setup both CARLA and ROS2.

## CARLA
For the carla installation I've followed the steps described on this page: https://carla.readthedocs.io/en/latest/start_quickstart/ everything went pretty well until I tried to launch the simulator. When I launched the simulator for some reason instead of using my dedicated NVIDIA GPU it used my intel integrated GPU, this caused the simulator to malfunction. It was almost impossible to navigate in the CARLA world because of the lag and my hole pc started to go slow.

After some research on CARLA forums I found this page: https://github.com/carla-simulator/carla/issues/4716
Here a found an user that sugested to run carla with the ***-prefernvidia*** flag, this way: /opt/carla-simulator/CarlaUE4.sh -prefernvidia this flag forces CARLA to run using your computers NVIDIA GPU. This way I was able to launch  CARLA  correctly

## ROS2
As I've mentioned before, ROS2 foxy is the ROS2 distribution we will be working with, so for the installation, I went to the official ROS2 installation  page https://docs.ros.org/en/foxy/index.html and follow the steps, in this case, I didn't have any problem and I was able to perform the installation with no major difficulty 


## carla_to_ros_bridge
Now we have both ROS2 and CARLA installed, but how do we make them communicate with each other? how can we make CARLA understand rostopics? 
fortunately, CARLA developers already thought of this problem and on the CARLA main GitHub page there is a repository called carla_to_ros_bridge, this software translates rostopics so that CARLA can understand them. 
The carla_to_ros_bridge repo can be found here [carla_to_ros_bridge](https://github.com/carla-simulator/ros-bridge)

![ros-bridge](https://github.com/RoboticsLabURJC/2022-tfg-juancamilo-carmona/assets/78978326/af450c6f-efde-4e7f-8e35-4a9b027876b8)
