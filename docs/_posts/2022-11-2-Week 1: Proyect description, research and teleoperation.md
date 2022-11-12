# Week 1: Project description and general research.

This first week job has been focused mainly on understanding and discussing a good topic to orientate the project and do some general research on it.

## Topic: Autonomous driving based on reinforcement learning.

In this TFG we will be attempting to create a simulated ecosystem where a car can drive autonomously in a track using reinforcement learning. The main focus of this project is to program, train and analyze the results of a model and compare it to different approaches of autonomous driving, showing the advantages and disadvantages of this model. 

We will be using an enviroment similar to the one the [AWS DEEPRACER](https://aws.amazon.com/es/deepracer/) competition uses. We will be focused on the model of the car, using similar sensors and actuators to the one the [actual car](https://www.amazon.com/dp/B07JMHRKQG) uses. 

## General research

In this first week we have visited lots of sites looking for information related with the robot. The main objective was to find a track world and an appropiate car model to create a simulation and be able to start proggraming. The models found were the following:

![Car and Track Models](https://user-images.githubusercontent.com/78983070/200606631-ddd94abf-cb84-48a3-8e6e-ec10365f5201.png)

As we can see, the model is nowhere near the physical appearence of the AWS Deepracer, but it has its key sensors and actuators: Four mobile wheels, directional front axis and a camera to sense its surroundings.

From it we have built a package where we can launch both the track and the car.

## Teleoperating 

The next objective will be to build a teleoperator to be able to move the car along the track. To do this, we firstly needed to acceed to the camera image. We have adapted the plugin that was already written into the URDF file of the robot to publish on the desired topic. We can collect and convert that image to openCV using a python script. 

In that same script we also need to be able to generate a graphical interface for the user to be able to move the car along. That interface should include buttons to manage the power of the car as well as the direction.

The problem we are facing at the moment involves implementing the controller for the wheels, as the topic where the data should be published to move the wheels is not appearing. 

Next week we will be hoping to solve this problem and finish up the teleoperator.
