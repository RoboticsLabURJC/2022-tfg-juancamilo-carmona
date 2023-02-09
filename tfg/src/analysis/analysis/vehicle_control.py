import pygame
import sys
import numpy
import rclpy
from rclpy.node import Node

from pygame.locals import K_ESCAPE
from pygame.locals import K_DOWN
from pygame.locals import K_LEFT
from pygame.locals import K_RIGHT
from pygame.locals import K_UP
from pygame.locals import KEYDOWN
from pygame.locals import K_a
from pygame.locals import K_s
from pygame.locals import K_w
from pygame.locals import K_d
from sensor_msgs.msg import Image
from threading import Thread
from carla_msgs.msg import CarlaEgoVehicleControl
import time




class VehicleControl(Node):
    def __init__(self):
        super().__init__("Vehicle_teleop")

        self.clock = pygame.time.Clock()
        self.vehicle_control_publisher = self.create_publisher( CarlaEgoVehicleControl, "/carla/ego_vehicle/vehicle_control_cmd_manual", 10)       

        self.throttle = 0.0
        self.steer = 0.0 
        self.brake = 0.0
        self.hand_brake = False
        self.reverse = False
        self.gear = 1
        self.manual_gear_shift = False

    def control_vehicle(self):        
        msg = CarlaEgoVehicleControl()
        for event in pygame.event.get():

            self.get_logger().error(str(event.type))            
            if event.type == KEYDOWN:
                keys = pygame.key.get_pressed()

                self.get_logger().error(str(keys[K_UP]))
                if (keys[K_DOWN]):
                    msg.brake = 1

                elif (keys[K_UP]):
                    msg.throttle = 1          
                    self.get_logger().error("adelante")

                elif (keys[K_LEFT]):
                    msg.steer = -1            

                elif (keys[K_RIGHT]):
                    msg.steer = 1

        msg.hand_brake = self.hand_brake
        msg._manual_gear_shift = self.manual_gear_shift

        self.vehicle_control_publisher.publish(msg)


def main(args=None):

    pygame.init()
    rclpy.init(args=args)

    teleop = VehicleImages()

    executor = MultiThreadedExecutor()   
    executor.add_node(teleop)
    
    executor.spin()

    executor.shutdown()
    teleop.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()