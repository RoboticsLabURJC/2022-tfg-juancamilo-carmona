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
from pygame.locals import KEYUP
from pygame.locals import K_a
from pygame.locals import K_s
from pygame.locals import K_w
from pygame.locals import K_d
from sensor_msgs.msg import Image
from threading import Thread
from carla_msgs.msg import CarlaEgoVehicleControl
from rclpy.executors import MultiThreadedExecutor
from std_msgs.msg import Bool
from carla_msgs.msg import CarlaStatus



class VehicleImages(Node):
    def __init__(self):
        super().__init__("Vehicle_teleop")

        self.role_name = "ego_vehicle"
        #suscritor de la imagenes
        self.image_surface = None
        size = 800, 600
        self.screen = pygame.display.set_mode(size)
        pygame.display.set_caption("teleop screen")
        
        self.image_subscriber = self.create_subscription( Image, "/carla/ego_vehicle/rgb_view/image", self.car_image_cb, 10 )
        self.vehicle_control_publisher = self.create_publisher( CarlaEgoVehicleControl, "/carla/ego_vehicle/vehicle_control_cmd", 10)       
        self.vehicle_control_manual_override_publisher = self.create_publisher( Bool, "/carla/ego_vehicle/vehicle_control_manual_override",10)
        self.auto_pilot_enable_publisher = self.create_publisher(Bool,"/carla/ego_vehicle/enable_autopilot",10)

        self.control_msg = CarlaEgoVehicleControl()

        self.control_msg.throttle = 0.0
        self.control_msg.steer = 0.0 
        self.control_msg.brake = 0.0
        self.control_msg.hand_brake = False
        self.control_msg.reverse = False
        self.control_msg.gear = 0
        self.control_msg.manual_gear_shift = False


        self.set_autopilot()
        self.set_vehicle_control_manual_override()


    def car_image_cb(self, image):

        array = numpy.frombuffer(image.data, dtype=numpy.dtype("uint8"))
        array = numpy.reshape(array, (image.height, image.width, 4))
        array = array[:, :, :3]
        array = array[:, :, ::-1]
        image_surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
        self.screen.blit(image_surface, (0,0))                
        pygame.display.flip()

    def control_vehicle(self):        

        self.set_autopilot()
        self.set_vehicle_control_manual_override()

        for event in pygame.event.get():

            if event.type == KEYDOWN:
                keys = pygame.key.get_pressed()

                if (keys[K_DOWN]):
                    self.get_logger().error("down")
                    self.control_msg.brake = 1.0

                if (keys[K_UP]):
                    self.get_logger().error("adelante")
                    self.control_msg.throttle = 1.0          

                if (keys[K_LEFT]):
                    self.get_logger().error("left")
                    self.control_msg.steer = -1.0            

                if (keys[K_RIGHT]):
                    self.get_logger().error("derecha")
                    self.control_msg.steer = 1.0
            elif event.type == KEYUP :
                keys = pygame.key.get_pressed()
                if not(keys[K_DOWN]):
                    self.get_logger().error("suelta freno")
                    self.control_msg.brake = 0.0

                if not(keys[K_UP]):
                    self.get_logger().error("suelta acelerador")
                    self.control_msg.throttle = 0.0          

                if not(keys[K_LEFT]):
                    self.get_logger().error("suelta dir")
                    self.control_msg.steer = 0.0            

                if not(keys[K_RIGHT]):
                    self.get_logger().error("suelta dir")
                    self.control_msg.steer = 0.0



        self.vehicle_control_publisher.publish(self.control_msg)


    def set_vehicle_control_manual_override(self):
        """
        Set the manual control override
        """
        self.vehicle_control_manual_override_publisher.publish((Bool(data=True)))

    def set_autopilot(self):
        """
        enable/disable the autopilot
        """
        self.auto_pilot_enable_publisher.publish(Bool(data=False))

def main(args=None):

    pygame.init()
    rclpy.init(args=args)

    teleop = VehicleImages()
    rclpy.spin_once(teleop)

    while rclpy.ok():

        teleop.control_vehicle()
        rclpy.spin_once(teleop)

    teleop.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()


