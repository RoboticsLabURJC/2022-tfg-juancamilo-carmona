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
from pygame.locals import K_a
from pygame.locals import K_s
from pygame.locals import K_w
from pygame.locals import K_d
from ros_compatibility.node import CompatibleNode
from carla_msgs.msg import CarlaEgoVehicleControl
from sensor_msgs.msg import Image
from ros_compatibility.qos import QoSProfile, DurabilityPolicy
import ros_compatibility as roscomp



# es importante heredar el CompatibleNode porque nos permite usar la misma api aunque usemos ros1 y ros2
class VehicleTeleop(Node):
    def __init__(self):
        super().__init__("Vehicle_teleop")

        self.role_name = "ego_vehicle"
        super(VehicleTeleop, self).__init__("carla_teleop")
        # self.vehicle_control = CarlaEgoVehicleControl()
        self.image_subscriber = self.create_subscription( Image, "/carla/ego_vehicle/rgb_view/image", self.car_image_callback, 10 )
        Interface(self.role_name)

    def car_image_callback(self, image):

        self.get_logger().error("ejecucion del callback")
        self.get_logger().info("ejecucion del callback")
        self.get_logger().log("ejecucion del callback")
        self.get_logger().debug("ejecucion del callback")

        array = numpy.frombuffer(image.data, dtype=numpy.dtype("uint8"))
        array = numpy.reshape(array, (image.height, image.width, 4))
        array = array[:, :, :3]
        array = array[:, :, ::-1]
        self.image_surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))



class Interface(Node):
    def __init__(self, role_name):
        #suscritor de la imagenes
        self.role_name = role_name
        self.image_surface = None
        #self.image_subscriber = self.create_subscription( Image, "/carla/ego_vehicle/rgb_view/image", self.car_image, 10 )


        size = 800, 600
        screen = pygame.display.set_mode(size)
        pygame.display.set_caption("teleop screen")

        while rclpy.ok():

            #self.get_logger().error(str(self.image_surface))
            if self.image_surface is not None:
                self.get_logger().error("no es none")
                screen.blit(self.image_surface, (0.0))
                
            pygame.display.flip()

    



def main(args=None):

    pygame.init()
    rclpy.init(args=args)
    
    teleop = VehicleTeleop()
    rclpy.spin(teleop)

    teleop.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()





