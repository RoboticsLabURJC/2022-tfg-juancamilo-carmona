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
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
import cv2
from cv_bridge import CvBridge
import time



class VehicleTeleop(Node):
    def __init__(self):
        super().__init__("Vehicle_teleop")

        self.bridge = CvBridge()

        image_callback_group = MutuallyExclusiveCallbackGroup()
        self._default_callback_group = image_callback_group
        #subscritor de la imagenes
        self.image_surface = None
        size = 1600, 600
        self.screen = pygame.display.set_mode(size)
        pygame.display.set_caption("teleop_screen")

        self.image_subscriber = self.create_subscription( Image, "/carla/ego_vehicle/rgb_view/image", self.third_person_image_cb, 10)
        self.image_subscriber = self.create_subscription( Image, "/carla/ego_vehicle/rgb_front/image", self.first_person_image_cb, 10 )
        self.clock = pygame.time.Clock()
        self.fps = 0
        self.last_fps = 0
        self.start_time = 0


        self.role_name = "ego_vehicle"

        image_callback_group = MutuallyExclusiveCallbackGroup()
        self._default_callback_group = image_callback_group        

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
        self.vehicle_control_thread()


    def third_person_image_cb(self, image):

        array = numpy.frombuffer(image.data, dtype=numpy.dtype("uint8"))
        array = numpy.reshape(array, (image.height, image.width, 4))
        array = array[:, :, :3]
        array = array[:, :, ::-1]
        image_surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
        self.screen.blit(image_surface, (0,0))    

        pygame.display.flip()


    def first_person_image_cb(self, image):

        if self.fps == 0:
            self.start_time = time.time()

        self.fps = self.fps + 1

        if time.time() - self.start_time >= 1:
            self.last_fps = self.fps
            self.fps = 0
                
        filter_img = self.line_filter(image)
        array = numpy.frombuffer(filter_img.data, dtype=numpy.dtype("uint8"))
        array = numpy.reshape(array, (image.height, image.width, 4))
        array = array[:, :, :3]
        array = array[:, :, ::-1]
        image_surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
        self.screen.blit(image_surface, (800,0))           
        pygame.display.flip()




    def control_vehicle(self):        

        self.set_autopilot()
        self.set_vehicle_control_manual_override()

        while True:

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


    def vehicle_control_thread(self):
        spin_thread = Thread(target=self.control_vehicle)
        spin_thread.start()


    def line_filter(self, ros_img):

        img = self.bridge.imgmsg_to_cv2(ros_img, desired_encoding='passthrough')

        #converted = convert_hls(img)
        image = cv2.cvtColor(img,cv2.COLOR_BGR2HLS)
        #lower = numpy.uint8([0, 150, 150])
        #upper = numpy.uint8([255, 151, 200])

        #white_mask = cv2.inRange(image, lower, upper)
        # yellow color mask
        lower = numpy.uint8([10, 50,   100])
        upper = numpy.uint8([20, 180, 200])
        yellow_mask = cv2.inRange(image, lower, upper)
        # combine the mask
        #skel = cv2.bitwise_or(white_mask, yellow_mask)
        skel = yellow_mask

        result = img.copy()
        edges = cv2.Canny(skel, 50, 150)

        #cv2.imshow('mascara ', edges)
        #cv2.waitKey(0)

        lines = cv2.HoughLinesP(edges,1,numpy.pi/180,40,minLineLength=30,maxLineGap=40)
        i = 0
        if lines is not None:
            for x1,y1,x2,y2 in lines[0]:
                i+=1
                cv2.line(result,(x1,y1),(x2,y2),(255,0,0),3)

        final_img = self.show_fps(result)

        ros_image = self.bridge.cv2_to_imgmsg(final_img, encoding="passthrough")  

        return ros_image


    def show_fps(self, img):
        #fps = int(self.clock.get_fps())
        image = cv2.putText(img, 'FPS: ' + str(self.last_fps) , (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.5, (255, 0, 0), 1, cv2.LINE_AA)
        self.clock.tick(60)
        
        return image
        


def main(args=None):

    pygame.init()
    rclpy.init(args=args)
    
    teleop = VehicleTeleop()
    
    rclpy.spin(teleop)

    teleop.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()


