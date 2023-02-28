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
import matplotlib.pyplot as plt
import math


class VehicleTeleop(Node):
    def __init__(self):
        super().__init__("Vehicle_teleop")

        self.bridge = CvBridge()

        image_callback_group = MutuallyExclusiveCallbackGroup()
        self._default_callback_group = image_callback_group
        #subscritor de la imagenes
        self.image_surface = None
        size = 800, 600
        self.screen = pygame.display.set_mode(size)
        pygame.display.set_caption("teleop_screen")

        #self.image_subscriber = self.create_subscription( Image, "/carla/ego_vehicle/rgb_view/image", self.third_person_image_cb, 10)
        self.image_subscriber = self.create_subscription( Image, "/carla/ego_vehicle/rgb_front/image", self.first_person_image_cb, 10 )
        self.clock = pygame.time.Clock()
        self.fps = 0
        self.last_fps = 0
        self.start_time = 0

        self.lane_mean_x = (0,0)
        self.lane_mean_y = (0,0)


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
        self.error = 0

        self.left_a = []
        self.left_b = []
        self.left_c = []
        self.right_a = []
        self.right_b = []
        self.right_c = []


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


    def first_person_image_cb(self, ros_img):

        img = self.bridge.imgmsg_to_cv2(ros_img, desired_encoding='passthrough')

        filter_img = self.line_filter(img)

        if self.fps == 0:
            self.start_time = time.time()

        self.fps = self.fps + 1

        if time.time() - self.start_time >= 1:
            self.last_fps = self.fps
            self.fps = 0

        self.show_fps(filter_img)

        final_image = self.bridge.cv2_to_imgmsg(filter_img, encoding="passthrough")  

                
        array = numpy.frombuffer(filter_img.data, dtype=numpy.dtype("uint8"))
        array = numpy.reshape(array, (ros_img.height, ros_img.width, 4))
        array = array[:, :, :3]
        array = array[:, :, ::-1]
        image_surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
        self.screen.blit(image_surface, (0,0))           

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
    
    """
    def control_vehicle(self):        

        self.set_autopilot()
        self.set_vehicle_control_manual_override()

        Counter = 0
        acelerate = 0
        actual_error = 0
        i_error = 0
        last_error = 0
        INITIAL_CX = 0
        INITIAL_CY = 0
        speedup = 0


        kp_straight = 0.05  #0.05 does well in my pc
        kd_straight  = 0.7 #0.7 does well in my pc
        ki_straight = 0.000053; #0.000053 vdoes well in my pc

        kp_turn = 0.45 #0.45does well in my pc
        kd_turn= 2.5 #2.5 does well in my pc
        ki_turn = 0.0053; #0.0053 does well in my pc

        
        while True:


            actual_error = self.error
            actual_error = (actual_error) / 100  #error
            d_error =  actual_error - last_error #derivative erro
            
            i_error = i_error + actual_error #integral

            if ((actual_error < 0.35) and ( actual_error > -0.35)):
                self.control_msg.throttle = 1.0                          
                self.control_msg.steer = actual_error* kp_straight + d_error*kd_straight + i_error*ki_straight
                
            else :
                self.control_msg.throttle = -1.0          
                self.control_msg.steer = actual_error*kp_turn + d_error*kd_turn + i_error*ki_turn
                
      
    
            last_error = actual_error
            self.vehicle_control_publisher.publish(self.control_msg)
    """

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

    #para detectar blanco utiliza valores  s_thresh=(100, 255), sx_thresh=(15, 255)
  

    def perspective_warp(self,img, 
                        dst_size=(800,600),
                        src=numpy.float32([(0.43,0.65),(0.58,0.65),(0.1,1),(1,1)]),
                        dst=numpy.float32([(0,0), (1, 0), (0,1), (1,1)])):
        img_size = numpy.float32([(img.shape[1],img.shape[0])])
        src = src* img_size
        # For destination points, I'm arbitrarily choosing some points to be
        # a nice fit for displaying our warped result 
        # again, not exact, but close enough for our purposes
        dst = dst * numpy.float32(dst_size)
        
        # Given src and dst points, calculate the perspective transform matrix
        M = cv2.getPerspectiveTransform(src, dst)
        # Warp the image using OpenCV warpPerspective()
        warped = cv2.warpPerspective(img, M, dst_size)

        return warped


    def inv_perspective_warp(self,img, 
                        dst_size=(800,600),
                        src=numpy.float32([(0,0), (1, 0), (0,1), (1,1)]),
                        dst=numpy.float32([(0.43,0.65),(0.58,0.65),(0.1,1),(1,1)])):
        img_size = numpy.float32([(img.shape[1],img.shape[0])])
        src = src* img_size
        # For destination points, I'm arbitrarily choosing some points to be
        # a nice fit for displaying our warped result 
        # again, not exact, but close enough for our purposes
        dst = dst * numpy.float32(dst_size)
        # Given src and dst points, calculate the perspective transform matrix
        M = cv2.getPerspectiveTransform(src, dst)
        # Warp the image using OpenCV warpPerspective()
        warped = cv2.warpPerspective(img, M, dst_size)
        return warped


    def draw_lanes(self,img, left_fit, right_fit):
        ploty = numpy.linspace(0, img.shape[0]-1, img.shape[0])
        color_img = numpy.zeros_like(img)
        
        left = numpy.array([numpy.transpose(numpy.vstack([left_fit, ploty]))])
        right = numpy.array([numpy.flipud(numpy.transpose(numpy.vstack([right_fit, ploty])))])
        points = numpy.hstack((left, right))
        
        cv2.fillPoly(color_img, numpy.int_(points), (0,200,255))

        inv_perspective = self.inv_perspective_warp(color_img)
        inv_perspective = cv2.addWeighted(img, 1, inv_perspective, 0.7, 0)

        return inv_perspective

    def draw_centers(self, img):
        
        lane = []
        for i in range(800):
            px = img[ 450, i] 
            if px[1] == 255 and px[2] == 255:
                lane.append(i)

        center = numpy.mean(lane)
                
        center_x = int(img.shape[1]/2)
        cv2.circle(img, (center_x, 450), 8, (0,255,0), -1)
        cv2.circle(img, (int(center), 450), 4, (255,0,0), -1)


    def region_of_interest(self, img, vertices):
        mask = numpy.zeros_like(img)    
        match_mask_color = 255 # <-- This line altered for grayscale.
        
        cv2.fillPoly(mask, vertices, match_mask_color)
        masked_image = cv2.bitwise_and(img, mask)
        return masked_image

    def draw_lines(self, img, lines, color=[255, 0, 0], thickness=3):

        if lines is None:
            return    

        for line in lines:
            for x1, y1, x2, y2 in line:
                cv2.line(img, (x1, y1), (x2, y2), color, thickness)    
  
        return img

    def line_filter(self, img):

        # Convert to grayscale here.
        gray_image = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)# Call Canny Edge Detection here.
        cannyed_image = cv2.Canny(gray_image, 100, 150)

        h, w = cannyed_image.shape[:2]
        region_of_interest_vertices = [ (0, h),(w/2 , h/2) ,(w, h), ]
        cropped_image = self.region_of_interest(cannyed_image, numpy.array([region_of_interest_vertices], numpy.int32))


        lines = cv2.HoughLinesP(cropped_image,rho=9,theta=numpy.pi / 60, threshold=100,lines=numpy.array([]),minLineLength=20,maxLineGap=25)

        left_line_x = []
        left_line_y = []
        right_line_x = []
        right_line_y = []
    
        if lines is None:
            return img
        else:
            for line in lines:
                for x1, y1, x2, y2 in line:
                    slope = (y2 - y1) / (x2 - x1)

                if math.fabs(slope) < 0.5:
                    continue

                if slope <= 0:
                    left_line_x.extend([x1, x2])
                    left_line_y.extend([y1, y2])

                else:
                    right_line_x.extend([x1, x2])
                    right_line_y.extend([y1, y2])    

            min_y = int(img.shape[0] * (3 / 5))
            max_y = int(img.shape[0])    

            if not left_line_x or not left_line_y:
                return img 

            if not right_line_x or not right_line_y:
                return img


            poly_left = numpy.poly1d(numpy.polyfit(left_line_y,left_line_x,deg=1))
        
            left_x_start = int(poly_left(max_y))
            left_x_end = int(poly_left(min_y))
        
            poly_right = numpy.poly1d(numpy.polyfit(right_line_y,right_line_x,deg=1))
        
            right_x_start = int(poly_right(max_y))
            right_x_end = int(poly_right(min_y))    

            line_image = self.draw_lines(img,[[[left_x_start, max_y, left_x_end, min_y],[right_x_start, max_y, right_x_end, min_y],]],thickness=5,)
            
            lane_mean_x = int((left_x_start + left_x_end + right_x_start + right_x_end)/4)  

            image_center = int(img.shape[1]/2)

            cv2.line(img, (image_center, max_y), (image_center, min_y), [0, 255, 0], 2)    
            cv2.line(img, (lane_mean_x, max_y), (lane_mean_x, min_y), [0, 0, 255], 1)    

            return line_image

    def lane_center(self, left_line_y, left_line_x, right_line_y, right_line_x):


        y_lines = numpy.array( left_line_y + right_line_y )
        x_lines = numpy.array( left_line_x + right_line_x )

        print(y_lines)
        print(x_lines)

        y_mean = numpy.mean(y_lines, axis=0)
        x_mean = numpy.mean(x_lines, axis=0)

        return ( x_mean,y_mean )


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


