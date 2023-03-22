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
from std_msgs.msg import Float32
from threading import Thread
from carla_msgs.msg import CarlaEgoVehicleControl
from sensor_msgs.msg import NavSatFix
from std_msgs.msg import Bool
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
import cv2
from cv_bridge import CvBridge
import time
import math
import csv
import psutil
import os
import signal


class VehicleTeleop(Node):
    def __init__(self):
        super().__init__("Vehicle_teleop")

        self.bridge = CvBridge()

        file_name = '/home/camilo/Escritorio/tfg_metrics/hsv_metrics_5.csv'
        self.csv_file = open(file_name, mode='w', newline='')
        # Abre el archivo CSV en modo escritura
        self.csv_writer = csv.writer(self.csv_file)        
        self.csv_writer.writerow(['time','fps','cpu usage','Memory usage','PID curling','PID adjustment intesity','latitude','longitude', 'line detected num'])

        image_callback_group = MutuallyExclusiveCallbackGroup()
        self._default_callback_group = image_callback_group
        #subscritor de la imagenes
        self.image_surface = None
        size = 1600, 600
        self.screen = pygame.display.set_mode(size)
        pygame.display.set_caption("teleop_screen")

        self.image_subscriber = self.create_subscription( Image, "/carla/ego_vehicle/rgb_view/image", self.third_person_image_cb, 10)
        self.image_subscriber = self.create_subscription( Image, "/carla/ego_vehicle/rgb_front/image", self.first_person_image_cb, 10 )
        self.spedometer_subscriber= self.create_subscription( Float32, "/carla/ego_vehicle/speedometer", self.speedometer_cb, 10 )
        self.spedometer_subscriber= self.create_subscription( NavSatFix, "/carla/ego_vehicle/gnss", self.position_cb, 10 )


        self.speed = 0
        self.clock = pygame.time.Clock()
        self.fps = 0
        self.last_fps = 0
        self.start_time = 0

        self.left_x_start = 0
        self.max_y = 0
        self.left_x_end = 0
        self.min_y = 0
        self.right_x_start = 0
        self.max_y = 0
        self.right_x_end = 0
        self.min_y = 0

        self.latitude = 0
        self.longitude = 0

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
        self.line_detected_num = 0

        self.left_a = []
        self.left_b = []
        self.left_c = []
        self.right_a = []
        self.right_b = []
        self.right_c = []


        self.set_autopilot()
        self.set_vehicle_control_manual_override()
        #self.vehicle_control_thread()
        

        self.program_start_time = -100
        self.Counter = 0
        self.acelerate = 0
        self.actual_error = 0
        self.i_error = 0
        self.last_error = 0
        self.INITIAL_CX = 0
        self.INITIAL_CY = 0
        self.speedup = 0

        self.kp_straight = 0.08
        self.kd_straight  = 0.1
        self.ki_straight = 0.000002

        self.kp_turn = 0.1
        self.kd_turn= 0.15
        self.ki_turn = 0.000004

        self.adjustment_num = 0

        self.process = psutil.Process( os.getpid() )


    def position_cb(self, pos):
        
        self.latitude = pos.latitude
        self.longitude = pos.longitude
        if pos.latitude < 0.0001358:
            self.archivo_csv.close()
            exit()        
            

    def speedometer_cb(self, speed):
        self.speed = speed.data
        
    def third_person_image_cb(self, image):

        array = numpy.frombuffer(image.data, dtype=numpy.dtype("uint8"))
        array = numpy.reshape(array, (image.height, image.width, 4))
        array = array[:, :, :3]
        array = array[:, :, ::-1]
        image_surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
        self.screen.blit(image_surface, (800,0))    

        pygame.display.flip()


    def first_person_image_cb(self, ros_img):

        if self.program_start_time == -100:
            self.program_start_time = time.time()


        img = self.bridge.imgmsg_to_cv2(ros_img, desired_encoding='passthrough')

        filter_img = self.line_filter(img)

        if self.fps == 0:
            self.start_time = time.time()

        self.fps = self.fps + 1

        if time.time() - self.start_time >= 1:
            self.last_fps = self.fps
            self.fps = 0

        self.show_fps(filter_img)

        #final_image = self.bridge.cv2_to_imgmsg(filter_img, encoding="passthrough")  
                
        array = numpy.frombuffer(filter_img.data, dtype=numpy.dtype("uint8"))
        array = numpy.reshape(array, (ros_img.height, ros_img.width, 4))
        array = array[:, :, :3]
        array = array[:, :, ::-1]
        image_surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
        self.screen.blit(image_surface, (0,0))           

        pygame.display.flip()

        self.control_vehicle()



    def controlador(self, sig, frame):
        self.archivo_csv.close()
        exit()

    """"
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


        kp_straight = 0.08
        kd_straight  = 0.1
        ki_straight = 0.000002

        kp_turn = 0.1
        kd_turn= 0.15
        ki_turn = 0.000004

        adjustment_num = 0

        # Abre el archivo CSV en modo escritura
        csv_writer = csv.writer(self.archivo_csv)
        
        csv_row = ['fps','cpu usage','Memory usage','PID adjustment num','PID adjustment intesity']
        csv_writer.writerow(csv_row)
        
        while True:

            actual_error = self.error            

            actual_error = (actual_error) / 100  #error
            d_error =  actual_error - last_error #derivative erro
            
            i_error = i_error + actual_error #integral
            
            if actual_error >= 0 and last_error < 0:
                adjustment_num = adjustment_num + 1

            if actual_error <= 0 and last_error > 0:
                adjustment_num = adjustment_num + 1
            


            if ((actual_error < 50/100) and ( actual_error > -50/100)):

                #self.get_logger().error("straight " + str(stering))
                stering = actual_error* kp_straight + d_error*kd_straight + i_error*ki_straight

                if stering > 1:
                    self.control_msg.steer = 1.0

                elif stering <  -1.0:                    
                    self.control_msg.steer = -1.0

                else:
                    self.control_msg.steer = stering

                #if(actual_error == 0):
                    #self.control_msg.throttle = 1.0                          
                #else:
                    #self.control_msg.throttle = 1/actual_error* kp_straight + d_error*kd_straight + i_error*ki_straight                    
            if ((actual_error < 10/100) and ( actual_error > -10/100)):
                self.control_msg.steer = 0.0
            else :
                stering = actual_error*kp_turn + d_error*kd_turn + i_error*ki_turn

                if stering > 1:
                    self.control_msg.steer = 1.0

                elif stering <  -1.0:
                    self.control_msg.steer = -1.0

                else:
                    self.control_msg.steer = stering

                #if(actual_error == 0):
                    #self.control_msg.throttle = 1.0                          
                #else:
                    #self.control_msg.throttle = 1/actual_error* kp_straight + d_error*kd_straight + i_error*ki_straight
      
            if self.speed >= 20:
                self.control_msg.throttle = 0.0
            else:
                self.control_msg.throttle = 1.0                          

            last_error = actual_error
            self.vehicle_control_publisher.publish(self.control_msg)
            
            #el pid puede obtenerse fuera del bucle
            pid = os.getpid()
            process = psutil.Process(pid)
            memory_usage = process.memory_info().rss    

            #cpu_percent = psutil.cpu_percent()
            time.sleep(0.1)

            cpu_percent = process.cpu_percent(interval=0.1)
            csv_writer.writerow([self.last_fps, cpu_percent , memory_usage, adjustment_num, stering])
    """
    def control_vehicle(self):        

        actual_error = self.error            

        actual_error = (actual_error) / 100  #error
        d_error =  actual_error - self.last_error #derivative erro
        
        self.i_error = self.i_error + actual_error #integral
        
        if actual_error >= 0:
            self.curling = 1

        if actual_error <= 0:
            self.curling = -1
        
        if ((actual_error < 10/100) and ( actual_error > -10/100)):
            stering = 0.0
            self.control_msg.steer = stering
            self.curling = 0.0

        elif ((actual_error < 50/100) and ( actual_error > -50/100)):
            #self.get_logger().error("straight " + str(stering))
            stering = actual_error* self.kp_straight + d_error*self.kd_straight + self.i_error*self.ki_straight

            if stering > 1:
                self.control_msg.steer = 1.0

            elif stering <  -1.0:                    
                self.control_msg.steer = -1.0

            else:
                self.control_msg.steer = stering
        else :
            stering = actual_error*self.kp_turn + d_error*self.kd_turn + self.i_error*self.ki_turn

            if stering > 1:
                self.control_msg.steer = 1.0

            elif stering <  -1.0:
                self.control_msg.steer = -1.0

            else:
                self.control_msg.steer = stering

        if self.speed >= 20:
            self.control_msg.throttle = 0.0            
        else:
            self.control_msg.throttle = 1.0                          

        self.last_error = actual_error
        self.vehicle_control_publisher.publish(self.control_msg)
        
        #el pid puede obtenerse fuera del bucle

        memory_usage = self.process.memory_info().rss    
        cpu_percent = self.process.cpu_percent()
        
        self.csv_writer.writerow([time.time() - self.program_start_time , self.last_fps, cpu_percent , memory_usage/(1024*1024) , self.curling, abs(stering), self.latitude, self.longitude, self.line_detected_num])
        self.line_detected_num = 0

            
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

        hsv_image = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        lower_white = numpy.array([0, 0, 200])
        upper_white = numpy.array([255, 255, 255])
        #lower_white = numpy.array([0, 0, 160])
        #upper_white = numpy.array([130, 20, 250])
        color_mask = cv2.inRange(hsv_image, lower_white, upper_white)
        filtered_image = cv2.bitwise_and(img, img, mask=color_mask)
        #cv2.imshow('image',filtered_image)
        #cv2.waitKey(0)

        # Convert to grayscale and perform Canny edge detection
        gray_image = cv2.cvtColor(filtered_image, cv2.COLOR_RGB2GRAY)
        cannyed_image = cv2.Canny(gray_image, 100, 150)

        h, w = cannyed_image.shape[:2]
        region_of_interest_vertices = [ (0, h*0.75),(w/2 , h/2) ,(w, h*0.75), ]
        cropped_image = self.region_of_interest(cannyed_image, numpy.array([region_of_interest_vertices], numpy.int32))
        #cv2.imshow('images', cropped_image)
        #cv2.waitKey(0)

        lines = cv2.HoughLinesP(cropped_image,rho=9,theta=numpy.pi / 60, threshold=50,lines=numpy.array([]),minLineLength=20,maxLineGap=25)

        left_line_x = []
        left_line_y = []
        right_line_x = []
        right_line_y = []
        outter_left_line_x = []
        outter_left_line_y = []
        outter_right_line_x = []
        outter_right_line_y = []
        

        min_y = int(img.shape[0] * (3 / 5))
        max_y = int(img.shape[0])   
           
        if lines is None:
            return img
        
        else:
            for line in lines:
                for x1, y1, x2, y2 in line:
                    slope = (y2 - y1) / (x2 - x1)

                if math.fabs(slope) > 0.8:

                    if slope <= 0:
                        left_line_x.extend([x1, x2])
                        left_line_y.extend([y1, y2])

                    else:
                        right_line_x.extend([x1, x2])
                        right_line_y.extend([y1, y2])
                else:
                    if math.fabs(slope) > 0.2:
                        if slope <= 0:
                            outter_left_line_x.extend([x1, x2])
                            outter_left_line_y.extend([y1, y2])
                        else:
                            outter_right_line_x.extend([x1, x2])
                            outter_right_line_y.extend([y1, y2])  

            
            if outter_left_line_x and outter_left_line_y :

                outter_poly_left = numpy.poly1d(numpy.polyfit(outter_left_line_y,outter_left_line_x,deg=1))

                outter_left_x_start = int(outter_poly_left(max_y))
                outter_left_x_end = int(outter_poly_left(min_y))
                img = self.draw_lines(img,[[[outter_left_x_start, max_y, outter_left_x_end, min_y],]],thickness=5,color=[0,0,255])
                self.line_detected_num = self.line_detected_num  + 1

            if outter_right_line_x and outter_right_line_x:
                
                outter_poly_left = numpy.poly1d(numpy.polyfit(outter_right_line_y,outter_right_line_x,deg=1))

                outter_right_x_start = int(outter_poly_left(max_y))
                outter_right_x_end = int(outter_poly_left(min_y))
                img = self.draw_lines(img,[[[outter_right_x_start, max_y, outter_right_x_end, min_y],]],thickness=5,color=[0,0,255])
                self.line_detected_num = self.line_detected_num  + 1


            if left_line_x and left_line_y:

                poly_left = numpy.poly1d(numpy.polyfit(left_line_y,left_line_x,deg=1))
                left_x_start = int(poly_left(max_y))
                left_x_end = int(poly_left(min_y))

                self.left_x_start = left_x_start
                self.left_x_end = left_x_end
                img = self.draw_lines(img,[[[left_x_start, max_y, left_x_end, min_y]]],thickness=5,)

            #else: 
                #self.archivo_csv.close()
                #exit()   
                #line_image = self.draw_lines(img,[[[self.left_x_start, self.max_y, self.left_x_end, self.min_y],[self.right_x_start, self.max_y, self.right_x_end, self.min_y],]],thickness=5,)

                #lane_mean_x = int(( self.left_x_start + self.left_x_end + self.right_x_start + self.right_x_end)/4)  
                #image_center = int(line_image.shape[1]/2)
                #cv2.line(line_image, (image_center, self.max_y), (image_center, self.min_y), [0, 255, 0], 2)    
                #cv2.line(line_image, (lane_mean_x, self.max_y), (lane_mean_x, self.min_y), [0, 0, 255], 1)

            if right_line_x and right_line_y:

                poly_right = numpy.poly1d(numpy.polyfit(right_line_y,right_line_x,deg=1))
                right_x_start = int(poly_right(max_y))
                right_x_end = int(poly_right(min_y))

                self.right_x_start = right_x_start
                self.right_x_end = right_x_end
                img = self.draw_lines(img,[[[right_x_start, max_y, right_x_end, min_y],]],thickness=5,)

            #else:
                #self.archivo_csv.close()
                #exit()                   #line_image = self.draw_lines(img,[[[self.left_x_start, self.max_y, self.left_x_end, self.min_y],[self.right_x_start, self.max_y, self.right_x_end, self.min_y],]],thickness=5,)
                #lane_mean_x = int(( self.left_x_start + self.left_x_end + self.right_x_start + self.right_x_end)/4)  
                #image_center = int(line_image.shape[1]/2)
                #cv2.line(line_image, (image_center, self.max_y), (image_center, self.min_y), [0, 255, 0], 2)    
                #cv2.line(line_image, (lane_mean_x, self.max_y), (lane_mean_x, self.min_y), [0, 0, 255], 1)

            if not right_line_x and not right_line_y and not left_line_x and not left_line_y:
                self.archivo_csv.close()
                exit()  
        
            #line_image = self.draw_lines(img,[[[left_x_start, max_y, left_x_end, min_y],[right_x_start, max_y, right_x_end, min_y],]],thickness=5,)

            image_center = int(img.shape[1]/2)

            if right_line_x and right_line_y and left_line_x and left_line_y:
                lane_mean_x = int((left_x_end + right_x_end)/2)  
                cv2.line(img, (lane_mean_x, max_y), (lane_mean_x, min_y), [0, 0, 255], 1)    
                self.error =  lane_mean_x - image_center
            
            cv2.line(img, (image_center, max_y), (image_center, min_y), [0, 255, 0], 2)    


            self.max_y = max_y
            self.min_y = min_y
            self.max_y = max_y
            self.min_y = min_y

            return img


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
    
    signal.signal(signal.SIGINT, teleop.controlador)            

    rclpy.spin(teleop)

    teleop.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()


