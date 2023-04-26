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
from std_msgs.msg import Bool
from sensor_msgs.msg import NavSatFix
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
import cv2
from cv_bridge import CvBridge
import time
import math
import csv
import psutil
import os
import signal
from sensor_msgs.msg import NavSatFix
import torch


class VehicleTeleop(Node):
    def __init__(self):
        super().__init__("Vehicle_teleop")

        self.bridge = CvBridge()

        file_name = '/home/camilo/Escritorio/tfg_metrics/deeplearning_metrics_1.csv'
        self.csv_file = open(file_name, mode='w', newline='')
        # Abre el archivo CSV en modo escritura
        self.csv_writer = csv.writer(self.csv_file)        
        self.csv_writer.writerow(['time','fps','cpu usage','Memory usage','PID curling','PID adjustment intesity','latitude', 'longitude','lines detected num'])

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
        self.line_detected_num = 0

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

        self.kp_straight = 0.07
        self.kd_straight  = 0.09
        self.ki_straight = 0.000003

        self.kp_turn = 0.1
        self.kd_turn= 0.15
        self.ki_turn = 0.000004

        self.center = 0
        self.adjustment_num = 0

        self.deeplearning_model = torch.load('/home/camilo/2022-tfg-juancamilo-carmona/tfg/src/line_follower/model/fastai_torch_lane_detector_model.pth')
        self.deeplearning_model.eval()
        self.image_counter = 0

        self.process = psutil.Process( os.getpid() )


    def position_cb(self, pos):

        self.latitude = pos.latitude
        self.longitude = pos.longitude
        if pos.latitude < 0.0001358:
            self.csv_file.close()
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
        self.csv_file.close()
        exit()

    def control_vehicle(self):        

        actual_error = (self.error)/100            

        if (abs(actual_error - self.last_error)) < 0.3:

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
                    
            if self.last_fps != 0:
                self.csv_writer.writerow([time.time() - self.program_start_time , self.last_fps, cpu_percent , memory_usage/(1024*1024) , self.curling, stering, self.latitude, self.longitude, self.line_detected_num ])

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


    def get_prediction(self, img_array):
        with torch.no_grad():
            image_tensor = img_array.transpose(2,0,1).astype('float32')/255
            x_tensor = torch.from_numpy(image_tensor).to("cuda").unsqueeze(0)
            model_output = torch.softmax( self.deeplearning_model.forward(x_tensor), dim=1 ).cpu().numpy()

        return model_output

    def lane_detection_overlay(self, image, left_mask, right_mask):
        res = numpy.copy(image)
        # We show only points with probability higher than 0.5
        res[left_mask > 0.5, :] = [255,0,0]
        res[right_mask > 0.5,:] = [255, 0, 0]
        return res
    


    def draw_centers(self, img):
        
        lane = []
        for i in range(1024):
            px = img[ 304, i] 
            if px[0] == 255:
                lane.append(i)

        center = numpy.mean(lane)
        center_x = int(img.shape[1]/2)
        
        cv2.line(img, (center_x, 400), (center_x, 512), [0, 0, 255], 2)    
        cv2.line(img, (center_x-5, 304), (center_x+5, 304), [0, 0, 255], 1)
        cv2.line(img, (int(center), 400), (int(center), 512), [0, 255, 0], 1)

        self.error =  center - center_x


    def find_lane_center(self,left_mask, right_mask):
        # Suma las máscaras a lo largo del eje de las columnas
        left_sum = numpy.sum(left_mask, axis=0)
        right_sum = numpy.sum(right_mask, axis=0)

        # Encuentra la posición de la columna con la suma más alta para cada máscara
        left_index = numpy.argmax(left_sum)
        right_index = numpy.argmax(right_sum)

        # Calcula el centro del carril como el punto medio entre los dos índices
        lane_center = int((left_index + right_index) / 2)

        return lane_center


    def draw_lanes(self,img, left, right):
        color_img = numpy.zeros_like(img)
        
        points = numpy.hstack((left, right))
        
        cv2.fillPoly(color_img, numpy.int_(points), (0,200,255))
        final_img = cv2.addWeighted(img, 1, color_img, 0.7, 0)

        return final_img
    
    def line_filter(self, img):
        
        resized_img = cv2.resize(img, (1024, 512) )

        resized_img = cv2.cvtColor(resized_img, cv2.COLOR_BGRA2RGB)

        back, left, right = self.get_prediction(resized_img)[0]

        filtered_img = self.lane_detection_overlay(resized_img, left, right)

        #filtered_img = self.draw_lanes(filtered_img, left, right )

        self.draw_centers(filtered_img)

        final_img = cv2.cvtColor(filtered_img, cv2.COLOR_RGB2BGRA)
        final_img = cv2.resize(final_img, (800, 600) )


        return final_img


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


