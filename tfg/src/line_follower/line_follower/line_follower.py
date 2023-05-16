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
import random


class Actions:
    def __init__(self):
        # Acciones posibles
        self.actions = ['forward', 
                        'turn_left_1', 'turn_left_2', 'turn_left_3', 
                        'turn_right_1', 'turn_right_2', 'turn_right_3']
        
        self.intesity_1 = 0.1
        self.intesity_2 = 0.2
        self.intesity_3 = 0.3

    def forward(self):

        if self.speed >= 20:
            self.control_msg.throttle = 0.0            
        else:
            self.control_msg.throttle = 1.0        

    def turn_left(self, stering):
        """
        Método para girar a la izquierda con una cierta intensidad. 
        Aquí deberías implementar el código que gira tu vehículo a la izquierda.
        """

        if stering > 1:
            self.control_msg.steer = 1.0

        elif stering <  -1.0:                    
            self.control_msg.steer = -1.0

        else:
            self.control_msg.steer = stering


    def turn_right(self, stering):
        """
        Método para girar a la derecha con una cierta intensidad. 
        Aquí deberías implementar el código que gira tu vehículo a la derecha.
        """

        if stering > 1:
            self.control_msg.steer = 1.0

        elif stering <  -1.0:
            self.control_msg.steer = -1.0

        else:
            self.control_msg.steer = stering

    def execute_action(self, action):
        """
        Método para ejecutar una acción dada.
        """
        if action == 'forward':
            self.forward()

        elif action == 'turn_left_1':
            stering = -self.intensity_1
            self.turn_left(stering)

        elif action == 'turn_left_2':
            stering = -self.intensity_2
            self.turn_right(stering)

        elif action.startswith == 'turn_left_3':
            stering = -self.intensity_2
            self.turn_left(stering)

        elif action.startswith == 'turn_right_1':
            stering = self.intensity_1
            self.turn_right(stering)

        elif action.startswith == 'turn_right_2':
            stering = self.intensity_2
            self.turn_left(stering)

        elif action.startswith == 'turn_right_3':
            stering = self.intensity_3
            self.turn_right(stering)

        else:
            raise ValueError(f"Acción desconocida: {action}")


class RewardFunction:
    def __init__(self):
        # Ajustes para la recompensa/punición
        self.collision_penalty = -1000
        self.off_road_penalty = -100
        self.center_reward_factor = 10
        self.image_center = 400  # Mitad de la imagen

    def get_reward(self, car):
        """
        Función de recompensa para un coche dado.

        Args:
            car: Un objeto que representa el coche.

        Returns:
            Un valor de recompensa flotante.
        """
        if car.crashed:
            return self.collision_penalty

        if not car.on_road:
            return self.off_road_penalty

        # Calcula la distancia al centro de la imagen
        distance_to_center = abs(self.lane_center - self.image_center)

        # Da recompensa por estar cerca del centro de la imagen
        reward = self.center_reward_factor / (distance_to_center + 1.0)

        return reward


import numpy as np

class VehicleControl:
    def __init__(self, num_states, num_actions, alpha=0.5, gamma=0.9, epsilon=0.1):

        #Qlearning variables
        self.num_states = num_states
        self.num_actions = num_actions
        self.alpha = alpha  # Tasa de aprendizaje
        self.gamma = gamma  # Factor de descuento
        self.epsilon = epsilon  # Probabilidad de exploración
        self.q_table = np.zeros((num_states, num_actions))  # Inicializar la tabla Q
        self.actions = Actions()
        self.reward_function = RewardFunction()

        #Metrics file variables 
        file_name = '/home/camilo/Escritorio/tfg_metrics/sliding_window_metrics_1.csv'
        self.csv_file = open(file_name, mode='w', newline='')
        self.csv_writer = csv.writer(self.csv_file)        

        self.csv_writer.writerow(['time','fps','cpu usage','Memory usage','PID curling','PID adjustment intesity','latitude', 'longitude','lines detected num','processing_time'])


        #vehicle control variables 
        self.spedometer_subscriber= self.create_subscription( Float32, "/carla/ego_vehicle/speedometer", self.speedometer_cb, 10 )
        self.spedometer_subscriber= self.create_subscription( NavSatFix, "/carla/ego_vehicle/gnss", self.position_cb, 10 )

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

        self.speed = 0

        self.latitude = 0
        self.longitude = 0

        self.process = psutil.Process( os.getpid() )

        self.set_autopilot()
        self.set_vehicle_control_manual_override()
        

    def position_cb(self, pos):

        self.latitude = pos.latitude
        self.longitude = pos.longitude
        if pos.latitude < 0.0001358:
            self.csv_file.close()
            exit()


    def speedometer_cb(self, speed):
        self.speed = speed.data
        
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



    def get_state(self, car, lane):
        # Obtén la posición del centro del carril
        lane_center_position = car.lane_center

        # Discretiza la posición del centro del carril
        # Supongamos que lane_center_position puede variar de 0 a 800 (si la imagen es de 800 píxeles de ancho)
        lane_center_bins = np.linspace(0, 800, num=20)  # Ajusta el número de bins según tus necesidades
        lane_center_bin = np.digitize(lane_center_position, lane_center_bins)

        # Haz lo mismo para las líneas laterales del carril
        left_line_position, right_line_position = car.lane_lines
        line_bins = np.linspace(0, 800, num=20)
        left_line_bin = np.digitize(left_line_position, line_bins)
        right_line_bin = np.digitize(right_line_position, line_bins)

        # Combina los bins en un único número de estado
        state = lane_center_bin * len(line_bins)**2 + left_line_bin * len(line_bins) + right_line_bin

        return state

    def choose_action(self, state):
        """
        Elige una acción basada en la política epsilon-greedy.
        """
        if np.random.uniform() < self.epsilon:
            # Exploración: elige una acción aleatoria
            action = np.random.choice(self.num_actions)
        else:
            # Explotación: elige la acción con el mayor valor Q para el estado actual
            action = np.argmax(self.q_table[state, :])
            
        return action

    def update_q_table(self, state, action, reward, next_state):
        """
        Actualiza la tabla Q basada en la recompensa recibida por tomar una acción en un estado.
        """
        current_q = self.q_table[state, action]
        new_q = reward + self.gamma * np.max(self.q_table[next_state, :])
        self.q_table[state, action] = (1 - self.alpha) * current_q + self.alpha * new_q

    def control_vehicle(self, car, lane):
        """
        Controla el vehículo durante un episodio de aprendizaje.
        """
        state = self.get_state(car, lane)
        action = self.choose_action(state)
        self.actions.execute_action(action)
        reward = self.reward_function.get_reward(car, lane)
        next_state = self.get_state(car, lane)
        self.update_q_table(state, action, reward, next_state)


    def control_vehicle(self, car, lane):
        # Observa el estado actual.
        next_state = VehicleControl.get_state(car, lane)

        # Si es la primera vez, no hay acción ni estado anteriores, por lo que debemos seleccionar una acción inicial.
        if self.current_state is None:
            next_action = random.choice(Actions.ALL)
        else:
            # Calcula la recompensa basada en el estado actual y la acción actual.
            reward = self.reward_function.get_reward(car)

            # Actualiza la tabla Q.
            old_value = self.q_table[self.current_state][self.current_action]
            future_rewards = [self.q_table[next_state][action] for action in Actions.ALL]
            new_value = reward + self.gamma * max(future_rewards)
            self.q_table[self.current_state][self.current_action] = (1 - self.alpha) * old_value + self.alpha * new_value

            # Selecciona la siguiente acción basada en la tabla Q.
            next_action = Actions.ALL[np.argmax(self.q_table[next_state])]

        # Aplica la acción seleccionada.
        car.apply_action(next_action)

        # Actualiza el estado y la acción actuales.
        self.current_state = next_state
        self.current_action = next_action
    
    """
    def control_vehicle(self, lane_center, left_lane, right_lane):        

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
                
        if self.last_fps != 0:
            self.csv_writer.writerow([time.time() - self.program_start_time , self.last_fps, cpu_percent , memory_usage/(1024*1024) , self.curling, stering, self.latitude, self.longitude, self.line_detected_num, self.processing_time ])

        self.line_detected_num = 0
    """

class VehiclePerception(Node):
    def __init__(self, vehicle_controller):
        super().__init__("Vehicle_perception")

        self. vehicle_controller = vehicle_controller
        self.bridge = CvBridge()

        #subscritor de la imagenes
        self.image_surface = None
        size = 1600, 600
        self.screen = pygame.display.set_mode(size)
        pygame.display.set_caption("sliding window algorithm")

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

        self.lane_center = 0
        self.left_lane = []
        self.right_lane = []
        
        self.program_start_time = -100
        self.Counter = 0
        self.acelerate = 0
        self.actual_error = 0
        self.i_error = 0
        self.last_error = 0
        self.INITIAL_CX = 0
        self.INITIAL_CY = 0
        self.speedup = 0

        self.center = 0


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


        self.processing_start_time = time.time()
        filter_img = self.line_filter(img)
        self.processing_time = time.time() - self.processing_start_time

        if self.fps == 0:
            self.start_time = time.time()

        self.fps = self.fps + 1

        if time.time() - self.start_time >= 1:
            self.last_fps = self.fps
            self.fps = 0

        self.show_fps(filter_img)
                
        array = numpy.frombuffer(filter_img.data, dtype=numpy.dtype("uint8"))
        array = numpy.reshape(array, (ros_img.height, ros_img.width, 4))
        array = array[:, :, :3]
        array = array[:, :, ::-1]
        image_surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
        self.screen.blit(image_surface, (0,0))           

        pygame.display.flip()

        self.vehicle_controller.control_vehicle(self.lane_center, self.left_lane, self.right_lane)


    """"
    def controlador(self, sig, frame):
        self.csv_file.close()
        exit()
    
    def vehicle_control_thread(self):
        spin_thread = Thread(target=self.control_vehicle)
        spin_thread.start()
    """

  
    #para detectar blanco utiliza valores  s_thresh=(100, 255), sx_thresh=(15, 255)
    def pipeline(self,img, s_thresh=(200, 255), sx_thresh=(50, 255)):

        img = numpy.copy(img)

        # Convert to HLS color space and separate the V channel
        hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS).astype(numpy.float)
        l_channel = hls[:,:,1]
        s_channel = hls[:,:,2]
        h_channel = hls[:,:,0]


        # Sobel x detecta lor bordes en x
        sobelx = cv2.Sobel(l_channel, cv2.CV_64F, 1, 1) # Take the derivative in x
        abs_sobelx = numpy.absolute(sobelx) # Absolute x derivative to accentuate lines away from horizontal
        scaled_sobel = numpy.uint8(255*abs_sobelx/numpy.max(abs_sobelx))
        
        # Threshold x gradient
        sxbinary = numpy.zeros_like(scaled_sobel)
        sxbinary[(scaled_sobel >= sx_thresh[0]) & (scaled_sobel <= sx_thresh[1])] = 1
        
        # Threshold color channel
        s_binary = numpy.zeros_like(s_channel)
        s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1
        
        #color_binary = numpy.dstack((numpy.zeros_like(sxbinary), sxbinary, s_binary)) * 255
        
        combined_binary = numpy.zeros_like(sxbinary)
        combined_binary[(s_binary == 1) | (sxbinary == 1)] = 1

        return combined_binary

    def perspective_warp(self,img, 
                        dst_size=(800,600),
                        src=numpy.float32([(0.39,0.57),(0.62,0.57),(0.1,1),(1,1)]),
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
                        dst=numpy.float32([(0.39,0.57),(0.62,0.57),(0.1,1),(1,1)])):
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

    def get_hist(sef,img):
        hist = numpy.sum(img[img.shape[0]//2:,:], axis=0)
        return hist



    def sliding_window(self,img, nwindows=10, margin=150, minpix = 1, draw_windows=True):
        left_fit_= numpy.empty(3)
        right_fit_ = numpy.empty(3)
        out_img = numpy.dstack((img, img, img))*255

        histogram = self.get_hist(img)
        # find peaks of left and right halves
        midpoint = int(histogram.shape[0]/2)
        leftx_base = numpy.argmax(histogram[:midpoint])
        rightx_base = numpy.argmax(histogram[midpoint:]) + midpoint

        # Set height of windows
        window_height = numpy.int(img.shape[0]/nwindows)
        # Identify the x and y positions of all nonzero pixels in the image
        nonzero = img.nonzero()
        nonzeroy = numpy.array(nonzero[0])
        nonzerox = numpy.array(nonzero[1])
        # Current positions to be updated for each window
        leftx_current = leftx_base
        rightx_current = rightx_base
                
        # Create empty lists to receive left and right lane pixel indices
        left_lane_inds = []
        right_lane_inds = []

        # Step through the windows one by one
        for window in range(nwindows):
            # Identify window boundaries in x and y (and right and left)
            win_y_low = img.shape[0] - (window+1)*window_height
            win_y_high = img.shape[0] - window*window_height
            win_xleft_low = leftx_current - margin
            win_xleft_high = leftx_current + margin
            win_xright_low = rightx_current - margin
            win_xright_high = rightx_current + margin
            # Draw the windows on the visualization image
            if draw_windows == True:
                cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),
                (100,255,255), 3) 
                cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),
                (100,255,255), 3) 

            # Identify the nonzero pixels in x and y within the window
            good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
            (nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
            good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
            (nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]
            # Append these indices to the lists
            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)

            # If you found > minpix pixels, recenter next window on their mean position
            if len(good_left_inds) > minpix:
                leftx_current = numpy.int(numpy.mean(nonzerox[good_left_inds]))
            if len(good_right_inds) > minpix:        
                rightx_current = numpy.int(numpy.mean(nonzerox[good_right_inds]))
            
        # Concatenate the arrays of indices
        left_lane_inds = numpy.concatenate(left_lane_inds)
        right_lane_inds = numpy.concatenate(right_lane_inds)

        
        # Extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds] 
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds] 

        # Fit a second order polynomial to each

        ploty = numpy.linspace(0, img.shape[0]-1, img.shape[0] )


        if lefty.any() and leftx.any() :

            left_fit = numpy.polyfit(lefty, leftx, 2)
            self.line_detected_num = self.line_detected_num +1

            self.left_a.append(left_fit[0])
            self.left_b.append(left_fit[1])
            self.left_c.append(left_fit[2])

            left_fit_[0] = numpy.mean(self.left_a[-10:])
            left_fit_[1] = numpy.mean(self.left_b[-10:])
            left_fit_[2] = numpy.mean(self.left_c[-10:])

            left_fitx = left_fit_[0]*ploty**2 + left_fit_[1]*ploty + left_fit_[2]
        else:
            left_fitx = numpy.empty(0)


        if righty.any() and rightx.any() :
            right_fit = numpy.polyfit(righty, rightx, 2)
            self.line_detected_num = self.line_detected_num +1
        
            self.right_a.append(right_fit[0])
            self.right_b.append(right_fit[1])
            self.right_c.append(right_fit[2])
            
            right_fit_[0] = numpy.mean(self.right_a[-10:])
            right_fit_[1] = numpy.mean(self.right_b[-10:])
            right_fit_[2] = numpy.mean(self.right_c[-10:])
            # Generate x and y values for plotting
            right_fitx = right_fit_[0]*ploty**2 + right_fit_[1]*ploty + right_fit_[2]
        else:
            right_fitx = numpy.empty(0)
        
        return (left_fitx, right_fitx)

    def get_curve(self,img, leftx, rightx):
        ploty = numpy.linspace(0, img.shape[0]-1, img.shape[0])
        y_eval = numpy.max(ploty)
        ym_per_pix = 30.5/600 # meters per pixel in y dimension
        xm_per_pix = 3.7/800 # meters per pixel in x dimension

        # Fit new polynomials to x,y in world space
        left_fit_cr = numpy.polyfit(ploty*ym_per_pix, leftx*xm_per_pix, 2)
        right_fit_cr = numpy.polyfit(ploty*ym_per_pix, rightx*xm_per_pix, 2)
        # Calculate the new radii of curvature
        left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / numpy.absolute(2*left_fit_cr[0])
        right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / numpy.absolute(2*right_fit_cr[0])

        car_pos = img.shape[1]/2
        l_fit_x_int = left_fit_cr[0]*img.shape[0]**2 + left_fit_cr[1]*img.shape[0] + left_fit_cr[2]
        r_fit_x_int = right_fit_cr[0]*img.shape[0]**2 + right_fit_cr[1]*img.shape[0] + right_fit_cr[2]
        lane_center_position = (r_fit_x_int + l_fit_x_int) /2
        center = (car_pos - lane_center_position) * xm_per_pix / 10
        self.get_logger().error(str(center))
        self.center = center

        # Now our radius of curvature is in meters
        return (left_curverad, right_curverad, center)

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
            px = img[ 350, i] 
            if px[1] == 255 and px[2] == 255:
                lane.append(i)

        center = numpy.mean(lane)
        self.lane_center = center
        center_x = int(img.shape[1]/2)

        cv2.line(img, (center_x, 450), (center_x, 600), [0, 255, 0], 2)    
        cv2.line(img, (int(center)-5, 350), (int(center)+5, 350), [0, 0, 255], 1)    
        cv2.line(img, (int(center), 450), (int(center), 600), [0, 0, 255], 1)

        self.error =  center - center_x



    def line_filter(self, img):

        img_ = self.pipeline(img)
        img_ = self.perspective_warp(img_)
        curves = self.sliding_window(img_, draw_windows=True)
        self.left_lane = curves[0]
        self.right_lane = curves[1]
        if curves[0].any() and curves[1].any():
            img = self.draw_lanes(img, curves[0], curves[1])
            self.draw_centers(img)
        else:
            center_x = int(img.shape[1]/2)        
            cv2.line(img, (center_x, 450), (center_x, 600), [0, 255, 0], 2)    


        return img


    def show_fps(self, img):
        #fps = int(self.clock.get_fps())
        image = cv2.putText(img, 'FPS: ' + str(self.last_fps) , (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.5, (255, 0, 0), 1, cv2.LINE_AA)
        self.clock.tick(60)
        
        return image



class VehicleNavigation:
    def __init__(self):       
        vehicle_controller = VehicleControl()
        perception = VehiclePerception(vehicle_controller)


def main(args=None):

    pygame.init()
    rclpy.init(args=args)

    navigation = VehicleNavigation()
    #signal.signal(signal.SIGINT, navigation.controlador)                
    rclpy.spin(navigation)

    navigation.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()


