import pygame
import sys
import numpy

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
import carla



class VehiclePerception():
    def __init__(self):     
        self.bridge = CvBridge()

        self.vehicle = 0
        self.program_start_time = -100

        self.init_enviroment()
        #subscritor de la imagenes
        self.image_surface = None
        size = 1600, 600
        self.screen = pygame.display.set_mode(size)
        pygame.display.set_caption("qlearning and DL")

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


        self.left_a = []
        self.left_b = []
        self.left_c = []
        self.right_a = []
        self.right_b = []
        self.right_c = []

        self.lane_center = 0
        self.left_lane = []
        self.right_lane = []
        
        self.Counter = 0
        self.acelerate = 0
        self.actual_error = 0
        self.i_error = 0
        self.last_error = 0
        self.INITIAL_CX = 0
        self.INITIAL_CY = 0

        self.center = 0


    def init_enviroment(self):
        # Conéctate al servidor de CARLA
        client = carla.Client('localhost', 2000)
        client.set_timeout(5.0) # Tiempo de espera en segundos

        # Obtiene el mundo
        world = client.get_world()

        # Define la ubicación donde quieres spawnear el vehículo
        spawn_location = carla.Transform(carla.Location(x=230, y=195, z=40), carla.Rotation(yaw=180))

        blueprint_library = world.get_blueprint_library()
        #vehicle_bp = blueprint_library.find('vehicle.tesla.model3')
        
        #Para un Ford Mustang
        vehicle_bp = blueprint_library.find('vehicle.ford.mustang')

        # Para un Porsche 911
        #vehicle_bp = blueprint_library.find('vehicle.porsche.911')

        # Spawn the vehicle
        self.vehicle = world.spawn_actor(vehicle_bp, spawn_location)

        # Busca el blueprint de la cámara
        camera_bp = blueprint_library.find('sensor.camera.rgb')

        # Configura la cámara
        camera_bp.set_attribute('image_size_x', '800')
        camera_bp.set_attribute('image_size_y', '600')
        camera_bp.set_attribute('fov', '110')

        # Añade la primera cámara (dashcam) al vehículo
        dashcam_location = carla.Location(x=1.5, y=0.0, z=1.4)
        dashcam_rotation = carla.Rotation(pitch=-15, yaw=0, roll=0)
        dashcam_transform = carla.Transform(dashcam_location, dashcam_rotation)
        dashcam = world.spawn_actor(camera_bp, dashcam_transform, attach_to=self.vehicle)

        # Añade la segunda cámara (vista en tercera persona) al vehículo
        third_person_cam_location = carla.Location(x=-5.5, y=0.0, z=2.8)
        third_person_cam_rotation = carla.Rotation(pitch=-20, yaw=0, roll=0)
        third_person_cam_transform = carla.Transform(third_person_cam_location, third_person_cam_rotation)
        third_person_cam = world.spawn_actor(camera_bp, third_person_cam_transform, attach_to=self.vehicle)

        # Asocia la función callback con las cámaras
        dashcam.listen(lambda image: self.first_person_image_cb(image, 'Dashcam'))
        third_person_cam.listen(lambda image: self.third_person_image_cb(image, 'Third person camera'))

        # Busca el blueprint del sensor GNSS
        gnss_bp = blueprint_library.find('sensor.other.gnss')

        # Añade el sensor GNSS al vehículo
        gnss = world.spawn_actor(gnss_bp, carla.Transform(carla.Location(x=1.0, z=2.8)), attach_to=self.vehicle)

        gnss.listen(self.position_cb)


    def calculate_speed(self):
        velocity = self.vehicle.get_velocity()
        speed = math.sqrt(velocity.x**2 + velocity.y**2)

        return 3.6 * speed  # convert m/s to km/h


    def first_person_image_cb(self, image, camera_name):
        # Convierte la imagen en una matriz numpy
        array = numpy.frombuffer(image.raw_data, dtype=numpy.dtype('uint8'))
        array = numpy.reshape(array, (image.height, image.width, 4))
        img = array[:, :, :3]

        if self.program_start_time == -100:
                self.program_start_time = time.time()

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
        array = numpy.reshape(array, (img.height, img.width, 4))
        array = array[:, :, :3]
        array = array[:, :, ::-1]
        image_surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
        self.screen.blit(image_surface, (0,0))           

        pygame.display.flip()

        #self.vehicle_controller.control_vehicle(self.lane_center, self.left_lane, self.right_lane)

    
    def position_cb(self, pos):

        self.latitude = pos.latitude
        self.longitude = pos.longitude
        if pos.latitude < 0.0001358:
            self.csv_file.close()
            exit()


        
    def third_person_image_cb(self, image, camera_name ):

        array = numpy.frombuffer(image.raw_data, dtype=numpy.dtype('uint8'))
        array = numpy.reshape(array, (image.height, image.width, 4))
        array = array[:, :, :3]
        array = array[:, :, ::-1]
        image_surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
        self.screen.blit(image_surface, (800,0))    

        pygame.display.flip()



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
        hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS).astype(float)
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
        #vehicle_controller = VehicleControl()
        perception = VehiclePerception()


def main(args=None):

    pygame.init()
    navigation = VehicleNavigation()
    #signal.signal(signal.SIGINT, navigation.controlador)                

if __name__ == '__main__':
    main()


