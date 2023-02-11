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

        """
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
    def pipeline(self,img, s_thresh=(100, 255), sx_thresh=(15, 255)):
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
        
        color_binary = numpy.dstack((numpy.zeros_like(sxbinary), sxbinary, s_binary)) * 255
        
        combined_binary = numpy.zeros_like(sxbinary)
        combined_binary[(s_binary == 1) | (sxbinary == 1)] = 1

        return combined_binary

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

    #################################################

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

    def get_hist(sef,img):
        hist = numpy.sum(img[img.shape[0]//2:,:], axis=0)
        return hist



    def sliding_window(self,img, nwindows=9, margin=150, minpix = 1, draw_windows=True):
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
            
            
    #        if len(good_right_inds) > minpix:        
    #            rightx_current = numpy.int(numpy.mean([leftx_current +900, numpy.mean(nonzerox[good_right_inds])]))
    #        elif len(good_left_inds) > minpix:
    #            rightx_current = numpy.int(numpy.mean([numpy.mean(nonzerox[good_left_inds]) +900, rightx_current]))
    #        if len(good_left_inds) > minpix:
    #            leftx_current = numpy.int(numpy.mean([rightx_current -900, numpy.mean(nonzerox[good_left_inds])]))
    #        elif len(good_right_inds) > minpix:
    #            leftx_current = numpy.int(numpy.mean([numpy.mean(nonzerox[good_right_inds]) -900, leftx_current]))


        # Concatenate the arrays of indices
        left_lane_inds = numpy.concatenate(left_lane_inds)
        right_lane_inds = numpy.concatenate(right_lane_inds)

        # Extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds] 
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds] 

        # Fit a second order polynomial to each
        left_fit = numpy.polyfit(lefty, leftx, 2)
        right_fit = numpy.polyfit(righty, rightx, 2)
        
        self.left_a.append(left_fit[0])
        self.left_b.append(left_fit[1])
        self.left_c.append(left_fit[2])
        
        self.right_a.append(right_fit[0])
        self.right_b.append(right_fit[1])
        self.right_c.append(right_fit[2])
        
        left_fit_[0] = numpy.mean(self.left_a[-10:])
        left_fit_[1] = numpy.mean(self.left_b[-10:])
        left_fit_[2] = numpy.mean(self.left_c[-10:])
        
        right_fit_[0] = numpy.mean(self.right_a[-10:])
        right_fit_[1] = numpy.mean(self.right_b[-10:])
        right_fit_[2] = numpy.mean(self.right_c[-10:])
        
        # Generate x and y values for plotting
        ploty = numpy.linspace(0, img.shape[0]-1, img.shape[0] )
        left_fitx = left_fit_[0]*ploty**2 + left_fit_[1]*ploty + left_fit_[2]
        right_fitx = right_fit_[0]*ploty**2 + right_fit_[1]*ploty + right_fit_[2]

        out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 100]
        out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 100, 255]
        
        return out_img, (left_fitx, right_fitx), (left_fit_, right_fit_), ploty

    def get_curve(sef,img, leftx, rightx):
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
        
    #################################################


    def line_filter(self, ros_img):

        img = self.bridge.imgmsg_to_cv2(ros_img, desired_encoding='passthrough')

        img_ = self.pipeline(img)
        img_ = self.perspective_warp(img_)
        out_img, curves, lanes, ploty = self.sliding_window(img_, draw_windows=False)

        curverad =self.get_curve(img, curves[0], curves[1])

        lane_curve = numpy.mean([curverad[0], curverad[1]])
        img = self.draw_lanes(img, curves[0], curves[1])
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        fontColor = (0, 0, 0)
        fontSize=0.5
        cv2.putText(img, 'Lane Curvature: {:.0f} m'.format(lane_curve), (570, 620), font, fontSize, fontColor, 2)
        cv2.putText(img, 'Vehicle offset: {:.4f} m'.format(curverad[2]), (570, 650), font, fontSize, fontColor, 2)


        ros_image = self.bridge.cv2_to_imgmsg(img, encoding="passthrough")  

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


