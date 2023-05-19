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
import torch



class VehiclePerception():
    def __init__(self):     
        self.bridge = CvBridge()

        self.vehicle = 0
        self.program_start_time = -100
        self.deeplearning_model = torch.load('/home/camilo/2022-tfg-juancamilo-carmona/tfg/src/line_follower/model/fastai_torch_lane_detector_model.pth')

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
        self.deeplearning_model.eval()
        self.image_counter = 0
        self.processing_start_time = 0
        self.processing_time = 0

        self.process = psutil.Process( os.getpid() )




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


