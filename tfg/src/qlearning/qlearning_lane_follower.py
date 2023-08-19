import carla
import random
import pygame
import numpy as np
import cv2
import time
import torch
from carla import VehicleControl
import pickle
import numpy as np
import csv
from prettytable import PrettyTable
import math
import threading
import matplotlib.pyplot as plt

class QLearningVehicleControl:
    def __init__(self,vehicle, num_actions=21, num_states=23):
        self.learning_rate = 0.5
        self.discount_factor = 0.95
        self.exploration_rate = 0.0
        self.num_actions = num_actions
        self.exploration_rate_counter = 0
        self.vehicle = vehicle
        self.lane_lines = 100
        self.start = True
        self.latitude = 100
        self.longitude = 100
        self.random_counter = 0
        self.table_counter = 0
        self.steer = 0.0
        self.speed = 4.0
        self.object_in_front = False
        
        self.lane_center_error = 0 
        self.lane_center = 0  
        self.episode_actions = []
        self.ACTIONS = [ 
            'forward',  
            'left_1',  
            'left_2',
            'left_3',  
            'left_4',  
            'left_5', 
            'left_6',  
            'left_7',
            'left_8',  
            'left_9', 
            'left_10',  
            'right_1',  
            'right_2',
            'right_3',  
            'right_4',  
            'right_5', 
            'right_6', 
            'right_7', 
            'right_8', 
            'right_9',
            'right_10'  
        ]
        self.SPEED = [ 
            'speed_1',  
            'speed_2',  
            'speed_3',
            'speed_4'
        ]

        self.q_table = np.zeros((num_states, len(self.ACTIONS), len(self.SPEED), 2 ))
        self.load_q_table("/home/camilo/2022-tfg-juancamilo-carmona/tfg/src/qlearning/q_table.pkl")


    def load_q_table(self, file_path):
        with open(file_path, 'rb') as f:
            loaded_q_table = pickle.load(f)
        if loaded_q_table.shape == self.q_table.shape:
            self.q_table = loaded_q_table
            print("Loaded Q-table successfully.")
        else:
            print("Loaded Q-table has incompatible shape.")

    def set_new_actuators(self, vehicle):
        self.vehicle = vehicle

    def get_lane_center(self):
        return self.lane_center

    def get_lane_center_error(self):
        return self.lane_center_error

    def set_lane_center_error(self, error):
        self.lane_center_error = error

    def set_lane_center(self, center):
        self.lane_center = center

    def get_qlearning_parameters(self):
        return self.learning_rate, self.discount_factor, self.exploration_rate
    
    def set_vehicle(self, vehicle):
        self.vehicle = vehicle

    def choose_action(self, state):
        if np.random.uniform(0, 1) < self.exploration_rate:
            action = np.random.randint(len(self.ACTIONS))
            self.random_counter += 1
        else:
            action_values = self.q_table[state]
            action = np.unravel_index(action_values.argmax(), action_values.shape)[0]
            self.table_counter += 1

        return action
    
    def choose_speed(self, state):
        action_values = self.q_table[state]
        if np.random.uniform(0, 1) < self.exploration_rate:
            action = np.random.randint(len(self.SPEED))
            self.random_counter += 1
        else:
            action_values = self.q_table[state]
            action = np.unravel_index(action_values.argmax(), action_values.shape)[1]
            self.table_counter += 1

        return action
        

    
    def get_random_counter(self):
        return self.random_counter
    def get_table_counter(self):
        return self.table_counter
    def set_random_counter(self, value):
        self.random_counter = value
    def set_table_counter(self, value):
        self.table_counter = value

    def update_q_table(self, current_state, steering_action, acceleration_action,reward,next_state, object_in_front,next_object_in_front):
        # Obtener el máximo valor Q para el próximo estado y siguiente valor de la variable object_in_front
        future_max_q = np.max(self.q_table[next_state, :, :, int(next_object_in_front)])

        # Calcular el nuevo valor Q para el estado y acción actual
        current_q_value = self.q_table[current_state, steering_action, acceleration_action, int(object_in_front)]
        new_q = (1 - self.learning_rate) * current_q_value + \
                self.learning_rate * (reward + self.discount_factor * future_max_q)

        # Actualizar la tabla Q con el nuevo valor
        self.q_table[current_state, steering_action, acceleration_action, int(object_in_front)] = new_q

        if self.exploration_rate_counter > 200:            
            self.exploration_rate = self.exploration_rate - 0.1
            self.exploration_rate_counter = 0

        # Optional: Consider removing this part or adjusting it as per your requirements.
        #if self.exploration_rate < 0.01:
            #print("exploration rate is lower than 0.1 finishing training")
            #exit()


    def increment_exploration_counter(self):
        self.exploration_rate_counter += 1

    def set_exploration_rate(self, exploration_rate):
        self.exploration_rate =  exploration_rate

    def get_state(self, center_of_lane):

        object_in_front = self.object_in_front
        #threshold for the lines that define the stastes
        thresholds = np.array([0,292,312,332,352,372,392,412,432,452,472,492,512,532,552,572,592,612,632,652,672,692,712,1025]) 
        #thresholds = np.array([0,312,352,392,432,472,492,512,532,552,592,632,672,712,1025]) 
        for i in range( len(thresholds) - 1 ):
            if thresholds[i] <= center_of_lane < thresholds[i + 1]:
                return i, object_in_front

        return int(len(thresholds) / 2), object_in_front

    #we use an exponencial function to calculate the reward
    def reward_function(self, error, angle_error,car_crashed):

        normalized_error = abs(error)

        # Estrategia para el error original
        reward = ((((1 / (normalized_error + 1)) + self.speed/100) - angle_error/100))

        #if we stop without any obstacle we set a small penalization
        if not self.object_in_front and self.speed == 0.0:
            reward = reward - 0.1
            print("penaliza por no parar ", self.speed)

        elif self.object_in_front and self.speed == 0.0:
            reward = reward + 0.1
            print("premia por parar ", self.speed)

        # Si no detectamos ambas líneas del carril, se aplica una gran penalización
        if self.lane_lines < 1:
            reward = reward - 20
            
        #if car crashes we give a big penalization
        if car_crashed:
            reward = reward - 20

        #print("reward: ", reward)
        return reward
    
    def accelerate(self):
        control = VehicleControl()  
        control.throttle = 0.5
        self.vehicle.apply_control(control)      

    def perform_action(self, action, speed):

        control = VehicleControl()
        if action == 'forward':
            control.steer = 0.0

        elif action == 'left_1':
            control.steer = -0.01

        elif action == 'left_2':
            control.steer = -0.02

        elif action == 'left_3':
            control.steer = -0.04

        elif action == 'left_4':
            control.steer = -0.05

        elif action == 'left_5':
            control.steer = -0.06

        elif action == 'left_6':
            control.steer = -0.08

        elif action == 'left_7':
            control.steer = -0.1
            
        elif action == 'left_8':
            control.steer = -0.11

        elif action == 'left_9':
            control.steer = -0.12

        elif action == 'right_1':
            control.steer = 0.01

        elif action == 'right_2':
            control.steer = 0.02

        elif action == 'right_3':
            control.steer = 0.04

        elif action == 'right_4':
            control.steer = 0.05

        elif action == 'right_5':
            control.steer = 0.06

        elif action == 'right_6':
            control.steer = 0.08

        elif action == 'right_7':
            control.steer = 0.1
        
        elif action == 'right_8':
            control.steer = 0.11

        elif action == 'right_9':
            control.steer = 0.12

        elif action == 'right_14':
            control.steer = 0.18
        
        elif speed == 'speed_1':
            self.speed = 4.0

        elif speed == 'speed_2':
            self.speed = 4.5

        elif speed == 'speed_3':
            self.speed = 5.0

        elif speed == 'speed_4':
            self.speed = 0.0

        #we try to mantain the same speed all the time
        velocity = self.vehicle.get_velocity()
        speed = math.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)

        self.steer = control.steer

        if speed == 0.0:
            control.brake = 1.0
            control.throttle = 0.0            

        else:
            if speed >= self.speed:
                control.brake = 0.0
                control.throttle = 0.0
            else:
                control.brake = 0.0
                control.throttle = 0.5

        #print("stering: ",self.steer, "    speed: ", self.speed)
        self.episode_actions.append( [control.steer, self.speed] )
        self.vehicle.apply_control(control)

    def calculate_lane_angle_error(self, right_lane_x,right_lane_y ):

        if self.lane_lines > 1:
            # Extraer las coordenadas x y y de los puntos
            x, y = right_lane_x, right_lane_y
            
            # Ajuste lineal a los puntos
            m, b = np.polyfit(x, y, 1)  # Esto devuelve la pendiente (m) y la intercepción (b) de la línea
            
            # Calcular el ángulo entre la línea del carril y el eje horizontal
            theta_horizontal = np.arctan(m)
            
            # Convertirlo a grados
            theta_horizontal_deg = np.degrees(theta_horizontal)
            
            # Ángulo entre la línea del carril y el eje vertical
            theta_vertical = 90 - theta_horizontal_deg

            angle_error = abs(55 - theta_vertical)
        else:
            angle_error = 10

        return angle_error

class RenderObject(object):
    def __init__(self):
        init_image = np.random.randint(0,255,(800,600,3),dtype='uint8')
        init_image = np.random.randint(0,255,(800,600,3),dtype='uint8')
        self.surface = pygame.surfarray.make_surface(init_image.swapaxes(0,1))
        self.surface2 = pygame.surfarray.make_surface(init_image.swapaxes(0,1))

# Render object to keep and pass the PyGame surface
class Metrics():
    def __init__(self):
        self.program_start_time = -100
        self.processing_time = 0
        self.last_fps = 0
        self.fps = 0
        self.start_time = 0

def get_prediction( img_array, deeplearning_model):
    with torch.no_grad():
        image_tensor = img_array.transpose(2,0,1).astype('float32')/255
        x_tensor = torch.from_numpy(image_tensor).to("cuda").unsqueeze(0)
        model_output = torch.softmax( deeplearning_model.forward(x_tensor), dim=1 ).cpu().numpy()

    return model_output

def lane_detection_overlay( image, left_mask, right_mask):

    global right_lane_y
    global right_lane_x

    res = np.copy(image)

    # We use only points with probability higher than 0.5 of being a lane
    res[left_mask > 0.6, :] = [255,0,0]
    res[right_mask > 0.6,:] = [255, 0, 0]

    left_y, left_x = np.where(left_mask > 0.5)
    right_y, right_x = np.where(right_mask > 0.5)
    
    right_lane_y = right_y
    right_lane_x = right_x

    #cv2.line(res, (0, 400), (1000, 400), [255, 0, 255], 1)    

    left_lane_coordinates = left_x[(left_y < 350)]
    right_lane_coordinates = right_x[(right_y < 350)]

    
    left_lane_coordinates_list = left_lane_coordinates.tolist()
    right_lane_coordinates_list = right_lane_coordinates.tolist()

    return res, left_lane_coordinates_list, right_lane_coordinates_list


#draw both lane center and image center
def draw_centers( img, VehicleQlearning, left_line, right_line ):

    if(not right_line):
        print("no right line")
        VehicleQlearning.lane_lines = 0

    elif (not left_line and right_line):
        VehicleQlearning.lane_lines = 1
        
        center = int((np.mean(right_line)) - 123)
        center_x = int(img.shape[1]/2)
    
        cv2.line(img, (center_x, 400), (center_x, 512), [0, 0, 255], 2)    
        cv2.line(img, (center_x-5, 304), (center_x+5, 304), [0, 0, 255], 1)
        cv2.line(img, (int(center), 400), (int(center), 512), [0, 255, 0], 1)


        VehicleQlearning.set_lane_center(center)
        VehicleQlearning.set_lane_center_error(center_x - center)        

    else:
        if (np.mean(right_line) - np.mean(left_line)  < 450):
            VehicleQlearning.lane_lines = 2            
            center = int((np.mean(left_line) + np.mean(right_line)) / 2)

            center_x = int(img.shape[1]/2)
        
            cv2.line(img, (center_x, 400), (center_x, 512), [0, 0, 255], 2)    
            cv2.line(img, (center_x-5, 304), (center_x+5, 304), [0, 0, 255], 1)
            cv2.line(img, (int(center), 400), (int(center), 512), [0, 255, 0], 1)


            VehicleQlearning.set_lane_center(center)
            VehicleQlearning.set_lane_center_error(center_x - center)
        else:
            print("to big lane distance")
            VehicleQlearning.lane_lines = 0


    thresholds = np.array([292,312,332,352,372,392,412,432,452,472,492,512,532,552,572,592,612,632,652,672,692,712]) 
    for i in thresholds:
        cv2.line(img, (i, 0), (i, 600), [0, 255, 255], 1)


def line_filter( img, dl_model, VehicleQlearning):
    
    resized_img = cv2.resize(img, (1024, 512) )

    resized_img = cv2.cvtColor(resized_img, cv2.COLOR_BGRA2RGB)

    back, left, right = get_prediction(resized_img,dl_model)[0]

    filtered_img, left_line, right_line = lane_detection_overlay(resized_img, left, right)

    draw_centers(filtered_img, VehicleQlearning, left_line, right_line)

    final_img = cv2.cvtColor(filtered_img, cv2.COLOR_RGB2BGRA)
    final_img = cv2.resize(final_img, (800, 600) )


    return final_img


def show_fps( img, metrics):
    image = cv2.putText(img, 'FPS: ' + str(metrics.last_fps) , (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 
                0.5, (255, 0, 0), 1, cv2.LINE_AA)
    
    return image
        
#callback for the car first person camera
def first_person_image_cb(image, obj, metrics, dl_model, VehicleQlearning):

    array = np.frombuffer(image.raw_data, dtype=np.dtype('uint8'))
    array = np.reshape(array, (image.height, image.width, 4))
    img = array[:, :, :3]

    if metrics.program_start_time == -100:
        metrics.program_start_time = time.time()

    processing_start_time = time.time()

    filter_img = line_filter(img, dl_model, VehicleQlearning)
    
    metrics.processing_time = time.time() - processing_start_time

    if metrics.fps == 0:
        metrics.start_time = time.time()

    metrics.fps = metrics.fps + 1

    if time.time() - metrics.start_time >= 1:
        metrics.last_fps = metrics.fps
        metrics.fps = 0

    bgr_img_with_fps = show_fps(filter_img,metrics)
    rgb_img = cv2.cvtColor(bgr_img_with_fps, cv2.COLOR_BGR2RGB)
    rgb_img = np.rot90(np.fliplr(rgb_img))

    obj.surface = pygame.surfarray.make_surface(rgb_img)



def position_cb( pos, metrics, vehicleQlearning):

    latitude = pos.latitude
    longitude = pos.longitude

    vehicleQlearning.latitude = latitude
    vehicleQlearning.longitude = longitude


def third_person_image_cb(image, obj, metrics, dl_model, VehicleQlearning):

    array = np.frombuffer(image.raw_data, dtype=np.dtype('uint8'))
    array = np.reshape(array, (image.height, image.width, 4))
    array = array[:, :, :3]
    array = array[:, :, ::-1]
    obj.surface2 = pygame.surfarray.make_surface(array.swapaxes(0,1))
    

#choose the vehicle initial location from a pool of locations
def choose_vehicle_location():
    """
    locations = [(carla.Location(x=-26.48, y=-249.39, z=0.2), 
                  carla.Rotation(pitch=-1.19, yaw=128, roll=0)), 
                   (carla.Location(x=-65.03, y=-199.5, z=0.2), 
                    carla.Rotation(pitch=-6.46, yaw=133.11, roll=0)),
                    (carla.Location(x=-65.380, y=-199.5546, z=0.2), 
                    carla.Rotation(pitch=-2.0072, yaw=132.0, roll=0)),
                    (carla.Location(x=-108.05, y=-158.886, z=0.3), 
                    carla.Rotation(pitch=-1.85553, yaw=142.7858, roll=0)),
                    (carla.Location(x=-157.8591, y=-125.4512, z=0.5), 
                    carla.Rotation(pitch=-4.850, yaw=158.7178, roll=0))   ]
    """
    locations = [(carla.Location(x=-26.48, y=-249.39, z=0.2), 
                  carla.Rotation(pitch=-1.19, yaw=128, roll=0)), 
                   (carla.Location(x=-65.03, y=-199.5, z=0.2), 
                    carla.Rotation(pitch=-6.46, yaw=133.11, roll=0)),
                    (carla.Location(x=-65.380, y=-199.5546, z=0.2), 
                    carla.Rotation(pitch=-2.0072, yaw=132.0, roll=0)) ]
    
    location, rotation = random.choice(locations)

    return location, rotation

#This funcion waitf for the first camera image to arrive, we use it to start each episode lane detection
def wait_for_detection(vehicleQlearning, gameDisplay, renderObject):

    vehicleQlearning.lane_lines = 100
    while vehicleQlearning.lane_lines == 100:
        world.tick()
        gameDisplay.blit(renderObject.surface, (0,0))
        gameDisplay.blit(renderObject.surface2, (800,0))
        pygame.display.flip()
        time.sleep(1.0)         
        

def wait_for_spawning(vehicleQlearning, gameDisplay, renderObject):

    start = time.time()
    while time.time() - start < 1:
        world.tick()
        gameDisplay.blit(renderObject.surface, (0,0))
        gameDisplay.blit(renderObject.surface2, (800,0))
        pygame.display.flip()
        vehicleQlearning.accelerate()
        
def save_data(csv_writer, episode,acum_reward ,vehicleQlearning):   
        
    learning_rate,discount_factor,exploration_rate = vehicleQlearning.get_qlearning_parameters()
    file_name = '/home/alumnos/camilo/Escritorio/qlearning_metrics/metrics_1.csv'
    #file_name = '/home/camilo/Escritorio/qlearning_metrics/metrics_1.csv'
    with open(file_name, 'a') as csv_file:
        csv_writer = csv.writer(csv_file)        
        csv_writer.writerow([ episode, learning_rate , discount_factor,exploration_rate, acum_reward])

def save_action(actions):   
    file_name = '/home/alumnos/camilo/Escritorio/qlearning_metrics/actions_1.csv'
    # file_name = '/home/camilo/Escritorio/qlearning_metrics/actions_1.csv'
    
    with open(file_name, 'a') as csv_file:
        csv_writer = csv.writer(csv_file)
        
        # Assuming 'episode' is defined somewhere outside of this function or you can pass it as an argument
        for action in actions:
            csv_writer.writerow([ action[0], action[1] ])

def show_data( episode,acum_reward ,vehicleQlearning):   
    learning_rate, discount_factor, exploration_rate = vehicleQlearning.get_qlearning_parameters()

    table = PrettyTable()
    table.field_names = ["Episode", "Learning Rate", "Discount Factor", "Exploration Rate", "Acumulated Reward"]
    table.add_row([episode, learning_rate, discount_factor, exploration_rate, acum_reward])
    print(table)
    print('random: ',vehicleQlearning.get_random_counter(),' table: ',vehicleQlearning.get_table_counter())
    vehicleQlearning.set_random_counter(0)
    vehicleQlearning.set_table_counter(0)


def visualize_lidar(lidar_data):
    """
    Visualiza los datos del LIDAR en un plot 2D y muestra puntos de objetos detectados.
    
    :param lidar_data: Datos recibidos del sensor LIDAR.
    """
    # Convertir datos LIDAR en un array numpy
    points = np.frombuffer(lidar_data.raw_data, dtype=np.dtype('f4'))
    points = np.reshape(points, (int(points.shape[0] / 3), 3))

    # Para una vista superior, consideraremos x como "adelante" y y como "lado".
    forward, side = points[:, 0], points[:, 1]
    
    # Filtrar solo puntos que estén delante del vehículo entre 0.5 y 8 metros
    mask = (1 <= forward) & (forward <= 8) & (side <= 0.3) & (side >= -0.3)
    forward_filtered = forward[mask]
    side_filtered = side[mask]

    if len(forward_filtered) > 50:
        print("Objetos detectados en las siguientes coordenadas:")


    plt.figure(figsize=(10, 10))

    # Dibujar puntos LIDAR
    plt.scatter(side_filtered, forward_filtered, s=1)

    # Configuración de la visualización
    plt.xlim(-8, 8)  # Ajustado para centrar en el vehículo
    plt.ylim(0, 8)  # Ajustado para mostrar solo puntos delante del coche
    plt.xlabel('Lado (m)')
    plt.ylabel('Adelante (m)')
    plt.title('Visualización LIDAR 2D')
    plt.grid(True)
    plt.axvline(0, color='grey', linewidth=1)
    plt.axhline(0.5, color='grey', linewidth=1)  # Línea inicial del rango
    plt.gca().set_aspect('equal', adjustable='box')

    # Mostrar visualización
    plt.show()


def lidar_callback(point_cloud, vehicleQlearning):
    # Convertir datos a un numpy array
    points = np.frombuffer(point_cloud.raw_data, dtype=np.dtype('f4'))
    
    # Revisamos que la longitud de los puntos sea divisible por 3
    if len(points) % 3 != 0:
        return

    # Cambiamos la forma del array para tener puntos en 3D (x, y, z)
    points = points.reshape(-1, 3)

    # Para una vista superior, consideraremos x como "adelante" y y como "lado".
    forward, side = points[:, 0], points[:, 1]

    # Filtrar solo puntos que estén delante del vehículo entre 1 y 8 metros y lateralmente dentro de +/- 0.3 metros
    mask = (1 <= forward) & (forward <= 8) & (side <= 0.3) & (side >= -0.3)
    forward_filtered = forward[mask]
    side_filtered = side[mask]
    
    # Si hay puntos en ese rango, entonces hay un objeto en frente
    if len(forward_filtered) > 50:
        vehicleQlearning.object_in_front = True
        print("object in front")
    else:
        vehicleQlearning.object_in_front = False



def spawn_vehicle(renderObject):
    #spawn vehicle
    blueprint_library = world.get_blueprint_library()
    vehicle_bp = blueprint_library.find('vehicle.tesla.model3')
    location,rotation = choose_vehicle_location()
    transform = carla.Transform(location, rotation)
    vehicle = world.spawn_actor(vehicle_bp, transform)

    actors = []
    actors.append(vehicle)

    #spawn camera first person view
    camera_bp = blueprint_library.find('sensor.camera.rgb')
    camera_bp.set_attribute('image_size_x', '800')
    camera_bp.set_attribute('image_size_y', '600')
    camera_bp.set_attribute('fov', '120')
    dashcam_location = carla.Location(x=1.9, y=0.0, z=2)
    dashcam_rotation = carla.Rotation(pitch=-20, yaw=0, roll=0)
    dashcam_transform = carla.Transform(dashcam_location, dashcam_rotation)
    dashcam = world.spawn_actor(camera_bp, dashcam_transform, attach_to=vehicle)
    actors.append(dashcam)


    dl_model = torch.load('/home/alumnos/camilo/2022-tfg-juancamilo-carmona/tfg/src/qlearning/model/fastai_torch_lane_detector_model.pth')
    #dl_model = torch.load('/home/camilo/2022-tfg-juancamilo-carmona/tfg/src/qlearning/model/fastai_torch_lane_detector_model.pth')

    dashcam.listen(lambda image: first_person_image_cb(image, renderObject, metrics, dl_model, vehicleQlearning))

    # Spawn camera third person view
    camera_bp_third_person = blueprint_library.find('sensor.camera.rgb')
    camera_bp_third_person.set_attribute('image_size_x', '800')
    camera_bp_third_person.set_attribute('image_size_y', '600')
    camera_bp_third_person.set_attribute('fov', '120')

    # Posición y rotación detrás y por encima del vehículo.
    third_person_location = carla.Location(x=-5, y=0.0, z=3)  # 'x=-5' posiciona la cámara detrás del vehículo
    third_person_rotation = carla.Rotation(pitch=-10, yaw=0, roll=0)  # 'pitch=-10' es un ligero ángulo hacia abajo
    third_person_transform = carla.Transform(third_person_location, third_person_rotation)

    third_person_cam = world.spawn_actor(camera_bp_third_person, third_person_transform, attach_to=vehicle)
    actors.append(third_person_cam)

    # Escuchar imágenes de la cámara en tercera persona
    third_person_cam.listen(lambda image: third_person_image_cb(image, renderObject, metrics, dl_model, vehicleQlearning))


    #spawn gnss sensor
    gnss_bp = blueprint_library.find('sensor.other.gnss')
    gnss = world.spawn_actor(gnss_bp, carla.Transform(carla.Location(x=1.0, z=2.8)), attach_to=vehicle)
    gnss.listen(lambda data: position_cb(data,metrics, vehicleQlearning))
    actors.append(gnss)

    collision_sensor = world.spawn_actor(blueprint_library.find('sensor.other.collision'),carla.Transform(), attach_to=vehicle)
    collision_sensor.listen(lambda event: collision_cb(event))
    actors.append(collision_sensor)

    #spawn lidar sensor
    lidar_bp = blueprint_library.find('sensor.lidar.ray_cast')
    lidar_bp.set_attribute('channels', '32')
    lidar_bp.set_attribute('points_per_second', '100000')
    lidar_bp.set_attribute('rotation_frequency', '10')
    lidar_bp.set_attribute('range', '100')
    lidar_location = carla.Location(x=1.5, y=0, z=1.2)
    lidar_rotation = carla.Rotation(pitch=0, yaw=0, roll=0)
    lidar_transform = carla.Transform(lidar_location, lidar_rotation)
    lidar = world.spawn_actor(lidar_bp, lidar_transform, attach_to=vehicle)
    lidar.listen(lambda data: lidar_callback(data, vehicleQlearning))
    # lidar.listen(lambda point_cloud: lidar_callback(point_cloud))
    actors.append(lidar)

    collision_sensor = world.spawn_actor(blueprint_library.find('sensor.other.collision'),carla.Transform(), attach_to=vehicle)
    collision_sensor.listen(lambda event: collision_cb(event))
    actors.append(collision_sensor)
    
    return vehicle, actors

def choose_random_obstacle_location():
    
    locations = [   (carla.Location(x=-108.05, y=-158.886, z=0.3), 
                    carla.Rotation(pitch=-1.85553, yaw=142.7858, roll=0)),
                    (carla.Location(x=-157.8591, y=-125.4512, z=0.5), 
                    carla.Rotation(pitch=-4.850, yaw=158.7178, roll=0))   ]


    location, rotation = random.choice(locations) 
    return location, rotation

def wait_for_action(gameDisplay, renderObject):

    start_time = time.time()
    while time.time() - start_time < 0.3:
        world.tick()
        gameDisplay.blit(renderObject.surface, (0,0))
        gameDisplay.blit(renderObject.surface2, (800,0))
        pygame.display.flip()

def collision_cb(event):
    global car_crashed
    car_crashed = True

def destroy_actor(actor):
    try:
        actor.destroy()
    except:
        pass

def destroy_actor_after_delay(actor, delay):
    threading.Timer(delay, lambda: destroy_actor(actor)).start()

    
pygame.init()
image_surface = None
size = 1600, 600
gameDisplay = pygame.display.set_mode(size)
pygame.display.set_caption("qlearning and DL")
pygame.display.flip()


right_lane_y = []
right_lane_x = []

file_name = '/home/alumnos/camilo/Escritorio/qlearning_metrics/metrics_1.csv'
#file_name = '/home/camilo/Escritorio/qlearning_metrics/metrics_1.csv'
with open(file_name, 'w') as csv_file:
    csv_writer = csv.writer(csv_file)      
    csv_writer.writerow(['num episodio','learning constant','discount factor','exploration factor','acumulated reward'])

# Connect to the client and retrieve the world object
client = carla.Client('localhost', 2016)
world = client.get_world()

try:
    world = client.load_world('Town04')
except RuntimeError:
    print("No se pudo cargar Town04. Comprueba si esta ciudad está?? disponible en tu versió??n de CARLA.")

# Set the weather to be sunny and without wind, wind affects steering
weather = carla.WeatherParameters(
    cloudiness=40.0,
    precipitation=0.0,
    sun_altitude_angle=90.0,
    wind_intensity=0.0
)
world.set_weather(weather)

# Set up the simulator in synchronous mode
settings = world.get_settings()
settings.synchronous_mode = True
settings.fixed_delta_seconds = 0.1
world.apply_settings(settings)
spectator = world.get_spectator()

metrics = Metrics()

renderObject = RenderObject()

vehicle, actors = spawn_vehicle(renderObject)
vehicleQlearning = QLearningVehicleControl(vehicle)

num_episodes = 6000
finished_laps_counter = 0

car_crashed = False
start = True
obstacle_control = False
while start:

    for episode in range(num_episodes):

        wait_for_detection(vehicleQlearning, gameDisplay, renderObject)     
        wait_for_spawning(vehicleQlearning, gameDisplay, renderObject) 
        current_state, current_object_in_front = vehicleQlearning.get_state(vehicleQlearning.get_lane_center())

        world.tick()        

        done = False
        acum_reward = 0
        while not done:

            if np.random.uniform(0, 1) < 0.5:

                blueprint_library = world.get_blueprint_library()
                vehicle_bp = blueprint_library.find('vehicle.tesla.model3')
                location, rotation = choose_random_obstacle_location()
                transform = carla.Transform(location, rotation)
                obstacle = world.spawn_actor(vehicle_bp, transform)
                destroy_actor_after_delay(obstacle, 120)
                actors.append(obstacle)

            gameDisplay.blit(renderObject.surface, (0,0))
            gameDisplay.blit(renderObject.surface2, (800,0))
            pygame.display.flip()
                
            action = vehicleQlearning.choose_action(current_state)
            speed = vehicleQlearning.choose_speed(current_state)
            vehicleQlearning.perform_action(vehicleQlearning.ACTIONS[action], vehicleQlearning.SPEED[speed])
            world.tick()
            lane_center= vehicleQlearning.get_lane_center()
            next_state, next_object_in_front = vehicleQlearning.get_state(lane_center)

            lane_center_error = vehicleQlearning.get_lane_center_error()
            angle_error = vehicleQlearning.calculate_lane_angle_error( right_lane_x, right_lane_y )
            reward = vehicleQlearning.reward_function(lane_center_error, angle_error, car_crashed)
            acum_reward = acum_reward + reward

            if vehicleQlearning.latitude < 0.0001358:
                done = True
                reward = reward + 50
                finished_laps_counter += 1
                if finished_laps_counter > 25:
                    print("algorithm converged! finishing training")
                    q_table = vehicleQlearning.q_table
                    print(vehicleQlearning.q_table)
                    with open('q_table.pkl', 'wb') as f:
                        pickle.dump(q_table, f)

            else:
                finished_laps_counter = 0

            if vehicleQlearning.lane_lines < 1:
                done = True

            if car_crashed:
                done = True
                car_crashed = False
            
                        
            vehicleQlearning.update_q_table(current_state, action, speed , reward, next_state, current_object_in_front, next_object_in_front)

            current_state = next_state
            current_object_in_front = next_object_in_front


            world.tick()

            

        for actor in actors:
            actor.destroy()

        del actors

        actors = []               

        q_table = vehicleQlearning.q_table
        with open('q_table.pkl', 'wb') as f:
            pickle.dump(q_table, f)

        save_data(csv_writer,episode,acum_reward ,vehicleQlearning)
        show_data(episode,acum_reward ,vehicleQlearning)
        save_action(vehicleQlearning.episode_actions)

        
        vehicle, actors = spawn_vehicle(renderObject)
        vehicleQlearning.set_vehicle(vehicle)

        if vehicleQlearning.exploration_rate > 0.05:
            vehicleQlearning.increment_exploration_counter()
        else:
            vehicleQlearning.set_exploration_rate(0.001)


            

    start = False
    pygame.quit()
