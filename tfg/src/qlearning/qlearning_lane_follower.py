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

class QLearningVehicleControl:
    def __init__(self,vehicle, num_actions=13, num_states=9):
        self.learning_rate = 0.5
        self.discount_factor = 0.95
        self.exploration_rate = 0.95
        self.num_actions = num_actions
        self.exploration_rate_counter = 0
        self.q_table = np.zeros((num_states, num_actions))
        self.vehicle = vehicle
        self.lane_lines = 100
        self.start = True
        self.latitude = 100
        self.longitude = 100
        self.random_counter = 0
        self.table_counter = 0
        
        self.lane_center_error = 0 
        self.lane_center = 0  

        self.ACTIONS = [ 
            'forward',  
            'left_1',  
            'left_2',
            'left_3',  
            'left_4',  
            'left_5', 
            'left_6',  
            'right_1',  
            'right_2',
            'right_3',  
            'right_4',  
            'right_5', 
            'right_6', 
        ]

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
            action = np.random.randint(self.num_actions)
            self.random_counter += 1
        else:
            action = np.argmax(self.q_table[state])
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

    #function to update the qtale
    def update_q_table(self, current_state, action, reward, next_state):
        future_max_q = np.max(self.q_table[next_state])

        # new q table entry value
        new_q = (1 - self.learning_rate) * self.q_table[current_state][action] + \
                self.learning_rate * (reward + self.discount_factor * future_max_q)

        self.q_table[current_state][action] = new_q


        if self.exploration_rate_counter > 5:            
            #self.exploration_rate = self.exploration_rate - 0.0015
            self.exploration_rate = self.exploration_rate - 0.005
            self.exploration_rate_counter = 0
        
        #if self.exploration_rate < 0.01:
            #print("exploration rate is lower than 0.1 finishing training")
            #exit()

    def increment_exploration_counter(self):
        self.exploration_rate_counter += 1

    def set_exploration_rate(self, exploration_rate):
        self.exploration_rate =  exploration_rate

    def get_state(self, center_of_lane):

        #threshold for the lines that define the stastes
        thresholds = np.array([0,412,462,487,500,524,537,562,612,1025]) 
        for i in range( len(thresholds) - 1 ):
            if thresholds[i] <= center_of_lane < thresholds[i + 1]:
                return i

        return int(len(thresholds) / 2)

    #we use an exponencial function to calculate the reward
    def reward_function(self, error):

        normalized_error = abs(error) / 1024
        reward = np.exp(-normalized_error)

        # if we are not detecting both lane lines reward gets a big penalization
        if self.lane_lines < 1:
            reward = reward - 1000

        return reward
    
    def perform_action(self, action):

        control = VehicleControl()
        if action == 'forward':
            control.steer = 0.0

        elif action == 'left_1':
            control.steer = -0.01

        elif action == 'left_2':
            control.steer = -0.05

        elif action == 'left_3':
            control.steer = -0.1

        elif action == 'left_4':
            control.steer = -0.15

        elif action == 'left_5':
            control.steer = -0.2

        elif action == 'left_6':
            control.steer = -0.3

        elif action == 'right_1':
            control.steer = 0.01

        elif action == 'right_2':
            control.steer = 0.05

        elif action == 'right_3':
            control.steer = 0.1

        elif action == 'right_4':
            control.steer = 0.15

        elif action == 'right_5':
            control.steer = 0.2

        elif action == 'right_6':
            control.steer = 0.3
        

        
        #we try to mantain the same speed all the time
        velocity = self.vehicle.get_velocity()
        speed = math.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)

        if speed >= 4.0:
            control.throttle = 0.0
        else:
            control.throttle = 1.0

        self.vehicle.apply_control(control)


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
    res = np.copy(image)

    # We use only points with probability higher than 0.5 of being a lane
    res[left_mask > 0.5, :] = [255,0,0]
    res[right_mask > 0.5,:] = [255, 0, 0]

    left_y, left_x = np.where(left_mask > 0.5)
    right_y, right_x = np.where(right_mask > 0.5)

    #cv2.line(res, (0, 400), (1000, 400), [255, 0, 255], 1)    

    left_lane_coordinates = left_x[(left_y < 400)]
    right_lane_coordinates = right_x[(right_y < 400)]

    #left_lane_coordinates = left_x[(left_y < 350)]
    #right_lane_coordinates = right_x[(right_y < 350)]
    
    left_lane_coordinates_list = left_lane_coordinates.tolist()
    right_lane_coordinates_list = right_lane_coordinates.tolist()

    return res, left_lane_coordinates_list, right_lane_coordinates_list


#draw both lane center and image center
def draw_centers( img, VehicleQlearning, left_line, right_line ):
    if ((not left_line and not right_line) or (left_line and not right_line)):
        VehicleQlearning.lane_lines = 0

    elif (not left_line and right_line):
        if (left_line or right_line):
            VehicleQlearning.lane_lines = 1
            if left_line:
                center = int((np.mean(left_line)) + 125)
            else:
                center = int((np.mean(right_line)) - 125)


            center_x = int(img.shape[1]/2)
        
            cv2.line(img, (center_x, 400), (center_x, 512), [0, 0, 255], 2)    
            cv2.line(img, (center_x-5, 304), (center_x+5, 304), [0, 0, 255], 1)
            cv2.line(img, (int(center), 400), (int(center), 512), [0, 255, 0], 1)


            VehicleQlearning.set_lane_center(center)
            VehicleQlearning.set_lane_center_error(center_x - center)        

    else:
        if (np.mean(right_line) - np.mean(left_line)  < 400):
            VehicleQlearning.lane_lines = 2            
            center = int((np.mean(left_line) + np.mean(right_line)) / 2)

            center_x = int(img.shape[1]/2)
        
            cv2.line(img, (center_x, 400), (center_x, 512), [0, 0, 255], 2)    
            cv2.line(img, (center_x-5, 304), (center_x+5, 304), [0, 0, 255], 1)
            cv2.line(img, (int(center), 400), (int(center), 512), [0, 255, 0], 1)


            VehicleQlearning.set_lane_center(center)
            VehicleQlearning.set_lane_center_error(center_x - center)
        else:
            VehicleQlearning.lane_lines = 0


    thresholds = np.array([412,462,487,500,524,537,562,612]) 
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


def third_person_image_cb(image, obj ):

    array = np.frombuffer(image.raw_data, dtype=np.dtype('uint8'))
    array = np.reshape(array, (image.height, image.width, 4))
    array = array[:, :, :3]
    array = array[:, :, ::-1]
    obj.surface2 = pygame.surfarray.make_surface(array.swapaxes(0,1))
    

#choose the vehicle initial location from a pool of locations
def choose_vehicle_location():
    locations = [(carla.Location(x=-26.48, y=-249.39, z=0.5), 
                  carla.Rotation(pitch=-1.19, yaw=131, roll=0)), 
                   (carla.Location(x=-65.03, y=-199.5, z=0.5), 
                    carla.Rotation(pitch=-6.46, yaw=133.11, roll=0)),
                    (carla.Location(x=-65.380, y=-199.5546, z=0.5), 
                    carla.Rotation(pitch=-2.0072, yaw=127.517, roll=0)),
                    (carla.Location(x=-108.05, y=-158.886, z=0.3), 
                    carla.Rotation(pitch=-1.85553, yaw=142.7858, roll=0)),
                    (carla.Location(x=-157.8591, y=-125.4512, z=0.5), 
                    carla.Rotation(pitch=-4.850, yaw=160.7178, roll=0))   ]

    
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
        

def save_data(csv_writer, episode,acum_reward ,vehicleQlearning):   
        
    learning_rate,discount_factor,exploration_rate = vehicleQlearning.get_qlearning_parameters()
    #file_name = '/home/alumnos/camilo/Escritorio/qlearning_metrics/metrics_1.csv'
    file_name = '/home/camilo/Escritorio/qlearning_metrics/metrics_1.csv'
    with open(file_name, 'a') as csv_file:
        csv_writer = csv.writer(csv_file)        
        csv_writer.writerow([ episode, learning_rate , discount_factor,exploration_rate, acum_reward])


def show_data( episode,acum_reward ,vehicleQlearning):   
    learning_rate, discount_factor, exploration_rate = vehicleQlearning.get_qlearning_parameters()

    table = PrettyTable()
    table.field_names = ["Episode", "Learning Rate", "Discount Factor", "Exploration Rate", "Acumulated Reward"]
    table.add_row([episode, learning_rate, discount_factor, exploration_rate, acum_reward])
    print(table)
    print('random: ',vehicleQlearning.get_random_counter(),' table: ',vehicleQlearning.get_table_counter())
    vehicleQlearning.set_random_counter(0)
    vehicleQlearning.set_table_counter(0)


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


    #dl_model = torch.load('/home/alumnos/camilo/2022-tfg-juancamilo-carmona/tfg/src/qlearning/model/fastai_torch_lane_detector_model.pth')
    dl_model = torch.load('/home/camilo/2022-tfg-juancamilo-carmona/tfg/src/qlearning/model/fastai_torch_lane_detector_model.pth')

    dashcam.listen(lambda image: first_person_image_cb(image, renderObject, metrics, dl_model, vehicleQlearning))

    #spawn gnss sensor
    gnss_bp = blueprint_library.find('sensor.other.gnss')
    gnss = world.spawn_actor(gnss_bp, carla.Transform(carla.Location(x=1.0, z=2.8)), attach_to=vehicle)
    gnss.listen(lambda data: position_cb(data,metrics, vehicleQlearning))
    actors.append(gnss)

    return vehicle, actors

def tick_world(vehicleQlearning, world):
    if vehicleQlearning.lane_lines > 0:
        world.tick()


pygame.init()
image_surface = None
size = 800, 600
gameDisplay = pygame.display.set_mode(size)
pygame.display.set_caption("qlearning and DL")
pygame.display.flip()

#file_name = '/home/alumnos/camilo/Escritorio/qlearning_metrics/metrics_1.csv'
file_name = '/home/camilo/Escritorio/qlearning_metrics/metrics_1.csv'
with open(file_name, 'w') as csv_file:
    csv_writer = csv.writer(csv_file)      
    csv_writer.writerow(['num episodio','learning constant','discount factor','exploration factor','acumulated reward'])

# Connect to the client and retrieve the world object
client = carla.Client('localhost', 2015)
world = client.get_world()

try:
    world = client.load_world('Town04')
except RuntimeError:
    print("No se pudo cargar Town04. Comprueba si esta ciudad está?? disponible en tu versió??n de CARLA.")

# Set up the simulator in synchronous mode
settings = world.get_settings()
settings.synchronous_mode = True
settings.fixed_delta_seconds = 0.05
world.apply_settings(settings)
spectator = world.get_spectator()

metrics = Metrics()

renderObject = RenderObject()

vehicle, actors = spawn_vehicle(renderObject)
vehicleQlearning = QLearningVehicleControl(vehicle)

num_episodes = 6000
finished_laps_counter = 0

start = True
while start:

    for episode in range(num_episodes):

        wait_for_detection(vehicleQlearning, gameDisplay, renderObject)
        world.tick()        
        
        current_state = vehicleQlearning.get_state(vehicleQlearning.get_lane_center())

        done = False
        acum_reward = 0
        while not done:
            gameDisplay.blit(renderObject.surface, (0,0))
            gameDisplay.blit(renderObject.surface2, (800,0))
            pygame.display.flip()

            if vehicleQlearning.lane_lines < 1:
                done = True
                
            action = vehicleQlearning.choose_action(current_state)
            vehicleQlearning.perform_action(vehicleQlearning.ACTIONS[action])

            lane_center= vehicleQlearning.get_lane_center()
            next_state = vehicleQlearning.get_state(lane_center)

            lane_center_error = vehicleQlearning.get_lane_center_error()
            reward = vehicleQlearning.reward_function(lane_center_error)
            acum_reward = acum_reward + reward

            current_state = next_state

            if vehicleQlearning.latitude < 0.0001358:
                reward = reward + 500
                finished_laps_counter += 1
                if finished_laps_counter > 5:
                    print("algorithm converged! finishing training")
                    q_table = vehicleQlearning.q_table

                    with open('q_table.pkl', 'wb') as f:
                        pickle.dump(q_table, f)

            else:
                finished_laps_counter = 0

            vehicleQlearning.update_q_table(current_state, action, reward, next_state)

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

        
        vehicle, actors = spawn_vehicle(renderObject)
        vehicleQlearning.set_vehicle(vehicle)

        if vehicleQlearning.exploration_rate > 0.05:
            vehicleQlearning.increment_exploration_counter()
        else:
            vehicleQlearning.set_exploration_rate(0.001)


            

    start = False
    pygame.quit()