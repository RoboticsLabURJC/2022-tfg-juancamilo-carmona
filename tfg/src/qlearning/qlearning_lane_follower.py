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

class QLearningVehicleControl:
    def __init__(self,vehicle, learning_rate=0.5, discount_factor=0.95, exploration_rate=0.5, num_actions=7):
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.num_actions = num_actions
        self.q_table = np.zeros((num_actions, num_actions))
        self.vehicle = vehicle
        self.lane_lines = 100
        self.start = 0
        self.latitude = 100
        self.longitude = 100
        
        self.lane_center_error = 0  # Inicializamos la variable lane_center_error

        self.ACTIONS = [ 
            'forward',  # sigue recto
            'slight_left',  # giro suave a la izquierda
            'medium_left',  # giro medio a la izquierda
            'hard_left',  # giro duro a la izquierda
            'slight_right',  # giro suave a la derecha
            'medium_right',  # giro medio a la derecha
            'hard_right',  # giro duro a la derecha
        ]

    def set_new_actuators(self, vehicle):
        self.vehicle = vehicle


    def get_lane_center_error(self):
        return self.lane_center_error

    def set_lane_center_error(self, error):
        self.lane_center_error = error


    def choose_action(self, state):
        if np.random.uniform(0, 1) < self.exploration_rate:
            # Explora: elige una acció??n aleatoria
            action = np.random.randint(self.num_actions)
        else:
            # Explota: elige la acció??n con el mayor valor Q para este estado
            action = np.argmax(self.q_table[state])
        return action

    def update_q_table(self, current_state, action, reward, next_state):
        # Calcula el valor Q esperado para la pró??xima acció??n, si se elige la mejor
        future_max_q = np.max(self.q_table[next_state])

        # Calcula el nuevo valor Q para la acció??n tomada en el estado actual
        new_q = (1 - self.learning_rate) * self.q_table[current_state][action] + \
                self.learning_rate * (reward + self.discount_factor * future_max_q)

        # Actualiza la tabla Q
        self.q_table[current_state][action] = new_q

    def get_state(self, center_of_lane):
        # Asumimos que 'center_of_lane' es la posició??n en pí??xeles del centro del carril en la imagen.
        # Definimos los umbrales de pí??xeles para cada franja.
        thresholds = np.linspace(0, 800, num=5)  # Esto nos da 7 franjas de igual tamañ??o.

        # Verificamos en qué?? franja cae el centro del carril.
        for i in range(4):
            if thresholds[i] <= center_of_lane < thresholds[i + 1]:
                return i

        # Si el centro del carril está?? en el extremo derecho de la imagen, lo consideramos en la franja má??s a la derecha.
        return 6

    def reward_function(self, error):
        # Asumimos que 'error' es la diferencia en pí??xeles entre el centro del carril y el centro de la imagen.
        # Normalizamos el error dividié??ndolo por el ancho de la imagen (por ejemplo, 800 pí??xeles).
        normalized_error = abs(error) / 800

        # Usamos una funció??n exponencial negativa para convertir el error en una recompensa.
        reward = np.exp(-normalized_error)

        return reward
    
    def perform_action(self, action):
        # Define la acció??n que el vehí??culo debe tomar en funció??n de la acció??n especificada
        control = VehicleControl()
        if action == 'forward':
            control.throttle = 0.5
            control.steer = 0.0
        elif action == 'slight_left':
            control.throttle = 0.4
            control.steer = -0.01
        elif action == 'medium_left':
            control.throttle = 0.4
            control.steer = -0.05
        elif action == 'hard_left':
            control.throttle = 0.4
            control.steer = -0.1
        elif action == 'slight_right':
            control.throttle = 0.4
            control.steer = 0.01
        elif action == 'medium_right':
            control.throttle = 0.4
            control.steer = 0.05
        elif action == 'hard_right':
            control.throttle = 0.4
            control.steer = 0.1
        self.vehicle.apply_control(control)

    def train(self, num_episodes):
        for episode in range(num_episodes):
            # Reinicia el estado del entorno para el comienzo de cada episodio
            current_state = self.get_state(self.get_lane_center_error())

            done = False
            while not done:
                # Elige una acció??n
                action = self.choose_action(current_state)

                # Realiza la acció??n
                self.perform_action(self.ACTIONS[action])

                # Obté??n el nuevo estado y la recompensa
                lane_center_error = self.get_lane_center_error()
                next_state = self.get_state(lane_center_error)
                reward = self.reward_function(lane_center_error)

                # Aquí??, puede ser ú??til definir una condició??n de "finalizació??n" para el episodio, por ejemplo, si el vehí??culo
                # se sale de la carretera o llega a su destino.
                # En este ejemplo, simplemente definimos 'done' como False para que el episodio continú??e indefinidamente.
                # Tendrá??s que modificar esto para adaptarlo a tu caso de uso especí??fico.

                # Actualiza la tabla Q
                self.update_q_table(current_state, action, reward, next_state)

                # El nuevo estado se convierte en el estado actual para la pró??xima iteració??n
                current_state = next_state




# Render object to keep and pass the PyGame surface
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
    # We show only points with probability higher than 0.5
    res[left_mask > 0.5, :] = [255,0,0]
    res[right_mask > 0.5,:] = [255, 0, 0]

    left_lane_coordinates = np.where(left_mask > 0.5)[1]
    right_lane_coordinates = np.where(right_mask > 0.5)[1]

    # convert numpy arrays to lists of coordinates
    left_lane_coordinates_list = left_lane_coordinates.tolist()
    right_lane_coordinates_list = right_lane_coordinates.tolist()

    return res, left_lane_coordinates_list, right_lane_coordinates_list


        

def draw_centers( img, VehicleQlearning, left_line, right_line ):
    
    if  left_line and right_line :

        VehicleQlearning.lane_lines = 2            
        center = int((np.mean(left_line) + np.mean(right_line)) / 2)
        center_x = int(img.shape[1]/2)
    
        cv2.line(img, (center_x, 400), (center_x, 512), [0, 0, 255], 2)    
        cv2.line(img, (center_x-5, 304), (center_x+5, 304), [0, 0, 255], 1)
        cv2.line(img, (int(center), 400), (int(center), 512), [0, 255, 0], 1)

        VehicleQlearning.set_lane_center_error(center - center_x)

    else:
        center_x = int(img.shape[1]/2)
        VehicleQlearning.set_lane_center_error(center_x)
        VehicleQlearning.lane_lines = 0
        VehicleQlearning.set_lane_center_error(0)



    #print(VehicleQlearning.lane_lines)

    thresholds = np.linspace(0, 800, num=5)  # Esto nos da 7 franjas de igual tamañ??o.
    for i in thresholds:
        cv2.line(img, (int(i), 0), (int(i), 600), [0, 255, 255], 1)
        

def draw_lanes(img, left, right):
    color_img = np.zeros_like(img)
    
    points = np.hstack((left, right))
    
    cv2.fillPoly(color_img, np.int_(points), (0,200,255))
    final_img = cv2.addWeighted(img, 1, color_img, 0.7, 0)

    return final_img

def line_filter( img, dl_model, VehicleQlearning):
    
    resized_img = cv2.resize(img, (1024, 512) )

    resized_img = cv2.cvtColor(resized_img, cv2.COLOR_BGRA2RGB)

    back, left, right = get_prediction(resized_img,dl_model)[0]

    filtered_img, left_line, right_line = lane_detection_overlay(resized_img, left, right)

    #filtered_img = self.draw_lanes(filtered_img, left, right )

    draw_centers(filtered_img, VehicleQlearning, left_line, right_line)

    final_img = cv2.cvtColor(filtered_img, cv2.COLOR_RGB2BGRA)
    final_img = cv2.resize(final_img, (800, 600) )


    return final_img


def show_fps( img, metrics):
    #fps = int(self.clock.get_fps())
    image = cv2.putText(img, 'FPS: ' + str(metrics.last_fps) , (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 
                0.5, (255, 0, 0), 1, cv2.LINE_AA)
    
    return image
        
def first_person_image_cb(image, obj, metrics, dl_model, VehicleQlearning):

    if VehicleQlearning.start == False:
        VehicleQlearning.start = True

    # Convierte la imagen en una matriz np
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

    # Mostrar el FPS en la imagen en formato BGR
    bgr_img_with_fps = show_fps(filter_img,metrics)
    rgb_img = cv2.cvtColor(bgr_img_with_fps, cv2.COLOR_BGR2RGB)
    rgb_img = np.rot90(np.fliplr(rgb_img))

    obj.surface = pygame.surfarray.make_surface(rgb_img)

    #self.vehicle_controller.control_vehicle(self.lane_center, self.left_lane, self.right_lane)


def position_cb( pos, metrics, vehicleQlearning):

    latitude = pos.latitude
    longitude = pos.longitude

    vehicleQlearning.latitude = latitude
    vehicleQlearning.longitude = longitude

    #if pos.latitude < 0.0001358:
        #self.csv_file.close()
        #exit()

def third_person_image_cb(image, obj ):

    array = np.frombuffer(image.raw_data, dtype=np.dtype('uint8'))
    array = np.reshape(array, (image.height, image.width, 4))
    array = array[:, :, :3]
    array = array[:, :, ::-1]
    obj.surface2 = pygame.surfarray.make_surface(array.swapaxes(0,1))


def reset_simulation(actors):
    # Destruye los actores actuales
    for actor in actors:
        if actor is not None:
            actor.destroy()
    actors.clear()
    

def choose_vehicle_location():
    locations = [(carla.Location(x=-26.48, y=-249.39, z=0.2) , carla.Rotation(pitch=-1.19, yaw=131, roll=0)), (carla.Location(x=-47.41, y=-214.90, z=0.3) , carla.Rotation(pitch=-5.579, yaw=132.02, roll=0)),(carla.Location(x=-65.03, y=-198.54, z=0.3) , carla.Rotation(pitch=-6.46, yaw=133.11, roll=0)) ]
    
    # Selecciona una ubicació??n aleatoria de la lista
    location, rotation = random.choice(locations)
    return location, rotation

def wait_spawn(vehicleQlearning, world):
    vehicleQlearning.start = False
    while not vehicleQlearning.start :
        world.tick()



# Initialise the display
pygame.init()
image_surface = None
size = 800, 600
gameDisplay = pygame.display.set_mode(size)
pygame.display.set_caption("qlearning and DL")
pygame.display.flip()

# Connect to the client and retrieve the world object
client = carla.Client('localhost', 2015)
world = client.get_world()

# Cargamos la ciudad deseada
try:
    world = client.load_world('Town04')
except RuntimeError:
    print("No se pudo cargar Town04. Comprueba si esta ciudad está?? disponible en tu versió??n de CARLA.")

# Set up the simulator in synchronous mode
settings = world.get_settings()
settings.synchronous_mode = True # Enables synchronous mode
settings.fixed_delta_seconds = 0.05
world.apply_settings(settings)


# We will aslo set up the spectator so we can see what we do
spectator = world.get_spectator()

# Define la ubicació??n donde quieres spawnear el vehí??culo
spawn_points = world.get_map().get_spawn_points()

blueprint_library = world.get_blueprint_library()
#vehicle_bp = blueprint_library.find('vehicle.tesla.model3')

#Para un Ford Mustang
#vehicle_bp = blueprint_library.find('vehicle.ford.mustang')

# Para un Porsche 911
vehicle_bp = blueprint_library.find('vehicle.tesla.model3')

# Spawn the vehicle
# Creamos una Transform con la ubicació??n y orientació??n que queremos

location,rotation = choose_vehicle_location()

transform = carla.Transform(location, rotation)

actors = []
# Generamos el vehí??culo
vehicle = world.spawn_actor(vehicle_bp, transform)
actors.append(vehicle)

# Busca el blueprint de la cá??mara
camera_bp = blueprint_library.find('sensor.camera.rgb')

# Configura la cá??mara
camera_bp.set_attribute('image_size_x', '800')
camera_bp.set_attribute('image_size_y', '600')
camera_bp.set_attribute('fov', '110')

# Añ??ade la primera cá??mara (dashcam) al vehí??culo
dashcam_location = carla.Location(x=1.9, y=0.0, z=1.4)
dashcam_rotation = carla.Rotation(pitch=-15, yaw=0, roll=0)
dashcam_transform = carla.Transform(dashcam_location, dashcam_rotation)
dashcam = world.spawn_actor(camera_bp, dashcam_transform, attach_to=vehicle)
actors.append(dashcam)

# Añ??ade la segunda cá??mara (vista en tercera persona) al vehí??culo
"""
third_person_cam_location = carla.Location(x=-5.5, y=0.0, z=2.8)
third_person_cam_rotation = carla.Rotation(pitch=-20, yaw=0, roll=0)
third_person_cam_transform = carla.Transform(third_person_cam_location, third_person_cam_rotation)
third_person_cam = world.spawn_actor(camera_bp, third_person_cam_transform, attach_to=vehicle)
actors.append(third_person_cam)
"""


# Instantiate objects for rendering and vehicle control
renderObject = RenderObject()

#dl_model = torch.load('/home/alumnos/camilo/2022-tfg-juancamilo-carmona/tfg/src/qlearning/model/fastai_torch_lane_detector_model.pth')
dl_model = torch.load('/home/camilo/2022-tfg-juancamilo-carmona/tfg/src/qlearning/model/fastai_torch_lane_detector_model.pth')

# Asocia la funció??n callback con las cá??maras
metrics = Metrics()

vehicleQlearning = QLearningVehicleControl(vehicle)
dashcam.listen(lambda image: first_person_image_cb(image, renderObject, metrics, dl_model, vehicleQlearning))
#third_person_cam.listen(lambda image:third_person_image_cb(image, renderObject))

# Busca el blueprint del sensor GNSS
gnss_bp = blueprint_library.find('sensor.other.gnss')

# Añ??ade el sensor GNSS al vehí??culo
gnss = world.spawn_actor(gnss_bp, carla.Transform(carla.Location(x=1.0, z=2.8)), attach_to=vehicle)
gnss.listen(lambda data: position_cb(data,metrics, vehicleQlearning))
actors.append(gnss)

#gnss.listen(position_cb)

vehicle.set_autopilot(True)  # Activating autopilot
num_episodes = 1000000

start = True
while start:
    if vehicleQlearning.start == True:

        for episode in range(num_episodes):

            print(episode)
            q_table = vehicleQlearning.q_table
            # Guardar la tabla Q
            with open('q_table.pkl', 'wb') as f:
                pickle.dump(q_table, f)
              
            # Reinicia el estado del entorno para el comienzo de cada episodio
            current_state = vehicleQlearning.get_state(vehicleQlearning.get_lane_center_error())

            done = False
            while not done:

                world.tick()
                gameDisplay.blit(renderObject.surface, (0,0))
                gameDisplay.blit(renderObject.surface2, (800,0))
                pygame.display.flip()

                # Elige una acci????n
                action = vehicleQlearning.choose_action(current_state)

                # Realiza la acci????n
                vehicleQlearning.perform_action(vehicleQlearning.ACTIONS[action])

                # Obt????n el nuevo estado y la recompensa
                lane_center_error = vehicleQlearning.get_lane_center_error()
                next_state = vehicleQlearning.get_state(lane_center_error)
                reward = vehicleQlearning.reward_function(lane_center_error)

                # Aqu????, puede ser ????til definir una condici????n de "finalizaci????n" para el episodio, por ejemplo, si el veh????culo
                # se sale de la carretera o llega a su destino.
                # En este ejemplo, simplemente definimos 'done' como False para que el episodio contin????e indefinidamente.
                # Tendr????s que modificar esto para adaptarlo a tu caso de uso espec????fico.

                # Actualiza la tabla Q
                vehicleQlearning.update_q_table(current_state, action, reward, next_state)

                # El nuevo estado se convierte en el estado actual para la pr????xima iteraci????n
                current_state = next_state

                if vehicleQlearning.lane_lines < 2:
                    done = True

                if vehicleQlearning.latitude < 0.0001358:
                    print("Goal reached! finishing training")
                    q_table = vehicleQlearning.q_table

                    # Guardar la tabla Q
                    with open('q_table.pkl', 'wb') as f:
                        pickle.dump(q_table, f)
                    exit()


            #dashcam.stop()
            #third_person_cam.stop()
            
            #reset_simulation(actors)

            location,rotation = choose_vehicle_location()

            transform = carla.Transform(location, rotation)
            vehicleQlearning.vehicle.set_transform(transform)

            #vehicle = world.spawn_actor(vehicle_bp, transform)
            #vehicleQlearning.set_new_actuators(vehicle)
            #actors.append(vehicle)

            #dashcam = world.spawn_actor(camera_bp, dashcam_transform, attach_to=vehicle)
            #actors.append(dashcam)

            #third_person_cam = world.spawn_actor(camera_bp, third_person_cam_transform, attach_to=vehicle)
            #actors.append(third_person_cam)

            #gnss = world.spawn_actor(gnss_bp, carla.Transform(carla.Location(x=1.0, z=2.8)), attach_to=vehicle)
            #actors.append(gnss)

            # Reasocia la funci????n callback con las c????maras
            #dashcam.listen(lambda image: first_person_image_cb(image, renderObject, metrics, dl_model, vehicleQlearning))
            #third_person_cam.listen(lambda image:third_person_image_cb(image, renderObject))
            wait_spawn(vehicleQlearning, world)

        start = False
        pygame.quit()

    world.tick()