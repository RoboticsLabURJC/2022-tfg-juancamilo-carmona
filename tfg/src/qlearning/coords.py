import carla
import random
import pygame
import numpy as np
import cv2
import time

client = carla.Client('localhost', 2000)
world = client.get_world()

# Cargamos la ciudad deseada
try:
    world = client.load_world('Town04')
except RuntimeError:
    print("No se pudo cargar Town04. Comprueba si esta ciudad está disponible en tu versión de CARLA.")


# Set up the spectator so we can see what we do
spectator = world.get_spectator()

while True:

    # Obtenemos la transformación de la cámara espectadora
    spectator_transform = spectator.get_transform()

    # Obtenemos la ubicación y rotación de la cámara espectadora
    spectator_location = spectator_transform.location
    spectator_rotation = spectator_transform.rotation

    # Imprimimos la ubicación y rotación de la cámara espectadora
    print(f"Ubicación de la cámara: {spectator_location}")
    print(f"Rotación de la cámara: {spectator_rotation}")

    time.sleep(0.5)