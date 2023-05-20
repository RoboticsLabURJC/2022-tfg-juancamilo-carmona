#!/usr/bin/python3
#
# RoboticsLabURJC - Roberto Calvo
# Model trained using kaggle dataset -> https://www.kaggle.com/datasets/thomasfermi/lane-detection-for-carla-driving-simulator/download?datasetVersionNumber=1
#

import torch
import torchvision
import cv2
import matplotlib.pyplot as plt
import numpy as np

# This model ouputs 3 probability mask. They are same size of the original image
# back -> Mask that represents no line
# left -> Mask that represents left line of the lane
# right -> Mask that represents right line of the lane

model = torch.load('/home/camilo/2022-tfg-juancamilo-carmona/tfg/src/line_follower/model/fastai_torch_lane_detector_model.pth')
model.eval()

def get_prediction(model, img_array):
    with torch.no_grad():
        image_tensor = img_array.transpose(2,0,1).astype('float32')/255
        x_tensor = torch.from_numpy(image_tensor).to("cuda").unsqueeze(0)
        model_output = torch.softmax( model.forward(x_tensor), dim=1 ).cpu().numpy()
    return model_output

def lane_detection_overlay(image, left_mask, right_mask):
    res = np.copy(image)
    # We show only points with probability higher than 0.5
    res[left_mask > 0.5, :] = [255,0,0]
    res[right_mask > 0.5,:] = [0, 0, 255]
    return res

img = cv2.imread('/home/camilo/2022-tfg-juancamilo-carmona/tfg/src/line_follower/model/test.png')
back, left, right = get_prediction(model,img)[0]

plt.imshow(lane_detection_overlay(img, left, right))
plt.show()
#print("llega al final")

# Next line compute 100 iterations to compute the mean time of inference (notebook)
# Results: GPU NVIDIA A100 ~ 90-100 FPS
# %timeit get_prediction(model,img)



