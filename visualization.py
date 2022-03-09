# import necessary libraries
from matplotlib import axes
import numpy as np
import scipy
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import uuid
import tensorflow as tf
import cv2
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

"""
Visualization for Classification
Parameters:
  - image: 
  - image_label: The anomaly or normal image label
  - y_pred: predicted image from the network
  - network: the network variable
"""
def viz_classification(image, image_label, y_pred, network='resnet'):
  folder_name = 'resnet_code' if network == 'resnet' else 'resnet_code'
  y_pred_test = pickle.load(open("models/" + folder_name + "/y_pred_test.pkl", "rb"))
  y_pred_classes = np.zeros(len(y_pred_test))
  y_pred_classes[np.array(y_pred_test).flatten() > 0.5] = 1
  
  plt.scatter(np.arange(0,len(y_pred_test[y_pred_classes==0])), y_pred_test[y_pred_classes==0], color='green', label='Normal Images')
  plt.scatter(np.arange(0,len(y_pred_test[y_pred_classes==1])), y_pred_test[y_pred_classes==1], color='orange', label='Anomaly Images')
  plt.legend()
  plt.scatter([250], y_pred, facecolors=None, edgecolors=['red' if image_label == "Anomaly" else 'blue'], s=80, lw=10)
  
  filename = 'images/temp/' + str(uuid.uuid4()) + ".jpg"
  print("Filename: ", filename)
  plt.savefig(filename)
  plt.close()
  
  return filename

"""
Visualization for Anomaly Detection
Parameters:
  - in_image: Input Image
  - out_image: Output Image
  - SSIMLoss: SSIMLoss
"""
def viz_vae_ssim(in_image, out_image, SSIMLoss):
  ssim_test_noaug = pickle.load(open("models/vae_code/ssim_test_noaug.pkl", "rb"))
  s1 = np.array([s_.numpy() for s_ in ssim_test_noaug[0]])
  s2 = np.array([s_.numpy() for s_ in ssim_test_noaug[1]])
  plt.hist(s2, color='red')
  plt.hist(s1, color='blue')
  
  ssimloss = SSIMLoss((in_image.reshape(-1,56,56,1).astype(np.float32)-127.5)/127.5, out_image.reshape(-1,56,56,1).astype(np.float32))
  plt.vlines(x=ssimloss, ymax=80, ymin=0.0, linestyles='dashed', colors='green', lw=5)
  
  filename = 'images/temp/' + str(uuid.uuid4()) + ".jpg"
  plt.savefig(filename)
  
  plt.close()
  
  image_label = 'Anomaly' if ssimloss > 0.55 else 'Normal'
  
  return filename, image_label
