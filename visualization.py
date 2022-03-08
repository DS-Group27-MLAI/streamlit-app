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

def viz_classification(image, image_label, y_pred, network='resnet'):
  folder_name = 'resnet_code' if network == 'resnet' else 'densenet_code'
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

def KL_DIVERGENCE(X_pca, pca, score1, score2, s1=1, s2=0, size_of_golden_set=3):
  Q_j_i_array = []
  for i in range(size_of_golden_set):
    for j in range(1):
      Q_j_i = np.exp(-1 * np.linalg.norm(score1[i] - score2[j], 2)/(2*np.std(score1[i])**2))
      Q_j_i_array.append(Q_j_i)
  Q = np.array(Q_j_i_array) / np.sum(Q_j_i_array)

  P_j_i_array = []
  X_pca_score = pca.transform(score2.reshape(-1,20))
  for i in range(size_of_golden_set):
    for j in range(1):
      P_j_i = (1+(X_pca[i]-X_pca_score[j])**2)**-1
      P_j_i_array.append(P_j_i)
  P = np.array(P_j_i_array) / np.sum(P_j_i_array)

  return np.sum(scipy.special.rel_entr(P, Q))

def generate_results_data_of_images(anomaly_scores_test):
  normal_index = [18, 60, 198]
  anomaly_index = [387, 90, 378]

  normal_test = np.array(anomaly_scores_test[0]).reshape(-1,20)
  anomaly_test = np.array(anomaly_scores_test[1]).reshape(-1,20)

  normal_test_gt = np.array(anomaly_scores_test[0]).reshape(-1,20)[normal_index]
  anomaly_test_gt = np.array(anomaly_scores_test[1]).reshape(-1,20)[anomaly_index]

  results_normal = pd.DataFrame()
  results_anomaly = pd.DataFrame()
  diff_normal = []
  diff_anomaly = []
  diff_normal_anomaly = []
  diff_anomaly_normal = []
  pca1 = PCA(n_components=1)
  X_pca1 = pca1.fit_transform(normal_test)
  pca2 = PCA(n_components=1)
  X_pca2 = pca2.fit_transform(anomaly_test)
  for i in range(0,len(normal_test)):
      diff_normal.append(KL_DIVERGENCE(X_pca1, pca1, normal_test_gt, normal_test[i:i+2], 0, 0, 3))
      diff_anomaly_normal.append(KL_DIVERGENCE(X_pca2, pca1, anomaly_test_gt, normal_test[i:i+2], 1, 0, 3))
  for i in range(0,len(anomaly_test)):
      diff_anomaly.append(KL_DIVERGENCE(X_pca2, pca2, anomaly_test_gt, anomaly_test[i:i+2], 1, 1, 3))
      diff_normal_anomaly.append(KL_DIVERGENCE(X_pca1, pca2, normal_test_gt, anomaly_test[i:i+2], 0, 1, 3))
  results_normal['normal'] = diff_normal
  results_anomaly['anomaly'] = diff_anomaly
  results_anomaly['normal_anomaly'] = diff_normal_anomaly
  results_normal['anomaly_normal'] = diff_anomaly_normal
  
  return results_normal, results_anomaly

def mapping_y_pred_labels_gmm(gmm, y_pred_labels, results_data):
    cluster_x = gmm.predict(results_data[results_data[:,0].argmax(), :].reshape(-1,2))
    cluster_y = gmm.predict(results_data[results_data[:,1].argmax(), :].reshape(-1,2))
    
    y_pred_new_labels = np.zeros_like(y_pred_labels)
    y_pred_new_labels[y_pred_labels==cluster_x] = 0
    y_pred_new_labels[y_pred_labels==cluster_y] = 1
    
    return y_pred_new_labels

def viz_cvae_gmm(in_image, out_data):
  gmm = pickle.load(open("models/vae_code/GaussianMixtureModel.pkl", "rb"))
  anomaly_scores_test = pickle.load(open("models/vae_code/anomaly_scores_test.pkl", "rb"))
  results_normal, results_anomaly = generate_results_data_of_images(anomaly_scores_test)
  
  results_data = np.concatenate([results_normal.values, results_anomaly.values], axis=0)
  print(results_normal.head(), results_anomaly.head())
  gmm.fit(results_data)
  y_pred_labels = gmm.predict(results_data)
  plt.scatter(results_data[y_pred_labels==1, 0], results_data[y_pred_labels==1, 1], color='green', label='Cluster 1')
  plt.scatter(results_data[y_pred_labels==0, 0], results_data[y_pred_labels==0, 1], color='orange', label='Cluster 0')
  plt.xlabel("Scores of Normal Compared \nwith Normal Image and \nAnomaly Compared with Anomaly Image")
  plt.ylabel("Scores of Normal Compared \nwith Anomaly Image and \nAnomaly Compared with Normal Image")
  
  # latent space and predicted image
  pred, z = out_data
  
  normal_index = [18, 60, 198]
  anomaly_index = [387, 90, 378]

  normal_test = np.array(anomaly_scores_test[0]).reshape(-1,20)
  anomaly_test = np.array(anomaly_scores_test[1]).reshape(-1,20)

  normal_test_gt = np.array(anomaly_scores_test[0]).reshape(-1,20)[normal_index]
  anomaly_test_gt = np.array(anomaly_scores_test[1]).reshape(-1,20)[anomaly_index]
  
  randomised_anomaly_set = []
  
  pca1 = PCA(n_components=1)
  X_pca1 = pca1.fit_transform(anomaly_test)
  pca2 = PCA(n_components=1)
  X_pca2 = pca2.fit_transform(normal_test)
  
  # Calculation using KL Divergence
  n3 = KL_DIVERGENCE(X_pca2, pca1, normal_test_gt, np.array([z]), 0, 0, size_of_golden_set=min(len(normal_test_gt), len(anomaly_test_gt)))
  n4 = KL_DIVERGENCE(X_pca2, pca1, normal_test_gt, np.array([z]), 0, 1, size_of_golden_set=min(len(normal_test_gt), len(anomaly_test_gt)))
  a3 = KL_DIVERGENCE(X_pca1, pca1, anomaly_test_gt, np.array([z]), 1, 0, size_of_golden_set=min(len(normal_test_gt), len(anomaly_test_gt)))
  a4 = KL_DIVERGENCE(X_pca1, pca1, anomaly_test_gt, np.array([z]), 1, 1, size_of_golden_set=min(len(normal_test_gt), len(anomaly_test_gt)))

  randomised_anomaly_set.append((n3,a3))
  randomised_anomaly_set.append((a4,n4))
  
  y_pred_single_label = gmm.predict(np.array(randomised_anomaly_set))
  print("Cluster: ", y_pred_single_label)
  y_pred_single_label = mapping_y_pred_labels_gmm(gmm, y_pred_single_label, results_data)
  print("Labels: ", y_pred_single_label)

  proba = gmm.predict_proba(np.array(randomised_anomaly_set))
  print("Product of predict proba: \n", np.prod(proba, axis=1))
  print("Argmax of predict proba: \n", np.argmax(proba, axis=1))

  print("Proba: ", proba)
  
  # determining the outcome
  # first proba is that of comparison of unknown image with normal set and anomaly set:
  max_proba_of_normal_configuration = proba[0].max()
  
  # second proba is that of comparison of unknown image with anomaly set and normal set:
  max_proba_of_anomaly_configuration = proba[1].max()
  
  # first proba is that of comparison of unknown image with normal set and anomaly set:
  min_proba_of_normal_configuration = proba[0].min()
  
  # second proba is that of comparison of unknown image with anomaly set and normal set:
  min_proba_of_anomaly_configuration = proba[1].min()
  
  s = None
  
  # if cluster predictions are the same, use cluster labels
  if y_pred_single_label[0] == y_pred_single_label[1]:
    s = 'Normal' if y_pred_single_label[0] == 0 else 'Anomaly'
  
  # if both the maximum values are close, check for lowest probabilities
  if s is None and np.isclose(max_proba_of_normal_configuration, max_proba_of_anomaly_configuration, atol=1e-2):
    if np.isclose(min_proba_of_normal_configuration, min_proba_of_anomaly_configuration, atol=1e-12):
      print("Predicted 'Unknown'")
      s = 'Unknown'
    else:
      if min_proba_of_normal_configuration > min_proba_of_anomaly_configuration:
        print("Normal Image")
        s = 'Normal'
      else:
        print("Anomaly Image")
        s = 'Anomaly'
  elif s is None:
    if max_proba_of_normal_configuration > max_proba_of_anomaly_configuration:
      print("Normal Image")
      s = 'Normal'
    else:
      print("Anomaly Image")
      s = 'Anomaly'
  
  noanswer = False
  if s == 'Normal':
    idx = 0
  elif s == "Anomaly":
    idx = 1
  else:
    noanswer = True
  if not noanswer:
    plt.scatter(np.array(randomised_anomaly_set)[idx,0], np.array(randomised_anomaly_set)[idx,1], s=80, facecolors='none', edgecolors=['b', 'r'][idx], lw=10)
  plt.legend()
  plt.text(0.4, 0.85, s + " Image", transform = plt.gca().transAxes)
  
  filename = 'images/temp/' + str(uuid.uuid4()) + ".jpg"
  plt.savefig(filename)
  plt.close()

  return filename, s