# Import necessary libraries
from keras.models import load_model
import cv2
import numpy as np
import uuid
import matplotlib.pyplot as plt
import visualization
import tensorflow as tf

"""
SSIMLoss: which calculate the structural similarity between two images using tensorflow
"""
def SSIMLoss(y_true, y_pred):
  return 1 - tf.reduce_mean(tf.image.ssim(y_true, y_pred, 1.0))


"""
Inference using the ResNet Model
"""
def resnet_infer_model(model_name, image):
    model = load_model(model_name)
    y_pred = model.predict((image.reshape(-1, 224, 224, 1)) / 255)
    
    return True if y_pred[0] > 0.5 else False, y_pred


"""
Inference using the DenseNet Model
"""
def densenet_infer_model(model_name, image):
    model = load_model(model_name)
    y_pred = model.predict((image.reshape(-1, 224, 224, 3)) / 255)
    
    return True if y_pred[0] > 0.5 else False


"""
Inference using the Convolutional AE model
"""
def conv_ad_infer_model(model_name, image, image28x28):
    # conv ae
    model = load_model(model_name)
    result = model.predict([image.reshape(-1, 224, 224, 1)/255, (image28x28.reshape(-1,28,28, 1))/255])
    result = (result*255).reshape(224,224)
    filename = 'images/temp/' + str(uuid.uuid4()) + ".jpg"
    plt.imsave(filename, np.clip(result.astype(np.int), a_min=0, a_max=255), cmap='gray')
    
    return filename


"""
Inference using the variational autoencoder (VAE) Model
"""
def vae_ad_infer_model(model_name, image):
    # vae
    model = load_model(model_name, {'SSIMLoss': SSIMLoss})
    pred = model.predict((image.reshape(-1, 56, 56)-127.5)/127.5)
    result = (pred*127.5 + 127.5).reshape(56,56)
    filename = 'images/temp/' + str(uuid.uuid4()) + ".jpg"
    plt.imsave(filename, np.clip(result.astype(np.int), a_min=0, a_max=255), cmap='gray')
    
    return filename, pred


"""
Inference using Conditional variational autoencoder (CVAE) model
"""
def cvae_ad_infer_model(model_name, image):
    # vae
    model = load_model(model_name, {'SSIMLoss': SSIMLoss})
    pred, z = model.predict([(image.reshape(-1, 56, 56, 1)-127.5)/127.5, np.random.randint(0,2,(1,1))])
    pred_image = (pred*127.5 + 127.5).reshape(56,56)
    filename = 'images/temp/' + str(uuid.uuid4()) + ".jpg"
    plt.imsave(filename, np.clip(pred_image.astype(np.int), a_min=0, a_max=255), cmap='gray')
    
    return filename, (pred, z)

"""
Choose the best model in Classification based on Input Parameter
Parameters:
    - idx: Index of the model list
    - model_list: list of all models
    - image: input image in 224x224 size
"""
# image: form input from api
def classification_best_model(idx, model_list, image):
    # for i in range(len(model_list)):
    if idx == 0:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        result, y_pred = resnet_infer_model(model_list[idx], image)
        image_label = "Normal" if result == False else "Anomaly"
        viz_result = visualization.viz_classification(image, image_label, y_pred)
    elif idx == 1:
        result = densenet_infer_model(model_list[idx], image)
        image_label = "Normal" if result == False else "Anomaly"
        viz_result = visualization.viz_classification(image)

    return result, viz_result, image_label


"""
Choose the best model in Anomaly Detection based on Input Parameter
Parameters:
    - idx: Index of the model list
    - model_list: list of all models
    - image: input image in 224x224 size
    - image_28x28: input image in 28x28 size
"""
def ad_best_model(idx, model_list, image, image28x28=None):
    # for i in range(len(model_list)):
    # conv ae
    if idx == 2:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        image28x28 = cv2.resize(image, (28,28))
        result = conv_ad_infer_model(model_list[idx], image, image28x28)
    # vae
    elif idx == 1:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        image = cv2.resize(image, (56,56))
        result, out_image = vae_ad_infer_model(model_list[idx], image)
        viz_result, image_label = visualization.viz_vae_ssim(image, out_image, SSIMLoss)
    # cvae
    elif idx == 0:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        image = cv2.resize(image, (56,56))
        result, out_image = cvae_ad_infer_model(model_list[idx], image)
        viz_result, image_label = visualization.viz_cvae_hist(image, out_image, SSIMLoss)

    return result, viz_result, image_label