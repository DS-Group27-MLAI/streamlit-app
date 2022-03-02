from keras.models import load_model
import cv2
import numpy as np
import uuid
import matplotlib.pyplot as plt


def resnet_infer_model(model_name, image):
    model = load_model(model_name)
    y_pred = model.predict((image.reshape(-1, 224, 224)) / 255)
    
    return True if y_pred[0] > 0.5 else False


def ad_infer_model(model_name, image, image28x28=None):
    model = load_model(model_name)
    # conv ae
    if image28x28 is not None:
        result = model.predict([image.reshape(-1, 224, 224)/255, (image28x28.reshape(-1,28,28)-127.5)/127.5])
        result = result*255
    # vae
    else:
        result = model.predict((image.reshape(-1, 56, 56)-127.5)/127.5)
        result = result*127.5 + 127.5
    filename = 'images/temp/' + str(uuid.uuid4()) + ".jpg"
    print(result.shape)
    plt.imsave(filename, np.clip(result.astype(np.int), a_min=0, a_max=255), cmap='gray')
    
    return filename


# image: form input from api
def classification_best_model(model_list, image):
    for i in range(len(model_list)):
        if i == 0:
            result = resnet_infer_model(model_list[i], image)
        else:
            pass
          
    return result


def ad_best_model(model_list, image, image28x28=None):
    for i in range(len(model_list)):
        # conv ae
        if i == 2:
            image28x28 = cv2.resize(image, (28,28))
            result = ad_infer_model(model_list[i], image, image28x28)
        # vae
        # elif i == 1:
        #     image = cv2.resize(image, (56,56))
        #     result = ad_infer_model(model_list[i], image)
          
    return result