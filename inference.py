from keras.models import load_model

model_list = [
    'models/vae/model_best_weights_anomaly_detection_vae_designed_completion.h5',
    'models/vae/model_best_weights_anomaly_detection_vae_existing.h5',
    'model_best_weights_anomaly_detection_convae_designed.h5',
    'model_best_weights_classification_resnet_existing_completion.h5'
]


def resnet_infer_model(model_name, image):
    model = load_model(model_name)
    y_pred = model.predict((image.reshape(-1, 224, 224)) / 255)
    return True if y_pred[0] > 0.5 else False


# image: form input from api
def best_model(model_list, image):
    for i in range(len(model_list)):
        if i == 3:
            result = resnet_infer_model(model_list[i], image)
        else:
            pass
          
    return result