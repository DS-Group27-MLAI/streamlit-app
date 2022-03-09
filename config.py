# backend/config.py

MODEL_PATH = "./models/"

STYLES = {
    "candy": "candy",
    "composition 6": "composition_vii",
    "feathers": "feathers",
    "la_muse": "la_muse",
    "mosaic": "mosaic",
    "starry night": "starry_night",
    "the scream": "the_scream",
    "the wave": "the_wave",
    "udnie": "udnie",
}

API_HOST = 'http://localhost:8080/'

# in order of performance
anomaly_detection_models = [
    'models/vae/model_best_weights_anomaly_detection_vae_designed.h5',
    'models/vae/model_best_weights_anomaly_detection_vae.h5',
    'models/convae/model_best_weights_anomaly_convae.h5'
]

# in order of performance
classification_models = [
    'models/resnet/model_best_weights_classification_resnet_existing_completion.h5',
    'models/densenet/model_best_weights_classification_densenet_existing_completion.h5'
]

# Best Model with AD (Anomaly Detection) with index 1 (CVAE Model)
best_model_anomaly_detection = 0

# Best Model with Clasisification with index 0 (ResNet)
best_model_classification = 0