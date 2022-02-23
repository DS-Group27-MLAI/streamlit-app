import numpy as np
import cv2

def prepare_image_from_bytes(element, bytes_data, size=(224,224)):
    element.image(bytes_data, caption="Uploaded Image")
    bytes_data = np.frombuffer(bytes_data, dtype=np.int8)
    image = cv2.imdecode(bytes_data, cv2.IMREAD_UNCHANGED)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    image = cv2.resize(image, size)

    return image

