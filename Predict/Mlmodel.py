import numpy as np
from tensorflow.keras.preprocessing import image as keras_image # type: ignore
from tensorflow.keras.applications.vgg16 import preprocess_input # type: ignore

def predict_chest_xray(img_data, model):
    try:
        # Preprocess the image data
        img_data = preprocess_input(img_data)
        
        # Make prediction
        classes = model.predict(img_data)
        result = int(classes[0][0])
        return result, None
    except Exception as e:
        return None, f"Error making prediction: {str(e)}"
