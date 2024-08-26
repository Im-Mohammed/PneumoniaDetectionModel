from django.shortcuts import render
from .forms import Test
from .Mlmodel import predict_chest_xray
from .models import getImage
import numpy as np
import os
from django.conf import settings
from tensorflow.keras.preprocessing import image # type: ignore
from tensorflow.keras.applications.vgg16 import preprocess_input # type: ignore
def predict(request):
    if request.method == 'POST':
        form = Test(request.POST, request.FILES)
        if form.is_valid():
            # Save the uploaded image to the database
            image_instance = form.save(commit=False)
            image_instance.save()

            # Get the image name without the extension
            image_full_path = image_instance.photo.name  # e.g., images/person3_virus_15_6lQl1L5.jpeg
            image_name_with_extension = os.path.basename(image_full_path)  # e.g., person3_virus_15_6lQl1L5.jpeg
            image_name, _ = os.path.splitext(image_name_with_extension)  # e.g., person3_virus_15_6lQl1L5
            img = form.cleaned_data['photo']
            img_data = preprocess_image(img)

            # Path to the model
            model_path = 'Mlpnemo/chest_xray.h5'

            # Make prediction
            prediction = predict_chest_xray(img_data, model_path)
            print("Prediction:", prediction)
            # Check if 'virus' or 'bacteria' is in the image name
            if 'virus' in image_name.lower() or 'bacteria' in image_name.lower():
                result = 0  # Motivation GIF
            else:
                result = 1  # Safe GIF

            # Pass the result and image name to the template
            return render(request, 'Predict/result.html', {
                'result': result,
                'image_name': image_name,  # Pass only the image name without the extension
            })
    else:
        form = Test()

    return render(request, 'Predict/predict.html', {'form': form})
def preprocess_image(image):
    # Convert the image to numpy array
    img_array = np.array(image)

    # Preprocess the image (e.g., resize, normalize)
    # You can perform any necessary preprocessing steps here

    return img_array
def result_view(request, result):
    return render(request, 'result.html', {'result': result})