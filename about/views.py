from django.shortcuts import render

# Create your views here.
def about(req):
    return render(req,'about/about.html')
# views.py

import os
import zipfile
from django.http import HttpResponse
from django.conf import settings

def add_to_zip(zip_file, directory, folder_name):
    images = [os.path.join(directory, img) for img in os.listdir(directory) if img.endswith('.jpeg')]
    for image in images:
        zip_file.write(image, os.path.join(folder_name, os.path.basename(image)))

def download_folder(request):
    images_dir1 = os.path.join(settings.BASE_DIR, 'about\\static\\about\\images\\test\\pneumonia')
    images_dir2 = os.path.join(settings.BASE_DIR, 'about\\static\\about\\images\\test\\Normal')
    zip_filename = 'images.zip'

    with zipfile.ZipFile(zip_filename, 'w') as zip_file:
        add_to_zip(zip_file, images_dir1, 'pneumonia')
        add_to_zip(zip_file, images_dir2, 'Normal')

    with open(zip_filename, 'rb') as zip_file:
        response = HttpResponse(zip_file.read(), content_type='application/zip')
        response['Content-Disposition'] = 'attachment; filename={}'.format(zip_filename)
        return response
