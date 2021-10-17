import numpy as np
# import os
import tensorflow as tf
import cv2
import pathlib
from django.core.files.storage import FileSystemStorage



from django.shortcuts import render



def index(request):

    return render(request, "index.html")


def predict(request):

    file_obj = request.FILES['filepath']
    fs = FileSystemStorage()
    file_name = fs.save(file_obj.name, file_obj)
    file_name = fs.url(file_name)
    # file_name = pathlib.Path(file_name)

    my_img = cv2.imread('.'+file_name)
    my_img = cv2.resize(my_img, (180, 180))
    my_img = np.array(my_img)
    my_img = my_img/255
    my_img = my_img.reshape(1, 180, 180, 3)

    my_model = tf.keras.models.load_model(f"glasses_model")
    my_pred = my_model.predict(my_img)
    my_pred = np.argmax(my_pred)

    if my_pred == 1:
        prediction = "This Person Does not Wear Glasses"
    elif my_pred == 0:
        prediction = "This Person Wears Glasses"
    else:
        prediction = "Model Can't Recognise the features"





    params = {
        "file_name": file_name,
        "prediction": prediction
    }

    return render(request, "result.html", params)
