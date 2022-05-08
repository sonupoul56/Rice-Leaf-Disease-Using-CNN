
import jsonify
import pickle
import numpy as np
import requests
from flask import Flask, render_template, request, url_for, redirect
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.models import model_from_json
from keras.preprocessing import image
from keras.preprocessing.image import load_img
from keras.preprocessing.image import image
from keras.applications.resnet import preprocess_input
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
from werkzeug.datastructures import  FileStorage
from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
import os
import base64
from PIL import Image
from io import BytesIO
import re
import json

# Define a flask app
app = Flask(__name__)

# opening and store file in a variable
json_file = open('model.json','r')
loaded_model_json = json_file. read()
json_file. close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model.
loaded_model.load_weights("model.h5")

def model_predict(img_path, model):
    print(img_path)
    img = image.load_img(img_path, target_size=(250, 250))

    # Preprocessing the image
    x = image.img_to_array(img)
    # x = np.true_divide(x, 255)
    ## Scaling
    x = x / 255
    x = np.expand_dims(x, axis=0)

    preds = loaded_model.predict(x)
    preds = np.argmax(preds, axis=1)
    if preds == 0:

        preds = "The leaf is diseased bacterial leaf blight"
    elif preds == 1:

        preds = "The leaf is diseased brown spot"
    elif preds == 2:
        preds = "The leaf is diseased leaf smut"

    return preds


@app.route('/',methods=['GET'])
def Home():
    return render_template('index.html')



@app.route('/predict',methods=['GET','POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname("uploads")
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        preds = model_predict(file_path, loaded_model)
        result=preds
        return result
    return None

if __name__ == '__main__':
    app.run(debug=True, port=8000)