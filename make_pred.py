#!flask/bin/python
from flask import request
from flask import Flask
from flask_cors import CORS
app = Flask(__name__)
CORS(app)

import cgi, cgitb
cgitb.enable()


#the cgi library gets vars from html
data = cgi.FieldStorage()

from keras.models import load_model
import imageio
import numpy as np
from math import floor, ceil

model = load_model("/Users/arushigupta/Desktop/iam_modelII_addtn.h5")

infi = "CanvasDemo-2 copy.png"

max_width = 231
max_height = 1934



@app.route("/", methods = ['POST', 'GET'])
def predict():

    loc_fil = '/Users/arushigupta/Downloads/' + request.args['finam'] + '.png'
    
    img = imageio.imread(loc_fil)
    img = img [130:300, :]
    newimg = np.full((169, 400), 250)

    for i in range(0, 169):
        for j in range(0, 400):
            if img[i, j, 3] > 0:
                newimg[i,j] = 0
    img = newimg
    im_w = np.shape(img)[0]
    im_h = np.shape(img)[1] 
    pad_0_l = floor((max_width - im_w)/2.0) 
    pad_0_r = ceil((max_width - im_w)/2.0)
    pad_1_l  = floor((max_height - im_h)/2.0)
    pad_1_r = ceil((max_height - im_h)/2.0)
    padded_img =  np.pad(img, [(pad_0_l, pad_0_r), (pad_1_l, pad_1_r)], mode = 'constant', constant_values = 0)

    padded_img  = np.reshape( padded_img, (1, 231, 1934, 1) )


    if np.argmax(model.predict(padded_img, batch_size = 1)) == 17:
        return "not" 
    else :
        return "the"

    


if __name__ == '__main__':
    # run!
    app.run()
