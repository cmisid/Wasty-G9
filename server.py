from io import BytesIO

from flask import Flask, jsonify
from PIL import Image
import cv2
import requests
import numpy

#from Classification.classification import classify
from Treatment.sift import sift_descriptor

app = Flask(__name__)

@app.route('/')
def index():
    payload = {
        'confiance': 0.8,
        'tag': 'meuble'
    }
    return jsonify(payload)

@app.route('/classify/<path:url>')
def classifier_image(url):

    return None

@app.route('/descriptor')
def hello():
    return 'HELLO'
@app.route('/descriptor/<path:url>')
def descriptor_image(url):
    response = requests.get(url)
    payload = {
        'url': url
    }
    if response.status_code == 200:
        img = Image.open(BytesIO(response.content))
        img = numpy.array(img)
        image_descriptor = sift_descriptor(img)
        payload = { **payload , **image_descriptor }
    return jsonify(payload)

if __name__ == '__main__':
    app.run(debug=True)
