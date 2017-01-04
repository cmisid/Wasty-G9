from io import BytesIO

from flask import Flask, jsonify
from PIL import Image
import requests

from classification import classify


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

    response = requests.get(url)

    payload = {
        'url': url
    }

    if response.status_code == 200:

        img = Image.open(BytesIO(response.content))
        (width, height) = img.size
        payload['img_width'] = width
        payload['img_height'] = height

    return jsonify(payload)

if __name__ == '__main__':
    app.run(debug=True)
