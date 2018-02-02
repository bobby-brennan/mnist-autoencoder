from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
from model import Model
from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin
app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

m = Model()

mnist = input_data.read_data_sets('/tmp/MNIST_data')
images = mnist.train.images[0:50];
labels = mnist.train.labels[0:50];

labeled_images = [None] * 50
for i in range(0, 50):
    image = np.reshape(images[i], [28, 28])
    labeled_images[i] = {
        'image': image.tolist(),
        'label': labels[i].astype('int'),
        'encoding': m.encode(image).tolist(),
    }
    labeled_images[i]['decoded'] = m.decode(labeled_images[i]['encoding']).tolist()

@app.route('/images')
def get_images():
  return jsonify(labeled_images)

@app.route('/encode', methods=["POST"])
@cross_origin()
def encode():
  return jsonify(m.encode(request.json).tolist())

@app.route('/decode', methods=["POST"])
@cross_origin()
def decode():
  return jsonify(m.decode(request.json).tolist())

