import os
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
from model import Model
from flask import Flask, request, jsonify, send_from_directory
import json
BROWSER_DIR = os.path.dirname(__file__) + "/web/dist/browser/"
app = Flask(__name__, static_folder=BROWSER_DIR, static_url_path="/")
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0

strategy = "gan"
MODEL_DIR = "./models/" + strategy + "/model.ckpt"
m = Model(MODEL_DIR)

NUM_SAMPLES = 100

mnist = input_data.read_data_sets('/tmp/MNIST_data')
images = mnist.train.images[0:NUM_SAMPLES];
labels = mnist.train.labels[0:NUM_SAMPLES];

labeled_images = [None] * NUM_SAMPLES
for i in range(0, NUM_SAMPLES):
    image = np.reshape(images[i], [28, 28])
    labeled_images[i] = {
        'image': image.tolist(),
        'label': labels[i].astype('int'),
        'encoding': m.encode([images[i]])[0].tolist(),
    }
    decoded = m.decode([labeled_images[i]['encoding']])
    labeled_images[i]['decoded'] = np.reshape(decoded[0], [28, 28]).tolist()

@app.route('/reload_model')
def reload_model():
    m = Model(MODEL_DIR)
    return jsonify(ok=True)

@app.route('/images')
def get_images():
  return jsonify(labeled_images)

@app.route('/encode', methods=["POST"])
def encode():
  image = np.reshape(request.json, [28*28])
  return jsonify(m.encode([image])[0].tolist())

@app.route('/decode', methods=["POST"])
def decode():
  val = m.decode([request.json])
  return jsonify(np.reshape(val[0], [28, 28]).tolist())

@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def catch_all(path):
    return app.send_static_file("index.html")

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)

app.json_encoder = NpEncoder
