from tensorflow.examples.tutorials.mnist import input_data
from model import Model
from flask import Flask, request, jsonify
app = Flask(__name__)

mnist = input_data.read_data_sets('/tmp/MNIST_data')
first_batch = mnist.train.next_batch(50)

m = Model()
encoding = [.5, .5]

@app.route('/encode', methods=["POST"])
def encode():
  return jsonify(m.encode(request.json).tolist())

@app.route('/decode', methods=["POST"])
def decode():
  return jsonify(m.decode(request.json).tolist())

