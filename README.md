# MNIST experiments

## Setup
Working with python3.7
```
sudo apt install libpython3.7-dev
python -m pip install -r requirements.txt
```

## Train models
`python autoencoder/mnist.py`

## Run the server
```
cd web
npm install
npm run build
cd ..
FLASK_APP=server.py flask run -h 0.0.0.0 -p 3000
```
