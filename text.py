from model import Model
from autoencoder.wiki import *

MODEL_DIR = "./models/wiki/model.ckpt"

m = Model(MODEL_DIR)
e = m.encode(encode_text("hello there! you're a =  person. this is some &*( text"))
print(e)
d = m.decode(e)
print(decode_text(d))
