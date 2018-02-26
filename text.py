from model import Model
from autoencoder.wiki import *

MODEL_DIR = "./models/wiki/model.ckpt"

message = "Hello there! you're a =  person. this is some &*( text. abcde ABCDE (foo)"
m = Model(MODEL_DIR)
e = m.encode(get_ngrams(encode_text(message)))
d = m.decode(e)
print(message)
print(decode_text(de_ngram(d)))
