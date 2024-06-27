from keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


def read_image(path:str):
    image = Image.open(path).convert("L")
    image = np.asarray(image.resize((128,128)))/255

    return image

def array_to_PIL_image(arr:np.ndarray):
    arr = (arr * 255).astype(np.uint8)
    return Image.fromarray(arr)


def predict_image(path:str,h5_model:str):

    model = load_model(h5_model)

    image = read_image(path)

    mask = model.predict(np.expand_dims(image,axis=0))

    seg = np.squeeze(image).copy()
    seg[np.squeeze(mask)<0.2] = 0

    origin = np.squeeze(image) 
    segment = seg
    masked = np.squeeze(mask[0])

    origin = array_to_PIL_image(origin)
    segment = array_to_PIL_image(segment)
    masked = array_to_PIL_image(masked)

    return origin,segment,masked