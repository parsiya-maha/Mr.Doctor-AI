from .PredictImage import predict_image
from keras.models import load_model
import os


base_path = os.path.join(os.getcwd(),"AI\LungMask")

model_h5 = os.path.join(base_path,"u_net.h5")


def LungMaskLoadModel():
    return load_model(model_h5)

def LungMaskPredictImage(image_path:str):
    return predict_image(image_path,model_h5)
