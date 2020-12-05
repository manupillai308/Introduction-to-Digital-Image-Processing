from Preprocess import preprocess
from keras.models import load_model
import numpy as np
import json

classes = json.load(open("./classes.json", "r"))

model = load_model("./Model.h5")

def predict(image):

    pp_image = preprocess(image)
    pp_image = pp_image.reshape(-1, 45, 45, 1)

    prediction = model.predict(pp_image)
    cls = classes[str(np.argmax(prediction))]

    return cls