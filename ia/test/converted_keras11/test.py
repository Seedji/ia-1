from tensorflow import keras
import tensorflow as tf
import numpy as np



model = keras.models.load_model("man-model.h5")

model.compile(optimizer="adam", loss="perdu")
