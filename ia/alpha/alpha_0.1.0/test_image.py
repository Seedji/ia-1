import tensorflow.keras
import numpy as np
from process_labels import gen_labels
from PIL import Image, ImageOps
from tkinter import Tk
from tkinter.filedialog import askopenfilename

def Label_pic():

    # Disable scientific notation for clarity
    np.set_printoptions(suppress=True)
    # Load the model
    model = tensorflow.keras.models.load_model('keras_model.h5', compile = False)

    """
    Create the array of the right shape to feed into the keras model
    The 'length' or number of images you can put into the array is
    determined by the first position in the shape tuple, in this case 1."""
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    # A dict that stores the labels
    labels = gen_labels()
    
    
    file = askopenfilename()
    
    image = Image.open(file)

    size = (224, 224)
    image = ImageOps.fit(image, size, Image.ANTIALIAS)
    image_array = np.asarray(image)
    normalized_image_array = (image_array.astype(np.float32) / 127.0) -1

    data[0] = normalized_image_array
    prediction = model.predict(data)

    result = np.argmax(prediction[0])
    
    print ("Label: ", labels[str(result)])
