from tkinter import *
from tkinter.filedialog import askopenfilename
from tkinter.messagebox import showinfo
from tkinter.ttk import *
from time import *

import cv2
import numpy as np
from PIL import Image, ImageOps
from keras.models import load_model


def gen_labels():
    labels = {}
    with open("labels.txt", "r") as label:
        text = label.read()
        lines = text.split("\n")
        for line in lines[0:-1]:
            hold = line.split(" ", 1)
            labels[hold[0]] = hold[1]
    return labels

# Couleurs
Color_name = ['Bleu',
                   'Jaune',
                   'Noir',
                   'Rose',
                   'Rouge',
                   'Vert',
                   'Rien']

Color = [(255, 0, 0),
              (5, 209, 242),
              (0, 0, 0),
              (151, 115, 242),
              (0, 0, 255),
              (0, 255, 0),
              (255, 255, 255)]

# Initialisation de la fenetre
window = Tk()

window.title("IA Dragibus")
window.configure(width=300, height=150)
window.iconbitmap('icon.ico')

# On centre la fenetre au milieu de l'écran
winWidth = window.winfo_reqwidth()
winHeight = window.winfo_reqheight()
posRight = int(window.winfo_screenwidth() / 2 - winWidth / 2)
posDown = int(window.winfo_screenheight() / 2 - winHeight / 2)
window.geometry("+{}+{}".format(posRight, posDown))

def ia_pic():
    labels = gen_labels()
    filename = askopenfilename(
        title="Open a file",
    )
    # Disable scientific notation for clarity
    np.set_printoptions(suppress=True)
    # Load the model
    model = load_model('keras_model.h5', compile=False)

    """
    Create the array of the right shape to feed into the keras model
    The 'length' or number of images you can put into the array is
    determined by the first position in the shape tuple, in this case 1."""
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    # A dict that stores the labels

    image = Image.open(filename)

    size = (224, 224)
    image = ImageOps.fit(image, size, Image.ANTIALIAS)
    image_array = np.asarray(image)
    normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1

    data[0] = normalized_image_array
    prediction = model.predict(data)

    result = np.argmax(prediction[0])
    r_final = labels[str(result)]
    # print(r_final)

    # txt2 = Label(window, text=r_final, font=("Arial", 20))
    # txt2.place(relx=0.5, rely=0.6, anchor=CENTER)
    showinfo("Result", r_final)

# Contenu de la fenetre
txt1 = Label(window, text="Mode selection", font=("Arial", 20))
txt1.place(relx=0.5, rely=0.3, anchor=CENTER)

# Déclencheur ia par image
btn1 = Button(
    window,
    text="Picture",
    command=ia_pic
)
btn1.place(relx=0.2, rely=0.6)

# Déclencheur ia par caméra
btn2 = Button(
    window,
    text="Camera(marche pas trop)",
    command=ia_cam
)
btn2.place(relx=0.5, rely=0.6)

window.mainloop()
