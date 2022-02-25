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

def ia_cam():
    labels = gen_labels()
    # Disable scientific notation for clarity
    np.set_printoptions(suppress=True)

    image = cv2.VideoCapture(0)
    # Load the model
    model = load_model('keras_model.h5', compile=False)

    """
    Create the array of the right shape to feed into the keras model
    The 'length' or number of images you can put into the array is
    determined by the first position in the shape tuple, in this case 1."""
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

    while True:
        # Choose a suitable font
        text_font = cv2.FONT_HERSHEY_SIMPLEX
        ret, frame = image.read()
        if not ret:
            continue
        frame2 = frame
        # resize the image to a 224x224 with the same strategy as in TM2:
        # resizing the image to be at least 224x224 and then cropping from the center
        frame2 = cv2.resize(frame2, (224, 224))
        # turn the image into a numpy array
        image_array = np.asarray(frame2)
        # Normalize the image
        normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1
        # Load the image into the array
        data[0] = normalized_image_array
        pred = model.predict(data)
        result = np.argmax(pred[0])
        r_final = labels[str(result)]

        # Récupère la couleur en BGR
        for n in range(0, 7):
            if r_final == Color_name[n]:
                txt_color = Color[n]
                # print(txt_color)

        # Print the predicted label into the screen.
        cv2.putText(frame,
                    r_final,
                    (280, 400),
                    text_font,
                    1,
                    txt_color,
                    2,
                    cv2.LINE_AA
                    )
        # Exit, when 'q' is pressed on the keyboard
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        # Show the frame
        cv2.imshow('Frame',
                   frame)
        sleep(0.1)

    image.release()
    cv2.destroyAllWindows()


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
