labels = gen_labels()
filename = askopenfilename(
    title="Open a file",
)
# Disable scientific notation for clarity
np.set_printoptions(suppress=True)
# Load the model
model = load_model('keras_model.h5', compile = False)

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
normalized_image_array = (image_array.astype(np.float32) / 127.0) -1

data[0] = normalized_image_array
prediction = model.predict(data)

result = np.argmax(prediction[0])
r_final = labels[str(result)]
#print(r_final)

txt2 = Label(window, text=r_final, font=("Arial", 20))
txt2.place(relx=0.5, rely=0.6, anchor=CENTER)