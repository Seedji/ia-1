from tkinter import *
from tkinter.ttk import *
from test_image import Label_pic

#Initialisation de la fenetre
window = Tk()

window.title("IA Dragibus")
window.configure(width=640, height=480)

#On centre la fenetre au milieu de l'écran
winWidth = window.winfo_reqwidth()
winHeight = window.winfo_reqheight()
posRight = int(window.winfo_screenwidth() / 2 - winWidth /2)
posDown = int(window.winfo_screenheight() / 2 - winHeight /2)
window.geometry("+{}+{}".format(posRight, posDown))

#Contenu de la fenetre
txt1 = Label(window, text = "Mode selection", font=("Arial", 20))
txt1.place(relx=0.5, rely=0.1, anchor=CENTER)

btn1 = Button(window, text="Picture", command = Label_pic)
btn1.place(relx=0.3, rely=0.5)

btn2 = Button(window, text="Camera")
btn2.place(relx=0.6, rely=0.5)

txt2 = Label(window, text = "Result :"labels[str(result)], font=("Arial", 16))


window.mainloop()

