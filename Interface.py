from tkinter import *
from tkinter import filedialog
import Final
from Final import images


root = Tk()
root.title('Gender and Age Prediction')
root.geometry('809x559')
root.resizable(0, 0)
root.config(bg='black')


root.iconbitmap('icon.ico')
bg = PhotoImage(file = "Background-01-01.png")
label = Label(root, image=bg)
label.place(x=0, y=0)

M = Label(root, text= "Gender and Age Prediction", font= ('Helvetica 25 bold'), bg="black", fg="yellow")
M.pack(padx=50, pady=50)

# Button for opening camera
B = Button(root, text ="Open Camera", activebackground="#FFBD35", bg="#3030EF",fg="white", font=('Helvetica 13 bold'), command = Final.cam)
B.pack(padx=0, pady=20)

#Function for image selection
def handle_file():
    path = open()
    if images(path) is None:
        no_image_label = Label(root, text="No Face Detected", font= ('Helvetica 10 bold'), bg="yellow", fg="black")
        no_image_label.pack(padx=0, pady=1)
    
def open():
    global filepath
    filepath = filedialog.askopenfilename(title="Select a File")
    return filepath

U = Button(root, text ="Upload Image", activebackground="#FFBD35", bg="#3030EF",fg="white", font=('Helvetica 13 bold'), command = handle_file)
U.pack(padx=0, pady=30)



# Exit
Q = Button(root, text="           Quit           ", font= ('Helvetica 10'), command=root.destroy, bg="#EF353F", fg="white")
Q.pack(padx=0, pady=70)


root.mainloop()


