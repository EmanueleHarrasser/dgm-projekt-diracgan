import tkinter
from tkinter import *
import model
import plotter

def show_window():
    parameters, gradients = dirac_gan.get_vectors()
    plotter.plot_vectors(parameters,gradients)

    points = dirac_gan.train()
    plotter.plot_points(points)
    plotter.init_plot()
    plotter.show_plot()

def save(dirac_gan,loss_type):
    print('loss type: ',loss_type.get())
    dirac_gan.set_loss(loss_type.get())

dirac_gan = model.Model()
root = Tk()
root.title('Dirac-GAN')
root.geometry("200x200")
Train_Button = Button( text ="Train",command=show_window)
Train_Button.place(x=50,y=50)
loss_type = StringVar()
GAN_box = Radiobutton(root, text="GAN",  variable = loss_type, value = 'GAN').grid(row = 0, column = 0)
NGAN_box = Radiobutton(root, text="NGAN", variable = loss_type, value = 'NGAN').grid(row = 0, column = 1)
loss_type.set('GAN')
plotter.init_plot()


Save_Button = Button( text = "Save", command= lambda: save(dirac_gan,loss_type))
Save_Button.place(x = 50,y = 100)





root.mainloop()

