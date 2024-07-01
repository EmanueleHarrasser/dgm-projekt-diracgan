import tkinter
from tkinter import *
import model
import plotter

def show_window():
    vectors = dirac_gan.get_vectors()
    plotter.plot_vectors(vectors)

    x_coord,y_coord = dirac_gan.train()
    plotter.plot_points(x_coord,y_coord)
    plotter.init_plot()
    plotter.show_plot()

def save(dirac_gan,loss_type):
    print('loss type: ',loss_type.get())
    dirac_gan.set_loss(loss_type.get())
    print('regularization: ', regularization.get())
    dirac_gan.set_regularization_loss(regularization.get())
    print('instance noise: ',instance_noise.get())
    dirac_gan.set_instance_noise(instance_noise.get())

dirac_gan = model.Model()
root = Tk()
root.title('Dirac-GAN')
root.geometry("300x300")
Train_Button = Button( text ="Train",command=show_window)
Train_Button.place(x=50,y=70)
loss_type = StringVar()
regularization = StringVar()
instance_noise = BooleanVar()

GAN_box = Radiobutton(root, text="GAN",  variable = loss_type, value = 'GAN').grid(row = 0, column = 0)
NGAN_box = Radiobutton(root, text="NGAN", variable = loss_type, value = 'NGAN').grid(row = 0, column = 1)
WGAN_box = Radiobutton(root, text="WGAN", variable = loss_type, value = 'WGAN').grid(row = 0, column = 2)
NO_REG_box = Radiobutton(root, text="NONE", variable = regularization, value = '').grid(row = 1, column = 0)
GP_REG_box = Radiobutton(root, text="GP", variable = regularization, value = 'GP').grid(row = 1, column = 1)
WGP_REG_box = Radiobutton(root, text="WGP", variable = regularization, value = 'WGP').grid(row = 1, column = 2)
CRGP_REG_box = Radiobutton(root, text="CRGP", variable = regularization, value = 'CRGP').grid(row = 1, column = 3)
IN_check_box = Checkbutton(root, text="Instance Noise", variable=instance_noise).grid(row = 2, column = 0)

instance_noise.set(False)
loss_type.set('GAN')
regularization.set('')
plotter.init_plot()


Save_Button = Button( text = "Save", command= lambda: save(dirac_gan,loss_type))
Save_Button.place(x = 50,y = 130)





root.mainloop()

