import sys
import numpy as n
import os
from tkinter import Tk, Label,Button
import tkinter as tk
from tkinter import Message, Text
import shutil
import csv
import numpy as np
from PIL import Image, ImageTk
import pandas as pd
import tkinter.ttk as ttk
import tkinter.font as font
from pathlib import Path
from GUI import Facial
from GaitDemoV1 import Gait

root = Tk()
message = tk.Label(
    root, text ="Biometric Identification", 
     fg = "green", width = 50, 
    height = 2, font = ('times', 15, 'bold')) 
      
message.place(x = -150, y = 20)

button = Button(root,text="Gait",command=Gait,fg="white",bg="green",width=20 ,height =3 ,activebackground="gray58",
font =('times', 15, ' bold '))
button.place(x=25,y=100)
#button.pack()
button = Button(root,text="Facial",command=Facial,fg="white",bg="green",width=20 ,height =3 ,activebackground="gray58",
font =('times', 15, ' bold '))
button.place(x=25,y=200)
#button.pack()

root.geometry("300x300+120+120")
root.mainloop()
    