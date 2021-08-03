#%%
import numpy as np
import matplotlib.pyplot as plt
import csv
import warnings
from tkinter import *

def normalisation(x, base):
    return (x - np.min(base))/(np.max(base) - np.min(base))

def calculate(km, theta, base, window, label):
    try:
        km = int(km)
        price = theta[1] + theta[0] * normalisation(km, base[:, 0])
        label.set("Estimated price: "+str(int(price)))
        window.update_idletasks()
        plt.figure("Estimated price: "+str(int(price)))
        plt.scatter(base[:,0], base[:,1])
        plt.scatter(km, price, c='r', marker='x')
        plt.xlabel("Kilometers")
        plt.ylabel("Price")
        plt.show()
    except:
        label.set("Only numerical values are accepted")
        window.update_idletasks()

# Try to get theta value from file
warnings.filterwarnings("ignore")
try:
    theta = np.genfromtxt('theta', delimiter=',')
except:
    theta = np.array([0.0,0.0])

if theta.size == 0:
    print("theta file is empty, set default to [0 0]")
    theta = np.array([0.0, 0.0])

# Get original dataset from data file
x = np.array([])
y = np.array([])
with open('data.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    for row in csv_reader:
        if line_count == 0:
            line_count += 1
        else:
            x = np.append(x, int(row[0]))
            y = np.append(y, int(row[1]))

# Build the real matrice
base = np.array((x, y))
base = base.T

# Window for user input / graph
window = Tk()
window.title("Predict car price")
window.geometry("300x150")

theta_label = Label(window, text="Theta: "+str(theta))
theta_label.pack()

km = Label(window, text="Kilometers: ")
km.pack()


to_pred = Entry(window)
to_pred.pack()

label = StringVar()
label.set("")


predict = Button(window, text="Predict the price", command= lambda: calculate(to_pred.get(), theta, base, window, label))
predict.pack(pady=5)

price_label = Label(window, textvariable=label)
price_label.pack()

mainloop()
# %%
