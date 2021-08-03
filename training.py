#%%
import numpy as np
import matplotlib.pyplot as plt
import csv
from tkinter import *

# Button's function
def showDataGraph(x, y):
    plt.figure("Training Data Graph")
    plt.scatter(x, y)
    plt.xlabel("Kilometers")
    plt.ylabel("Price")
    plt.title("Data for the training")
    plt.show()

def showPredGraph(x, y, pred):
    plt.figure("Prediction on Data Graph")
    plt.plot(x, pred, c='r')
    plt.scatter(x, y)
    plt.xlabel("Kilometers")
    plt.ylabel("Price")
    plt.title("Prediction on data")
    plt.show()

def showCostGraph(cost):
    plt.figure("Cost function evaluation")
    plt.plot(range(1000), cost)
    plt.ylabel("Cost")
    plt.xlabel("Iteration")
    plt.title("Apprentice cost function")
    plt.show()


# Deep learning
def model(X, theta):
    return X.dot(theta)

def cost_function(X, y, theta):
    m = len(y)
    return 1/(2*m) * np.sum((model(X, theta) - y)**2)

def grad(X, y, theta):
    m = len(y)
    return 1/m * X.T.dot(model(X, theta) - y)

def gradient_descent(X, y, theta, learning_rate, n_iterations):
    cost = np.zeros(n_iterations)
    for i in range(0, n_iterations):
        theta = theta - learning_rate * grad(X, y, theta)
        cost[i] = cost_function(X, y, theta)
    return theta, cost

def normalisation(x):
    return (x - np.min(x))/(np.max(x) - np.min(x))

def coef_determination(y, pred):
    u = ((y - pred)**2).sum()
    v = ((y - y.mean())**2).sum()
    return 1 - u/v

# Dataset's vector
x = np.array([])
y = np.array([])

# Get datas from csv
with open('data.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    for row in csv_reader:
        if line_count == 0:
            print(f'Column names are {", ".join(row)}')
            line_count += 1
        else:
            print(f'\t{row[0]}KM for {row[1]}€')
            x = np.append(x, int(row[0]))
            y = np.append(y, int(row[1]))
            line_count += 1
    print(f'Processed {line_count} lines.')

# Save real kilometers values
km = x


x = normalisation(x)
x = x.reshape(x.shape[0], 1)
y = y.reshape(y.shape[0], 1)


X = np.hstack((x, np.ones(x.shape)))

# Generate random first theta
theta = np.random.randn(2,1)
theta = theta.reshape(theta.shape[0], 1)

# Pull down the theta value and save the cost of each iteration
theta_final, cost_history = gradient_descent(X, y, theta, 0.1, 1000)
pred = model(X, theta_final)

# Save final theta
file = open('theta', 'w+')
file.write(str(theta_final[0][0]) + "," + str(theta_final[1][0]))
file.close()

# Bonus part
window = Tk()
window.geometry("400x150")
window.title("ft_linear_regression")
window.resizable(False, False)

label = Label(window, text="Precision: " + str(coef_determination(y, pred) * 100)[0:5]+"%")
label.pack(pady=5)

show_data = Button(window, text="Show data graph", command =lambda: showDataGraph(km, y))
show_data.pack(pady=5)

show_pred =Button(window, text="Show prediction on data", command =lambda: showPredGraph(km, y, pred))
show_pred.pack(pady=5)

show_cost = Button(window, text="Show Cost Function", command =lambda: showCostGraph(cost_history))
show_cost.pack(pady=5)

mainloop()
# %%
