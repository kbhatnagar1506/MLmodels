"""
🏠 House Price Prediction with 5 Features using Linear Regression (NumPy Only)

Features:
1. Square Footage
2. Number of Bedrooms
3. Age of the House
4. Number of Bathrooms
5. Garage Space (number of cars it fits)

Target:
- House Price in dollars

Method:
- Linear Regression using Gradient Descent
"""

import numpy as np

# ----------------- STEP 1: Load Synthetic Data -----------------
def load_data():
    # X rows: [sqft, bedrooms, age, bathrooms, garage_space]
    x = np.array([
        [1400, 3, 20, 2, 1],
        [1600, 3, 15, 2, 1],
        [1700, 3, 18, 2, 2],
        [1875, 4, 12, 3, 2],
        [1100, 2, 30, 1, 1],
        [1550, 3, 20, 2, 2],
        [2350, 4, 10, 3, 2],
        [2450, 4, 8, 3, 2],
        [1425, 3, 15, 2, 1],
        [1700, 3, 14, 2, 1]
    ])
    y = np.array([245000, 312000, 279000, 308000, 199000, 219000, 405000, 324000, 319000, 255000])
    return x,y

def normalise(x,y):
    mean = np.mean(x, axis=0)
    std = np.std(x, axis=0)
    x_normalized = (x - mean) / std
    meany = np.mean(y, axis=0)
    stdy = np.std(y, axis=0)
    y_normalized = (y - meany) / stdy
    return x_normalized,y_normalized,meany,stdy,mean,std

def gradient(x,y,w,b):
    m, n = x.shape
    drv_w = np.zeros(n)
    drv_b = 0.0
    for  i in range(m):
        prediction = np.dot(w,x[i]) + b
        drv_w = drv_w + (prediction - y[i])*x[i]
        drv_b = drv_b + (prediction - y[i])
    drv_w = drv_w/m
    drv_b = drv_b/m
    return drv_w,drv_b


def gradient_descent(x, y, w_init, b_init, alpha, iterations):
    w = w_init.copy()
    b = b_init
    for i in range(iterations):
        dw,db = gradient(x,y,w,b)
        w = w - (alpha * dw)
        b = b - (alpha * db)
    return(w,b)

def predict(x,w,b):
    return (np.dot(w,x)) + b

x,y = load_data()
x_train,y_train,y_mean,y_std,x_mean,x_std = normalise(x,y)
alpha = 0.01
b_init = 0.0
w_init = np.zeros(x_train.shape[1])
iterations =  1500
w,b = gradient_descent(x_train,y_train,w_init, b_init, alpha, iterations)

# Predict a new house (normalize it first)
new_house = np.array([2000, 3, 15, 2, 2])
new_house_norm = (new_house - x_mean) / x_std
pred_norm = predict(new_house_norm, w, b)

# Denormalize prediction
pred_actual = pred_norm * y_std + y_mean

print(f"\n🏠 Predicted price for the house: ${pred_actual:.2f}")
