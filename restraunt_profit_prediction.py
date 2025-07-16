"""
📌 PROBLEM STATEMENT: Linear Regression to Predict Restaurant Profit

You're the CEO of a restaurant franchise looking to expand into new cities.
You already have profit data from 20 cities and their populations.

🎯 GOAL:
Build a simple machine learning model using **Linear Regression** to:
- Learn the relationship between a city's population and expected monthly profit.
- Predict profit for a new city given its population.

👨‍💻 WHAT YOU WILL BUILD:
- A machine learning model using **gradient descent** to minimize cost.
- A way to visualize both data and the fitted regression line.
- A predictor that can estimate profits for cities with different population sizes.
"""


#importing modules 

import numpy as np
import matplotlib.pyplot as plt

# loading data

def load_data():
    # Population of 20 cities (in units of 10,000 people)
    population = np.array([
        6.1101, 5.5277, 8.5186, 7.0032, 5.8598, 8.3829, 7.4764, 8.5781, 6.4862, 5.0546,
        5.7107, 14.164, 5.734, 8.4084, 5.6407, 5.3794, 6.3654, 5.1301, 6.4296, 7.0708
    ])
    # Monthly profit from each city (in units of $10,000)
    profit = np.array([
        17.592, 9.1302, 13.662, 11.854, 6.8233, 11.886, 4.3483, 12.0, 6.5987, 3.8166,
        3.2522, 15.505, 3.1551, 7.2258, 0.71618, 3.5129, 5.3048, 0.56077, 3.6518, 5.3893
    ])
    return population , profit
    
def compute_gradient(x,y,w,b):
    m = x.shape[0]
    drv_w = 0
    drv_b = 0 
    for i in range(m):
        prediction = x[i]* w + b
        drv_w = drv_w + (prediction - y[i])*x[i]
        drv_b = drv_b + prediction - y[i]
    drv_w = drv_w / m
    drv_b = drv_b /m 
    return drv_w , drv_b

def compute_gradient_decent(x , y ,tmp_w,tmp_b,alpha,iterations):
    b = tmp_b 
    w = tmp_w 
    for i in range(iterations):
        dj_dw, dj_db = compute_gradient(x, y, w, b)
        w -= alpha * dj_dw
        b -= alpha * dj_db
    return w,b

def predict(x,w,b):
    prediction = x*w + b 
    return prediction

x_train , y_train = load_data()

tmp_w = 0.0  
tmp_b = 0.0  
alpha = 0.01     
iterations = 1500  

w_final,b_final = compute_gradient_decent(x_train , y_train ,tmp_w,tmp_b,alpha,iterations)

pop1 = 3.5
pop2 = 7.0
print(f"\n📈 For population = 35,000 → Predicted profit = ${predict(pop1, w_final, b_final) * 10000:.2f}")
print(f"📈 For population = 70,000 → Predicted profit = ${predict(pop2, w_final, b_final) * 10000:.2f}")
