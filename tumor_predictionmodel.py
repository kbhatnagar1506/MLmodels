# -----------------------------------------------------------
# 📌 PROJECT: Tumor Classification using Logistic Regression
# -----------------------------------------------------------
# GOAL:
# Predict whether a tumor is malignant (1) or benign (0)
# using patient tumor features like radius, texture, perimeter, area, and smoothness.

# We use logistic regression with gradient descent from scratch.
# -----------------------------------------------------------


import numpy as np

# -------- STEP 1: Load Data --------
def load_data():
    # Features: [radius, texture, perimeter, area, smoothness]
    x = np.array([
        [14.5, 20.2, 95.2, 600.1, 0.08],
        [20.3, 25.5, 130.0, 1200.3, 0.10],
        [12.4, 15.0, 82.0, 520.5, 0.09],
        [22.1, 30.0, 140.5, 1300.0, 0.11],
        [10.2, 12.8, 70.5, 400.6, 0.07],
        [18.7, 22.1, 120.3, 1100.1, 0.12],
        [11.3, 13.3, 75.2, 420.8, 0.06],
        [16.0, 18.9, 98.1, 670.4, 0.09],
        [13.8, 17.6, 90.0, 570.3, 0.08],
        [24.5, 33.0, 155.0, 1500.5, 0.13]
    ])
    y = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])  # Labels
    return x, y

# -------- STEP 2: Normalize Data --------
def normalisation(x, y):
    x_mean = np.mean(x, axis=0)
    x_std = np.std(x, axis=0)
    return (x - x_mean)/x_std, y, x_mean, x_std

# -------- STEP 3: Compute Gradient --------
def gradient(x, y, w, b):
    m, n = x.shape
    dw = np.zeros(n)
    db = 0.0
    for i in range(m):
        z = np.dot(w, x[i]) + b
        prediction = 1 / (1 + np.exp(-z))
        dw += (prediction - y[i]) * x[i]
        db += (prediction - y[i])
    return dw / m, db / m

# -------- STEP 4: Gradient Descent --------
def gradient_descent(x, y, w_init, b_init, iterations, alpha):
    w = w_init
    b = b_init
    for i in range(iterations):
        dw, db = gradient(x, y, w, b)
        w -= alpha * dw
        b -= alpha * db
        if i % 100 == 0 or i == iterations - 1:
            z = np.dot(x, w) + b
            pred = 1 / (1 + np.exp(-z))
            cost = -np.mean(y * np.log(pred + 1e-15) + (1 - y) * np.log(1 - pred + 1e-15))
            print(f"Iteration {i}: Cost = {cost:.4f}")
    return w, b

# -------- STEP 5: Predict --------
def predict(x, w, b):
    z = np.dot(w, x) + b
    return 1 / (1 + np.exp(-z))

# -------- MAIN --------
x, y = load_data()
x_train, y_train, x_mean, x_std = normalisation(x, y)

# Parameters
init_w = np.zeros(x_train.shape[1])  # ✅ Fix: shape = (5,)
init_b = 0.0
alpha = 0.01
iterations = 1500

# Train
w, b = gradient_descent(x_train, y_train, init_w, init_b, iterations, alpha)

# Predict
new_sample = np.array([15.0, 18.0, 85.0, 650.0, 0.085])
new_sample_norm = (new_sample - x_mean) / x_std
prob = predict(new_sample_norm, w, b)

# Output
print(f"\n🧠 Prediction probability of being malignant: {prob:.4f}")
print("🔍 Diagnosis:", "Malignant" if prob > 0.5 else "Benign")
