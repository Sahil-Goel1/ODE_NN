import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import math
import matplotlib.pyplot as plt

# My First Order ODE: dy/dx = -y + sin(x)
def f(x, y):
    return -y + tf.sin(x)

# Loss function used for computing the residual of the ODE
def loss_fn(model, x_train):
    with tf.GradientTape() as tape:
        tape.watch(x_train)
        y_pred = model(x_train)             # Predict y(x) using the model
        dydx_pred = tape.gradient(y_pred, x_train)  # Compute dy/dx using autograd
    ode_residual = dydx_pred - f(x_train, y_pred)   # ODE residual
    return tf.reduce_mean(tf.square(ode_residual))  # Mean squared error


def initial_condition_loss(model, x0, y0):
    y0_pred = model(x0)
    return tf.reduce_mean(tf.square(y0_pred - y0))

# Training step that includes both ODE residual and initial condition
def train_step(model, x_train, x0, y0, optimizer):
    with tf.GradientTape() as tape:
        loss = loss_fn(model, x_train)      # ODE loss
        loss += initial_condition_loss(model, x0, y0)  # Add initial condition loss
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return loss

# Initial condition: y(0) = 1
x0 = tf.constant([[0.0]])  
y0 = tf.constant([[1.0]])  

# Create model and optimizer
model = models.Sequential([
        layers.Input(shape=(1,)),    #giving input in the model      
        layers.Dense(10, activation="tanh"), #these are hidden layers
        layers.Dense(10, activation="tanh"),
        layers.Dense(1)     #it is the output from the model                
    ])
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

x_train = tf.constant(np.linspace(0, 2, 100).reshape(-1, 1), dtype=tf.float32)

# Code to train the model
epochs = 1000
for epoch in range(epochs):
    loss_value = train_step(model, x_train, x0, y0, optimizer)
    if epoch % 100 == 0:
        print(f"Epoch {epoch}: Loss = {loss_value.numpy()}")

x_test = np.linspace(0, 2, 100).reshape(-1, 1).astype(np.float32)

y_pred = model.predict(x_test)

for i in range(len(x_test)):
    print(f"x = {x_test[i][0]:.4f}, y_pred = {y_pred[i][0]:.4f}")

x=[]
y_actual=[]
y=[]
for i  in range(len(x_test)):
    x.append(x_test[i][0])
    y.append(y_pred[i][0])
    y_actual.append(((math.sin(x_test[i][0])-math.cos(x_test[i][0]))/2) + (1.5*math.exp(-x_test[i][0])))
plt.plot(x,y_actual,label="actual")
plt.plot(x,y,label="predicted")
plt.xlabel("Values of X")
plt.ylabel("Values of Y")
plt.legend()
plt.show()

squared_error=[]
for i in range(len(x_test)):
    squared_error.append((y[i]-y_actual[i])**2)

plt.figure(figsize=(10, 6))
plt.plot(x,squared_error)
plt.xlabel("Values of X")
plt.ylabel("Squared of diffference b/w actual and predicted Y for given X")
plt.show()
