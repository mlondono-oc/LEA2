### Carga de paquetes y librerias
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier

import tensorflow as tf #Framework para deep learning
from tensorflow import keras #API que contiene la mayoría de funciones para las RN


### Carga de datos fasion_mnist
(x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()
x_train.shape, y_train.shape, x_test.shape, y_test.shape
np.unique(y_train, return_counts=True)
x_train[0]

### Visualización de algunos datos
plt.imshow(x_train[2800], cmap='gray')
y_train[2800]

### Estandarización de los datos
x_train2 = x_train/255 # Valores entre 0 y 1
x_test2 = x_test/255
x_train2.shape

### Reshape para el algoritmos de RFClassifier
filas_img = 28
columnas_img = 28
filasxcolumnas_img = filas_img*columnas_img

x_train2r = x_train2.reshape(x_train2.shape[0], filasxcolumnas_img)
x_train2r.shape

### Instanciar modelos

### Random Forest Classifier
rf = RandomForestClassifier(n_estimators=10)

#Para las capas ocultas se puede utilizar cualquier funcion de activación
#Relu -> retorna siempre valores positivos, los negativos los vuelve 0
#Sigmoid -> retorna valores entre 0 y 1
#Tanh -> Es la que mejor funciona en capas ocultas

#Para la capa de salida
# Clasificacion Binaria: Sigmoid
# Regresion: Relu (siempre positivos) - Ninguna funcion de activación (valores negativos)
# Calsificación multiple: Softmax

### ANN
ann1 = keras.models.Sequential(
    [keras.layers.Flatten(input_shape=[filas_img, columnas_img]),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(10, activation='softmax')]
)

## Ajuste el RF Classifier
rf.fit(x_train2r, y_train)

## Compilación ANN
ann1.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])
ann1.fit(x_train2, y_train, epochs=5, validation_data=(x_test2, y_test))