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

### Estandarización de los datos
x_train2 = x_train/255 # Valores entre 0 y 1
x_test2 = x_test/255
x_train2.shape

### Dimensiones de la imagen
filas_img = 28
columnas_img = 28

### ANN 1: red neuronal base
ann1 = keras.models.Sequential(
    [
        keras.layers.Flatten(input_shape = [filas_img, columnas_img]),
        keras.layers.Dense(128, activation = 'relu'),
        keras.layers.Dense(64, activation = 'relu'),
        keras.layers.Dense(10, activation = 'softmax')
    ]
)

### Compilacion de la ANN1
ann1.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])

### Entrenamiento de la ANN1
history = ann1.fit(x_train2, y_train, epochs = 15, validation_data = (x_test2, y_test))

### Listado de toda la data almacenada en 'history'
print(history.history.keys())

### Visualización de las curvas de error
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Perdida del modelo')
plt.xlabel('Tiempo de entrenamiento - Epochs')
plt.ylabel('Loss')
plt.legend(['train', 'val'])
plt.show()

### Visualización de las curvas de error
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Exactitud del modelo')
plt.xlabel('Tiempo de entrenamiento - Epochs')
plt.ylabel('Accuracy')
plt.legend(['train', 'val'])
plt.show()

##### REGULARIZACION #####
from tensorflow.keras import regularizers

### ANN 2: red neuronal regularizada
ann2 = keras.models.Sequential(
    [
        keras.layers.Flatten(input_shape = [filas_img, columnas_img]),
        keras.layers.Dense(128, activation = 'relu', kernel_regularizer = regularizers.L2(l2 = 0.01)),
        keras.layers.Dense(64, activation = 'relu'),
        keras.layers.Dense(10, activation = 'softmax')
    ]
)

### Compilacion de la ANN2
ann2.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])

### Entrenamiento de la ANN2
history_2 = ann2.fit(x_train2, y_train, epochs = 15, validation_data = (x_test2, y_test))

### Visualización de las curvas de error
plt.plot(history_2.history['loss'])
plt.plot(history_2.history['val_loss'])
plt.title('Perdida del modelo')
plt.xlabel('Tiempo de entrenamiento - Epochs')
plt.ylabel('Loss')
plt.legend(['train', 'val'])
plt.show()

### Visualización de las curvas de error
plt.plot(history_2.history['accuracy'])
plt.plot(history_2.history['val_accuracy'])
plt.title('Exactitud del modelo')
plt.xlabel('Tiempo de entrenamiento - Epochs')
plt.ylabel('Accuracy')
plt.legend(['train', 'val'])
plt.show()

### ANN 3: red neuronal con dropout
ann3 = keras.models.Sequential(
    [
        keras.layers.Flatten(input_shape = [filas_img, columnas_img]),
        keras.layers.Dense(128, activation = 'relu'),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(64, activation = 'relu'),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(10, activation = 'softmax')
    ]
)

### Compilacion de la ANN3
ann3.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])

### Entrenamiento de la ANN3
history_3 = ann3.fit(x_train2, y_train, epochs = 15, validation_data = (x_test2, y_test))

### Visualización de las curvas de error
plt.plot(history_3.history['loss'])
plt.plot(history_3.history['val_loss'])
plt.title('Perdida del modelo')
plt.xlabel('Tiempo de entrenamiento - Epochs')
plt.ylabel('Loss')
plt.legend(['train', 'val'])
plt.show()

### Visualización de las curvas de error
plt.plot(history_3.history['accuracy'])
plt.plot(history_3.history['val_accuracy'])
plt.title('Exactitud del modelo')
plt.xlabel('Tiempo de entrenamiento - Epochs')
plt.ylabel('Accuracy')
plt.legend(['train', 'val'])
plt.show()