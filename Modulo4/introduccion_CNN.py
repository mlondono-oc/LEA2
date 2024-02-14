### Carga de paquetes y librerias
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier

import tensorflow as tf #Framework para deep learning
from tensorflow import keras #API que contiene la mayoría de funciones para las RN


### Carga de datos fasion_mnist
(x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()

# Se especifica la dimension de las imagenes con el canal (escala de grises)
x_train = x_train.reshape(len(x_train), 28, 28, 1)
x_test = x_test.reshape(len(x_test), 28, 28, 1)
x_train.shape, y_train.shape, x_test.shape, y_test.shape

### Estandarización de los datos
x_train2 = x_train/255 # Valores entre 0 y 1
x_test2 = x_test/255
x_train2.shape

### Definición de la arquitectura para la CNN base
model = keras.models.Sequential()

# Definición de la primera capa convolucional
model.add(
    keras.layers.Conv2D(
        filters = 32, # Cantidad de filtros
        kernel_size = (3,3), # Tamaño de los filtros
        strides = (2,2), # Cantidad de pasos o zancada
        activation = 'relu', # Rectified Linear Unit (ReLU)
        input_shape = (28,28,1) # Tamaño de la imagen
    )
)

# Definición de la capa de agrupación
model.add(
    keras.layers.MaxPooling2D(
        pool_size = (2,2),
        strides = (2,2)
    )
)

# La salida de la capa anterior es un tensor 3D. Se debe conertir a un tensor de 1D
# antes de pasar a las capas densas (Flatten)
model.add(
    keras.layers.Flatten()  
)

# Definición de capa totalemente conectada
model.add(
    keras.layers.Dense(
        units = 128,
        activation = 'relu'
    )
)

# Definición de la capa de salida
model.add(
    keras.layers.Dense(
        units= 10,
        activation = 'softmax'
    )
)

# Compilación del modelo
model.compile(
    optimizer = 'adam',
    loss = 'sparse_categorical_crossentropy',
    metrics = ['accuracy']
)

model.summary()

# Representación de la arquitectura
keras.utils.plot_model(
    model,
    to_file = 'model.png',
    show_shapes = True,
    show_layer_names = True
)

### Entrenamiento de la CNN
history = model.fit(
    x_train2,
    y_train,
    epochs = 10,
    validation_split = 0.2
)

### Evaluación del modelo con dataset de test
from sklearn.metrics import classification_report
class_names = ['Camiseta', 'Pantalón', 'Sueco', 'Vestido', 'Abrigo', 'Sandalía', 'Camisa', 'Tenis', 'Bolso', 'Botas']

y_hat = np.argmax(model.predict(x_test2), axis = 1)

print(classification_report(y_test, y_hat, target_names=class_names))

### Errores en la red neuronal
errors = np.nonzero(y_hat != y_test)[0]

# Visualizar las primeras 10 predicciones erroneas
plt.figure(figsize=(16, 8))
for i, incorrect in enumerate(errors[0:10]):
    plt.subplot(2,5, i+1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(x_test[incorrect].reshape(28,28), cmap = 'Reds')
    plt.title("Prediccion: {}".format(class_names[y_hat[incorrect]]))
    plt.xlabel("Real: {}".format(class_names[y_test[incorrect]]))
