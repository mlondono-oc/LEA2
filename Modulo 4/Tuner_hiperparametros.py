### Carga de paquetes y librerias
#!pip install keras-tuner

import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier

import tensorflow as tf #Framework para deep learning
from tensorflow import keras
from keras_tuner.tuners import RandomSearch #API que contiene la mayoría de funciones para las RN

### Carga de datos fasion_mnist
(x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()
x_train.shape, y_train.shape, x_test.shape, y_test.shape
np.unique(y_train, return_counts=True)

### Estandarización de los datos
x_train2 = x_train/255 # Valores entre 0 y 1
x_test2 = x_test/255
x_train2.shape

### Definición del Hyper model
def build_model(hp):
    model = keras.Sequential()
    model.add(keras.layers.Flatten(input_shape=(28,28)))

    # Defenición de la primera capa oculta con ajuste de hiperparámetros
    # Elegir el valor óptimo entre 32 - 512 neuronas
    hp_units_1 = hp.Int('units_1', min_value=32, max_value=512, step=32)
    model.add(keras.layers.Dense(units=hp_units_1, activation='relu'))

    # Defenición de la segunda capa oculta con ajuste de hiperparámetros
    # Elegir el valor óptimo entre 32 - 512 neuronas
    hp_units_2 = hp.Int('units_2', min_value=32, max_value=512, step=32)
    model.add(keras.layers.Dense(units=hp_units_2, activation='relu'))

    # Definición de la capa de salida
    model.add(keras.layers.Dense(10, activation='softmax'))

    # Definición de la tasa de aprendizaje del optimizador
    # Elegir el valor óptimos entre 0.1, 0.01, 0.001, 0.0001
    hp_learning_rate = hp.Choice('learning_rate', values=[0.1, 0.01, 0.001, 0.0001])
    model.compile(
        optimizer = keras.optimizers.Adam(learning_rate= hp_learning_rate),
        loss = 'sparse_categorical_crossentropy',
        metrics = ['accuracy']
    )

    return model

### Definción del tuner
tuner = RandomSearch(
    build_model,
    objective= 'val_accuracy',
    max_trials=5,
    executions_per_trial= 3,
    directory = 'results_tuner',
    project_name = 'fashion_mnist'
    
)

### Ejecución del Tuner
tuner.search(x_train2, y_train, epochs=5, validation_data= (x_test2, y_test))

### Mostrar el mejor modelo
for h_param in [f"units_{i}" for i in range(1,3)] + ['learning_rate']:
                print(h_param, tuner.get_best_hyperparameters()[0].get(h_param))

### Almacenar el mejor modelo
best_model = tuner.get_best_models()[0]
### Definir la arquitectura del modelo según hiperparámetros optimos
best_model.build(x_train2.shape)
### Resumen de la arquitectura
best_model.summary()

### Ajuste de la red neuronal con hiperparámetros optimos
history = best_model.fit(x_train2, y_train, epochs=10, validation_data = (x_test2, y_test))

### Cual es el epoch con max val_accuracy
val_accuracy_per_epoch = history.history['val_accuracy']
best_epoch = val_accuracy_per_epoch.index(max(val_accuracy_per_epoch))+1
print(f"Best epoch: {best_epoch}")

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

### Utilizar el metodo evaluate para evaluar la red neuronal
test_loss, test_accuracy = best_model.evaluate(x_test2, y_test)
print('Test accuracy: ', test_accuracy)