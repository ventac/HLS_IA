#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 2019
Auteur: S. Bilavarn (adapté pour TensorFlow 2+)
"""

# Imports
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import mnist

# Chargement des données
(trainData, trainLabels), (testData, testLabels) = mnist.load_data()

# Reshape pour correspondre aux dimensions attendues par Conv2D et normalisation
trainData = trainData.reshape((trainData.shape[0], 28, 28, 1)).astype('float32') / 255.0
testData  = testData.reshape((testData.shape[0], 28, 28, 1)).astype('float32') / 255.0

# Conversion des labels en one-hot encoding
trainLabels = to_categorical(trainLabels)
testLabels  = to_categorical(testLabels)
num_classes = testLabels.shape[1]

print("Shape des données d'entraînement:", trainData.shape)

# Définition du modèle LeNet
model = Sequential([
    Conv2D(20, (5,5), padding='valid', activation='relu', input_shape=(28,28,1)),
    MaxPooling2D(pool_size=(2,2)),
    Conv2D(40, (5,5), padding='valid', activation='relu'),
    MaxPooling2D(pool_size=(2,2)),
    Flatten(),
    Dense(400, activation='relu'),
    Dense(num_classes, activation='softmax')
])

# Optimiseur
sgd = SGD(learning_rate=0.01)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

# Affichage du résumé du modèle
print(model.summary())

# Entraînement
model.fit(trainData, trainLabels, batch_size=128, epochs=20, verbose=1)

# Sauvegarde du modèle au format JSON
model_json = model.to_json()
with open('model.json', 'w') as json_file:
    json_file.write(model_json)

# Sauvegarde des poids (nom compatible TensorFlow 2.13+)
model.save_weights('lenet_weights.weights.h5')    

# Évaluation sur les données de test
scores = model.evaluate(testData, testLabels, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))

