#Imports
import numpy as np
import os
import PIL
import PIL.Image
import tensorflow as tf
import pathlib

print("Inicialização")
#Definição do Arquivo
dataset = "./dataset"
dataset = pathlib.Path(dataset)

#Contagem
contagem = len(list(dataset.glob('*/*.jpg')))
print(contagem)

#Teste de imagem 1
basophil = list(dataset.glob('basophil/*'))
PIL.Image.open(str(basophil[0]))

#Teste de imagem 2
basophil = list(dataset.glob('basophil/*'))
PIL.Image.open(str(basophil[1]))

#Parâmetros
batch = 32
height = 180
width = 180

#80% para o treino
#20% para o teste
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    dataset,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(height, width),
    batch_size=batch)

#80% para o treino
#20% para o teste
test_ds = tf.keras.preprocessing.image_dataset_from_directory(
    dataset,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(height, width),
    batch_size=batch)

#Nomes das classes
classes = train_ds.class_names
print(classes)

from tensorflow.keras import layers
normalization_layer = tf.keras.layers.experimental.preprocessing.Rescaling(1./255) #Camada de reescalonamento

normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y)) #Normalização de todo dataset de treino
image_batch, labels_batch = next(iter(normalized_ds))
first_image = image_batch[0]
print(np.min(first_image), np.max(first_image)) #Imprime o menor e o maior valor de pixel da imagem, portanto foi normalizado com sucesso

AUTOTUNE = tf.data.AUTOTUNE
#Dois metodos aplicados nos datasets de treino e teste
train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE) 
test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)

#Treinamento de Modelo

qtdClasses = 8

model = tf.keras.Sequential([
    layers.experimental.preprocessing.Rescaling(1./255),
    layers.Conv2D(32, 3, activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(32, 3, activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(32, 3, activation='relu'),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(qtdClasses)
])
print("Compilação")
model.compile(
    optimizer='adam',
    loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy'])

epocas = 1
model.fit(train_ds, validation_data=test_ds, epochs=epocas)

predicoes = model.predict(test_ds) #Passando o conjunto de imagens de teste
predicoes[0]

np.argmax(predicoes[0])

classes[0]