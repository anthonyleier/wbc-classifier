#Imports
import numpy as np
import os
import PIL
import PIL.Image
import tensorflow as tf
import pathlib


#####Leitura do Dataset
print("Leitura do Dataset")

dataset = "./dataset"
dataset = pathlib.Path(dataset)

#Contagem
contagem = len(list(dataset.glob('*/*.jpg')))
print("Existem", contagem, "imagens nesse dataset")


#####Pré-Processamento dos Dados
print("Pré-Processamento dos Dados")
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
print("Classes encontadas:", classes)


#####Padronização dos Dados
print("Padronização dos Dados")
from tensorflow.keras import layers
normalization_layer = tf.keras.layers.experimental.preprocessing.Rescaling(1./255) #Camada de reescalonamento

normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y)) #Normalização de todo dataset de treino
image_batch, labels_batch = next(iter(normalized_ds))
first_image = image_batch[0]
print(np.min(first_image), np.max(first_image)) #Imprime o menor e o maior valor de pixel da imagem, portanto foi normalizado com sucesso


#####Configurações de Desempenho
print("Configurações de Desempenho")
AUTOTUNE = tf.data.AUTOTUNE
#Dois metodos aplicados nos datasets de treino e teste
train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE) 
test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)


#####Treinamento do Modelo
print("Treinamento do Modelo")

qtdClasses = len(classes)

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

model.compile(
    optimizer='adam',
    loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy'])

epocas = 6
model.fit(train_ds, validation_data=test_ds, epochs=epocas)

#####Predição de Teste
print("Predição de Teste")

predicoes = model.predict(test_ds) #Passando o conjunto de imagens de teste
print(predicoes[0])

print(np.argmax(predicoes[0]))

print(classes[np.argmax(predicoes[0])])