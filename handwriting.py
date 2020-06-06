# -*- coding: utf-8 -*-
"""
Created on Fri May  1 13:37:09 2020

@author: uasmt
"""

import numpy as np #lineer cebir kütüphanesi
import pandas as pd #verileri işlemek için
import seaborn as sns #görselleştirme kütüphanesi
import matplotlib.pyplot as plt
from keras.utils.np_utils import to_categorical # encode etmek için 
from sklearn.model_selection import train_test_split #verileri ayırmak için
from sklearn.metrics import confusion_matrix #confusion matrix için
from keras.models import Sequential  #model oluşturmak için
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D #CNN modelini oluşturmak için 
from keras.optimizers import Adam #optimizer
from keras.preprocessing.image import ImageDataGenerator

# uarıları kapatmak için 
import warnings
warnings.filterwarnings('ignore')


#verilerimiz görsel değil dataset içinde flatten olarak tutulmaktadır.
# test ve train olarak ayrılan verisetimizi aynı şekilde pandas kütühanesini 
# csv okuma kullanarak verilerimizi import ettik

trainData = pd.read_csv("train.csv")
testData = pd.read_csv("test.csv")

# eğitim için labellerimizi yani sonuçlarımızı Y_train olarak
# pixellerimizi X_train olarak ayıralım
Y_train = trainData["label"]
X_train = trainData.drop(labels = ["label"],axis = 1) 

# Verilerimizde normalizasyon gerçekleştiriyoruz.
X_train = X_train / 255.0
testData = testData / 255.0

# keras kütühanesinde işleyeceğimiz için verilerimizi reshape yapıyoruz
X_train = X_train.values.reshape(-1,28,28,1)
testData = testData.values.reshape(-1,28,28,1)

# sonuçlarımızı verktorlere dönüştürmek için encode ediyoruz.
Y_train = to_categorical(Y_train, num_classes = 10)

# sklearn kütühanesini kullanarak verilerimizi fit etmek için validation ve train için ayırıyoruz.
# %10 test, %90 eğitime ayıralım
X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size = 0.1, random_state=2)

 
#CNN modelimizi oluşturalım.
# modelimiz conv -> max pool ->dropout ->conv -> max pool => 2 layerli cnn modeli
model = Sequential()

#CNN modeli için convulalution layer ekleyelim
#5,5 lik 8 filtre oluştuyoruz
#aktivasyon fonksiyonumuz relu olacak
model.add(Conv2D(filters = 16, kernel_size = (5,5),padding = 'Same', 
                 activation ='relu', input_shape = (28,28,1)))

# max pool ekleyelim ve %24 lik dropout yapalım
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.25))

#2. convulation layeri oluşturuyoruz
model.add(Conv2D(filters = 32, kernel_size = (3,3),padding = 'Same', 
                 activation ='relu'))
model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))
model.add(Dropout(0.25))

# flatten yapalım ve cnn modelimizi oluşturalım
# sonuç için softmax aktivasyonunu seçiyoruz 
model.add(Flatten())
model.add(Dense(256, activation = "relu"))
model.add(Dropout(0.5))
model.add(Dense(10, activation = "softmax"))

# optimizerimizi tanımlıyoruz
#Adam optimizer lr nin durumuna göre otomatik artıp azalmasını saylamak için
optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999)

# ve modelimizi compile ediyoruz.
model.compile(optimizer = optimizer , loss = "categorical_crossentropy", metrics=["accuracy"])

#epoch ve batch size belirliyoruz
epochs = 25
batch_size = 250

# over fittingi önlemek için önlemek içinaugmentation yapıyoruz.
# verilerimizden benzer yeni veriler olşturuyoruz
#bunun için ImageDataGenerator kullanıyoruz
#ve train verimize fit ediyoruz.
augData = ImageDataGenerator(
        featurewise_center=False,  
        samplewise_center=False,  
        featurewise_std_normalization=False, 
        samplewise_std_normalization=False, 
        zca_whitening=False,  
        rotation_range=0.5,  # rastgele rotasyonu 5 derece değiştir
        zoom_range = 0.5, # rastgele 5 derece zoom yap
        width_shift_range=0.5,  # genişliği değiştir
        height_shift_range=0.5,  # yükselik 5 derece değiştir
        horizontal_flip=False,  
        vertical_flip=False)

augData.fit(X_train)

# eğitimden sonra loss,accuarcy gibi modelin bilgilerine ulaşmak için history tutuyoruz.
history = model.fit_generator(augData.flow(X_train,Y_train, batch_size=batch_size),
                              epochs = epochs, validation_data = (X_val,Y_val), steps_per_epoch=X_train.shape[0] // batch_size)

# accuracyi görselleştirelim
plt.plot(history.history['val_loss'], color='b', label="validation loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()

# confusion matrix yapıyoruz.

# validationden predict ettiriyoruz.
prediction = model.predict(X_val)
# tahminlerimizi vektor olarak duzenliyoruz
prediction_class = np.argmax(prediction,axis = 1) 
# gerçek sonuçlarımızı vektör olarak düzenliyoruz
true = np.argmax(Y_val,axis = 1) 
# ve iki verileri kullnarak confusion matrix oluşturuyoruz.
conMatrix = confusion_matrix(true, prediction_class) 

# confusion matrix görselleştirme
_,ax = plt.subplots(figsize=(8, 8))
sns.heatmap(conMatrix, annot=True, linewidths=0.01,cmap="Greens",linecolor="gray", fmt= '.1f',ax=ax)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()