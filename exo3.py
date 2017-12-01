import numpy as np
from keras.models import Sequential
from keras.utils import np_utils
from keras import optimizers
model = Sequential()

from keras.layers import Dense, Flatten, Activation
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D

model.add(Conv2D(32,kernel_size=(5, 5),activation='sigmoid',input_shape=(28, 28, 1),padding='same'))
model.add(Activation('sigmoid'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64,kernel_size=(5, 5),activation='sigmoid',input_shape=(28, 28, 1),padding='same'))
model.add(Activation('sigmoid'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())

model.add(Dense(100, name='fc1'))
model.add(Activation('sigmoid'))
model.add(Dense(10, name='fc2'))
model.add(Activation('softmax'))

#Avec |Keras|, on va compiler le modèle en lui passant un loss (ici l’
#entropie croisée), une méthode d’optimisation (ici uns descente de
#gradient stochastique, stochatic gradient descent, sgd), et une métrique
#d’évaluation (ici le taux de bonne prédiction des catégories, accuracy):
#
learning_rate = 0.5
sgd = optimizers.SGD(learning_rate)
model.compile(loss='categorical_crossentropy',optimizer=sgd,metrics=['accuracy'])
#
#Enfin, l’apprentissage du modèle sur des données d’apprentissage est mis
#en place avec la méthode |fit| :
#
batch_size = 300
nb_epoch = 10
## convert class vectors to binary class matrices
from keras.datasets import mnist

# the data, shuffled and split between train and test sets
(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = X_train.reshape(60000, 784)
X_test = X_test.reshape(10000, 784)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')
Y_train = np_utils.to_categorical(y_train, 10)
Y_test = np_utils.to_categorical(y_test, 10)
X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)
input_shape = (28, 28, 1)
model.fit(X_train, Y_train,batch_size=batch_size, epochs=nb_epoch,verbose=1)
#from keras.callbacks import TensorBoard
#tensorboard = TensorBoard(log_dir="_mnist", write_graph=False, write_images=True)
#model.fit(X_train, Y_train,batch_size=batch_size, epochs=nb_epoch,verbose=1, callbacks=[tensorboard])
#
#  * batch_size correspond au nombre d’exemples utilisé pour estimer le
#    gradient de la fonction de coût.
#  * epochs est le nombre d’époques (/i.e./ passages sur l’ensemble des
#    exemples de la base d’apprentissage) lors de la descente de gradient.
#
#*N.B :* on rappelle que comme dans les TME précédents, les labels
#données par la supervision doivent être au format “one-hot encoding”.
#
#On peut ensuite évaluer les performances du modèle dur l’ensemble de
#test avec la fonction |evaluate|
#
scores = model.evaluate(X_test, Y_test, verbose=0)
print("%s: %.2f%%" % (model.metrics_names[0], scores[0]*100))
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

from keras.models import model_from_yaml

def saveModel(model, savename):
    # serialize model to YAML
    model_yaml = model.to_yaml()
    with open(savename+".yaml", "w") as yaml_file:
        yaml_file.write(model_yaml)
        print ("Yaml Model ",savename,".yaml saved to disk")
    # serialize weights to HDF5
    model.save_weights(savename+".h5")
    print ("Weights ",savename,".h5 saved to disk")

saveModel(model, "ConvNet_Exo3")
