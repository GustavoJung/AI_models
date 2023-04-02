import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras import Model
from sklearn.model_selection import train_test_split
import os, time
from skimage.transform import resize
from skimage.io import imread
import numpy as np

#X
dogs_folder_path = '../datasets/PetImages/Dog'
cats_folder_path = '../datasets/PetImages/Cat'
dogs_folder= os.listdir('../datasets/PetImages/Dog')
cats_folder = os.listdir('../datasets/PetImages/Cat')

images = []
labels = []

# Inicia o cronômetro
start_time = time.time()
print("Ler imagens gatos")
for file_name in cats_folder:
    # Ignora pastas e outros arquivos que não são imagens
    if not os.path.isfile(os.path.join(cats_folder_path, file_name)):
        continue
    if file_name.endswith('.jpg') or file_name.endswith('.png'):
        # Carrega a imagem e redimensiona para 100x100 pixels
        img = resize(imread(os.path.join(cats_folder_path, file_name)), (100, 100), anti_aliasing=True)
        if img.shape == (100, 100, 3):
            images.append(np.array(img))
            labels.append(1)

print("Ler imagens cachorros")
for file_name in dogs_folder:
    # Ignora pastas e outros arquivos que não são imagens
    if not os.path.isfile(os.path.join(dogs_folder_path, file_name)):
        continue
    if file_name.endswith('.jpg') or file_name.endswith('.png'):
        # Carrega a imagem e redimensiona para 100x100 pixels
        img = resize(imread(os.path.join(dogs_folder_path, file_name)), (100, 100), anti_aliasing=True)
        if img.shape == (100, 100, 3):
            images.append(np.array(img))
            labels.append(0)

# Converte as listas em arrays numpy
X = np.array(images)
Y = np.array(labels)


X = X/255.0

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

X_train = X_train[..., tf.newaxis].astype("float32")
X_test = X_test[..., tf.newaxis].astype("float32")


train_dataset = tf.data.Dataset.from_tensor_slices(
    (X_train, Y_train)).shuffle(10000).batch(32)

test_dataset = tf.data.Dataset.from_tensor_slices(
    (X_test, Y_test)).batch(32)


#model
class Modelo(Model):
    def __init__(self):
        super(Modelo, self).__init__()
        self.conv1 = Conv2D(32, 3, activation='relu')
        self.flatten = Flatten()
        self.d1 = Dense(128, activation='relu')
        self.d2 = Dense(10)

    def call(self, x):
        x = self.conv1(x)
        x = self.flatten(x)
        x = self.d1(x)
        return self.d2(x)
    
model = Modelo()

loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
optimizer = tf.keras.optimizers.Adam()

train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

test_loss = tf.keras.metrics.Mean(name='test_loss')
test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')

@tf.function
def train_step(images, labels):
    with tf.GradientTape() as tape:
        predictions = model(images, training=True)
        loss = loss_object(labels, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    train_loss(loss)
    train_accuracy(labels, predictions)


@tf.function
def test_step(images, labels):
    predictions = model(images, training=False)
    t_loss = loss_object(labels, predictions)

    test_loss(t_loss)
    test_accuracy(labels, predictions)

EPOCHS = 5

for epoch in range(EPOCHS):
    train_loss.reset_states()
    train_accuracy.reset_states()
    test_loss.reset_states()
    test_accuracy.reset_states()

    for images, labels in train_dataset:
        train_step(images, labels)
    
    for test_images, test_labels in test_dataset:
        test_step(test_images, test_labels)

    print(
        f'Epoch {epoch+1},'
        f'Loss: {train_loss.result()},'
        f'Accuracy: {train_accuracy.result() * 100},'
        f'Test Loss: {test_loss.result()},'
        f'Test Accuracy: {test_accuracy.result() * 100}'
    )
end_time = time.time()
training_time = end_time - start_time
print(f'Demorou {training_time} para finalizar')

"""
---------Modelo usando somente TensorFlow------------------
mnist = tf.keras.datasets.mnistw
(X_train, Y_train),(X_test, Y_test) = mnist.load_data()
X_train, X_test = X_train / 255.0, X_test / 255.0
len é 60000

# Add a channels dimension
X_train = X_train[..., tf.newaxis].astype("float32")
X_test = X_test[..., tf.newaxis].astype("float32")

train_dataset = tf.data.Dataset.from_tensor_slices(
    (X_train, Y_train)).shuffle(10000).batch(32)

test_dataset = tf.data.Dataset.from_tensor_slices(
    (X_test, Y_test)).batch(32)

                            
#model
class Modelo(Model):
    def __init__(self):
        super(Modelo, self).__init__()
        self.conv1 = Conv2D(32, 3, activation='relu')
        self.flatten = Flatten()
        self.d1 = Dense(128, activation='relu')
        self.d2 = Dense(10)

    def call(self, x):
        x = self.conv1(x)
        x = self.flatten(x)
        x = self.d1(x)
        return self.d2(x)
    
model = Modelo()

loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
optimizer = tf.keras.optimizers.Adam()

train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

test_loss = tf.keras.metrics.Mean(name='test_loss')
test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')

@tf.function
def train_step(images, labels):
    with tf.GradientTape() as tape:
        predictions = model(images, training=True)
        loss = loss_object(labels, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    train_loss(loss)
    train_accuracy(labels, predictions)


@tf.function
def test_step(images, labels):
    predictions = model(images, training=False)
    t_loss = loss_object(labels, predictions)

    test_loss(t_loss)
    test_accuracy(labels, predictions)

EPOCHS = 5

for epoch in range(EPOCHS):
    train_loss.reset_states()
    train_accuracy.reset_states()
    test_loss.reset_states()
    test_accuracy.reset_states()

    for images, labels in train_dataset:
        train_step(images, labels)
    
    for test_images, test_labels in test_dataset:
        test_step(test_images, test_labels)

    print(
        f'Epoch {epoch+1},'
        f'Loss: {train_loss.result()},'
        f'Accuracy: {train_accuracy.result() * 100},'
        f'Test Loss: {test_loss.result()},'
        f'Test Accuracy: {test_accuracy.result() * 100}'
    )"""