# 循环神经网络
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, Activation, Flatten, Embedding
from keras.utils import np_utils
from tensorflow.keras.optimizers import Adam
import numpy as np

input_size = 32
time_steps = 32
cell_size = 50

# 载入数据
(train_x, train_y), (test_x, test_y) = cifar10.load_data()


def rgb2grey(rgb):
    return np.dot(rgb[..., :3], [0.299, 0.587, 0.144])


train_x = rgb2grey(train_x)/255.0
test_x = rgb2grey(test_x)/255.0
train_y = np_utils.to_categorical(train_y, 10)
test_y = np_utils.to_categorical(test_y, 10)
train_dataset = tf.data.Dataset.from_tensor_slices((train_x, train_y))
train_dataset = train_dataset.shuffle(1000).batch(128)
test_dataset = tf.data.Dataset.from_tensor_slices((test_x, test_y))
test_dataset = test_dataset.batch(128)

# 创建模型
model = Sequential()
model.add(SimpleRNN(units=cell_size,
                    input_shape=train_x.shape[1:], unroll=True))
model.add(Dense(10, activation='softmax'))

model.summary()
model.compile(optimizer=Adam(lr=1e-3),
              loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(train_dataset, epochs=1000)
loss, accuracy = model.evaluate(test_dataset)
print('test loss', loss)
print('test accuracy', accuracy)
