from tensorflow import keras
import matplotlib.pyplot as pit
import tensorflow as tf
import numpy as np

fashion_mnist = keras.datasets.fashion_mnist
(train_images,train_lables),(test_images,test_lables) = fashion_mnist.load_data()
print(train_images.shape)
train_images=train_images/255.0
test_images=test_images/255.0
def create_model():
    model = tf.keras.models.Squential([
        keras.layers.Flatten(input_shape=(28,28)),
        keras.layers.Dense(128,activation="relu"),
        keras.layers.Dense(28)
    ])
    model.compile(
        optimizer="adam",
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[tf.metrics.SparseCategoricalAccuracy()]
    )
    return model

new_model = create_model()
new_model.fit(train_images,train_lables,epochs=100)
new_model.save("model/mymodel3.h5")
