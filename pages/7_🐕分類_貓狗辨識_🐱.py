import streamlit as st
from skimage import io, transform
import numpy as np
import tensorflow as tf
import keras

class CustomLayer(keras.layers.Layer):
    def __init__(self, sublayer, **kwargs):
        super().__init__(**kwargs)
        self.sublayer = sublayer

    def call(self, x):
        return self.sublayer(x)

    def get_config(self):
        base_config = super().get_config()
        config = {
            "sublayer": keras.saving.serialize_keras_object(self.sublayer),
        }
        return {**base_config, **config}

    @classmethod
    def from_config(cls, config):
        sublayer_config = config.pop("sublayer")
        sublayer = keras.saving.deserialize_keras_object(sublayer_config)
        return cls(sublayer, **config)

def custom_fn(x):
    return x**2

# Load the model with the custom layer and custom function
model = tf.keras.models.load_model('./model/cats_and_dogs_new_2.h5', custom_objects={"CustomLayer": CustomLayer, "custom_fn": custom_fn})

# Rest of your code...

# Use it in your code
custom_optimizer_instance = CustomRMSprop()

# Rest of your code...

