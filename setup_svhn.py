## setup_mnist.py -- mnist data and model loading code
##
## Copyright (C) 2017, Lily Weng  <twweng@mit.edu>
##                and Huan Zhang <ecezhang@ucdavis.edu>
## Copyright (C) 2016, Nicholas Carlini <nicholas@carlini.com>.
##
## This program is licenced under the BSD 2-Clause licence,
## contained in the LICENCE file in this directory.

import tensorflow as tf
import numpy as np
import os
import pickle
import gzip
import urllib.request
import data_load

from tensorflow.contrib.keras.api.keras.models import load_model
from tensorflow.contrib.keras.api.keras import backend as K


class SVHN:
    def __init__(self):
        dataObject = data_load.get_appropriate_data("svhn")()
        self.attack_data, self.attack_labels = dataObject.get_attack_data()


class SVHNModel:
    def __init__(self, restore = None, session=None, use_softmax=False):
        self.num_channels = 3
        self.image_size = 32
        self.num_labels = 10

        # load model from memory
        model = load_model(restore)

        # output log probability, used for black-box attack
        if not use_softmax:
            model.layers.pop()

        layer_outputs = []
        for layer in model.layers:
            if isinstance(layer, Conv2D) or isinstance(layer, Dense):
                layer_outputs.append(K.function([model.layers[0].input], [layer.output]))

        self.model = model
        self.layer_outputs = layer_outputs

    def predict(self, data):
        return self.model(data)

