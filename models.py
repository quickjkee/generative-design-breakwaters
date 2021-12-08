from keras.models import load_model
import keras.backend as K
import numpy as np

def custom_metric(y_true, y_pred):
  point = [127, 106]

  difference = K.abs((y_true - y_pred) / y_true)
  error = difference[:, point[0], point[1]]

  return K.mean(error)


def w_loss(y_true,y_pred):
    difference = K.square(y_true - y_pred)
    add = difference[:, 127, 106] * 5
    out = K.mean(add)
    out1 = K.mean(difference)
    #difference = difference * w_matrix

    return out + out1


def get_model():
    path_to_model = 'C:/Users/nano_user/swan_star/MODELS/ae_653_examples_new/ae_653_examples_new'
    my_model = load_model(path_to_model)

    return my_model

def get_classifier_model():
    path_to_model = 'C:/Users/nano_user/swan_star/MODELS/classifier_653_kakoito_pizdec/classifier_653_kakoito_pizdec'
    my_model = load_model(path_to_model)

    return my_model