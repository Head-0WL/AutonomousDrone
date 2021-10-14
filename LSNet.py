import math
import tensorflow as tf
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.metrics import *
from tensorflow.keras.losses import Loss
from tensorflow.python.keras.utils import losses_utils


def LSNetBN(pretrained_weights=None, input_size=(512, 512, 3)):
  # Feature extractor
  inputs = Input(input_size, name='main_input')
  conv1 = Conv2D(64, 3, padding='same')(inputs)
  conv1 = BatchNormalization()(conv1)
  conv1 = ReLU()(conv1)
  conv2 = Conv2D(64, 3, padding='same')(conv1)
  conv2 = BatchNormalization()(conv2)
  conv2 = ReLU()(conv2)
  conv3 = Conv2D(64, 3, strides=2, padding='same')(conv2)
  conv3 = BatchNormalization()(conv3)
  conv3 = ReLU()(conv3)
  conv4 = Conv2D(128, 3, padding='same')(conv3)
  conv4 = BatchNormalization()(conv4)
  conv4 = ReLU()(conv4)
  conv5 = Conv2D(128, 3, padding='same')(conv4)
  conv5 = BatchNormalization()(conv5)
  conv5 = ReLU()(conv5)
  conv6 = Conv2D(128, 3, strides=2, padding='same')(conv5)
  conv6 = BatchNormalization()(conv6)
  conv6 = ReLU()(conv6)
  conv7 = Conv2D(256, 3, padding='same')(conv6)
  conv7 = BatchNormalization()(conv7)
  conv7 = ReLU()(conv7)
  conv8 = Conv2D(256, 3, padding='same')(conv7)
  conv8 = BatchNormalization()(conv8)
  conv8 = ReLU()(conv8)
  conv9 = Conv2D(256, 3, strides=2, padding='same')(conv8)
  conv9 = BatchNormalization()(conv9)
  conv9 = ReLU()(conv9)
  conv10 = Conv2D(512, 3, padding='same')(conv9)
  conv10 = BatchNormalization()(conv10)
  conv10 = ReLU()(conv10)
  conv11 = Conv2D(512, 3, padding='same')(conv10)
  conv11 = BatchNormalization()(conv11)
  conv11 = ReLU()(conv11)
  conv12 = Conv2D(512, 3, strides=2, padding='same')(conv11)
  conv12 = BatchNormalization()(conv12)
  feat_out = ReLU()(conv12)

  # Classifier
  conv13 = Conv2D(512, 2, padding='valid')(feat_out)
  conv13 = BatchNormalization()(conv13)
  conv13 = ReLU()(conv13)
  conv14 = Conv2D(2, 1, padding='valid')(conv13)
  cls = Softmax()(conv14)

  # Regressor
  conv15 = Conv2D(512, 2, padding='valid')(feat_out)
  conv15 = ReLU()(conv15)
  reg = Conv2D(4, 1, padding='valid')(conv15)

  model = Model(inputs=inputs, outputs={'classifier': cls, 'regressor': reg})
  # model.summary()

  if pretrained_weights:
    model.load_weights(pretrained_weights)

  return model


def TinyLSNet(pretrained_weights=None, input_size=(512, 512, 3)):
    # Feature extractor
    inputs = Input(input_size, name='main_input')
    conv1 = Conv2D(64, 3, padding='same')(inputs)
    conv1 = BatchNormalization()(conv1)
    conv1 = ReLU()(conv1)
    conv3 = Conv2D(64, 3, strides=2, padding='same')(conv1)
    conv3 = BatchNormalization()(conv3)
    conv3 = ReLU()(conv3)
    conv4 = Conv2D(128, 3, padding='same')(conv3)
    conv4 = BatchNormalization()(conv4)
    conv4 = ReLU()(conv4)
    conv5 = Conv2D(64, 1, padding='same')(conv4) # 1x1
    conv5 = ReLU()(conv5)
    conv6 = Conv2D(128, 3, strides=2, padding='same')(conv5)
    conv6 = BatchNormalization()(conv6)
    conv6 = ReLU()(conv6)
    conv7 = Conv2D(256, 3, padding='same')(conv6)
    conv7 = BatchNormalization()(conv7)
    conv7 = ReLU()(conv7)
    conv8 = Conv2D(128, 1, padding='same')(conv7) # 1x1
    conv8 = ReLU()(conv8)
    conv9 = Conv2D(256, 3, strides=2, padding='same')(conv8)
    conv9 = BatchNormalization()(conv9)
    conv9 = ReLU()(conv9)
    conv10 = Conv2D(512, 3, padding='same')(conv9)
    conv10 = BatchNormalization()(conv10)
    conv10 = ReLU()(conv10)
    conv11 = Conv2D(256, 1, padding='same')(conv10) # 1x1
    conv11 = ReLU()(conv11)
    conv12 = Conv2D(512, 3, strides=2, padding='same')(conv11)
    conv12 = BatchNormalization()(conv12)
    feat_out = ReLU()(conv12)

    # Classifier
    conv13 = Conv2D(256, 1, padding='same')(feat_out) # 1x1
    conv13 = ReLU()(conv13)
    conv13 = Conv2D(512, 2, padding='valid')(conv13)
    conv13 = BatchNormalization()(conv13)
    conv13 = ReLU()(conv13)
    conv14 = Conv2D(2, 1, padding='valid')(conv13)
    cls = Softmax()(conv14)

    # Regressor
    conv15 = Conv2D(256, 1, padding='same')(feat_out) # 1x1
    conv15 = ReLU()(conv15)
    conv15 = Conv2D(512, 2, padding='valid')(conv15)
    conv15 = ReLU()(conv15)
    reg = Conv2D(4, 1, padding='valid')(conv15)

    model = Model(inputs=inputs, outputs={'classifier': cls, 'regressor': reg})
    #model.summary()

    if pretrained_weights:
        model.load_weights(pretrained_weights)

    return model

def TinierLSNet(pretrained_weights=None, input_size=(512, 512, 3), activation='relu'):
    act = ReLU()
    if activation == 'leaky_relu':
        act = LeakyReLU(alpha=0.1)
    elif activation == 'prelu':
        act = PReLU()

    # Feature extractor
    inputs = Input(input_size, name='main_input')
    conv1 = Conv2D(32, 3, padding='same')(inputs)
    conv1 = BatchNormalization()(conv1)
    conv1 = ReLU()(conv1)
    conv3 = Conv2D(32, 3, strides=2, padding='same')(conv1)
    conv3 = BatchNormalization()(conv3)
    conv3 = ReLU()(conv3)
    conv4 = Conv2D(64, 3, padding='same')(conv3)
    conv4 = BatchNormalization()(conv4)
    conv4 = ReLU()(conv4)
    conv5 = Conv2D(32, 1, padding='same')(conv4) # 1x1
    conv5 = ReLU()(conv5)
    conv6 = Conv2D(64, 3, strides=2, padding='same')(conv5)
    conv6 = BatchNormalization()(conv6)
    conv6 = ReLU()(conv6)
    conv7 = Conv2D(128, 3, padding='same')(conv6)
    conv7 = BatchNormalization()(conv7)
    conv7 = ReLU()(conv7)
    conv8 = Conv2D(64, 1, padding='same')(conv7) # 1x1
    conv8 = ReLU()(conv8)
    conv9 = Conv2D(128, 3, strides=2, padding='same')(conv8)
    conv9 = BatchNormalization()(conv9)
    conv9 = ReLU()(conv9)
    conv10 = Conv2D(256, 3, padding='same')(conv9)
    conv10 = BatchNormalization()(conv10)
    conv10 = ReLU()(conv10)
    conv11 = Conv2D(128, 1, padding='same')(conv10) # 1x1
    conv11 = ReLU()(conv11)
    conv12 = Conv2D(256, 3, strides=2, padding='same')(conv11)
    conv12 = BatchNormalization()(conv12)
    feat_out = ReLU()(conv12)

    # Classifier
    conv13 = Conv2D(128, 1, padding='same')(feat_out) # 1x1
    conv13 = ReLU()(conv13)
    conv13 = Conv2D(256, 2, padding='valid')(conv13)
    conv13 = BatchNormalization()(conv13)
    conv13 = ReLU()(conv13)
    conv14 = Conv2D(2, 1, padding='valid')(conv13)
    cls = Softmax()(conv14)

    # Regressor
    conv15 = Conv2D(128, 1, padding='same')(feat_out) # 1x1
    conv15 = ReLU()(conv15)
    conv15 = Conv2D(256, 2, padding='valid')(conv15)
    conv15 = ReLU()(conv15)
    reg = Conv2D(4, 1, padding='valid')(conv15)

    model = Model(inputs=inputs, outputs={'classifier': cls, 'regressor': reg})
    #model.summary()

    if pretrained_weights:
        model.load_weights(pretrained_weights)

    return model


class WingLoss(Loss):
    def __init__(self, reduction=tf.keras.losses.Reduction.NONE):
        super(WingLoss, self).__init__(reduction=reduction)

    def call(self, y_true, y_pred, w=10.0, epsilon=2.0):
        """
        :param y_true: Tensors with shape [batch_size, 31, 31, 5] where the first
                       column is 1 if the segment contains a wire and 0 otherwise.
                       This column is used as the Iverson bracket indicator function
                       since the loss should be 0 when there is no segment in the cell.
                       See section 3.3 for more info:
                       https://link.springer.com/content/pdf/10.1007/s00138-020-01138-6.pdf
        :param y_pred: Tensors with shape [batch_size, 31, 31, 4]
        :return: Wing loss
        """
        # Slice first column from y_true
        iver = tf.squeeze(tf.slice(y_true, [0, 0, 0, 0], [-1, -1, -1, 1]), axis=3)
        # iver = tf.squeeze(iverson, axis=3)
        # First coordinate of y_true
        y_true_first = tf.slice(y_true, [0, 0, 0, 1], [-1, -1, -1, 2])
        # Second coordinate of y_true}
        y_true_second = tf.slice(y_true, [0, 0, 0, 3], [-1, -1, -1, 2])
        # First coordinate of y_pred
        y_pred_first = tf.slice(y_pred, [0, 0, 0, 0], [-1, -1, -1, 2])
        # Second coordinate of y_pred
        y_pred_second = tf.slice(y_pred, [0, 0, 0, 2], [-1, -1, -1, 2])

        diff = tf.add(
            tf.reduce_sum(tf.abs(y_true_first - y_pred_first), axis=3),
            tf.reduce_sum(tf.abs(y_true_second - y_pred_second), axis=3))
        diff_swap = tf.add(
            tf.reduce_sum(tf.abs(y_true_first - y_pred_second), axis=3),
            tf.reduce_sum(tf.abs(y_true_second - y_pred_first), axis=3))

        d = tf.minimum(diff, diff_swap)  # Shape: [batch_size, 31, 31]

        C = w * (1.0 - math.log(1.0 + w / epsilon))
        losses = tf.where(
            tf.greater(w, d),
            w * tf.math.log(1.0 + d / epsilon),
            d - C)

        return iver * losses


# Wrapper class for RootMeanSquerdError to remove Iverson bracket indicator from
# y_true and return the error only in cells where there is a wire
class RootMeanSquaredErrorWrapper(RootMeanSquaredError):

    def __init__(self):
        super(RootMeanSquaredErrorWrapper, self).__init__()

    def update_state(self, y_true, y_pred, sample_weight=None):
        iverson = tf.slice(y_true, [0, 0, 0, 0], [-1, -1, -1, 1])
        y_true = tf.slice(y_true, [0, 0, 0, 1], [-1, -1, -1, -1])

        y_true = tf.cast(y_true, self._dtype)
        y_pred = tf.cast(y_pred, self._dtype)
        y_pred, y_true = losses_utils.squeeze_or_expand_dimensions(y_pred, y_true)
        error_sq = tf.math.squared_difference(y_pred, y_true)
        # Only return the squared error where the Iverson bracket indicator
        # function is not zero i.e. in cells that contain wires
        error_sq_where = tf.where(tf.greater(iverson, 0), error_sq, 0)

        return super(RootMeanSquaredError, self).update_state(error_sq_where, sample_weight=sample_weight)