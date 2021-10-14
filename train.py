import pathlib
import helpers
import argparse
import tensorflow as tf
import tensorflow_addons as tfa
from datetime import datetime
from tensorflow.keras.optimizers import Adam
from DataGenerator import DataGenerator
from LSNet import LSNetBN, TinyLSNet, TinierLSNet, WingLoss, RootMeanSquaredErrorWrapper


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--name', type=str, default='LSNet', help='name of base model to train / predict')
parser.add_argument('--model', type=str, default='LSNetBN', help='type of base model to train / predict')
parser.add_argument('--n_epochs', type=int, default=5, help='number of training epochs')
parser.add_argument('--batch_size', type=int, default=8, help='batch size')
parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
parser.add_argument('--augment', action='store_true', default=False, help='Apply augmentations to training data')
parser.add_argument('--use_640x480', action='store_true', default=False, help='Use images of size 640x480 for training')
opt = parser.parse_args()

helpers.test_gpu()

dataPath = 'Data/'
batch_size = opt.batch_size

if opt.use_640x480:
    coco_path_train = dataPath + 'train-640x480.json'
    coco_path_val = dataPath + 'val-640x480.json'
    images_path = dataPath + 'sized_data-640x480/'
    input_size = (480, 640, 3)
else:
    coco_path_train = dataPath + 'train.json'
    coco_path_val = dataPath + 'val.json'
    images_path = dataPath + 'sized_data/'
    input_size = (512, 512, 3)

train_gen = DataGenerator(coco_path=coco_path_train, images_path=images_path, batch_size=batch_size,
                          input_size=input_size, cell_size=(32, 32), shuffle=True, augment=opt.augment)
val_gen = DataGenerator(coco_path=coco_path_val, images_path=images_path, batch_size=batch_size, input_size=input_size,
                        cell_size=(32, 32), shuffle=True)

model = LSNetBN(input_size=input_size)
if opt.model == 'LSNetBN':
    print('Training LSNet with batch norm')
elif opt.model == 'TinyFasterLSNet':
    print('Training TinyFasterLSNet')
    model = TinyLSNet(input_size=input_size)
elif opt.model == 'TinierFasterLSNet':
    print('Training TinierFasterLSNet')
    model = TinierLSNet(input_size=input_size)

losses = {
    'classifier': tfa.losses.SigmoidFocalCrossEntropy(reduction=tf.keras.losses.Reduction.AUTO),
    'regressor': WingLoss(reduction=tf.keras.losses.Reduction.AUTO)
}
loss_weights = {
    'classifier': 100,
    'regressor': 1
}
metrics = {
    'classifier': 'accuracy',
    'regressor': RootMeanSquaredErrorWrapper()
}

model.compile(optimizer=Adam(lr=opt.lr), loss=losses, metrics=metrics, loss_weights=loss_weights)

# Callbacks
logdir = str(pathlib.Path('logs', datetime.now().strftime("%Y%m%d-%H%M%S")))
tensorboard_callback = tf.keras.callbacks.TensorBoard(logdir, histogram_freq=0, update_freq='batch')

weightsPath = 'weights'
tm = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=weightsPath + '/' + opt.name + '-lr' + str(opt.lr) + '-' + tm + '-best.hdf5',
    save_weights_only=True,
    monitor='val_loss',
    mode='min',
    save_best_only=True)

early_stopping_callback = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    min_delta=0.01,
    patience=4,
    verbose=1,
    mode='min',
    restore_best_weights=True)

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    min_delta=0.01,
    factor=0.1,
    patience=2,
    mode='min',
    verbose=1)

model.fit(train_gen(),
          epochs=opt.n_epochs,
          steps_per_epoch=(train_gen.num_images // batch_size),
          validation_data=val_gen(),
          validation_steps=(val_gen.num_images // batch_size),
          callbacks=[tensorboard_callback, model_checkpoint_callback, reduce_lr])

tm = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
model.save_weights(pathlib.Path(weightsPath, opt.name + '-lr' + str(opt.lr) + '_' + tm + '-last.hdf5'))
