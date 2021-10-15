from keras.engine import training
import tensorflow as tf
import numpy as np
import pandas as pd
import time
import os
import imgaug as ia
from imgaug import augmenters as iaa
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.python.keras.callbacks import TerminateOnNaN
from src.utils.cyclic_lr import CosineLRScheduler
from src.utils import image_utils
from src.models.resnet18 import ResNet18
from src.models.densenet import DenseNet
import argparse
from tensorflow_addons.optimizers import SGDW, AdamW
import shutil

# gpu set mem greowth
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
  tf.config.experimental.set_memory_growth(gpu, True)

IMAGE_SIZE = 64
SEED_NUM = 42     
BATCH_SIZE_TRAIN = 128
BATCH_SIZE_VAL = 128
NUM_TRAIN_IMAGES = 100000
NUM_VAL_IMAGES = 10000
NUM_EPOCHS = 40
N_CLASSES = 200
CYCLE_LENGTH = 3
EPS = 1e-3



@tf.function
def accuracy(Y_hat, Y):
  return tf.reduce_mean(tf.cast(tf.argmax(Y_hat, axis=1) == tf.argmax(Y, axis=1), tf.float32))


@tf.function
def entropy(prob, from_logits=False):
  if from_logits:
    prob = tf.nn.softmax(prob, axis=-1)
  return tf.reduce_mean(-1.0 * tf.reduce_mean(prob * tf.math.log(prob + EPS), axis=1)) 

@tf.function
def KLUnif(prob, from_logits=False):
  if from_logits:
    prob = tf.nn.softmax(prob, axis=-1)
  return tf.keras.losses.CategoricalCrossentropy(from_logits=False)(y_true=tf.ones_like(prob) / N_CLASSES, y_pred=prob)



class Baseline(tf.keras.Model):

  def __init__(self, model, args):
    super().__init__()
    self.args = args
    self.model = model
    # metrics
    self.ce_loss_tracker = tf.keras.metrics.Mean(name="ce_loss")
    self.ent_tracker = tf.keras.metrics.Mean(name="ent")
    self.acc_tracker = tf.keras.metrics.Mean(name='accuracy')

    
  @tf.function
  def train_step(self, data):
    X, Y = data
    with tf.GradientTape() as tape:
      Y_logits = self.model(X, training=True) 
      ce_loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True)(y_true=Y, y_pred=Y_logits)
    acc = accuracy(Y_logits, Y) 
    ent = entropy(Y_logits, from_logits=True)
    # Compute gradients
    gradients = tape.gradient(ce_loss, self.model.trainable_variables)
    # Update weights
    self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
    # Metrics
    self.ce_loss_tracker.update_state(ce_loss)
    self.ent_tracker.update_state(ent)
    self.acc_tracker.update_state(acc)
    return {"ce_loss": self.ce_loss_tracker.result(), "ent": self.ent_tracker.result(), "accuracy": self.acc_tracker.result()}

  @tf.function
  def test_step(self, data):
    X, Y = data
    Y_logits = self.model(X, training=False)
    ent = entropy(Y_logits, from_logits=True)
    ce_loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True)(y_true=Y, y_pred=Y_logits)
    acc = accuracy(Y_logits, Y)
    self.ce_loss_tracker.update_state(ce_loss)
    self.ent_tracker.update_state(ent)
    self.acc_tracker.update_state(acc)
    return {"ce_loss": self.ce_loss_tracker.result(), "ent": self.ent_tracker.result(), "accuracy": self.acc_tracker.result()}

  def call(self, X):
    pass

  @property
  def metrics(self):
    # We list our `Metric` objects here so that `reset_states()` can be
    # called automatically at the start of each epoch
    # or at the start of `evaluate()`.
    # If you don't implement this property, you have to call
    # `reset_states()` yourself at the time of your choosing.
    return [self.acc_tracker, self.ce_loss_tracker, self.ent_tracker]




def reset_graph(seed):
    tf.compat.v1.reset_default_graph()
    tf.compat.v1.set_random_seed(seed)
    np.random.seed(seed)


def fetch_data():
  if not os.path.isdir('datasets/tiny-imagenet-200'):
    os.chdir('datasets')
    os.system('wget http://cs231n.stanford.edu/tiny-imagenet-200.zip')
    os.system('unzip -qq tiny-imagenet-200.zip')
    os.chdir('..')


def get_generators(batch_size_train, batch_size_val, no_aug=False, img_size=(64,64)):
  
  Sometimes = lambda aug: iaa.Sometimes(0.5, aug)

  seq = iaa.Sequential([
    iaa.Fliplr(0.25),
    iaa.Sometimes(0.25, iaa.Affine(rotate=(-25, 25))),
    iaa.Sometimes(0.2, iaa.Affine(translate_percent={'x': (-0.1, 0.1), 'y': (-0.1, 0.1)})),
    iaa.Sometimes(0.2, iaa.Crop(percent=(0, 0.125))),
    iaa.Sometimes(0.2, iaa.OneOf([
    iaa.CoarseDropout(0.2, size_percent=(0.05, 0.1)),
    iaa.CoarseSalt(0.2, size_percent=(0.05, 0.1)),
    iaa.CoarsePepper(0.2, size_percent=(0.05, 0.1)),
    iaa.CoarseSaltAndPepper(0.2, size_percent=(0.05, 0.1))]))
    ,iaa.Sometimes(0.3,iaa.GammaContrast((0.8, 1.3)))
    ,iaa.Sometimes(0.2,iaa.GaussianBlur(sigma=(0.4, 1.3)))])


  if no_aug:
    train_datagen = ImageDataGenerator(
        rescale= 1./255)
  else:
    train_datagen = ImageDataGenerator(
        preprocessing_function=seq.augment_image,
        rescale= 1./255)
  valid_datagen = ImageDataGenerator(rescale=1./255)

  train_generator = train_datagen.flow_from_directory(
    r'datasets/tiny-imagenet-200/train/', 
    target_size=img_size, 
    color_mode='rgb',
    batch_size=batch_size_train, 
    class_mode='categorical', 
    shuffle=True, 
    seed=SEED_NUM)
  val_data = pd.read_csv('datasets/tiny-imagenet-200/val/val_annotations.txt', 
      sep='\t', header=None, names=['File', 'Class', 'X', 'Y', 'H', 'W'])
  val_data.drop(['X', 'Y', 'H', 'W'], axis=1, inplace=True)
  validation_generator = valid_datagen.flow_from_dataframe(
      val_data, 
      directory=r'datasets/tiny-imagenet-200/val/images/', 
      x_col='File', y_col='Class', 
      target_size=img_size,
      color_mode='rgb', 
      class_mode='categorical', 
      batch_size=batch_size_val, 
      shuffle=True, 
      seed=SEED_NUM
  )
  return train_generator, validation_generator


class LearningRateLoggingCallback(tf.keras.callbacks.Callback):

    def on_epoch_begin(self, batch, logs=None):
        print(f"Learning Rate: {round(self.model.optimizer.lr.numpy(), 5)}")
        

def process_example_dict(example_dict):
    image, label = example_dict['image'], example_dict['label']
    image = tf.cast(image, tf.float32) / 255.
    return image, label


def project_image(x, label):
  return tf.clip_by_value(x, 0, 1), label


def make_ds(ds, aug=False):
    ds = ds.map(
      process_example_dict, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    if aug:
      augmentations = [image_utils.flip, image_utils.color, image_utils.zoom, image_utils.rotate]
      # Add the augmentations to the dataset
      for f in augmentations:
        ds = ds.map(f, num_parallel_calls=tf.data.experimental.AUTOTUNE)
      # Make sure that the values are still in [0, 1]
      ds = ds.map(project_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds = ds.cache()
    ds = ds.batch(128)
    ds = ds.prefetch(256)
    ds = ds.repeat()
    return ds


def run():

  parser = argparse.ArgumentParser()
  parser.add_argument('--base-checkpoint-path', type=str, required=True)
  
  args = parser.parse_args()
  assert args.base_checkpoint_path.strip() != ''
  args.base_checkpoint_path = 'runs/tiny/' + args.base_checkpoint_path
  if os.path.exists(args.base_checkpoint_path):
    shutil.rmtree(args.base_checkpoint_path)
  os.mkdir(args.base_checkpoint_path)
  
  reset_graph(seed=SEED_NUM)

  print('\n--- Creating Dataset ---')
  fetch_data()
  train_generator, validation_generator = get_generators(
    BATCH_SIZE_TRAIN, BATCH_SIZE_VAL, no_aug=False, img_size=(64, 64))
  
  print('\n--- Creating Model ---')
  model = DenseNet(N_CLASSES, softmax=False)
  tf.autograph.experimental.do_not_convert(model.build)
  model.build(input_shape = (None, 64, 64, 3)) 
  print(model.summary())
  
  optimizer = SGDW(momentum=0.9, weight_decay=0.0001, nesterov=True, lr=0.1)
  filepath = args.base_checkpoint_path + "/model_best_val.hdf5"
  model_checkpoint = ModelCheckpoint(
    filepath, monitor='val_accuracy', 
    verbose=2, save_best_only=True, mode='max', save_weights_only=True)
  clr_triangular = CosineLRScheduler(
                      min_lr=0.0001,
                      max_lr=0.05,
                      cycle_length=CYCLE_LENGTH)
  lr_log_callback = LearningRateLoggingCallback()
  tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="runs/tiny/baseline")

  print('\n--- Model Compilation and Training ---')
  start = time.time()
  tf.keras.backend.clear_session()
  trainer = Baseline(model=model, args=args)
  trainer.compile(optimizer=optimizer)
  trainer.fit(train_generator,
    steps_per_epoch=NUM_TRAIN_IMAGES // BATCH_SIZE_TRAIN,
    validation_data=validation_generator,
    validation_steps=NUM_VAL_IMAGES // BATCH_SIZE_VAL,
    epochs=NUM_EPOCHS, 
    workers=tf.data.experimental.AUTOTUNE,
    callbacks=[clr_triangular, lr_log_callback, model_checkpoint, tensorboard_callback],
    verbose=1)
  end = time.time()

  print("LR Range : ", min(clr_triangular.history['lr']), max(clr_triangular.history['lr']))
    

if __name__ == '__main__':
  run()