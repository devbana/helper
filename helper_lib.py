
import tensorflow_hub as hub
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import layers
import datetime

INPUT_SHAPE = (224, 224)
BATCH = 32

def create_tensorborad_callback(dirname, experiment_name):
  log_dir = dirname + "/" + experiment_name + "/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
  tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir)
  print(f'Saving Tensorboard log files to {log_dir}')
  return tensorboard_callback

def create_model(model_url, no_of_classes):
  feature_layer = hub.KerasLayer(model_url, 
                 trainable=False, 
                 name='feature_extraction_layer',
                 input_shape=INPUT_SHAPE + (3,))
  trained_model = tf.keras.Sequential([
                       feature_layer,
                       layers.Dense(no_of_classes, activation='softmax', name='output_layer')
  ])
  return trained_model

def plot_loss_curve(history):
  """
  Returns seperate losss curves for training and validation metrics

  Args:
  history: Tensorflow history object

  Returns:
  plots of training and Validation loss and accuracy
  """
  loss = history.history['loss']
  val_loss = history.history['val_loss']

  acc = history.history['acc']
  val_acc = history.history['val_acc']

  epochs = range(len(history.history['loss']))
  plt.figure()
  plt.plot(epochs, loss, label='Training_loss')
  plt.plot(epochs, val_loss, label='val_loss')
  plt.title(label='Loss')
  plt.xlabel(xlabel='epochs')
  plt.legend()
  plt.figure()
  plt.plot(epochs, acc, label='Training_Accuracy')
  plt.plot(epochs, val_acc, label='val_acc')
  plt.title(label='Accuracy')
  plt.xlabel(xlabel='epochs')
  plt.legend()

def view_random_image(target_dir, target_class):
  target_folder = target_dir + target_class
  random_image = random.sample(os.listdir(target_folder), 1)
  img = mpimg.imread(target_folder + "/" + random_image[0])
  plt.imshow(img)
  plt.suptitle(target_class)
  plt.axis('off')
  print(f'Image Shape : {img.shape}')
  return img

def unzip_data(name_):
  import zipfile
  zip_ref = zipfile.ZipFile(name_)
  zip_ref.extractall()
  zip_ref.close()
  
def list_dir(name_):
  import os
  for dir_path, dirnames, filenames in os.walk('10_food_classes_10_percent'):
    print(f'There are {len(dirnames)} directories and {len(filenames)} images in {dir_path}')



