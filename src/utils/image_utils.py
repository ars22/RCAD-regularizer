import tensorflow as tf 
import numpy as np

def zoom(x, label):
  """Zoom augmentation

  Args:
      x: Image

  Returns:
      Augmented image
  """

  # Generate 20 crop settings, ranging from a 1% to 20% crop.
  scales = list(np.arange(0.8, 1.0, 0.01))
  boxes = np.zeros((len(scales), 4))

  for i, scale in enumerate(scales):
    x1 = y1 = 0.5 - (0.5 * scale)
    x2 = y2 = 0.5 + (0.5 * scale)
    boxes[i] = [x1, y1, x2, y2]

  def random_crop(img):
    # Create different crops for an image
    crops = tf.image.crop_and_resize([img], boxes=boxes, box_indices=np.zeros(len(scales)), crop_size=(64, 64))
    # Return a random crop
    return crops[tf.random.uniform(shape=[], minval=0, maxval=len(scales), dtype=tf.int32)]


  choice = tf.random.uniform(shape=[], minval=0., maxval=1., dtype=tf.float32)

  # Only apply cropping 50% of the time
  return tf.cond(choice < 0.5, lambda: x, lambda: random_crop(x)), label


def rotate(x, label):
    """Rotation augmentation

    Args:
        x: Image

    Returns:
        Augmented image
    """

    # Rotate 0, 90, 180, 270 degrees
    return tf.image.rot90(x, tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32)), label



def color(x, label):
    """Color augmentation

    Args:
        x: Image

    Returns:
        Augmented image
    """
    x = tf.image.random_hue(x, 0.08)
    x = tf.image.random_saturation(x, 0.6, 1.6)
    x = tf.image.random_brightness(x, 0.05)
    x = tf.image.random_contrast(x, 0.7, 1.3)
    return x, label


def flip(x, label):
    """Flip augmentation

    Args:
        x: Image to flip

    Returns:
        Augmented image
    """
    x = tf.image.random_flip_left_right(x)
    x = tf.image.random_flip_up_down(x)

    return x, label