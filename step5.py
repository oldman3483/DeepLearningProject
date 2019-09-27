
#=============================================================================
from PIL import Image
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import model
from input_data import get_files
 

def get_one_image(train):

    n = len(train)
    ind = np.random.randint(0, n)
    img_dir = train[ind]   
 
    img = Image.open(img_dir)
    plt.imshow(img)
    imag = img.resize([64, 64])  
    image = np.array(imag)
    return image
 
#--------------------------------------------------------------------
def evaluate_one_image(image_array):
    with tf.Graph().as_default():
       BATCH_SIZE = 1
       N_CLASSES = 4
 
       image = tf.cast(image_array, tf.float32)
       image = tf.image.per_image_standardization(image)
       image = tf.reshape(image, [1, 64, 64, 3])
 
       logit = model.inference(image, BATCH_SIZE, N_CLASSES)
 
       logit = tf.nn.softmax(logit)
 
       x = tf.placeholder(tf.float32, shape=[64, 64, 3])
 

       logs_train_dir = 'C:/zx/train/input_dog/'
 
       saver = tf.train.Saver()
 
       with tf.Session() as sess:
 
           print("Reading checkpoints...")
           ckpt = tf.train.get_checkpoint_state(logs_train_dir)
           if ckpt and ckpt.model_checkpoint_path:
               global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
               saver.restore(sess, ckpt.model_checkpoint_path)
               print('Loading success, global_step is %s' % global_step)
           else:
               print('No checkpoint file found')
 
           prediction = sess.run(logit, feed_dict={x: image_array})
           max_index = np.argmax(prediction)
           if max_index==0:
               print('This is a husky with possibility %.6f' %prediction[:, 0])
           elif max_index==1:
               print('This is a jiwawa with possibility %.6f' %prediction[:, 1])
           elif max_index==2:
               print('This is a poodle with possibility %.6f' %prediction[:, 2])
           else:
               print('This is a english sheep dog with possibility %.6f' %prediction[:, 3])
 
               
if __name__ == '__main__':
    
    train_dir = 'C:/zx/train/input_dog/'
    train, train_label, val, val_label = get_files(train_dir, 0.3)
    img = get_one_image(train)  
    evaluate_one_image(img)
