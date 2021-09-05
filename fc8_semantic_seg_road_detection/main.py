import tensorflow as tf
#import tensorflow.compat.v1 as tf
#tf.disable_v2_behavior() 
import warnings
import helper
tf.compat.v1.reset_default_graph()
tf.compat.v1.disable_eager_execution()
#tunable params
IMAGE_SHAPE = (160,576)
NUMBER_OF_CLASSES = 2
EPOCHS = 42
BATCH_SIZE = 16
DROPOUT = 0.75

#directory paths
data_dir = './data'
runs_dir = './runs'
training_dir = './data/data_road/training'
vgg_path = './data/vgg'

#placeholder tensors
correct_label = tf.compat.v1.placeholder(tf.float32, [None, IMAGE_SHAPE[0], IMAGE_SHAPE[1], NUMBER_OF_CLASSES])
learning_rate = tf.compat.v1.placeholder(tf.float32)
keep_prob = tf.compat.v1.placeholder(tf.float32)

#functions

def load_vgg(sess, vgg_path):
  model = tf.compat.v1.saved_model.loader.load(sess, ['vgg16'], vgg_path)    #loading model and weights
  #tensors to be returned from graph
  graph = tf.compat.v1.get_default_graph()
  image_input = graph.get_tensor_by_name('image_input:0')
  keep_prob = graph.get_tensor_by_name('keep_prob:0')
  layer3 = graph.get_tensor_by_name('layer3_out:0')
  layer4 = graph.get_tensor_by_name('layer4_out:0')
  layer7 = graph.get_tensor_by_name('layer7_out:0')
  return image_input, keep_prob, layer3, layer4, layer7

def layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, NUMBER_OF_CLASSES):
  layer3, layer4, layer7 = vgg_layer3_out, vgg_layer4_out, vgg_layer7_out #just renaming
  #apply 1x1 convolution in place of fully connected layer
  fcn8 = tf.compat.v1.layers.conv2d(layer7, filters=NUMBER_OF_CLASSES, kernel_size=1, name="fcn8")
  #upsample fcn8 to match size of layer 4 so that we can add skip connection with 4rth layer
  #print(layer4.get_shape().as_list()[-1])
  fcn9 = tf.compat.v1.layers.conv2d_transpose(fcn8, filters=layer4.get_shape().as_list()[-1], kernel_size=4, strides=(2,2), padding='SAME', name='fcn9')
  #adding skip conn between current final layer fcn8 and 4rth layer
  fcn9_skip_connected = tf.add(fcn9, layer4, name='fcn_plus_vgg_layer4')
  #upscale
  fcn10 = tf.compat.v1.layers.conv2d_transpose(fcn9_skip_connected, filters=layer3.get_shape().as_list()[-1],kernel_size=4, strides=(2,2), padding='SAME', name='fcn10_conv2d')
  #adding skip connection
  fcn10_skip_connected = tf.add(fcn10, layer3, name='fcn10_plus_vgg_layer3')
  #upsample again
  fcn11 = tf.compat.v1.layers.conv2d_transpose(fcn10_skip_connected, filters=NUMBER_OF_CLASSES, kernel_size=16, strides=(8,8), padding='SAME', name='fcn11')
  return fcn11

def optimize(nn_last_layer, correct_label, learning_rate, NUMBER_OF_CLASSES):
  #reshape 4D tensors to 2D, each row represents a pixel, each column a class
  logits = tf.reshape(nn_last_layer, (-1, NUMBER_OF_CLASSES), name='fcn_logits')
  correct_label_reshaped = tf.reshape(correct_label, (-1, NUMBER_OF_CLASSES))
  #calculate distance from actual labels using cross_entropy
  cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=correct_label_reshaped[:])
  #take mean for total loss
  loss_op = tf.reduce_mean(input_tensor=cross_entropy, name='fcn_loss')
  #the model implements this operation to find the weights/parameters that would yield correct pixel labels
  train_op = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss_op, name='fcn_train_op')
  return logits,train_op,loss_op

def train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_image, correct_label, keep_prob, learning_rate):
  keep_prob_value = 0.5
  learning_rate_value = 0.001
  for epoch in range(epochs):
    #create function to get batches
    total_loss = 0
    for X_batch, gt_batch in get_batches_fn(batch_size):
      loss, _ = sess.run([cross_entropy_loss, train_op], feed_dict={input_image: X_batch, correct_label: gt_batch, keep_prob: keep_prob_value, learning_rate: learning_rate_value})
      total_loss += loss
    print("EPOCH {} \n".format(epoch + 1))
    print("loss = {:.3f}\n".format(total_loss))
def run():
  #download pretrained vgg model
  helper.maybe_download_pretrained_vgg(data_dir)
  #function to get batches
  get_batches_fn = helper.gen_batch_function(training_dir, IMAGE_SHAPE)
  with tf.compat.v1.Session() as session:
    image_input, keep_prob, layer3, layer4, layer7 = load_vgg(session, vgg_path) # from vgg architecture
    model_output = layers(layer3, layer4, layer7, NUMBER_OF_CLASSES)
    #returns the out logits, training operation and cost operation
    #logits: each row represents a pixel, each column a class
    #train_op: function used to get the right parameters to the model to correctly label the pixels
    logits, train_op, cross_entropy_loss = optimize(model_output, correct_label, learning_rate, NUMBER_OF_CLASSES)
    #init all vars
    session.run(tf.compat.v1.global_variables_initializer()) 
    session.run(tf.compat.v1.local_variables_initializer())
    print("Model build successful, starting training")
    #train nn
    train_nn(session, EPOCHS, BATCH_SIZE, get_batches_fn, train_op, cross_entropy_loss, image_input, correct_label, keep_prob, learning_rate)
    #next, run with test images (road pained green)
    helper.save_inference_samples(runs_dir, data_dir, session, IMAGE_SHAPE, logits, keep_prob, image_input)
  
if __name__ == '__main__':
  run()