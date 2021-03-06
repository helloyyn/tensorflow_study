from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets
import tensorflow as tf

#read data
mnist = read_data_sets("MNIST_data", one_hot = True)

#def function
def ful_conn_layer(serial_num, in_layer, out_layer_node_num, activ_func=None):#Before calling this function, remenber to reshape your tensor.
    layer = 'FULLY_CONNECTED_LAYER%s' % serial_num
    with tf.name_scope(layer):
        w = tf.Variable(tf.random_uniform([in_layer.shape.as_list()[1], out_layer_node_num], -0.1, 0.1), dtype=tf.float32, name='weights')
        tf.summary.histogram('weight%s' % serial_num, w)
        b = tf.Variable(tf.zeros([1, out_layer_node_num]), dtype=tf.float32, name='biases')
        tf.summary.histogram('biases%s' % serial_num, b)
        out_layer = tf.add(tf.matmul(in_layer, w), b, name='outputs')
        tf.summary.histogram('output%s' % serial_num, out_layer)
        if activ_func is not None:
            out_layer = activ_func(out_layer)
    return out_layer

def conv_layer(serial_num, in_layer, filter_shape, activ_func=None):#Before calling this function, remenber to reshape your tensor.
    layer = 'CONVOLUTION_LAYER%s' % serial_num
    with tf.name_scope(layer):
        w = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name='weights')
        tf.summary.histogram('weight%s' % serial_num, w)
        b = tf.Variable(tf.constant(0.1, shape=filter_shape[3:]), name='biases')
        tf.summary.histogram('biases%s' % serial_num, b)
        out_layer = tf.add(tf.nn.conv2d(in_layer, w, strides=[1, 1, 1, 1], padding='SAME'), b, name='outputs')
        tf.summary.histogram('output%s' % serial_num, out_layer)
        if activ_func is not None:
            out_layer = activ_func(out_layer)
    return out_layer

def max_pool_layer(serial_num, in_layer):
    layer = 'CONVOLUTION_LAYER%s' % serial_num
    with tf.name_scope(layer):
        out_layer = tf.nn.max_pool(in_layer, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='outputs')
        tf.summary.histogram('output%s' % serial_num, out_layer)
    return out_layer
    

dirr = '/home/yoyo/Desktop/yoyo/log'
with tf.Graph().as_default():

    sess = tf.Session()
    
    #network
    x = tf.placeholder(tf.float32, [None, 784])
    x_bacup = tf.reshape(x, [-1, 28, 28, 1])
    layer1 = conv_layer('1', x_bacup, [5, 5, 1, 10], tf.nn.relu)
    layer2 = max_pool_layer('2', layer1)
    layer2_bacup = tf.reshape(layer2, [-1, 14*14*10])
    f = ful_conn_layer('3', layer2_bacup, 200, tf.nn.relu)
    keep_prob = tf.placeholder('float')
    f = tf.nn.dropout(f, keep_prob)
    y = ful_conn_layer('4', f, 10, tf.nn.softmax)
    
    #cost and accuracy
    y_ = tf.placeholder(tf.float32, [None, 10])
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_*tf.log(y)))
    tf.summary.scalar('loss', cross_entropy)
    train = tf.train.AdamOptimizer(0.0001).minimize(cross_entropy)
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.summary.scalar('accuracy', accuracy)
    
    if tf.gfile.Exists(dirr):
        tf.gfile.DeleteRecursively(dirr)
    tf.gfile.MakeDirs(dirr)  
    summary_op = tf.summary.merge_all()
    summary_writer = tf.summary.FileWriter(dirr, sess.graph)    
    
    #train
    
    
    
    init = tf.initialize_all_variables()
    sess.run(init)
    train_step = 10000
    batch_size = 128
    for i in range(train_step):
        batch = mnist.train.next_batch(batch_size)
        sess.run(train, feed_dict = {x:batch[0], y_:batch[1], keep_prob:0.8})
        if i%100 == 0:
            los = sess.run(cross_entropy, feed_dict = {x:batch[0], y_:batch[1], keep_prob:1.0})
            acc_result = sess.run([summary_op, accuracy], feed_dict = {x: mnist.test.images, y_: mnist.test.labels, keep_prob:1.0})
            summary_str = acc_result[0]
            acc = acc_result[1]
            print('Accuracy is: '+str(acc)+' and loss is '+str(los))
            summary_writer.add_summary(summary_str, i)
            summary_writer.flush()            
