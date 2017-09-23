# coding: utf-8
import os
import pandas as pd
import tensorflow as tf
import numpy as np

from utils import *

# Parameters

DATA_PATH = "/media/user001/SSD_Samsung_NTFS/images"
assert os.path.exists(DATA_PATH)

BATCH_SIZE = 100
BATCH_SIZE_TEST = 256
n_epochs = 1000
img_size = 180
img_ch = 3

freq_dev_eval=10000
freq_train_verbose=10

dev_size=20000


# Category mapping
category_names = pd.read_csv(os.path.join(os.path.split(DATA_PATH)[0], "category_names.csv"))

cat2code=dict(zip(category_names.category_id, category_names.index))
code2cat={v:k for k,v in cat2code.items()}

n_classes=len(np.unique(list(cat2code.values())))


# Load JL
jl = list(map(json.loads, codecs.open(os.path.join(DATA_PATH, "index.jl"), "r", "utf-8").readlines()))


#Randomize JL
random.seed(655321)
random.shuffle(jl)


# Split JL in Train and Dev
jl_train = jl[:-dev_size]
jl_dev = jl[-dev_size:]


# Helper functions
def conv2d(x, n_filters, filters_size=[3,3], strides=[1,1], stddev=0.01, name="conv2d"):
    with tf.variable_scope(name):
        w = tf.get_variable('w', filters_size + [x.get_shape()[-1], n_filters],
                            initializer=tf.truncated_normal_initializer(stddev=stddev))
        
        conv = tf.nn.conv2d(x, w, strides=[1] + strides + [1], padding='SAME')

        biases = tf.get_variable('biases', [n_filters], initializer=tf.constant_initializer(0.0))
        conv = tf.reshape(tf.nn.bias_add(conv, biases), [-1]+conv.get_shape().as_list()[1:])
        return(conv)

def conv_relu_block(x, n_filters, is_train=None, bn=False, n_convolutions=2, filters_size=[3,3], name="conv_block"):
    with tf.variable_scope(name):
        for i in range(n_convolutions-1):
            if bn:
                x = tf.contrib.layers.batch_norm(x, 
                                                 center=True, scale=True, 
                                                 is_training=is_train,
                                                 scope=name+'_bn_'+str(i),
                                                 updates_collections=None)
            x = conv2d(x=x, n_filters=n_filters, filters_size=filters_size, name=name+"_conv2d_"+str(i))
            x = tf.nn.relu(features=x, name=name+"_relu_"+str(i))
        pool = tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding="SAME", name = name+"_maxpooling")
    return(pool)


def dense(x, n_layers, is_train=None, relu=True, bn=False, stddev=0.01, bias_start=0.0, name="dense"):
    with tf.variable_scope(name):
        if bn:
            x = tf.contrib.layers.batch_norm(x, 
               	                         center=True, scale=True, 
                       	                 is_training=is_train,
                               	         scope=name+'_bn',
                                         updates_collections=None)   
        w = tf.get_variable("w", [x.get_shape().as_list()[1], n_layers], tf.float32, 
                            tf.random_normal_initializer(stddev=stddev))
        b = tf.get_variable("b", [n_layers],
                            initializer=tf.constant_initializer(bias_start))
        output = tf.matmul(x,  w) + b
        if relu:
            output = tf.nn.relu(features=output, name=name+"_relu")
        return(output)


# Architecture definition
tf.reset_default_graph()

x = tf.placeholder(dtype=tf.float32, shape=[None, img_size, img_size, img_ch], name="input")
y = tf.placeholder(dtype=tf.int64, shape=[None], name="target")
is_train = tf.placeholder(tf.bool, name='phase')

conv_1 = conv_relu_block(x, n_filters=64, n_convolutions=2, bn=True, is_train=is_train, name="conv_block_1")

conv_2 = conv_relu_block(conv_1, n_filters=128, n_convolutions=2, bn=True, is_train=is_train, name="conv_block_2")

conv_3 = conv_relu_block(conv_2, n_filters=256, n_convolutions=3, bn=True, is_train=is_train, name="conv_block_3")

conv_4 = conv_relu_block(conv_3, n_filters=512, n_convolutions=3, bn=True, is_train=is_train, name="conv_block_4")

flattened = tf.reshape(conv_4, shape=[-1, np.prod(conv_4.get_shape().as_list()[1:])], name="flattened")

dense_1 = dense(flattened, n_layers=4096, bn=True, is_train=is_train, relu=True, name="dense_1")
dense_2 = dense(dense_1, n_layers=4096, bn=True, is_train=is_train, relu=True, name="dense_2")
logits = dense(dense_2, n_layers=n_classes, relu=False, name="logits")


# Metrics and loss
loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels = y, logits=logits)
tf.identity(loss, name='loss')
predictions = tf.argmax(logits, 1, name='predictions')
accuracy = tf.reduce_mean(tf.cast(tf.equal(predictions, y), tf.float32, name='accuracy'))


# Optimizer
opt = tf.train.AdamOptimizer(learning_rate=0.0001)
train_op = opt.minimize(loss)


# Summaries
train_summaries = []
train_histogram_probes = {"x_train": x, "y_train": y, "mxpool_1_train": conv_1, "mxpool_2_train": conv_2, 
                           "mxpool_3_train": conv_3, "mxpool_4_train": conv_4, "dense_1": dense_1,
                           "dense_2": dense_2, "logits": logits, "y_pred": predictions}
train_scalar_probes = {"loss_train": tf.reduce_mean(loss), "acc_train": accuracy}
dev_scalar_probes = {"loss_dev": tf.reduce_mean(loss), "acc_dev": accuracy}

train_summaries = [tf.summary.histogram(k, v) for k, v in train_histogram_probes.items()]
train_summaries += [tf.summary.scalar(k, v) for k, v in train_scalar_probes.items()]
dev_summaries = [tf.summary.scalar(k, v) for k, v in dev_scalar_probes.items()]

train_summaries = tf.summary.merge(train_summaries)
dev_summaries = tf.summary.merge(dev_summaries)



sess = tf.Session()
summary_writer = tf.summary.FileWriter("./logs", graph_def=sess.graph_def)

init = tf.global_variables_initializer()
sess.run(init)


# Training loop
for epoch in range(n_epochs):
    train_batch_generator = batch_generator(jl=jl_train, data_path=DATA_PATH, batch_size=BATCH_SIZE, 
                                            random_seed=655321+epoch)
    
    for batch, (batch_id, batch_x, batch_y) in enumerate(train_batch_generator):
        _, loss_train, acc_train, summary_train = sess.run([train_op, loss, accuracy, train_summaries],
                                                                feed_dict={x: batch_x, y: list(map(cat2code.get, batch_y)), is_train:True})
        summary_writer.add_summary(summary_train, epoch*batch+batch)
        if batch % freq_train_verbose == 0:
            print("[E{0:02d}|B{1:06d}] Train Loss: {2:.3f} Train Accuracy: {3:.3f}".format(epoch, batch, np.mean(loss_train), acc_train))
        
        if batch % freq_dev_eval == 0:
            loss_dev=acc_dev=0
            dev_batch_generator = batch_generator(jl=jl_dev, data_path=DATA_PATH, batch_size=BATCH_SIZE_TEST)
            for batch_dev, (batch_id, batch_x, batch_y) in enumerate(dev_batch_generator):
                loss_dev_aux, acc_dev_aux = sess.run([loss, accuracy],
                        feed_dict={x: batch_x, y: list(map(cat2code.get, batch_y)), is_train:False})
                loss_dev+=np.mean(loss_dev_aux)
                acc_dev+=np.mean(acc_dev_aux)
            loss_dev=loss_dev/(batch_dev+1)
            acc_dev=acc_dev/(batch_dev+1)
            print("*** [E{0:02d}|B{1:06d}] DEV Loss: {2:.3f} | DEV Accuracy: {3:.3f} ***".format(epoch, batch, 
                                                                                               loss_dev, 
                                                                                               acc_dev))


# Store model
saver = tf.train.Saver()
saver.save(sess, "output/model_V1")


# Predict test
jl_test = list(map(json.loads, codecs.open(os.path.join(DATA_PATH+"_test", "index.jl"), "r", "utf-8").readlines()))
test_batch_generator = batch_generator(jl=jl_test, data_path=DATA_PATH+"_test", batch_size=BATCH_SIZE_TEST, 	                                   random_seed=655321)

test_ids = []
test_preds = []

for batch, (batch_id, batch_x, batch_y) in enumerate(test_batch_generator):
    print(batch)
    preds = sess.run(predictions, feed_dict={x: batch_x, is_train:False})
    preds_decoded = list(map(code2cat.get, preds))
    test_ids.extend(batch_id)
    test_preds.extend(preds_decoded)

assert len(test_ids) == len(test_preds)

df_results = pd.DataFrame({"_id": test_ids, "prediction": test_preds})
df_results.to_csv("output/test_prediction.csv", sep=";", index=False)

