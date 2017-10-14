# coding: utf-8
import os
import pandas as pd
import tensorflow as tf
import numpy as np
import bson
import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"

from utils import *

# Parameters

DATA_PATH = "/media/user001/SSD_Samsung_NTFS/images"
assert os.path.exists(DATA_PATH)

BATCH_SIZE = 90
BATCH_SIZE_TEST = 128
n_epochs = 1000
keep_prob_train = 1.0

img_size = 176
img_ch = 3

freq_dev_eval = 5000
freq_train_eval = 100

dev_size = 20000

version_alias = "res_0.1"


# Category mapping
category_names = pd.read_csv(os.path.join(os.path.split(DATA_PATH)[0], "category_names.csv"))
cat2code=dict(zip(category_names.category_id, category_names.index))
code2cat={v:k for k,v in cat2code.items()}
n_classes=len(np.unique(list(cat2code.values())))


# Load data
data = bson.decode_file_iter(open(os.path.join(os.path.split(DATA_PATH)[0], "train.bson"), 'rb'))
data = list(data)

data_test = bson.decode_file_iter(open(os.path.join(os.path.split(DATA_PATH)[0], "test.bson"), 'rb'))
data_test = list(data_test)

#Randomize data
random.seed(655321)
random.shuffle(data)


# Split data in Train and Dev
data_train = data[:-dev_size]
data_dev = data[-dev_size:]


# Helper functions
def conv2d(x, n_filters, filters_size=[3,3], strides=[1,1], stddev=0.01, name="conv2d"):
    with tf.variable_scope(name):
        w = tf.get_variable('w', filters_size + [x.get_shape()[-1], n_filters],
                            initializer=tf.truncated_normal_initializer(stddev=stddev))
        
        conv = tf.nn.conv2d(x, w, strides=[1] + strides + [1], padding='SAME')

        biases = tf.get_variable('biases', [n_filters], initializer=tf.constant_initializer(0.0))
        conv = tf.reshape(tf.nn.bias_add(conv, biases), [-1]+conv.get_shape().as_list()[1:])
        return(conv)

def conv_relu_block(x, n_filters, is_train=None, keep_prob=None, bn=False, n_convolutions=2, 
                    filters_size=[3,3], pool=True, name="conv_block"):
    with tf.variable_scope(name):
        for i in range(n_convolutions):
            if bn:
                x = tf.contrib.layers.batch_norm(x, 
                                                 center=True, scale=True, 
                                                 is_training=is_train,
                                                 scope=name+'_bn_'+str(i),
                                                 updates_collections=None)
            x = conv2d(x=x, n_filters=n_filters, filters_size=filters_size, name=name+"_conv2d_"+str(i))
            x = tf.nn.relu(features=x, name=name+"_relu_"+str(i))
            if keep_prob is not None:
                x = tf.nn.dropout(x, keep_prob=keep_prob, name=name+"_dropout")

        if pool:    
            pool = tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding="SAME", name = name+"_maxpooling")
            return(pool)
        else:
            return(x)


def residual_block(x, n_filters, is_train=None, keep_prob=None, filters_size=[3,3], pool=False, name="residual_block"):
    h = x
    h = tf.contrib.layers.batch_norm(h, 
                                     center=True, scale=True, 
                                     is_training=is_train,
                                     scope=name+"_bn_1",
                                     updates_collections=None)
    with tf.variable_scope(name):
        for i in range(2):
            h = conv2d(x=h, n_filters=n_filters, filters_size=filters_size, name=name+"_conv2d_"+str(i))
            h = tf.contrib.layers.batch_norm(h, 
                                             center=True, scale=True, 
                                             is_training=is_train,
                                             scope=name+'_bn_2_'+str(i),
                                             updates_collections=None)
            if i == 0 :
                h = tf.nn.relu(features=h, name=name+"_relu_1_"+str(i))
        if keep_prob is not None:
            h = tf.nn.dropout(h, keep_prob=keep_prob, name=name+"_dropout")
        h = tf.add(x, h, name=name+"_merge_add")
        h = tf.nn.relu(features=h, name=name+"_relu_2_"+str(i))
        if pool:
            pool = tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding="SAME", name = name+"_maxpooling")
            return(pool) 
    return(h)

def dense(x, n_layers, is_train=None, keep_prob=None, relu=True, bn=False, stddev=0.01, bias_start=0.0, name="dense"):
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
        if keep_prob is not None:
            output = tf.nn.dropout(output, keep_prob=keep_prob)

        return(output)


# Architecture definition

tf.reset_default_graph()

x = tf.placeholder(dtype=tf.float32, shape=[None, img_size, img_size, img_ch], name="input")
y = tf.placeholder(dtype=tf.int64, shape=[None], name="target")
is_train = tf.placeholder(tf.bool, name='phase')
keep_prob = None #tf.placeholder(tf.float32, name="keep_prob")

conv_1 = conv_relu_block(x, n_filters=64, n_convolutions=1, bn=True, filters_size=[9,9], 
                         is_train=is_train, keep_prob=keep_prob, pool=False, name="conv_block_1")
res_1_1 = residual_block(conv_1, n_filters=64,  is_train=is_train, keep_prob=keep_prob, pool=False, name="residual_block_1_1")
res_1_2 = residual_block(res_1_1, n_filters=64,  is_train=is_train, keep_prob=keep_prob, pool=True, name="residual_block_1_2")

conv_2 = conv_relu_block(res_1_2, n_filters=128, n_convolutions=1, bn=True, is_train=is_train, keep_prob=keep_prob, pool=False, name="conv_block_2")
res_2_1 = residual_block(conv_2, n_filters=128,  is_train=is_train, keep_prob=keep_prob, pool=False, name="residual_block_2_1")
res_2_2 = residual_block(res_2_1, n_filters=128,  is_train=is_train, keep_prob=keep_prob, pool=True, name="residual_block_2_2")

conv_3 = conv_relu_block(res_2_2, n_filters=256, n_convolutions=2, bn=True, is_train=is_train, keep_prob=keep_prob, pool=False, name="conv_block_3")
res_3_1 = residual_block(conv_3, n_filters=256,  is_train=is_train, keep_prob=keep_prob, pool=False, name="residual_block_3_1")
res_3_2 = residual_block(res_3_1, n_filters=256,  is_train=is_train, keep_prob=keep_prob, pool=True, name="residual_block_3_2")

conv_4 = conv_relu_block(res_3_2, n_filters=512, n_convolutions=2, bn=True, is_train=is_train, keep_prob=keep_prob, pool=False, name="conv_block_4")
res_4_1 = residual_block(conv_4, n_filters=512,  is_train=is_train, keep_prob=keep_prob, pool=False, name="residual_block_4_1")
res_4_2 = residual_block(res_4_1, n_filters=512,  is_train=is_train, pool=True, keep_prob=keep_prob, name="residual_block_4_2")

conv_5 = conv_relu_block(res_4_2, n_filters=512, n_convolutions=2, bn=True, is_train=is_train, keep_prob=keep_prob, pool=False, name="conv_block_5")
res_5_1 = residual_block(conv_5, n_filters=512,  is_train=is_train, keep_prob=keep_prob, pool=False, name="residual_block_5_1")
res_5_2 = residual_block(res_5_1, n_filters=512,  is_train=is_train, keep_prob=keep_prob, pool=False, name="residual_block_5_2")
res_5_3 = residual_block(res_5_2, n_filters=512,  is_train=is_train, keep_prob=keep_prob, pool=True, name="residual_block_5_3")

flattened = tf.reshape(res_5_3, shape=[-1, np.prod(res_5_3.get_shape().as_list()[1:])], name="flattened")

dense_1 = dense(flattened, n_layers=2000, bn=True, is_train=is_train, keep_prob=keep_prob, relu=True, name="dense_1")
dense_2 = dense(dense_1, n_layers=2000, bn=True, is_train=is_train, keep_prob=keep_prob, relu=True, name="dense_2")
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
train_histogram_probes = {"x_train": x, 
                          "y_train": y, 
                          "y_pred": predictions,
                          "logits": logits}

train_scalar_probes = {"loss_train": tf.reduce_mean(loss), "acc_train": accuracy}


train_summaries = [tf.summary.histogram(k, v) for k, v in train_histogram_probes.items()]
train_summaries += [tf.summary.scalar(k, v) for k, v in train_scalar_probes.items()]
#dev_summaries = [tf.summary.scalar(k, v) for k, v in dev_scalar_probes.items()]

train_summaries = tf.summary.merge(train_summaries)
#dev_summaries = tf.summary.merge(dev_summaries)


#sess = tf.Session()
saver = tf.train.Saver()
sess = tf.Session()
# Restore variables from disk.
saver = tf.train.import_meta_graph('output_/model_res_0.1-1760000.meta')
saver.restore(sess, "output_/model_res_0.1-1760000")
print("Model restored.")


summary_writer = tf.summary.FileWriter("./logs_{}".format(version_alias), graph_def=sess.graph_def)

#sess.run(tf.global_variables_initializer())


# Training loop
epoch = 0
batch = 0
train_batch_generator = batch_generator(data=data_train, batch_size=BATCH_SIZE, random_seed=655321+epoch)


while 1: # Epochs loop
    for batch_id, batch_x, batch_y in train_batch_generator:
        if batch % freq_train_eval == 0:
            _, loss_train, acc_train, summary_train = sess.run([train_op, loss, accuracy, train_summaries],
                                        feed_dict={x: batch_x, y: list(map(cat2code.get, batch_y)), is_train:True})
            print("[E{0:02d}|B{1:06d}] Train Loss: {2:.3f} Train Accuracy: {3:.3f}".format(epoch, batch, np.mean(loss_train), acc_train))
        else:
            _, summary_train = sess.run([train_op, train_summaries], feed_dict={x: batch_x, y: list(map(cat2code.get, batch_y)), is_train:True})
        
        summary_writer.add_summary(summary_train, epoch*batch+batch)

        
        if batch % freq_dev_eval == 0:
            print("\nEVALUATING DEVELOPMENT SET PERFORMANCE, PLEASE WAIT...")
            loss_dev=acc_dev=0
            dev_batch_generator = batch_generator(data=data_dev, batch_size=BATCH_SIZE_TEST)
            for batch_dev, (batch_id, batch_x, batch_y) in enumerate(dev_batch_generator):
                loss_dev_aux, acc_dev_aux = sess.run([loss, accuracy],
                                                     feed_dict={x: batch_x, 
                                                                y: list(map(cat2code.get, batch_y)), 
                                                                is_train:False})
                loss_dev += np.mean(loss_dev_aux)
                acc_dev += np.mean(acc_dev_aux)
            loss_dev=loss_dev/(batch_dev+1)
            acc_dev=acc_dev/(batch_dev+1)
            
            summary_acc_dev = tf.Summary(value=[tf.Summary.Value(tag="acc_dev", simple_value=acc_dev)])
            summary_loss_dev = tf.Summary(value=[tf.Summary.Value(tag="loss_dev", simple_value=loss_dev)])            
            summary_writer.add_summary(summary_acc_dev, epoch*batch+batch)
            summary_writer.add_summary(summary_loss_dev, epoch*batch+batch)

            print("*** [E{0:02d}|B{1:06d}] DEV Loss: {2:.3f} | DEV Accuracy: {3:.3f} ***\n".format(epoch, batch, 
                                                                                               loss_dev, 
                                                                                               acc_dev))
            saver.save(sess, "output/model_{}".format(version_alias), global_step=epoch*BATCH_SIZE+batch)
        batch += 1
    epoch += 1
    train_batch_generator = batch_generator(data=data_train, batch_size=BATCH_SIZE, random_seed=655321+epoch)


# Predict test
#test_batch_generator = batch_generator(data=data_test, batch_size=BATCH_SIZE_TEST, random_seed=655321)
#
#test_ids, test_preds = [], []
#
#for batch, (batch_id, batch_x, batch_y) in enumerate(test_batch_generator):
#    print(batch)
#    preds = sess.run(predictions, feed_dict={x: batch_x, is_train:False, keep_prob:1.0})
#    preds_decoded = list(map(code2cat.get, preds))
#    test_ids.extend(batch_id)
#    test_preds.extend(preds_decoded)
#
#assert len(test_ids) == len(test_preds)
#
#df_results = pd.DataFrame({"_id": test_ids, "prediction": test_preds})
#df_results.to_csv("output/test_prediction_{}_{}.csv".format(version_alias, epoch*batch+batch), sep=";", index=False)
