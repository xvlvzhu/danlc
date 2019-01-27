import numpy as np
import tensorflow as tf

from models import DANN
from utils.utils import *

batch_size = 64
num_classes = 5
epochs = 10
num = 7000
# input image dimensions
img_rows, img_cols = 64, 64

source_x = np.load('./data/source_x_paper.npy')
target_x = np.load('./data/target_x_paper.npy')
source_y = np.load('./data/source_y_paper.npy')
target_y = np.load('./data/target_y_paper.npy')

indices = list(range(num))
np.random.seed(1)
np.random.shuffle(indices)
split1 = int(num * 0.8)
split2 = int(num * 0.9)
train_idx, valid_idx, test_idx = indices[:split1], indices[split1:split2], indices[split2:]

source_train_data = source_x[train_idx]
source_train_labels = source_y[train_idx]
source_eval_data = source_x[valid_idx]
source_eval_labels = source_y[valid_idx]
source_test_data = source_x[test_idx]
source_test_labels = source_y[test_idx]

target_train_data = target_x[train_idx]
target_train_labels = target_y[train_idx]
target_eval_data = target_x[valid_idx]
target_eval_labels = target_y[valid_idx]
target_test_data = target_x[test_idx]
target_test_labels = target_y[test_idx]

graph = tf.get_default_graph()
with graph.as_default():
    model = DANN(batch_size=batch_size, num_class=num_classes, img_cols=img_cols, img_rows=img_rows)

    learning_rate = tf.placeholder(tf.float32, [])

    pred = model.pred
    pred_loss = tf.reduce_mean(model.pred_loss)
    domain_loss = tf.reduce_mean(model.domain_loss)
    total_loss = pred_loss + domain_loss

    regular_train_op = tf.train.MomentumOptimizer(learning_rate, 0.9).minimize(pred_loss)
    dann_train_op = tf.train.MomentumOptimizer(learning_rate, 0.9).minimize(total_loss)

    # Evaluation
    correct_label_pred = tf.equal(tf.argmax(model.classify_labels, 1), tf.argmax(model.pred, 1))
    label_acc = tf.reduce_mean(tf.cast(correct_label_pred, tf.float32))
    correct_domain_pred = tf.equal(tf.argmax(model.domain, 1), tf.argmax(model.domain_pred, 1))
    domain_acc = tf.reduce_mean(tf.cast(correct_domain_pred, tf.float32))


def train_and_evaluate(training_mode, graph, model, num_steps=5000, verbose=True, channel_begin=0, channel_end=3):
    """Helper to run the model with different training modes."""

    with tf.Session(graph=graph) as sess:
        tf.global_variables_initializer().run()

        # Batch generators
        gen_source_batch = batch_generator(
            [source_train_data, source_train_labels], batch_size // 2)
        gen_target_batch = batch_generator(
            [target_train_data, target_train_labels], batch_size // 2)
        gen_source_only_batch = batch_generator(
            [source_train_data, source_train_labels], batch_size)
        gen_target_only_batch = batch_generator(
            [target_train_data, target_train_labels], batch_size)

        domain_labels = np.vstack([np.tile([1., 0.], [batch_size // 2, 1]),
                                   np.tile([0., 1.], [batch_size // 2, 1])])

        idx1 = channel_begin
        idx2 = channel_end

        # Training loop
        for i in range(num_steps):

            # Adaptation param and learning rate schedule as described in the paper
            p = float(i) / num_steps
            l = 2. / (1. + np.exp(-10. * p)) - 1
            lr = 0.01 / (1. + 10 * p) ** 0.75

            if training_mode == 'dann':

                X0, y0 = next(gen_source_batch)
                X1, y1 = next(gen_target_batch)
                X = np.vstack([X0[:, :, :, idx1:idx2], X1[:, :, :, idx1:idx2]])
                y = np.vstack([y0, y1])

                _, batch_loss, dloss, ploss, d_acc, p_acc = sess.run(
                    [dann_train_op, total_loss, domain_loss, pred_loss, domain_acc, label_acc],
                    feed_dict={model.X: X, model.y: y, model.domain: domain_labels,
                               model.train: True, model.l: l, learning_rate: lr})

                if verbose and i % 100 == 0:
                    print('loss: {}  d_acc: {}  p_acc: {}  p: {}  l: {}  lr: {}'.format(
                        batch_loss, d_acc, p_acc, p, l, lr))

            elif training_mode == 'source':
                X, y = next(gen_source_only_batch)
                X = X[:, :, :, idx1:idx2]
                _, batch_loss = sess.run([regular_train_op, pred_loss],
                                         feed_dict={model.X: X, model.y: y, model.train: False,
                                                    model.l: l, learning_rate: lr})
                if verbose and i % 100 == 0:
                    print('loss: {}'.format(batch_loss))

            elif training_mode == 'target':
                X, y = next(gen_target_only_batch)
                X = X[:, :, :, idx1:idx2]
                _, batch_loss = sess.run([regular_train_op, pred_loss],
                                         feed_dict={model.X: X, model.y: y, model.train: False,
                                                    model.l: l, learning_rate: lr})

        # Compute final evaluation on test data
        source_acc = sess.run(label_acc,
                              feed_dict={model.X: source_test_data[:, :, :, idx1:idx2], model.y: source_test_labels,
                                         model.train: False})

        target_acc = sess.run(label_acc,
                              feed_dict={model.X: target_test_data[:, :, :, idx1:idx2], model.y: target_test_labels,
                                         model.train: False})

        #         test_domain_acc = sess.run(domain_acc,
        #                                    feed_dict={model.X: combined_test_imgs[:,:,:,0:3],
        #                                               model.domain: combined_test_domain, model.l: 1.0})

        #         test_emb = sess.run(model.feature_0, feed_dict={model.X: combined_test_imgs[:,:,:,0:3], model.X1:combined_test_imgs[:,:,:,3:6]})

        res_s = []
        res_t = []
        for i in range(5):
            idx = source_test_labels[:, i] == 1

            pred_s = sess.run(pred, feed_dict={model.X: source_test_data[idx][:, :, :, idx1:idx2], model.train: False})
            res_s.append(pred_s)

            idx = target_test_labels[:, i] == 1

            pred_t = sess.run(pred, feed_dict={model.X: target_test_data[idx][:, :, :, idx1:idx2], model.train: False})
            res_t.append(pred_t)

    return source_acc, target_acc, res_s, res_t


# RGB channels
print('\nSource only training')
source_acc, target_acc, source_res_s, source_res_t = train_and_evaluate(
    'source', graph, model, channel_begin=0, channel_end=3)
print('Source accuracy:', source_acc)
print('Target  accuracy:', target_acc)

print('\nDomain adaptation training')
source_acc, target_acc, dann_res_s, dann_res_t = train_and_evaluate(
    'dann', graph, model, channel_begin=0, channel_end=3)
print('Source  accuracy:', source_acc)
print('Target  accuracy:', target_acc)

# Texture channels
graph = tf.get_default_graph()
with graph.as_default():
    model = DANN(batch_size=batch_size, num_class=num_classes, img_cols=img_cols, img_rows=img_rows)

    learning_rate = tf.placeholder(tf.float32, [])

    pred = model.pred
    pred_loss = tf.reduce_mean(model.pred_loss)
    domain_loss = tf.reduce_mean(model.domain_loss)
    total_loss = pred_loss + domain_loss

    regular_train_op = tf.train.MomentumOptimizer(learning_rate, 0.9).minimize(pred_loss)
    dann_train_op = tf.train.MomentumOptimizer(learning_rate, 0.9).minimize(total_loss)

    # Evaluation
    correct_label_pred = tf.equal(tf.argmax(model.classify_labels, 1), tf.argmax(model.pred, 1))
    label_acc = tf.reduce_mean(tf.cast(correct_label_pred, tf.float32))
    correct_domain_pred = tf.equal(tf.argmax(model.domain, 1), tf.argmax(model.domain_pred, 1))
    domain_acc = tf.reduce_mean(tf.cast(correct_domain_pred, tf.float32))

print('\nSource only training')
source_acc, target_acc, source_res_s_text, source_res_t_text = train_and_evaluate(
    'source', graph, model, channel_begin=3, channel_end=6)
print('Source accuracy:', source_acc)
print('Target  accuracy:', target_acc)

print('\nDomain adaptation training')
source_acc, target_acc, dann_res_s_text, dann_res_t_text = train_and_evaluate(
    'dann', graph, model, channel_begin=3, channel_end=6)
print('Source  accuracy:', source_acc)
print('Target  accuracy:', target_acc)

# Fusion
l1 = 0
l2 = 0
fusion = []
alpha = 0.4733
for i in range(5):
    pred_fusion = alpha * np.array(dann_res_t[i]) + (1 - alpha) * np.array(dann_res_t_text[i])
    pred_fusion = np.argmax(pred_fusion, axis=1)
    l1 = l1 + np.sum(pred_fusion == i)
    l2 = l2 + len(pred_fusion)
    pred_vector = np.zeros((pred_fusion.shape[0], num_classes))
    pred_vector[np.arange(pred_fusion.shape[0]), pred_fusion] = 1
    fusion.append(np.sum(pred_vector, axis=0))

confusion_matrix = fusion / np.sum(fusion, axis=1)
print("acc: ", l1/l2)
print(confusion_matrix)