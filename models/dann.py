# import tensorflow as tf
from utils.utils import *


@tf.custom_gradient
def flip_grad_layer(x, l):
    def grad(dy):
        return tf.negative(dy) * l, None

    return tf.identity(x), grad


class DANN(object):
    """Simple MNIST domain adaptation model."""

    def __init__(self, batch_size=64, num_class=5, img_rows=64,img_cols=64):
        self._build_model(batch_size, num_class, img_rows,img_cols)

    def _build_model(self, batch_size, num_class, img_rows,img_cols):
        self.X = tf.placeholder(tf.float32, [None, img_rows, img_cols, 3])
        self.y = tf.placeholder(tf.float32, [None, num_class])
        self.domain = tf.placeholder(tf.float32, [None, 2])
        self.l = tf.placeholder(tf.float32, [])
        self.train = tf.placeholder(tf.bool, [])

        # CNN model for feature extraction
        with tf.variable_scope('feature_extractor'):
            W_conv0 = weight_variable([5, 5, 3, 32])
            b_conv0 = bias_variable([32])
            h_conv0 = tf.nn.relu(conv2d(self.X, W_conv0) + b_conv0)
            h_pool0 = max_pool_2x2(h_conv0)

            W_conv1 = weight_variable([5, 5, 32, 64])
            b_conv1 = bias_variable([64])
            h_conv1 = tf.nn.relu(conv2d(h_pool0, W_conv1) + b_conv1)
            h_pool1 = max_pool_2x2(h_conv1)

            # The domain-invariant feature
            self.feature = tf.reshape(h_pool1, [-1, img_rows * img_cols * 4])

        # MLP for class prediction
        with tf.variable_scope('label_predictor'):
            # Switches to route target examples (second half of batch) differently
            # depending on train or test mode.
            all_features = lambda: self.feature
            source_features = lambda: tf.slice(self.feature, [0, 0], [batch_size // 2, -1])
            classify_feats = tf.cond(self.train, source_features, all_features)

            all_labels = lambda: self.y
            source_labels = lambda: tf.slice(self.y, [0, 0], [batch_size // 2, -1])
            self.classify_labels = tf.cond(self.train, source_labels, all_labels)

            W_fc0 = weight_variable([img_rows * img_cols * 4, 100])
            b_fc0 = bias_variable([100])
            h_fc0 = tf.nn.relu(tf.matmul(classify_feats, W_fc0) + b_fc0)

            W_fc2 = weight_variable([100, num_class])
            b_fc2 = bias_variable([num_class])
            logits = tf.matmul(h_fc0, W_fc2) + b_fc2

            self.pred = tf.nn.softmax(logits)
            self.pred_loss = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=self.classify_labels)

        # Small MLP for domain prediction with adversarial loss
        with tf.variable_scope('domain_predictor'):
            # Flip the gradient when backpropagating through this operation
            feat = flip_grad_layer(self.feature, self.l)

            d_W_fc0 = weight_variable([img_rows * img_cols * 4, 100])
            d_b_fc0 = bias_variable([100])
            d_h_fc0 = tf.nn.relu(tf.matmul(feat, d_W_fc0) + d_b_fc0)

            d_W_fc1 = weight_variable([100, 2])
            d_b_fc1 = bias_variable([2])
            d_logits = tf.matmul(d_h_fc0, d_W_fc1) + d_b_fc1

            self.domain_pred = tf.nn.softmax(d_logits)
            self.domain_loss = tf.nn.softmax_cross_entropy_with_logits(logits=d_logits, labels=self.domain)
