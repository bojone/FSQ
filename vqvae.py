#! -*- coding: utf-8 -*-
# Keras简单实现VQ-VAE

import os
import cv2
import numpy as np
import tensorflow as tf
from keras import backend as K
from keras.models import Model
from keras.layers import *
from keras.callbacks import Callback
from keras.initializers import VarianceScaling
from keras_preprocessing.image import list_pictures
from bert4keras.layers import ScaleOffset
from bert4keras.optimizers import Adam
from bert4keras.optimizers import extend_with_layer_adaptation
import warnings

warnings.filterwarnings('ignore')  # 忽略keras带来的满屏警告

if not os.path.exists('samples'):
    os.mkdir('samples')

# 基本配置
imgs = list_pictures('/mnt/vepfs/sujianlin/CelebA-HQ/train/', 'png')
imgs += list_pictures('/mnt/vepfs/sujianlin/CelebA-HQ/valid/', 'png')
np.random.shuffle(imgs)
img_size = 128
batch_size = 64
embedding_size = 128
num_layers = 6
min_pixel = 16

# 超参数选择
num_codes = 1296


def imread(f, crop_size=None):
    """读取图片
    """
    x = cv2.imread(f)
    height, width = x.shape[:2]
    if crop_size is None:
        crop_size = min([height, width])
    else:
        crop_size = min([crop_size, height, width])
    height_x = (height - crop_size + 1) // 2
    width_x = (width - crop_size + 1) // 2
    x = x[height_x:height_x + crop_size, width_x:width_x + crop_size]
    if x.shape[:2] != (img_size, img_size):
        x = cv2.resize(x, (img_size, img_size))
    x = x.astype('float32')
    x = x / 255 * 2 - 1
    return x


def imwrite(path, figure):
    """归一化到了[-1, 1]的图片矩阵保存为图片
    """
    figure = (figure + 1) / 2 * 255
    figure = np.round(figure, 0).astype('uint8')
    cv2.imwrite(path, figure)


def data_generator():
    """图片读取
    """
    batch_imgs = []
    while True:
        for i in np.random.permutation(len(imgs)):
            batch_imgs.append(imread(imgs[i]))
            if len(batch_imgs) == batch_size:
                batch_imgs = np.array(batch_imgs)
                yield batch_imgs, batch_imgs
                batch_imgs = []


class GroupNorm(ScaleOffset):
    """定义GroupNorm，默认groups=32
    """
    def call(self, inputs):
        inputs = K.reshape(inputs, (-1, 32), -1)
        mean, variance = tf.nn.moments(inputs, axes=[1, 2, 3], keepdims=True)
        inputs = (inputs - mean) * tf.rsqrt(variance + 1e-6)
        inputs = K.flatten(inputs, -2)
        return super(GroupNorm, self).call(inputs)


class VectorQuantizer(Layer):
    """量化层
    """
    def __init__(self, num_codes, **kwargs):
        super(VectorQuantizer, self).__init__(**kwargs)
        self.num_codes = num_codes

    def build(self, input_shape):
        super(VectorQuantizer, self).build(input_shape)
        self.embeddings = self.add_weight(
            name='embeddings',
            shape=(self.num_codes, input_shape[-1]),
            initializer='lecun_normal'
        )

    def call(self, inputs):
        x1 = K.sum(inputs**2, -1, keepdims=True)
        x2 = K.sum(self.embeddings**2, -1)
        x3 = tf.einsum('bmnd,kd->bmnk', inputs, self.embeddings)
        D = x1 + x2 - 2 * x3
        codes = K.cast(K.argmin(D, -1), 'int32')
        code_vecs = K.gather(self.embeddings, codes)
        return [codes, code_vecs]

    def compute_output_shape(self, input_shape):
        return [input_shape[:-1], input_shape]


def dense(x, out_dim, activation=None, init_scale=1):
    """Dense包装
    """
    init_scale = max(init_scale, 1e-10)
    initializer = VarianceScaling(init_scale, 'fan_avg', 'uniform')
    return Dense(
        out_dim,
        activation=activation,
        use_bias=False,
        kernel_initializer=initializer
    )(x)


def conv2d(x, out_dim, activation=None, init_scale=1):
    """Conv2D包装
    """
    init_scale = max(init_scale, 1e-10)
    initializer = VarianceScaling(init_scale, 'fan_avg', 'uniform')
    return Conv2D(
        out_dim, (3, 3),
        padding='same',
        activation=activation,
        use_bias=False,
        kernel_initializer=initializer
    )(x)


def residual_block(x, out_dim):
    """残差block
    """
    in_dim = K.int_shape(x)[-1]
    if in_dim == out_dim:
        xi = x
    else:
        xi = dense(x, out_dim)
    x = conv2d(x, out_dim, 'swish', 0.1)
    x = conv2d(x, out_dim, 'swish', 0.1)
    x = Add()([x, xi])
    x = GroupNorm()(x)
    return x


def vq(x):
    return VectorQuantizer(num_codes)(x)[1]


def l2_loss(y_true, y_pred):
    """用l2距离为损失，不能用mse代替
    """
    return K.sum((y_true - y_pred)**2, axis=[1, 2, 3])


# 编码器
x_in = x = Input(shape=(img_size, img_size, 3))
x = conv2d(x, embedding_size)

skip_pooling = 0
for i in range(num_layers):
    x = residual_block(x, embedding_size)
    if min(K.int_shape(x)[1:3]) > min_pixel:
        x = AveragePooling2D((2, 2))(x)
    else:
        skip_pooling += 1

x = residual_block(x, embedding_size)
xq = vq(x)
encoder = Model(inputs=x_in, outputs=[x, xq])
encoder.summary()

# 解码器
x_in = x = Input(shape=K.int_shape(x)[1:])
x = conv2d(x, embedding_size)
x = residual_block(x, embedding_size)

for i in range(num_layers):
    if i >= skip_pooling:
        x = UpSampling2D((2, 2))(x)
    x = residual_block(x, embedding_size)

x = GroupNorm()(x)
x = conv2d(x, 3, 'tanh')
decoder = Model(inputs=x_in, outputs=x)
decoder.summary()

## 训练器
x_in = x = Input(shape=(img_size, img_size, 3))
xe, xq = encoder(x)
x = Lambda(lambda x: x[0] + K.stop_gradient(x[1] - x[0]))([xe, xq])
x = decoder(x)
train_model = Model(inputs=x_in, outputs=x)

x_loss = l2_loss(x_in, x)
q_loss = l2_loss(K.stop_gradient(xe), xq)
e_loss = l2_loss(K.stop_gradient(xq), xe)
train_model.add_loss(q_loss + 0.25 * e_loss)
train_model.add_metric(x_loss, name='x_loss')
train_model.add_metric(q_loss, name='q_loss')
train_model.add_metric(e_loss, name='e_loss')
OPT = extend_with_layer_adaptation(Adam)
optimizer = OPT(
    learning_rate=2e-3,
    exclude_from_layer_adaptation=['Norm', 'bias'],
)
train_model.compile(loss=l2_loss, optimizer=optimizer)
train_model.summary()


def sample_ae_1(path, n=8):
    """重构采样函数（连续编码）
    """
    figure = np.zeros((img_size * n, img_size * n, 3))
    for i in range(n):
        for j in range(n):
            if j % 2 == 0:
                x_sample = [imread(np.random.choice(imgs))]
            else:
                z_sample = encoder.predict(np.array(x_sample))[0]
                x_sample = decoder.predict(z_sample)
            digit = x_sample[0]
            figure[i * img_size:(i + 1) * img_size,
                   j * img_size:(j + 1) * img_size] = digit
    imwrite(path, figure)


def sample_ae_2(path, n=8):
    """重构采样函数（连续编码）
    """
    figure = np.zeros((img_size * n, img_size * n, 3))
    for i in range(n):
        for j in range(n):
            if j % 2 == 0:
                x_sample = [imread(np.random.choice(imgs))]
            else:
                z_sample = encoder.predict(np.array(x_sample))[1]
                x_sample = decoder.predict(z_sample)
            digit = x_sample[0]
            figure[i * img_size:(i + 1) * img_size,
                   j * img_size:(j + 1) * img_size] = digit
    imwrite(path, figure)


class Trainer(Callback):
    def __init__(self):
        self.batch = 0
        self.n_size = 9
        self.iters_per_sample = 100

    def on_batch_end(self, batch, logs=None):
        if self.batch % self.iters_per_sample == 0:
            sample_ae_1('samples/test_ae_1_%s.png' % self.batch)
            sample_ae_2('samples/test_ae_2_%s.png' % self.batch)
            train_model.save_weights('./train_model.weights')
        self.batch += 1


if __name__ == '__main__':

    trainer = Trainer()
    train_model.fit_generator(
        data_generator(),
        steps_per_epoch=1000,
        epochs=1000,
        callbacks=[trainer]
    )
