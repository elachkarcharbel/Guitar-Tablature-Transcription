from __future__ import print_function

import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Flatten, Reshape, Activation, Input, GlobalAveragePooling2D
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Conv1D, Lambda, concatenate, BatchNormalization, AveragePooling2D
from tensorflow.keras import backend as K
from DataGenerator_224 import DataGenerator
import pandas as pd
import numpy as np
import datetime
from Metrics import *
from functools import partial

import glob
AUTOTUNE = tf.data.experimental.AUTOTUNE


def read_tfrecord(example, labeled):
    tfrecord_format = (
        {
            "X": tf.io.FixedLenFeature([50176], tf.float32),
            "Y": tf.io.FixedLenFeature([126], tf.float32),
        }
        if labeled
        else {"X": tf.io.FixedLenFeature([50176], tf.float32),}
    )
    example = tf.io.parse_example(example, tfrecord_format)
    input = example["X"]
    input = tf.cast(input, tf.float32)
    input = tf.reshape(input, [224,224,1])

    if labeled:
        label = tf.cast(example["Y"], tf.float32)
        label = tf.reshape(label, [6,21])
        return input, label
    return input



def load_dataset(filenames, labeled=True):
    ignore_order = tf.data.Options()
    ignore_order.experimental_deterministic = False  # disable order, increase speed                                                                           
    dataset = tf.data.TFRecordDataset(
        filenames
    )  # automatically interleaves reads from multiple files                                                                                                   
    dataset = dataset.with_options(
        ignore_order
    )  # uses data as soon as it streams in, rather than in its original order                                                                                 
    dataset = dataset.map(partial(read_tfrecord, labeled=labeled), num_parallel_calls=AUTOTUNE)
    # returns a dataset of (image, label) pairs if labeled=True or just images if labeled=False                                                                
    return dataset


def get_dataset(filenames, batch,labeled=True):
    dataset = load_dataset(filenames, labeled=labeled).repeat()
    dataset = dataset.shuffle(batch)
    dataset = dataset.prefetch(buffer_size=AUTOTUNE)
    dataset = dataset.batch(batch)
    return dataset


def get_dataset_validation(filenames, batch,labeled=True):
    dataset = load_dataset(filenames, labeled=labeled)
    dataset = dataset.prefetch(buffer_size=AUTOTUNE)
    dataset = dataset.batch(batch)
    return dataset



class TabCNN:
    
    def __init__(self, 
                 batch_size=128, 
                 epochs=16,
                 con_win_size = 9,
                ## swin params ##
                patch_size = (2, 2),  # 2-by-2 sized patches
                dropout_rate = 0.03,  # Dropout rate
                num_heads = 8,  # Attention heads
                embed_dim = 64,  # Embedding dimension
                num_mlp = 256,  # MLP layer size
                qkv_bias = True,  # Convert embedded patches to query, key, and values with a learnable additive value
                window_size = 2,  # Size of attention window
                shift_size = 1,  # Size of shifting window
                image_dimension = 224,  # Initial image size
                learning_rate = 1e-3,
                validation_split = 0.1,
                weight_decay = 0.0001,
                label_smoothing = 0.1,
                ## swin params ##
                 spec_repr="c",
                 data_path="../data/spec_repr/",
                 id_file="id.csv",
                 save_path="../model/saved/"):   
        
        self.batch_size = batch_size
        self.epochs = epochs
        self.con_win_size = con_win_size
        self.spec_repr = spec_repr
        self.data_path = data_path
        self.id_file = id_file
        self.save_path = save_path
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate
        self.embed_dim = embed_dim
        self.num_mlp = num_mlp
        self.qkv_bias = qkv_bias
        self.window_size = window_size
        self.shift_size = shift_size
        self.image_dimension = image_dimension
        self.learning_rate = learning_rate
        self.validation_split = validation_split
        self.weight_decay = weight_decay
        self.label_smoothing = label_smoothing
        
        self.load_IDs()
        
        self.save_folder = self.save_path + self.spec_repr + "Swin_tfrec_224x224_" + datetime.datetime.now().strftime("%Y-%m-%d %H%M%S") + "/"
        if not os.path.exists(self.save_folder):
            os.makedirs(self.save_folder)
        self.log_file = self.save_folder + "log.txt"
        
        self.metrics = {}
        self.metrics["pp"] = []
        self.metrics["pr"] = []
        self.metrics["pf"] = []
        self.metrics["tp"] = []
        self.metrics["tr"] = []
        self.metrics["tf"] = []
        self.metrics["tdr"] = []
        self.metrics["acc"] = []
        self.metrics["data"] = ["g0","g1","g2","g3","g4","g5","mean","std dev"]
        
        if self.spec_repr == "c":
            self.input_shape = (224, 224, 1)
        self.num_classes = 21
        self.num_strings = 6

    
    def load_IDs(self):
        csv_file = self.data_path + self.id_file
        self.list_IDs = list(pd.read_csv(csv_file, header=None)[0])
        
    
    def partition_data(self, data_split):
        self.data_split = data_split
        self.partition = {}
        self.partition["training"] = []
        self.partition["validation"] = []
        for ID in self.list_IDs:
            guitarist = int(ID.split("_")[0])
            if guitarist == data_split:
                self.partition["validation"].append(ID)
            else:
                self.partition["training"].append(ID)
                
        file_list_train=glob.glob("../data/tfrecords_224x224/0[0-2|4]*.tfrecords")
        print("Training files",len(file_list_train))
        self.training_generator=get_dataset(file_list_train, self.batch_size)
        self.nb_files_train=len(file_list_train)
        
        file_list_val=glob.glob("../data/tfrecords_224x224/05*.tfrecords")
        print("Validation files",len(file_list_val))
        self.validation_generator=get_dataset_validation(file_list_val, len(file_list_val))
        
        self.split_folder = self.save_folder + str(self.data_split) + "/"
        if not os.path.exists(self.split_folder):
            os.makedirs(self.split_folder)


          
    def log_model(self):
        with open(self.log_file,'w') as fh:
            fh.write("\nbatch_size: " + str(self.batch_size))
            fh.write("\nepochs: " + str(self.epochs))
            fh.write("\nspec_repr: " + str(self.spec_repr))
            fh.write("\ndata_path: " + str(self.data_path))
            fh.write("\ncon_win_size: " + str(self.con_win_size))
            fh.write("\nid_file: " + str(self.id_file) + "\n")
            self.model.summary(print_fn=lambda x: fh.write(x + '\n'))
       
    
    def softmax_by_string(self, t):
        sh = K.shape(t)
        string_sm = []
        for i in range(self.num_strings):
            string_sm.append(K.expand_dims(K.softmax(t[:,i,:]), axis=1))
        return K.concatenate(string_sm, axis=1)
    
    
    def catcross_by_string(self, target, output):
        loss = 0
        for i in range(self.num_strings):
            loss += K.categorical_crossentropy(target[:,i,:], output[:,i,:])
        return loss
    
    
    def avg_acc(self, y_true, y_pred):
        return K.mean(K.equal(K.argmax(y_true, axis=-1), K.argmax(y_pred, axis=-1)))
           
    
    def build_model(self):
        
        self.num_patch_x = self.input_shape[0] // self.patch_size[0]
        self.num_patch_y = self.input_shape[1] // self.patch_size[1] 

        def window_partition(x, window_size):
            _, height, width, channels = x.shape
            patch_num_y = height // window_size
            patch_num_x = width // window_size
            x = tf.reshape(
                x, shape=(-1, patch_num_y, window_size, patch_num_x, window_size, channels)
            )
            x = tf.transpose(x, (0, 1, 3, 2, 4, 5))
            windows = tf.reshape(x, shape=(-1, window_size, window_size, channels))
            return windows


        def window_reverse(windows, window_size, height, width, channels):
            patch_num_y = height // window_size
            patch_num_x = width // window_size
            x = tf.reshape(
                windows,
                shape=(-1, patch_num_y, patch_num_x, window_size, window_size, channels),
            )
            x = tf.transpose(x, perm=(0, 1, 3, 2, 4, 5))
            x = tf.reshape(x, shape=(-1, height, width, channels))
            return x


        class DropPath(layers.Layer):
            def __init__(self, drop_prob=None, **kwargs):
                super(DropPath, self).__init__(**kwargs)
                self.drop_prob = drop_prob

            def call(self, x):
                input_shape = tf.shape(x)
                batch_size = input_shape[0]
                rank = x.shape.rank
                shape = (batch_size,) + (1,) * (rank - 1)
                random_tensor = (1 - self.drop_prob) + tf.random.uniform(shape, dtype=x.dtype)
                path_mask = tf.floor(random_tensor)
                output = tf.math.divide(x, 1 - self.drop_prob) * path_mask
                return output

        
        class WindowAttention(layers.Layer):
            def __init__(
                self, dim, window_size, num_heads, qkv_bias=True, dropout_rate=0.0, **kwargs
            ):
                super(WindowAttention, self).__init__(**kwargs)
                self.dim = dim
                self.window_size = window_size
                self.num_heads = num_heads
                self.scale = (dim // num_heads) ** -0.5
                self.qkv = layers.Dense(dim * 3, use_bias=qkv_bias)
                self.dropout = layers.Dropout(dropout_rate)
                self.proj = layers.Dense(dim)

            def build(self, input_shape):
                num_window_elements = (2 * self.window_size[0] - 1) * (
                    2 * self.window_size[1] - 1
                )
                self.relative_position_bias_table = self.add_weight(
                    shape=(num_window_elements, self.num_heads),
                    initializer=tf.initializers.Zeros(),
                    trainable=True,
                )
                coords_h = np.arange(self.window_size[0])
                coords_w = np.arange(self.window_size[1])
                coords_matrix = np.meshgrid(coords_h, coords_w, indexing="ij")
                coords = np.stack(coords_matrix)
                coords_flatten = coords.reshape(2, -1)
                relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
                relative_coords = relative_coords.transpose([1, 2, 0])
                relative_coords[:, :, 0] += self.window_size[0] - 1
                relative_coords[:, :, 1] += self.window_size[1] - 1
                relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
                relative_position_index = relative_coords.sum(-1)

                self.relative_position_index = tf.Variable(
                    initial_value=tf.convert_to_tensor(relative_position_index), trainable=False
                )

            def call(self, x, mask=None):
                _, size, channels = x.shape
                head_dim = channels // self.num_heads
                x_qkv = self.qkv(x)
                x_qkv = tf.reshape(x_qkv, shape=(-1, size, 3, self.num_heads, head_dim))
                x_qkv = tf.transpose(x_qkv, perm=(2, 0, 3, 1, 4))
                q, k, v = x_qkv[0], x_qkv[1], x_qkv[2]
                q = q * self.scale
                k = tf.transpose(k, perm=(0, 1, 3, 2))
                attn = q @ k

                num_window_elements = self.window_size[0] * self.window_size[1]
                relative_position_index_flat = tf.reshape(
                    self.relative_position_index, shape=(-1,)
                )
                relative_position_bias = tf.gather(
                    self.relative_position_bias_table, relative_position_index_flat
                )
                relative_position_bias = tf.reshape(
                    relative_position_bias, shape=(num_window_elements, num_window_elements, -1)
                )
                relative_position_bias = tf.transpose(relative_position_bias, perm=(2, 0, 1))
                attn = attn + tf.expand_dims(relative_position_bias, axis=0)

                if mask is not None:
                    nW = mask.get_shape()[0]
                    mask_float = tf.cast(
                        tf.expand_dims(tf.expand_dims(mask, axis=1), axis=0), tf.float32
                    )
                    attn = (
                        tf.reshape(attn, shape=(-1, nW, self.num_heads, size, size))
                        + mask_float
                    )
                    attn = tf.reshape(attn, shape=(-1, self.num_heads, size, size))
                    attn = tf.keras.activations.softmax(attn, axis=-1)
                else:
                    attn = tf.keras.activations.softmax(attn, axis=-1)
                attn = self.dropout(attn)

                x_qkv = attn @ v
                x_qkv = tf.transpose(x_qkv, perm=(0, 2, 1, 3))
                x_qkv = tf.reshape(x_qkv, shape=(-1, size, channels))
                x_qkv = self.proj(x_qkv)
                x_qkv = self.dropout(x_qkv)
                return x_qkv

        
        class SwinTransformer(layers.Layer):
            def __init__(
                self,
                dim,
                num_patch,
                num_heads,
                window_size=7,
                shift_size=0,
                num_mlp=1024,
                qkv_bias=True,
                dropout_rate=0.0,
                **kwargs,
            ):
                super(SwinTransformer, self).__init__(**kwargs)

                self.dim = dim  # number of input dimensions
                self.num_patch = num_patch  # number of embedded patches
                self.num_heads = num_heads  # number of attention heads
                self.window_size = window_size  # size of window
                self.shift_size = shift_size  # size of window shift
                self.num_mlp = num_mlp  # number of MLP nodes

                self.norm1 = layers.LayerNormalization(epsilon=1e-5)
                self.attn = WindowAttention(
                    dim,
                    window_size=(self.window_size, self.window_size),
                    num_heads=num_heads,
                    qkv_bias=qkv_bias,
                    dropout_rate=dropout_rate,
                )
                self.drop_path = DropPath(dropout_rate)
                self.norm2 = layers.LayerNormalization(epsilon=1e-5)

                self.mlp = keras.Sequential(
                    [
                        layers.Dense(num_mlp),
                        layers.Activation(keras.activations.gelu),
                        layers.Dropout(dropout_rate),
                        layers.Dense(dim),
                        layers.Dropout(dropout_rate),
                    ]
                )

                if min(self.num_patch) < self.window_size:
                    self.shift_size = 0
                    self.window_size = min(self.num_patch)

            def build(self, input_shape):
                if self.shift_size == 0:
                    self.attn_mask = None
                else:
                    height, width = self.num_patch
                    h_slices = (
                        slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None),
                    )
                    w_slices = (
                        slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None),
                    )
                    mask_array = np.zeros((1, height, width, 1))
                    count = 0
                    for h in h_slices:
                        for w in w_slices:
                            mask_array[:, h, w, :] = count
                            count += 1
                    mask_array = tf.convert_to_tensor(mask_array)

                    # mask array to windows
                    mask_windows = window_partition(mask_array, self.window_size)
                    mask_windows = tf.reshape(
                        mask_windows, shape=[-1, self.window_size * self.window_size]
                    )
                    attn_mask = tf.expand_dims(mask_windows, axis=1) - tf.expand_dims(
                        mask_windows, axis=2
                    )
                    attn_mask = tf.where(attn_mask != 0, -100.0, attn_mask)
                    attn_mask = tf.where(attn_mask == 0, 0.0, attn_mask)
                    self.attn_mask = tf.Variable(initial_value=attn_mask, trainable=False)

            def call(self, x):
                height, width = self.num_patch
                _, num_patches_before, channels = x.shape
                x_skip = x
                x = self.norm1(x)
                x = tf.reshape(x, shape=(-1, height, width, channels))
                if self.shift_size > 0:
                    shifted_x = tf.roll(
                        x, shift=[-self.shift_size, -self.shift_size], axis=[1, 2]
                    )
                else:
                    shifted_x = x

                x_windows = window_partition(shifted_x, self.window_size)
                x_windows = tf.reshape(
                    x_windows, shape=(-1, self.window_size * self.window_size, channels)
                )
                attn_windows = self.attn(x_windows, mask=self.attn_mask)

                attn_windows = tf.reshape(
                    attn_windows, shape=(-1, self.window_size, self.window_size, channels)
                )
                shifted_x = window_reverse(
                    attn_windows, self.window_size, height, width, channels
                )
                if self.shift_size > 0:
                    x = tf.roll(
                        shifted_x, shift=[self.shift_size, self.shift_size], axis=[1, 2]
                    )
                else:
                    x = shifted_x

                x = tf.reshape(x, shape=(-1, height * width, channels))
                x = self.drop_path(x)
                x = x_skip + x
                x_skip = x
                x = self.norm2(x)
                x = self.mlp(x)
                x = self.drop_path(x)
                x = x_skip + x
                return x


        class PatchExtract(layers.Layer):
            def __init__(self, patch_size, **kwargs):
                super(PatchExtract, self).__init__(**kwargs)
                self.patch_size_x = patch_size[0]
                self.patch_size_y = patch_size[0]

            def call(self, images):
                batch_size = tf.shape(images)[0]
                patches = tf.image.extract_patches(
                    images=images,
                    sizes=(1, self.patch_size_x, self.patch_size_y, 1),
                    strides=(1, self.patch_size_x, self.patch_size_y, 1),
                    rates=(1, 1, 1, 1),
                    padding="VALID",
                )
                patch_dim = patches.shape[-1]
                patch_num = patches.shape[1]
                return tf.reshape(patches, (batch_size, patch_num * patch_num, patch_dim))


        class PatchEmbedding(layers.Layer):
            def __init__(self, num_patch, embed_dim, **kwargs):
                super(PatchEmbedding, self).__init__(**kwargs)
                self.num_patch = num_patch
                self.proj = layers.Dense(embed_dim)
                self.pos_embed = layers.Embedding(input_dim=num_patch, output_dim=embed_dim)

            def call(self, patch):
                pos = tf.range(start=0, limit=self.num_patch, delta=1)
                return self.proj(patch) + self.pos_embed(pos)


        class PatchMerging(tf.keras.layers.Layer):
            def __init__(self, num_patch, embed_dim):
                super(PatchMerging, self).__init__()
                self.num_patch = num_patch
                self.embed_dim = embed_dim
                self.linear_trans = layers.Dense(2 * embed_dim, use_bias=False)

            def call(self, x):
                height, width = self.num_patch
                _, _, C = x.get_shape().as_list()
                x = tf.reshape(x, shape=(-1, height, width, C))
                x0 = x[:, 0::2, 0::2, :]
                x1 = x[:, 1::2, 0::2, :]
                x2 = x[:, 0::2, 1::2, :]
                x3 = x[:, 1::2, 1::2, :]
                x = tf.concat((x0, x1, x2, x3), axis=-1)
                x = tf.reshape(x, shape=(-1, (height // 2) * (width // 2), 4 * C))
                return self.linear_trans(x)

        
        input = layers.Input(self.input_shape)
        x = PatchExtract(self.patch_size)(input)
        x = PatchEmbedding(self.num_patch_x * self.num_patch_y, self.embed_dim)(x)
        x = SwinTransformer(
            dim=self.embed_dim,
            num_patch=(self.num_patch_x, self.num_patch_y),
            num_heads=self.num_heads,
            window_size=self.window_size,
            shift_size=0,
            num_mlp=self.num_mlp,
            qkv_bias=self.qkv_bias,
            dropout_rate=self.dropout_rate,
        )(x)
        x = SwinTransformer(
            dim=self.embed_dim,
            num_patch=(self.num_patch_x, self.num_patch_y),
            num_heads=self.num_heads,
            window_size=self.window_size,
            shift_size=self.shift_size,
            num_mlp=self.num_mlp,
            qkv_bias=self.qkv_bias,
            dropout_rate=self.dropout_rate,
        )(x)
        x = PatchMerging((self.num_patch_x, self.num_patch_y), embed_dim=self.embed_dim)(x)
        x = layers.GlobalAveragePooling1D()(x)
        output = layers.Dense(self.num_classes * self.num_strings)(x)
        output = Reshape((self.num_strings, self.num_classes))(output)
        output = Activation(self.softmax_by_string)(output)

        model = Model(inputs=input, outputs = output)

        model.compile(loss=self.catcross_by_string,
                      optimizer=tf.keras.optimizers.Adam(),
                      metrics=[self.avg_acc])
        
        self.model = model


    
    def train(self):
        self.model.fit(self.training_generator,
                    validation_data=None,
                    epochs=self.epochs,
                    steps_per_epoch = self.nb_files_train//128,
                    verbose=1,
                    use_multiprocessing=True,
                    workers=9)
        
    
    def save_weights(self):
        self.model.save_weights(self.split_folder + "weights.h5")
        
    
    def test(self):
        self.validation_array = self.validation_generator.unbatch()
        self.X_test = np.array(list(self.validation_array.map(lambda x, y: x)))
        self.y_gt = np.array(list(self.validation_array.map(lambda x, y: y)))
        print("X_test shape:", self.X_test.shape)
        print("y_gt shape:", self.y_gt.shape)
        self.y_pred = self.model.predict(self.X_test)
        print("y_pred shape:", self.y_pred.shape)
    
    def save_predictions(self):
        np.savez(self.split_folder + "predictions.npz", y_pred=self.y_pred, y_gt=self.y_gt)
        
    
    def evaluate(self):
        self.metrics["pp"].append(pitch_precision(self.y_pred, self.y_gt))
        self.metrics["pr"].append(pitch_recall(self.y_pred, self.y_gt))
        self.metrics["pf"].append(pitch_f_measure(self.y_pred, self.y_gt))
        self.metrics["tp"].append(tab_precision(self.y_pred, self.y_gt))
        self.metrics["tr"].append(tab_recall(self.y_pred, self.y_gt))
        self.metrics["tf"].append(tab_f_measure(self.y_pred, self.y_gt))
        self.metrics["tdr"].append(tab_disamb(self.y_pred, self.y_gt))
        self.metrics["acc"].append(float(self.model.evaluate(self.X_test, self.y_gt, batch_size=self.batch_size)[1]))
    
    def save_results_csv(self):
        output = {}
        for key in self.metrics.keys():
            if key != "data":
                vals = self.metrics[key]
                mean = np.mean(vals)
                std = np.std(vals)
                output[key] = vals + [mean, std]
        output["data"] =  self.metrics["data"]
        df = pd.DataFrame.from_dict(output)
        df.to_csv(self.save_folder + "results.csv") 
        
##################################
########### EXPERIMENT ###########
##################################

if __name__ == '__main__':
    tabcnn = TabCNN()
    tabcnn.build_model()
    tabcnn.log_model()

    for fold in range(1):
        print("\nfold " + str(fold))
        tabcnn.partition_data(fold)
        print("building model...")
        tabcnn.build_model()  
        print("training...")
        tabcnn.train()
        tabcnn.save_weights()
        print("testing...")
        tabcnn.test()
        tabcnn.save_predictions()
        print("evaluation...")
        tabcnn.evaluate()
    print("saving results...")
    tabcnn.save_results_csv()
