# Copyright 2018 The TensorFlow Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Functions to build an input pipeline that reads from TFRecord files."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import logging
import collections
import six
import tensorflow as tf
import pandas as pd
import numpy as np
import random
from astronet.astro_cnn_model import configurations
from scipy import stats

def build_dataset(file_pattern,
                  input_config,
                  batch_size,
                  include_labels=True,
                  shuffle_filenames=False,
                  shuffle_values_buffer=0,
                  repeat=1,
                  include_identifiers=False):

    def parse_example(serialized_example):
        """Parses a single tf.Example into feature and label tensors."""
        
        data_fields = {
            feature_name: tf.io.FixedLenFeature(feature.shape, tf.float32)
            for feature_name, feature in input_config.features.items()
        }
        if include_labels:
            for n in input_config.label_columns:
                data_fields[n] = tf.io.FixedLenFeature([], tf.int64)
        if include_identifiers:
            assert "astro_id" not in data_fields
            data_fields["astro_id"] = tf.io.FixedLenFeature([], tf.int64)


        parsed_features = tf.io.parse_single_example(serialized_example, features=data_fields)
        
        
        if include_labels:
            label_features = [parsed_features.pop(name) for name in input_config.label_columns]
            labels = tf.stack(label_features)
            labels_f = tf.cast(labels, tf.float32)
            labels = tf.cast(tf.minimum(labels, 1), tf.float32)
            weights = tf.reduce_max(labels_f) / tf.maximum(tf.reduce_sum(labels_f), 1.0)
            if labels[input_config.primary_class] < 1:
                weights /= 2.0

        if include_identifiers:
            identifiers = parsed_features.pop("astro_id")
        else:
            assert "astro_id" not in parsed_features

        features = {}
        assert set(parsed_features.keys()) == set(input_config.features.keys())
        for name, value in parsed_features.items():
            cfg = input_config.features[name]
            if not cfg.is_time_series:
                if getattr(cfg, "scale", None) == "log":
                    value = tf.cast(value, tf.float64)
                    value = tf.maximum(value, cfg.min_val)
                    value = tf.minimum(value, cfg.max_val)
                    value = value - cfg.min_val + 1
                    value = tf.math.log(value) / tf.math.log(tf.constant(cfg.max_val, tf.float64))
                    value = tf.cast(value, tf.float32)
                elif getattr(cfg, "scale", None) == "norm":
                    value = (value - cfg["mean"]) / cfg["std"]
            features[name.lower()] = value
        if include_labels:
            return features, labels, weights
        elif include_identifiers:
            return features, identifiers
        return features


    filenames = tf.constant(tf.io.gfile.glob(file_pattern), dtype=tf.string)
    ds = tf.data.Dataset.from_tensor_slices(filenames)
    ds = ds.flat_map(tf.data.TFRecordDataset)
    ds = ds.map(parse_example)

###### rough code for checking whether the candidate has been assigned disps E,N,B, or S, and only using these candidates in the data augmentation. There is probably a more concise way to do this ######


#     values = np.array([x[1].numpy() for x in ds])
#    truth_array = []
#    for array in values:
#        z = array == [1., 1., 1., 1., 1.]
#        if z[0] or z[1] or z[3] or z[4]:
#            truth_array.append(True)
#        else:
#            truth_array.append(False)       
#    truth_data = tf.data.Dataset.from_tensor_slices(truth_array)
#    ds_new = tf.data.Dataset.zip((ds,truth_data))
#    new_len = [x for x in ds_new]
#    print("number trues =",np.array(truth_array).sum())
#    ds_false = ds_new.filter(lambda image,y:y==False)
#    false_list = [x for x in ds_false]
#    randoms = []
#    for i in range(len(false_list)):
#        randoms.append(random.randint(0,1))
#    rand = tf.data.Dataset.from_tensor_slices(randoms)
#    false_rand = tf.data.Dataset.zip((ds_false,rand))
#    false_rand = false_rand.filter(lambda x,z: z==1)
#    ds_new = ds_new.filter(lambda image,y: y ==True)
    
    features = list(configurations.final_alpha_0()['inputs']['features'].keys())
    globals = [feature for feature in features if 'global' in feature]
    locals = [feature for feature in features if 'local' in feature]
    secondaries = [feature for feature in features if 'secondary' in feature]
    all = globals+locals+secondaries
    
    def noise_array(array):
      array_new = [feature for feature in array if 'mask' not in feature if 'std' not in feature]
      return(array_new)
                
    array_reverse = secondaries  
    array_noise = noise_array(secondaries)

    #if reversing entire dataset- not specific dispostions (so NOT using ds_new) image is a dictionary corresponding with all the lightcurve data, x is an array corresponding to the dispositions, y takes a value between 1 and 0
    
    def add_noise_all(image,x,y):
        for a in array_noise:
            z = image[0][a]
            std = 1.48*stats.median_absolute_deviation(z, scale=1,nan_policy='omit')
            noise = tf.random.normal(shape=z.get_shape(), mean=0.0, stddev=std, dtype=tf.float32)
            image[0][a] = tf.math.add(z,noise)
        return image[0],image[1],image[2]   
    
    def reverse_all(image,x,y):
        for a in all:
            z = image[a]
            image[a]=tf.reverse(z,[0])
        return image,x,y
    
    #if not reversing entire dataset- not specific dispostions (using ds_new)- image here is an array with the image dictionary, x, and y, and y here is the array corresponding to the dispositions, used to filter the dataset 
    
    def reverse(image,y):
        for a in all:
            z = image[0][a]
            image[0][a]= tf.reverse(z,[0])
        return image[0],image[1],image[2]
      
    def add_noise(image,y):
        for a in array_noise:
            z = image[0][a]
            std = 1.48*stats.median_absolute_deviation(z, scale=1,nan_policy='omit')
            noise = tf.random.normal(shape=z.get_shape(), mean=0.0, stddev=std, dtype=tf.float32)
            image[0][a] = tf.math.add(z,noise)
        return image[0],image[1],image[2]
    

###### Only augment data in the training set, us ds/ds_new depending on whether all data or only disps E,N,B, or S are being augmented  ######
    
    if shuffle_filenames:
        
        reversed =  ds.map(reverse_all)
        noisy = ds_new.map(add_noise)
        
        combined_data = ds.concatenate(reversed)
        combined_data = combined_data.concatenate(noisy)
        ds = combined_data

    
    if repeat != 1:
        ds = ds.cache()

    if shuffle_values_buffer > 0:
        ds = ds.shuffle(shuffle_values_buffer)
    if repeat != 1:
        ds = ds.repeat(repeat)
    ds = ds.batch(batch_size)
    ds = ds.prefetch(10)

    return ds
