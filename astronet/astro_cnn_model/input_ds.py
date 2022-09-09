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
    
    
    
    globals  = ['global_mask', 'global_std', 'global_transit_mask', 'global_view', 'global_view_0.3', 'global_view_5.0', 'global_view_double_period', 'global_view_double_period_0.3', 'global_view_double_period_5.0']
    locals   =['local_mask','local_std','local_std_even','local_std_odd','local_view','local_view_0.3','local_view_5.0','local_view_even','local_view_half_period_std','local_view_odd']
    secondaries = ['secondary_mask','secondary_std','secondary_view','secondary_view_0.3','secondary_view_5.0']
    all  = ['global_mask', 'global_std', 'global_transit_mask', 'global_view', 'global_view_0.3', 'global_view_5.0', 'global_view_double_period', 'global_view_double_period_0.3', 'global_view_double_period_5.0','local_mask','local_std','local_std_even','local_std_odd','local_view','local_view_0.3','local_view_5.0','local_view_even','local_view_half_period_std','local_view_odd','secondary_mask','secondary_std','secondary_view','secondary_view_0.3','secondary_view_5.0']
    

    def add_noise_global(image,y):
        for a in globals: 
            z = image[0][a]
            noise = tf.random.normal(shape=z.get_shape(), mean=0.0, stddev=0.1, dtype=tf.float32)
            image[0][a] = tf.math.add(z,noise)
        return image[0],image[1],image[2]
    
    def add_noise_local(image,y):
        for a in locals: 
            z = image[0][a]
            noise = tf.random.normal(shape=z.get_shape(), mean=0.0, stddev=0.1, dtype=tf.float32)
            image[0][a] = tf.math.add(z,noise)
        return image[0],image[1],image[2]
        
    def add_noise_secondary(image,y):
        for a in secondaries: 
            z = image[0][a]
            noise = tf.random.normal(shape=z.get_shape(), mean=0.0, stddev=0.1, dtype=tf.float32)
            image[0][a] = tf.math.add(z,noise)
        return image[0],image[1],image[2]
    
    
    def reverse_global(image,x,y):
        for g in globals:
            z = image[g]
            image[g] = tf.reverse(z,[0])
        return image,x,y
    
    def reverse_local(image,y):
        for g in locals:
            z = image[0][g]
            image[0][g] = tf.reverse(z,[0])
        return image[0],image[1],image[2]
    
    def reverse_secondary(image,y):
        for g in secondaries:
            z = image[0][g]
            image[0][g] = tf.reverse(z,[0])
        return image[0],image[1],image[2]
    
    def reverse_all(image,y):
        for g in all:
            z = image[0][g]
            image[0][g] = tf.reverse(z,[0])
        return image[0],image[1],image[2]
    
    
    def repeat_disps(image,y):
        z = image[0]
        
        return z,image[1],image[2]
    
    def return_false(x,y):
        return x[0][0],x[0][1],x[0][2]
    
    def return_true(x,y):
        return x[0],x[1],x[2]    
    
  

###### Only augment data in the training set, us ds/ds_new depending on whether all data or only disps E,N,B, or S are being augmented  ######
    
    if shuffle_filenames:
        
    #    false_ds = false_rand.map(return_false)
    #    true_ds = ds_new.map(return_true)
    #    ds = false_ds.concatenate(true_ds)                 
    #    ds_repeat = ds_new.map(repeat_disps)
        ds_local =  ds_new.map(reverse_local)
        ds_global = ds_new.map(add_noise_global)
        
        combined_data = ds.concatenate(ds_local)
        combined_data = combined_data.concatenate(ds_global)
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
