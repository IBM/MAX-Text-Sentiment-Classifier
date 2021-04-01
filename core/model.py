#
# Copyright 2018-2019 IBM Corp. All Rights Reserved.
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
#

from maxfw.model import MAXModelWrapper

import logging
from config import DEFAULT_MODEL_PATH, MODEL_META_DATA as model_meta

from core.bert.run_classifier import convert_single_example, MAXAPIProcessor
from core.bert import tokenization
from tensorflow.python.saved_model import tag_constants
import tensorflow as tf
import numpy as np

logger = logging.getLogger()


class ModelWrapper(MAXModelWrapper):

    MODEL_META_DATA = model_meta

    def __init__(self, path=DEFAULT_MODEL_PATH):
  
        download_base = 'https://max-cdn.cdn.appdomain.cloud/' \
                    'max-object-detector/1.0.1/'
        model_file = 'ssd_mobilenet_v1_coco_2018_01_28.tar.gz'

        tar_path = os.path.join('assets/', model_file)

        if not os.path.exists(tar_path):
            print('Downloading model checkpoint...')
            opener = urllib.request.URLopener()
            opener.retrieve(download_base + model_file, tar_path)
        else:
            print('Model found.')

        with tarfile.open(tar_path) as tar:
            for member in tar.getmembers():
                # Flatten the directory.
                member.name = os.path.basename(member.name)
                if 'model.ckpt' in member.name:
                    print('Extracting {}...'.format(member.name))
                    tar.extract(member, path='assets/')

        self.max_seq_length = 128
        self.do_lower_case = True

        # Set Logging verbosity
        tf.logging.set_verbosity(tf.logging.INFO)

        # Loading the tf Graph
        self.graph = tf.Graph()
        self.sess = tf.Session(graph=self.graph)
        tf.saved_model.loader.load(self.sess, [tag_constants.SERVING], DEFAULT_MODEL_PATH)

        # Validate init_checkpoint
        tokenization.validate_case_matches_checkpoint(self.do_lower_case,
                                                      DEFAULT_MODEL_PATH)

        # Initialize the dataprocessor
        self.processor = MAXAPIProcessor()

        # Get the labels
        self.label_list = self.processor.get_labels()

        # Initialize the tokenizer
        self.tokenizer = tokenization.FullTokenizer(
            vocab_file=f'{DEFAULT_MODEL_PATH}/vocab.txt', do_lower_case=self.do_lower_case)

        logger.info('Loaded model')

    def _pre_process(self, input):
        '''Preprocessing of the input is not required as it is carried out by the BERT model (Tokenization).'''
        return input

    def _post_process(self, result):
        '''Reformat the results if needed.'''
        return result

    def _predict(self, x, predict_batch_size=32):
        '''Predict the class probabilities using the BERT model.'''

        # Get the input examples
        predict_examples = self.processor.get_test_examples(x)

        # Grab the input tensors of the Graph
        tensor_input_ids = self.sess.graph.get_tensor_by_name('input_ids_1:0')
        tensor_input_mask = self.sess.graph.get_tensor_by_name('input_mask_1:0')
        tensor_label_ids = self.sess.graph.get_tensor_by_name('label_ids_1:0')
        tensor_segment_ids = self.sess.graph.get_tensor_by_name('segment_ids_1:0')
        tensor_outputs = self.sess.graph.get_tensor_by_name('loss/Softmax:0')

        # Grab the examples, convert to features, and create batches. In the loop,
        # Go over all examples in chunks of size `predict_batch_size`.
        predictions = []

        for i in range(0, len(predict_examples), predict_batch_size):
            examples = predict_examples[i:i+predict_batch_size]

            tf.logging.info(
                f"{i} out of {len(predict_examples)} examples done ({round(i * 100 / len(predict_examples))}%).")

            # Convert example to feature in batches.
            input_ids, input_mask, label_ids, segment_ids = zip(
                *tuple(convert_single_example(i + j, example, self.label_list, self.max_seq_length, self.tokenizer)
                       for j, example in enumerate(examples)))

            # Convert to a format that is consistent with input tensors
            feed_dict = {}
            feed_dict[tensor_input_ids] = np.vstack(
                tuple(np.array(arr).reshape(-1, self.max_seq_length) for arr in input_ids))
            feed_dict[tensor_input_mask] = np.vstack(
                tuple(np.array(arr).reshape(-1, self.max_seq_length) for arr in input_mask))
            feed_dict[tensor_label_ids] = np.vstack(
                tuple(np.array(arr) for arr in label_ids)).flatten()
            feed_dict[tensor_segment_ids] = np.vstack(
                tuple(np.array(arr).reshape(-1, self.max_seq_length) for arr in segment_ids))

            # Make a prediction
            result = self.sess.run(tensor_outputs, feed_dict=feed_dict)
            # Add the predictions
            predictions.extend(p.tolist() for p in result)

        return predictions
