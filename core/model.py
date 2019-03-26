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
        logger.info('Loading model from: {}...'.format(path))

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

        # Grab the examples, convert to features, and create batches
        predictions = []
        batch = {}
        for i, example in enumerate(predict_examples):

            # convert example to feature
            input_ids, input_mask, label_ids, segment_ids = convert_single_example(i, example, self.label_list,
                                                                                   self.max_seq_length, self.tokenizer)

            # add to batch
            if batch == {}:
                batch['tensor_input_ids'] = np.array(input_ids).reshape(-1, self.max_seq_length)
                batch['tensor_input_mask'] = np.array(input_mask).reshape(-1, self.max_seq_length)
                batch['tensor_label_ids'] = np.array(label_ids)
                batch['tensor_segment_ids'] = np.array(segment_ids).reshape(-1, self.max_seq_length)
            else:
                batch['tensor_input_ids'] = np.vstack(
                    [batch['tensor_input_ids'], np.array(input_ids).reshape(-1, self.max_seq_length)])
                batch['tensor_input_mask'] = np.vstack(
                    [batch['tensor_input_mask'], np.array(input_mask).reshape(-1, self.max_seq_length)])
                batch['tensor_label_ids'] = np.vstack([batch['tensor_label_ids'], np.array(label_ids)])
                batch['tensor_segment_ids'] = np.vstack(
                    [batch['tensor_segment_ids'], np.array(segment_ids).reshape(-1, self.max_seq_length)])

            if batch['tensor_input_ids'].shape[0] == predict_batch_size:
                # Make a prediction
                result = self.sess.run(tensor_outputs, feed_dict={
                    tensor_input_ids: batch['tensor_input_ids'],
                    tensor_input_mask: batch['tensor_input_mask'],
                    tensor_label_ids: batch['tensor_label_ids'].reshape(-1, ),
                    tensor_segment_ids: batch['tensor_segment_ids'],
                })
                # Add the predictions
                predictions.extend(result)
                # Emtpy the batch
                batch = {}

            if i > 0 and i % 100 == 0:
                tf.logging.info(
                    f"example {i} out of {len(predict_examples)} done ({round(i * 100 / len(predict_examples))}%).")

        # The last batch
        if batch != {}:
            # Make a prediction
            result = self.sess.run(tensor_outputs, feed_dict={
                tensor_input_ids: batch['tensor_input_ids'],
                tensor_input_mask: batch['tensor_input_mask'],
                tensor_label_ids: batch['tensor_label_ids'].reshape(-1, ),
                tensor_segment_ids: batch['tensor_segment_ids'],
            })
            # Add the predictions
            predictions.extend(result)

        return [[p[0], p[1]] for p in predictions]
