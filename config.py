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

# Flask settings
DEBUG = False

# Flask-restplus settings
RESTPLUS_MASK_SWAGGER = False
SWAGGER_UI_DOC_EXPANSION = 'none'

# API metadata
API_TITLE = 'MAX Text Sentiment Classifier'
API_DESC = 'Detect the sentiment captured in short pieces of text. ' \
           'The model was finetuned on the IBM Project Debater Claim Sentiment dataset.'
API_VERSION = '2.0.0'

# default model
MODEL_NAME = 'sentiment_BERT_base_uncased'
DEFAULT_MODEL_PATH = 'assets'

# the metadata of the model
MODEL_META_DATA = {
    'id': 'max-text-sentiment-classifier',
    'name': 'Bert Base Uncased TensorFlow Model',
    'description': 'BERT Base finetuned on the IBM Project Debater Claim Sentiment dataset.',
    'type': 'Text Classification',
    'source': 'https://developer.ibm.com/exchanges/models/all/max-text-sentiment-classifier/',
    'license': 'Apache V2'
}
