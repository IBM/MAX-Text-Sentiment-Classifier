# Flask settings
DEBUG = False

# Flask-restplus settings
RESTPLUS_MASK_SWAGGER = False
SWAGGER_UI_DOC_EXPANSION = 'none'

# API metadata
API_TITLE = 'MAX Text Sentiment Classifier'
API_DESC = 'Detect the sentiment captured in short pieces of text. ' \
           'The model was finetuned on the IBM Project Debater Claim Sentiment dataset.'
API_VERSION = '1.0.1'

# default model
MODEL_NAME = 'sentiment_BERT_base_uncased'
DEFAULT_MODEL_PATH = 'assets/{}'.format(MODEL_NAME)

# the metadata of the model
MODEL_META_DATA = {
    'id': 'max-text-sentiment-classifier',
    'name': 'Bert Base Uncased TensorFlow Model',
    'description': 'BERT Base finetuned on the IBM Project Debater Claim Sentiment dataset.',
    'type': 'Text Classification',
    'source': 'https://developer.ibm.com/exchanges/models/all/max-text-sentiment-classifier/',
    'license': 'Apache V2'
}
