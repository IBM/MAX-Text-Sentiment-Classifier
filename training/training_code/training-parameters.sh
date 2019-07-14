# The pre-trained BERT model to use. For different options, see https://github.com/google-research/bert
# the download url will be equal to "$(MODEL_DOWNLOAD_BASE)$(MODEL_FILE)"
MODEL_DOWNLOAD_BASE="https://storage.googleapis.com/bert_models/2018_10_18/"
MODEL_FILE="uncased_L-12_H-768_A-12.zip"
MODEL_FOLDER="uncased_L-12_H-768_A-12"

# Hyperparameters
# model-specific:
DO_LOWER_CASE=true  # changing this parameter also requires changing the same parameter in `core/model.py` for inference
MAX_SEQ_LENGTH=128  # changing this parameter also requires changing the same parameter in `core/model.py` for inference
# general:
NUM_TRAIN_EPOCHS=10.0
TRAIN_BATCH_SIZE=16
WARMUP_PROPORTION=0.1
LEARNING_RATE=0.00005
