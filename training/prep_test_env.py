import os
import glob

from ruamel.yaml import YAML
from utils.cos import COSWrapper

# update the yaml file with the corresponding buckets
yaml_file = glob.glob('*.yaml')[0]

yaml = YAML(typ='rt')
yaml.allow_duplicate_keys = True
yaml.preserve_quotes = True
yaml.indent(mapping=6, sequence=4)

# open the file
with open(yaml_file, 'r') as fp:
    # Loading configuration file
    config = yaml.load(fp)

# set input bucket
config['train']['data_source']['bucket'] = os.environ.get('COS_INPUT_BUCKET')
config['train']['data_source']['training_data']['bucket'] = os.environ.get('COS_INPUT_BUCKET')
# set output bucket
config['train']['model_training_results']['bucket'] = os.environ.get('COS_OUTPUT_BUCKET')
config['train']['model_training_results']['trained_model']['bucket'] = os.environ.get('COS_OUTPUT_BUCKET')

# save the file
with open(yaml_file, 'w') as fp:
    yaml.dump(config, fp)

# clear the input and output bucket
cw = COSWrapper(os.environ.get('AWS_ACCESS_KEY_ID'),
                os.environ.get('AWS_SECRET_ACCESS_KEY'))

cw.clear_bucket(os.environ.get('COS_INPUT_BUCKET'))
cw.clear_bucket(os.environ.get('COS_OUTPUT_BUCKET'))

# upload sample training data to the bucket
for fp in glob.glob('sample_training_data/*'):
    cw.upload_file(file_name=fp,
                   bucket_name=os.environ.get('COS_INPUT_BUCKET'),
                   key_name=fp.replace('sample_training_data/', ''))
