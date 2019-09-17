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

"""
This script makes sure the $(DATA_DIR) and $(RESULT_DIR) are set up correctly.
For this BERT model, this includes downloading the appropriate pre-trained model file.
"""
import os
import shutil
import urllib
import zipfile

import tensorflow as tf

flags = tf.flags
FLAGS = flags.FLAGS

# Load the $(DATA_DIR) and $(RESULT_DIR) variables
flags.DEFINE_string("DATA_DIR", None, "The input data bucket.")
flags.DEFINE_string("RESULT_DIR", None, "The output data bucket.")

flags.DEFINE_string("MODEL_DOWNLOAD_BASE", None, "The data bucket.")
flags.DEFINE_string("MODEL_FILE", None, "The data bucket.")
flags.DEFINE_string("MODEL_FOLDER", None, "The data bucket.")


def force_create_dir(base, dirName):
    path = os.path.join(base, dirName)
    if os.path.exists(path) and os.path.isdir(path):
        shutil.rmtree(path)
    os.makedirs(path)
    return path


def soft_create_dir(base, dirName):
    path = os.path.join(base, dirName)
    if not os.path.exists(path):
        os.makedirs(path)
    return path


def main():
    ############################################################################
    # Create Directories
    ############################################################################

    pretrained_model_dir = soft_create_dir(FLAGS.RESULT_DIR, 'pretrained_models')
    force_create_dir(FLAGS.RESULT_DIR, 'temp')
    soft_create_dir(FLAGS.RESULT_DIR, 'model')

    ############################################################################
    # Download Pre-trained BERT Model
    ############################################################################
    download_base = FLAGS.MODEL_DOWNLOAD_BASE
    model_file = FLAGS.MODEL_FILE
    model_folder = FLAGS.MODEL_FOLDER

    archive_path = os.path.join(pretrained_model_dir, model_file)
    folder_path = os.path.join(pretrained_model_dir, model_folder)

    if not os.path.exists(folder_path):
        # Downloading the model file (if the unzipped file doesnt exist yet)
        print('Downloading pre-trained model file...')
        opener = urllib.request.URLopener()
        opener.retrieve(download_base + model_file, archive_path)

        # Unzipping
        zip_ref = zipfile.ZipFile(archive_path, 'r')
        zip_ref.extractall(pretrained_model_dir)
        zip_ref.close()

        # Remove the archive file
        os.remove(archive_path)
    else:
        print('Pre-trained model already exists. Skipping the download.')


if __name__ == "__main__":
    flags.mark_flag_as_required("DATA_DIR")
    flags.mark_flag_as_required("RESULT_DIR")
    flags.mark_flag_as_required("MODEL_DOWNLOAD_BASE")
    flags.mark_flag_as_required("MODEL_FILE")
    flags.mark_flag_as_required("MODEL_FOLDER")
    main()
