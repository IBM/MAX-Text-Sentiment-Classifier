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

#!/bin/bash

# uncomment to enable debug output
#set -x

# --------------------------------------------------------------------
#  Standard training wrapper script for Model Asset Exchange models
#  Complete the following IBM customization steps and remove the TODO
#  comments.
# --------------------------------------------------------------------

SUCCESS_RETURN_CODE=0
TRAINING_FAILED_RETURN_CODE=1
POST_PROCESSING_FAILED=2
PACKAGING_FAILED_RETURN_CODE=3
CUSTOMIZATION_ERROR_RETURN_CODE=4
ENV_ERROR_RETURN_CODE=5

# --------------------------------------------------------------------
#  Verify that the required environment variables are defined
# --------------------------------------------------------------------

# DATA_DIR identifies the directory where the training data is located.
# The specified directory must exist and be readable.
if [ -z ${DATA_DIR+x} ]; then
  echo "Error. Environment variable DATA_DIR is not defined."
  exit $ENV_ERROR_RETURN_CODE
fi

if [ ! -d ${DATA_DIR} ]; then 
  echo "Error. Environment variable DATA_DIR (\"$DATA_DIR\") does not identify an existing directory."
  exit $ENV_ERROR_RETURN_CODE
fi

# Verify whether the DATA_DIR contains only one tsv file, and assign it to the TRAINING_DATA variable.
tsv_count=`ls ${DATA_DIR}/*.tsv 2>/dev/null | wc -l`
if [ $tsv_count -gt 1 ]; then
    echo "ERROR: There are multiple .tsv files present in the data directory."
    exit $ENV_ERROR_RETURN_CODE
else
    # Assign it to the TRAIN_DATA variable
    TRAINING_DATA=`ls ${DATA_DIR}/*.tsv`
fi

# RESULT_DIR identifies the directory where the training output is stored.
# The specified directory must exist and be writable.
if [ -z ${RESULT_DIR+x} ]; then
  echo "Error. Environment variable RESULT_DIR is not defined."
  exit $ENV_ERROR_RETURN_CODE
fi

if [ ! -d ${RESULT_DIR} ]; then 
  echo "Error. Environment variable RESULT_DIR (\"$RESULT_DIR\") does not identify an existing directory."
  exit $ENV_ERROR_RETURN_CODE
fi

# ---------------------------------------------------------------
# Perform pre-training tasks
# (1) Verify that environment variables are defined
# (2) Install prerequisite packages
# ---------------------------------------------------------------

echo "# ************************************************************"
echo "# Preparing for model training"
echo "# ************************************************************"

# Prior to launching this script, WML copies the training data from 
# Cloud Object Storage to the $DATA_DIR directory. Use this environment
# variable to access the data.  
echo "Training data is stored in $DATA_DIR"

# The WML stores work files in the $RESULT_DIR.
echo "Training work files and results will be stored in $RESULT_DIR"

# Install prerequisite packages
echo "Installing prerequisite packages ..."
pip install -r training_requirements.txt

# Important: Trained model artifacts must be stored in ${RESULT_DIR}/model
# Make sure the directory exists
mkdir -p ${RESULT_DIR}/model

# ---------------------------------------------------------------
# Perform training environment setup tasks
# ---------------------------------------------------------------

MODEL_DOWNLOAD_BASE="https://storage.googleapis.com/bert_models/2018_10_18/"
MODEL_FILE="uncased_L-12_H-768_A-12.zip"
MODEL_FOLDER="uncased_L-12_H-768_A-12"

ENVIRONMENT_SETUP_CMD="python3 prepare_environment.py --DATA_DIR=${DATA_DIR} --RESULT_DIR=${RESULT_DIR} --MODEL_DOWNLOAD_BASE=${MODEL_DOWNLOAD_BASE} --MODEL_FILE=${MODEL_FILE} --MODEL_FOLDER=${MODEL_FOLDER}"

# display environment setup command
echo "Running environment setup command \"$ENVIRONMENT_SETUP_CMD\""

# run environment setup command
$ENVIRONMENT_SETUP_CMD

# capture return code
RETURN_CODE=$?
if [ $RETURN_CODE -gt 0 ]; then
  echo "Error: Environment setup run exited with status code $RETURN_CODE"
  exit $TRAINING_FAILED_RETURN_CODE
fi

echo "Environment setup completed."

# ---------------------------------------------------------------
# Perform model training tasks
# ---------------------------------------------------------------

echo "# ************************************************************"
echo "# Training model ..."
echo "# ************************************************************"

# IBM TODO: Specify the training command; the cwd contains the files from 
#           the training_code directory
# Example: "python3 train-dcgan.py --dataset ${DATA_DIR}/aligned --epoch 20"
# start training and capture return code
TRAINING_CMD="python3 finetune.py --data_type=tsv --train_data=${TRAINING_DATA} --output_dir=${RESULT_DIR} --pretrained_model_folder=${MODEL_FOLDER}"

# display training command
echo "Running training command \"$TRAINING_CMD\""

# run training command
$TRAINING_CMD

# capture return code
RETURN_CODE=$?
if [ $RETURN_CODE -gt 0 ]; then
  # the training script returned an error; exit with TRAINING_FAILED_RETURN_CODE
  echo "Error: Training run exited with status code $RETURN_CODE"
  exit $TRAINING_FAILED_RETURN_CODE
fi

echo "Training completed. Output is stored in $RESULT_DIR."

echo "# ************************************************************"
echo "# Post processing ..."
echo "# ************************************************************"

# according to WML coding guidelines the trained model should be 
# saved in ${RESULT_DIR}/model
cd ${RESULT_DIR}/model

# ---------------------------------------------------------------
# Prepare for packaging
# (1) create the staging directory structure
# (2) copy the trained model artifacts
# ---------------------------------------------------------------

cd ${RESULT_DIR}

BASE_STAGING_DIR=${RESULT_DIR}/output
# subdirectory where trained model artifacts will be stored
TRAINING_STAGING_DIR=${BASE_STAGING_DIR}/trained_model

#
# 1. make the directories
#
mkdir -p $TRAINING_STAGING_DIR

# TensorFlow-specific directories
# IBM TODO: 
#   1) Identify the serialization format that the model-serving Docker image uses
#   2) Uncomment the appropriate variable assignment 
#   3) Do NOT change the directory name
#   4) Remove the TODO comment
#MODEL_ARTIFACT_TARGET_PATH=${TRAINING_STAGING_DIR}/tensorflow/checkpoint
#MODEL_ARTIFACT_TARGET_PATH=${TRAINING_STAGING_DIR}/tensorflow/frozen_graph_def
#MODEL_ARTIFACT_TARGET_PATH=${TRAINING_STAGING_DIR}/tensorflow/saved_model
if [ -z ${MODEL_ARTIFACT_TARGET_PATH+x} ];
  then "Error. This script was not correctly customized."
  exit $CUSTOMIZATION_ERROR_RETURN_CODE
fi
mkdir -p $MODEL_ARTIFACT_TARGET_PATH

#
# 2. copy trained model artifacts to $MODEL_ARTIFACT_TARGET_PATH
# IBM TODO: 
#   1) Add commands that copy the model training artifacts from 
#   ${RESULT_DIR}/model/... to $MODEL_ARTIFACT_TARGET_PATH
#   2) Remove the TODO comment
# Example for tensorflow checkpoint files:
#if [ -d ${RESULT_DIR}/model/checkpoint ]; then
#  cp -R ${RESULT_DIR}/model/checkpoint ${TRAINING_STAGING_DIR}/tensorflow/
#fi
# Example for tensorflow frozen graph files:
#if [ -d ${RESULT_DIR}/model/frozen_graph_def ]; then
#  cp -R ${RESULT_DIR}/model/frozen_graph_def ${TRAINING_STAGING_DIR}/tensorflow/
#fi
# Example for tensorflow saved_model files:
#if [ -d ${RESULT_DIR}/model/saved_model ]; then
#  cp -R ${RESULT_DIR}/model/saved_model ${TRAINING_STAGING_DIR}/tensorflow/
#fi

# The following files should now be present in BASE_STAGING_DIR
#   trained_model/<framework-name>/<serialization-format>/file1
#   trained_model/<framework-name>/<serialization-format>subdirectory/file2
#   trained_model/<framework-name>/<serialization-format-2>file3
#   trained_model/<framework-name-2>/<serialization-format>file4
#   ...
# Example:
#   trained_model/tensorflow/checkpoint/checkpoint
#   trained_model/tensorflow/checkpoint/DCGAN.model-21.meta
#   trained_model/tensorflow/checkpoint/DCGAN.model-21.index
#   trained_model/tensorflow/checkpoint/DCGAN.model-21.data-00000-of-00001
#   trained_model/tensorflow/frozen_graph_def/frozen_inference_graph.pb

# ----------------------------------------------------------------------
# Create a compressed TAR archive containing files from $BASE_STAGING_DIR
# NO CODE CUSTOMIZATION SHOULD BE REQUIRED BEYOND THIS POINT
# ----------------------------------------------------------------------

echo "# ************************************************************"
echo "# Packaging artifacts"
echo "# ************************************************************"

# standardized archive name; do NOT change
OUTPUT_ARCHIVE=${RESULT_DIR}/model_training_output.tar.gz

CWD=`pwd`
cd $BASE_STAGING_DIR
# Create compressed archive from $BASE_STAGING_DIR 
echo "Creating downloadable archive \"$OUTPUT_ARCHIVE\"."
tar cvfz ${OUTPUT_ARCHIVE} .
RETURN_CODE=$?
if [ $RETURN_CODE -gt 0 ]; then
  # the tar command returned an error; exit with PACKAGING_FAILED_RETURN_CODE
  echo "Error: Packaging command exited with status code $RETURN_CODE."
  exit $PACKAGING_FAILED_RETURN_CODE
fi
cd $CWD

# remove the staging directory
rm -rf $BASE_STAGING_DIR

echo "Model training and packaging completed."
exit $SUCCESS_RETURN_CODE

#
# Expected result:
#  - $OUTPUT_ARCHIVE contains
#     trained_model/<framework-name>/<serialization-format>/file1
#     trained_model/<framework-name>/<serialization-format>subdirectory/file2
#     trained_model/<framework-name>/<serialization-format-2>file3
#     trained_model/<framework-name-2>/<serialization-format>file4
#     ...
