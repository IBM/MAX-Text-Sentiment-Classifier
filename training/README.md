## Train the Model with Your Own Data

This document provides instructions to train the model on Watson Machine Learning, an offering of IBM Cloud. The instructions in this document assume that you already have an IBM Cloud account. If not, please create an [IBM Cloud](https://ibm.biz/Bdz2XM) account. 

- [Prepare Data for Training](#prepare-data-for-training)
- [Train the Model](#train-the-model)
- [Rebuild the Model Serving Microservice](#rebuild-the-model-serving-microservice)

## Prepare Data for Training

To prepare your data for training complete the steps listed in [data_preparation/README.md](data_preparation/README.md).

## Train the Model

- [Install Local Prerequisites](#install-local-prerequisites)
- [Run the Setup Script](#run-the-setup-script)
- [Customize Training](#customize-training)
- [Train the Model Using Watson Machine Learning](#train-the-model-using-watson-machine-learning)

In this document `$MODEL_REPO_HOME_DIR` refers to the cloned MAX model repository directory, e.g. `/users/gone_fishing/MAX-Text-Sentiment-Classifier`. 

### Install Local Prerequisites

Open a terminal window, change dir into `$MODEL_REPO_HOME_DIR/training` and install the Python prerequisites. (Model training requires Python 3.6 or above.)

   ```
   $ cd training/

   $ pip install -r requirements.txt
    ... 
   ```

The directory contains two Python scripts, `setup_max_model_training.py` and `train_max_model.py`, which you'll use to prepare your environment for model training and to perform model training on Watson Machine Learning.

### Run the Setup Script

To perform model training, you need access to a Watson Machine Learning service instance and a Cloud Object Storage service instance on IBM Cloud. The `setup_max_model_training.py` script prepares your IBM Cloud resources for model training and configures your local environment.

#### Steps

1. Open a terminal window.

1. Locate the training configuration file. It is named `max-text-classifier-training-config.yaml`.

   ```

   $ ls *.yaml
     max-text-classifier-training-config.yaml
   ```

2. Run `setup_max_model_training.py` and follow the prompts to configure model training.

   ```
    $ python setup_max_model_training.py max-text-classifier-training-config.yaml
     ...
     ------------------------------------------------------------------------------
     Model training setup is complete and your configuration file was updated.
     ------------------------------------------------------------------------------
     Training data bucket name   : text-classifier-sample-input
     Local data directory        : sample_training_data/
     Training results bucket name: text-classifier-sample-output
     Compute configuration       : k80     
   ```

   The setup script updates the training configuration file using the information you've provided. For security reasons, confidential information, such as API keys or passwords, are _not_ stored in this file. Instead the script displays a set of environment variables that you must define to make this information available to the training script.
   
3. Once setup is completed, define the displayed environment variables. The model training script `train_max_model.py` uses those variables to access your training resources.

   MacOS/Linux example:
   
   ```
   $ export ML_APIKEY=...
   $ export ML_INSTANCE=...
   $ export ML_ENV=...
   $ export AWS_ACCESS_KEY_ID=...
   $ export AWS_SECRET_ACCESS_KEY=...
   ```

   Microsoft Windows:
   
   ```
   $ set ML_APIKEY=...
   $ set ML_INSTANCE=...
   $ set ML_ENV=...
   $ set AWS_ACCESS_KEY_ID=...
   $ set AWS_SECRET_ACCESS_KEY=...
   ```

   > If you re-run the setup script and select a different Watson Machine Learning service instance or Cloud Object Storage service instance the displayed values will change. The values do not change if you modify any other configuration setting, such as the input data bucket or the compute configuration.

### Prepare Data for Training

You can test the model training process using the sample data in the `sample_training_data` directory. To use your own data, follow the instructions in [data_preparation/README.md](data_preparation/README.md).

### Customize Training

If you wish to change the network architecture or training hyper-parameters like `epochs` etc, change the corresponding arguments in `$MODEL_REPO_HOME_DIR/training/training_code/training-parameters.sh`.

### Train the Model Using Watson Machine Learning

The `train_max_model.py` script verifies your configuration settings, packages the model training code, uploads it to Watson Machine Learning, launches the training run, monitors the training run, and downloads the trained model artifacts.

Complete the following steps in the terminal window where the earlier mentioned environment variables are defined. 

#### Steps

1. Verify that the training preparation steps complete successfully.

   ```
    $ python train_max_model.py max-text-classifier-training-config.yaml prepare
     ...
     # --------------------------------------------------------
     # Checking environment variables ...
     # --------------------------------------------------------
     ...
   ```

   If preparation completed successfully:

    - Training data is present in the Cloud Object Storage bucket that WML will access during model training.
    - Model training code is packaged `max-text-classifier-model-building-code.zip`

1. Start model training.

   ```
   $ python train_max_model.py max-text-classifier-training-config.yaml package
    ...
    # --------------------------------------------------------
    # Starting model training ...
    # --------------------------------------------------------
    Training configuration summary:
    Training run name     : train-max-...
    Training data bucket  : ...
    Results bucket        : ...
    Model-building archive: max-text-classifier-model-building-code.zip
    Model training was started. Training id: model-...
    ...
   ```

1. Note the displayed `Training id`. It uniquely identifies your training run in Watson Machine Learning.

1. Monitor training progress.

   ```
   ...
   Checking model training status every 15 seconds. Press Ctrl+C once to stop monitoring or  press Ctrl+C twice to cancel training.
   Status - (p)ending (r)unning (e)rror (c)ompleted or canceled:
   ppppprrrrrrr...
   ```

   To **stop** monitoring (but continue model training), press `Ctrl+C` once.
 
   To **restart** monitoring, run the following command, replacing `<training-id>` with the id that was displayed when you started model training. 
   
      ```
      python train_max_model.py max-text-classifier-training-config.yaml package <training-id>
      ```
  
   To **cancel** the training run, press `Ctrl+C` twice.

   After training has completed the training log file `training-log.txt` is downloaded along with the trained model artifacts.

   ```
   ...
   # --------------------------------------------------------
   # Downloading training log file "training-log.txt" ...
   # --------------------------------------------------------
   Downloading "training-.../training-log.txt" from bucket "..." to "training_output/training-log.txt"
   ..
   # --------------------------------------------------------
   # Downloading trained model archive "model_training_output.tar.gz" ...
   # --------------------------------------------------------
   Downloading "training-.../model_training_output.tar.gz" from bucket "..." to "training_output/model_training_output.tar.gz"
   ....................................................................................
   ```

   If training was terminated early due to an error only the log file is downloaded. Inspect it to identify the problem.

   ```
   $ ls training_output/
     model_training_output.tar.gz
     trained_model/
     training-log.txt 
   ```

4. Return to the parent directory `$MODEL_REPO_HOME_DIR/training`.

   ```
   $ cd ..
   ```

## Rebuild the Model-Serving Microservice

The model-serving microservice out of the box serves the pre-trained model which was trained on [IBM Claim Stance Dataset](http://www.research.ibm.com/haifa/dept/vst/debating_data.shtml). To serve the model trained on your dataset you have to rebuild the Docker image:

1. Rebuild the Docker image. In `$MODEL_REPO_HOME_DIR` run

   ```
   $ docker build -t max-text-sentiment-classifier --build-arg use_pre_trained_model=false . 
    ...
   ```
   
   > If the optional parameter `use_pre_trained_model` is set to `true` or if the parameter is not defined the Docker image will be configured to serve the pre-trained model.
   
 2. Run the customized Docker image.
 
    ```
    $ docker run -it -p 5000:5000 max-text-sentiment-classifier
    ```
