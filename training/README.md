## Train the Model with Your Own Data

This document provides instructions to train the model on Watson Machine Learning, an offering of IBM Cloud. The instructions in this document assume that you already have an IBM Cloud account. If not, please create an [IBM Cloud](https://ibm.biz/Bdz2XM) account. 

- [Prepare Data for Training](#prepare-data-for-training)
- [Train the Model](#train-the-model)
- [Rebuild the Model Serving Microservice](#rebuild-the-model-serving-microservice)

## Prepare Data for Training

To prepare your data for training complete the steps listed in [data_preparation/README.md](data_preparation/README.md).

## Train the Model

In this document `$MODEL_REPO_HOME_DIR` refers to the cloned MAX model repository directory, e.g.
`/users/hi_there/MAX-Object-Detector`. 

### Install Local Prerequisites

Open a terminal window, change dir into `$MODEL_REPO_HOME_DIR/training` and install the Python prerequisites. (Model training requires Python 3.6 or above.)

   ```
   $ cd training/

   $ pip install -r requirements.txt
    ... 
   ```
### Use Pre Trained Weights

If you wish to perform transfer learning or resume from a previous checkpoint, place the checkpoint files in the `$MODEL_REPO_HOME_DIR/training/sample_training_data/initial_model/` folder. <Any other info about the nature of ckpt files or any specific requirements are listed here>.

### Customize Model Specific Parameters
<Any model specific changes which the user can make go here>
For example: 
>If you wish to change training hyper-parameters like `num_iterations`, `learning_rate` etc, pass the 
>corresponding arguments to `$MODEL_REPO_HOME_DIR/training/training_code/training_command.sh`. Look for `#TODO`s on the file >which will guide you. You can also change the backbone/model type to either `full` (which uses the `xception_65` >architecture) or the faster `mobile`(which uses a `mobilenet_v2` architecture). 


### Run the Setup Script

The `wml_setup.py` script prepares your local environment and your IBM Cloud resources for model training.

1. Locate the training configuration file. It is named `...-training-config.yaml`.

   ```

   $ ls *.yaml
     <...-training-config.yaml> 
   ```

1. Configure your environment for model training. Run `wml_setup.py` and follow the prompts.

   ```
    $ python wml_setup.py <...-training-config.yaml> 
     ...
   ```
   
1. After setup has completed, define the displayed environment variables. These variables provide the model training script with access credentials for your Watson Machine Learning service and Cloud Object Storage service. 

   MacOS example:

   ```
   $ export ML_INSTANCE=...
   $ export ML_USERNAME=...
   $ export ML_PASSWORD=...
   $ export ML_ENV=...
   $ export AWS_ACCESS_KEY_ID=...
   $ export AWS_SECRET_ACCESS_KEY=...
   ```
   
   > The training script `wml_train.py` requires these environment variables. If they are not set, model training cannot be started.

### Train the Model Using Watson Machine Learning

1. Verify that the training preparation steps complete successfully. Replace `<model-name.yaml>` with your configuration file.

   ```
    $ python wml_train.py <...-training-config.yaml> prepare
     ...
     # --------------------------------------------------------
     # Checking environment variables ...
     # --------------------------------------------------------
     ...
   ```

   If preparation completed successfully:

    - The required environment variables are defined.
    - Training data is present in the Cloud Object Storage bucket that Watson Machine Learning will access to train the model.
    - The model training code is packaged in a ZIP file named `<model-name>-model-building-code.zip` that Watson Machine Learning uses to train the model.

1. Start model training.

   ```
   $ python wml_train.py <...-training-config.yaml> package
    ...
    # --------------------------------------------------------
    # Starting model training ...
    # --------------------------------------------------------
    Training configuration summary:
    Training run name     : train-max-...
    Training data bucket  : ...
    Results bucket        : ...
    Model-building archive: max-...-model-building-code.zip
    Model training was started. Training id: model-...
    ...
   ```
   
    > Take note of the training id.

1. Monitor the model training progress.

   ```
   ...
   Training status is updated every 15 seconds - (p)ending (r)unning (e)rror (c)ompleted: 
   ppppprrrrrrr...
   ```

   > Training continues should your training script get disconnected (e.g. because you terminated the script or lost network connectivity). You can resume monitoring by running `python wml_train.py <...-training-config.yaml> package <training-id>`.

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

   > If training was terminated early due to an error only the log file is downloaded. Inspect it to identify the problem.

   ```
   $ ls training_output/
     model_training_output.tar.gz
     trained_model/
     training-log.txt 
   ```

1. Return to the parent directory

### Rebuild the Model-Serving Microservice

The model-serving microservice out of the box serves the pre-trained model which was trained on [insert_standard_dataset_here](dataset_URL). To serve the model trained on your dataset you have to rebuild the Docker image:

1. [Build the Docker image](https://docs.docker.com/engine/reference/commandline/build/):

   ```
   $ docker build -t <max-model-name> --build-arg use_pre_trained_model=false . 
    ...
   ```
   
   > If the optional parameter `use_pre_trained_model` is set to `true` or if the parameter is not defined the Docker image will be configured to serve the pre-trained model.
   
1. Once the Docker image build completes start the microservice by [running the container](https://docs.docker.com/engine/reference/commandline/run/):
 
 ```
 $ docker run -it -p 5000:5000 <max-model-name>
 ...
 ```
