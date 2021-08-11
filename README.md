[![Build Status](https://travis-ci.com/IBM/MAX-Text-Sentiment-Classifier.svg?branch=master)](https://travis-ci.com/IBM/MAX-Text-Sentiment-Classifier) [![API demo](https://img.shields.io/website/http/max-text-sentiment-classifier.codait-prod-41208c73af8fca213512856c7a09db52-0000.us-east.containers.appdomain.cloud/swagger.json.svg?label=API%20demo&down_message=down&up_message=up)](http://max-text-sentiment-classifier.codait-prod-41208c73af8fca213512856c7a09db52-0000.us-east.containers.appdomain.cloud)

[<img src="docs/deploy-max-to-ibm-cloud-with-kubernetes-button.png" width="400px">](http://ibm.biz/max-to-ibm-cloud-tutorial)

# IBM Developer Model Asset Exchange: Text Sentiment Classifier

This repository contains code to instantiate and deploy a text sentiment classifier. This model is able to detect whether a text fragment leans towards a positive or a negative sentiment. Optimal input examples for this model are short strings (preferably a single sentence) with correct grammar, although not a requirement.

The model is based on the [pre-trained BERT-Base, English Uncased](https://github.com/google-research/bert/blob/master/README.md) model and was fine-tuned on the [IBM Claim Stance Dataset](http://www.research.ibm.com/haifa/dept/vst/debating_data.shtml). The model files are hosted on
[IBM Cloud Object Storage](https://max-cdn.cdn.appdomain.cloud/max-text-sentiment-classifier/1.2.0/assets.tar.gz).
The code in this repository deploys the model as a web service in a Docker container. This repository was developed
as part of the [IBM Developer Model Asset Exchange](https://developer.ibm.com/exchanges/models/) and the public API is powered by [IBM Cloud](https://ibm.biz/Bdz2XM).

## Model Metadata
| Domain | Application | Industry  | Framework | Training Data | Input Data |
| --------- | --------  | -------- | --------- | --------- | --------------- | 
| Natural Language Processing (NLP) | Sentiment Analysis | General | TensorFlow | [IBM Claim Stance Dataset](http://www.research.ibm.com/haifa/dept/vst/debating_data.shtml) | Text |

## Benchmark
In the table below, the prediction accuracy of the model on the test sets of three different datasets is listed. 

The first row showcases the generalization power of our model after fine-tuning on the IBM Claims Dataset.
 The Sentiment140 (Tweets) and IMDB Reviews datasets are only used for evaluating the transfer-learning capabilities of this model. The implementation in this repository was **not** trained or fine-tuned on the Sentiment140 or IMDB reviews datasets.
 
The second row describes the performance of the BERT-Base (English - Uncased) model when fine-tuned on the specific task. This was done simply for reference, and the weights are therefore not made available.


The generalization results (first row) are very good when the input data is similar to the data used for fine-tuning (e.g. Sentiment140 (tweets) when fine-tuned on the IBM Claims Dataset). However, when a different style of text is given as input, and with a longer median length (e.g. multi-sentence IMDB reviews), the results are not as good.

| Model Type | IBM Claims | Sentiment140 | IMDB Reviews |
| ------------- | --------  | -------- | -------------- | 
| This model (fine-tuned on IBM Claims) | 94% | 83.84% | 81% |
| Models fine-tuned on the specific dataset | 94% | 84% | 90% |

## References
* _J. Devlin, M. Chang, K. Lee, K. Toutanova_, [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805), arXiv, 2018.
* [Google BERT repository](https://github.com/google-research/bert)
* [IBM Claims Stance Dataset](http://www.research.ibm.com/haifa/dept/vst/debating_data.shtml#Project) and [IBM Project Debater](https://www.research.ibm.com/artificial-intelligence/project-debater/)

## Licenses
| Component | License | Link  |
| ------------- | --------  | -------- |
| This repository | [Apache 2.0](https://www.apache.org/licenses/LICENSE-2.0) | [LICENSE](https://github.com/IBM/MAX-Text-Sentiment-Classifier/blob/master/LICENSE) |
| Fine-tuned Model Weights | [Apache 2.0](https://www.apache.org/licenses/LICENSE-2.0) | [LICENSE](https://github.com/IBM/MAX-Text-Sentiment-Classifier/blob/master/LICENSE) |
| Pre-trained Model Weights | [Apache 2.0](https://www.apache.org/licenses/LICENSE-2.0) | [LICENSE](https://github.com/google-research/bert/blob/master/LICENSE) |
| Model Code (3rd party) | [Apache 2.0](https://www.apache.org/licenses/LICENSE-2.0) | [LICENSE](https://github.com/google-research/bert/blob/master/LICENSE) |
| IBM Claims Stance Dataset for fine-tuning | [CC-BY-SA](http://creativecommons.org/licenses/by-sa/3.0/) | [LICENSE 1](http://www.research.ibm.com/haifa/dept/vst/debating_data.shtml#Project) <br> [LICENSE 2](https://en.wikipedia.org/wiki/Wikipedia:Copyrights#Reusers.27_rights_and_obligations)|

## Pre-requisites:
* `docker`: The [Docker](https://www.docker.com/) command-line interface. Follow the [installation instructions](https://docs.docker.com/install/) for your system.
* The minimum recommended resources for this model is 4GB Memory and 4 CPUs.
* If you are on x86-64/AMD64, your CPU must support [AVX](https://en.wikipedia.org/wiki/Advanced_Vector_Extensions) at the minimum.

# Deployment options

* [Deploy from Quay](#deploy-from-quay)
* [Deploy on Red Hat OpenShift](#deploy-on-red-hat-openshift)
* [Deploy on Kubernetes](#deploy-on-kubernetes)
* [Run Locally](#run-locally)

## Deploy from Quay
To run the docker image, which automatically starts the model serving API, run:

```
$ docker run -it -p 5000:5000 quay.io/codait/max-text-sentiment-classifier
```

This will pull a pre-built image from the Quay.io container registry (or use an existing image if already cachedlocally) and run it.
If you'd rather checkout and build the model locally you can follow the [run locally](#run-locally) steps below.

## Deploy on Red Hat OpenShift

You can deploy the model-serving microservice on Red Hat OpenShift by following the instructions for the OpenShift web console or the OpenShift Container Platform CLI [in this tutorial](https://developer.ibm.com/tutorials/deploy-a-model-asset-exchange-microservice-on-red-hat-openshift/), specifying `quay.io/codait/max-text-sentiment-classifier` as the image name.

> Note that this model requires at least 4GB of RAM. Therefore this model will not run in a cluster that was provisioned under the [OpenShift Online starter plan](https://www.openshift.com/products/online/), which is capped at 2GB.

## Deploy on Kubernetes
You can also deploy the model on Kubernetes using the latest docker image on Quay.

On your Kubernetes cluster, run the following commands:

```
$ kubectl apply -f https://github.com/IBM/MAX-Text-Sentiment-Classifier/raw/master/max-text-sentiment-classifier.yaml
```

The model will be available internally at port `5000`, but can also be accessed externally through the `NodePort`.

A more elaborate tutorial on how to deploy this MAX model to production on [IBM Cloud](https://ibm.biz/Bdz2XM) can be found [here](http://ibm.biz/max-to-ibm-cloud-tutorial).

## Run Locally
1. [Build the Model](#1-build-the-model)
2. [Deploy the Model](#2-deploy-the-model)
3. [Use the Model](#3-use-the-model)
4. [Development](#4-development)
5. [Cleanup](#5-cleanup)


### 1. Build the Model
Clone this repository locally. In a terminal, run the following command:

```
$ git clone https://github.com/IBM/MAX-Text-Sentiment-Classifier.git
```

Change directory into the repository base folder:

```
$ cd MAX-Text-Sentiment-Classifier
```

To build the docker image locally, run: 

```
$ docker build -t max-text-sentiment-classifier .
```

All required model assets will be downloaded during the build process. _Note_ that currently this docker image is CPU only (we will add support for GPU images later).


### 2. Deploy the Model
To run the docker image, which automatically starts the model serving API, run:

```
$ docker run -it -p 5000:5000 max-text-sentiment-classifier
```

### 3. Use the Model

The API server automatically generates an interactive Swagger documentation page. Go to `http://localhost:5000` to load it. From there you can explore the API and also create test requests.

```
Example:
[
"The Model Asset Exchange is a crucial element of a developer's toolkit.",
"2008 was a dark, dark year for stock markets worldwide."
]

Result:
[
  {
    "positive": 0.9977352619171143,
    "negative": 0.002264695707708597
  }
],
[
  {
    "positive": 0.001138084102421999,
    "negative": 0.9988619089126587
  }
]
```


Use the `model/predict` endpoint to submit input text in json format. The json structure should have one key, `text`, with as value a list of input strings to be analyzed. An example can be found in the image below.

Submitting proper json data triggers the model and will return a json file with a `status` and a `predictions` key. With this `predictions` field, a list of class labels and their corresponding probabilities will be associated. The first element in the list corresponds to the prediction for the first string in the input list.


![Swagger UI Screenshot](docs/swagger-screenshot.png)

You can also test it on the command line, for example:

```bash
$ curl -d "{ \"text\": [ \"The Model Asset Exchange is a crucial element of a developer's toolkit.\" ]}" -X POST "http://localhost:5000/model/predict" -H "Content-Type: application/json"
```

You should see a JSON response like that below:

```json
{
  "status": "ok",
  "predictions": [
    [
      {
        "positive": 0.9977352619171143,
        "negative": 0.0022646968718618155
      }
    ]
  ]
}
```

### 4. Development
To run the Flask API app in debug mode, edit `config.py` to set `DEBUG = True` under the application settings. You will then need to rebuild the docker image (see [step 1](#1-build-the-model)).

### 5. Cleanup
To stop the Docker container, type `CTRL` + `C` in your terminal.

## Resources and Contributions
   
If you are interested in contributing to the Model Asset Exchange project or have any queries, please follow the instructions [here](https://github.com/CODAIT/max-central-repo).
