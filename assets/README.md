# Asset Details

## Model files

The original pre-trained model files are from the [BERT](https://github.com/google-research/bert) repository, where they are available under [Apache 2.0](https://github.com/google-research/bert/blob/master/LICENSE). This pre-trained model was then finetuned on the [IBM Claim Stance Dataset](http://www.research.ibm.com/haifa/dept/vst/debating_data.shtml).

_Note: the finetuned model files are hosted on [IBM Cloud Object Storage](http://s3.us-south.cloud-object-storage.appdomain.cloud/max-assets-prod/max-text-sentiment-classifier/1.0.0/assets.tar.gz)._

## Test Examples (assets/test-examples.tsv)

This tab-separated-values file contains a fraction of the [IBM Claim Stance Dataset](http://www.research.ibm.com/haifa/dept/vst/debating_data.shtml) ([CC-BY-SA](http://creativecommons.org/licenses/by-sa/3.0/)) not used for finetuning. In the first column, the claim is listed. In the second column, the corresponding sentiment ('pos' or 'neg') is listed. Claims in this file may be used to try out and benchmark the performance of this model.
