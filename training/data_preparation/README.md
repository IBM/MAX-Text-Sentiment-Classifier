## How to prepare your data for training

Follow the instructions in this document to prepare your data for model training.
- [Prerequisites](#prerequisites)
- [Preparing your data](#preparing-your-data)

## Prerequisites
No special prerequisites other than the standard data ETL tools to create a `.tsv`-formatted file.

## Preparing your data

Your training data must meet the following requirements:
- The data must be stored in a single `.tsv`-formatted file.
- The file must carry the `.tsv` extension. 
- The file must be stored in a directory named `data`.
- The file has to be made up of two columns exactly, separated by the tab-delimiter (`\t`) as is typical for a `tsv` file: 
  ```
  <text>\t<class_label>
  ```
- The entry before the `tab`-delimiter must equal a piece of text, and the element after the delimiter must be the class label corresponding to the text before the delimiter. The class label can be any string, and more than two different class labels are allowed (e.g. not just `negative` and `positive`, but also `neutral`). 
- The data file must not contain any headers.

An example can be found in the `sample_training_data` folder, for which a snippet from the `data/ibm_claim_stance_dataset_reformatted.tsv` file is given below.

```
Exposure to violent video games causes at least a temporary increase in aggression and this exposure correlates with aggression in the real world	neg
video game violence is not related to serious aggressive behavior in real life	pos
some violent video games may actually have a prosocial effect in some contexts	pos
```

This dataset is part of the [IBM Claim Stance Dataset](http://www.research.ibm.com/haifa/dept/vst/debating_data.shtml) and [IBM Project Debater](https://www.research.ibm.com/artificial-intelligence/project-debater/) (first 1000 lines), and is licensed under
[CC-BY-SA](http://creativecommons.org/licenses/by-sa/3.0/) ([LICENSE 1](http://www.research.ibm.com/haifa/dept/vst/debating_data.shtml#Project), [LICENSE 2](https://en.wikipedia.org/wiki/Wikipedia:Copyrights#Reusers.27_rights_and_obligations)).