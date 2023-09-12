# xpert
Code for XPERT algorithm from Personalized Retrieval over Millions of Items

This folder contains the code for XPERT and the Amazon 1M dataset 

## Download the data and model:
--------------------------------
1. Download the Amazon dataset as part of [data.zip](https://drive.google.com/file/d/1ySEZbqsNoZ_i1llj4az9xHJPof7G4VHo/view?usp=sharing). Extract and place in data folder
2. Download the trained model as part of [data.zip](https://drive.google.com/file/d/1ySEZbqsNoZ_i1llj4az9xHJPof7G4VHo/view?usp=sharing). Extract and place in models folder.

## Running the inference code of XPERT on Amazon 1M data:
--------------------------------------------------------

1. Set the environment variables in setup.sh file. 
Set CURR_DIR and PYTHONPATH with current_directory.
DATA_PATH  would be where the data is present (it would be CURR_DIR/data).

2. Run `source setup.sh`

3. We have provided a trained model in models/model_24.python, which will be used for evaluation.

3. Run `python src/evaluate_XPERT.py configs/evaluation.yaml`

## Data format:
-------------
1. item_features.txt contains the 768-dimensional embeddings of the Amazon product titles which were exracted from a pretrained 6-layered DistilBERT base mode.
The format of each row is: <item_id> <item_embedding>

2. final_data_test.txt and final_data_train.txt contains the test and train data respectively in the following format:
<user_id>   <label>   <label_time>  <history>
label = List of comma separated: <product_id> which are treated as the label
label_time = Timestamp of last reviewed product_id among labels
history = List of space separated: <product_id>:<timestamp> which are the user history

feat_data_bxml and user_data_test contains binarized files extracted from the files above, and are shared for fast inference.

## Cite as:
-----------
```bib
@inproceedings{vemuri2023personalized,
  title={Personalized Retrieval over Millions of Items},
  author={Vemuri, Hemanth and Agrawal, Sheshansh and Mittal, Shivam and Saini, Deepak and Soni, Akshay and Sambasivan, Abhinav V and Lu, Wenhao and Wang, Yajun and Parsana, Mehul and Kar, Purushottam and others},
  booktitle={Proceedings of the 46th International ACM SIGIR Conference on Research and Development in Information Retrieval},
  pages={1014--1022},
  year={2023}
}
```

## Currently added features:
--------------------------
1. Amazon-1M dataset.
2. Inference model and scripts for XPERT.

## Features to add:
---------------------
1. Add Amazon-10M dataset also.
2. Add text format for both Amazon-1M and Amazon-10M dataset.
3. Add dataset creation scripts.
4. Add base embedding extractions scripts and model.
5. Add global interest creation (clustering) code.
6. Add training scripts for morph operators.
