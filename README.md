# Hypothetical Training Framework
Codes and data for "Hypothetical Training for Robust Machine Reading Comprehension of Tabular Context", in Findings of ACL 2023, [pdf](https://aclanthology.org/2023.findings-acl.79.pdf). 

# Requirements
Please follow [TAT-QA repo](https://github.com/NExTplusplus/TAT-QA) to create the conda environment, and download `roberta.large` model. 
Our environment is Pytorch=1.7 CUDA=11.0. We use one 24GB RTX 3090. 

# Dataset

## Training data:
According to the paper, there are three types of questions used for hypothetical training. 
- The factual questions from `TAT-QA`: released in [this repo](https://github.com/NExTplusplus/TAT-QA). As described in the paper, we only utilize the answer types of `span`, `multi-span`, and `count`. 
- The hypothetical questions with changed answers from `TAT-HQA`: released in [this repo](https://github.com/NExTplusplus/TAT-HQA). We also use the answer types of `span`, `multi-span`, and `count`.  
- The hypothetical questions with unchanged answers, edited from TAT-HQA. We release our created data in `dataset/TAT-NHQA`. The `uid` of TAT-NHQA appends `_nhqa` to the corresponding TAT-HQA question uid, and the `original_question_uid` refers to the corresponding TAT-QA question uid. 

The training data statistics are shown as below. Note that not all factual questions have the two hypothetical questions. For validation set, we utilize the released validation set of TAT-QA&HQA for all compared methods, also filtered by the answer types, containing 1055 questions in total. 

| Data split | TAT-QA | TAT-HQA | TAT-NHQA|
| --- | --- | --- | --- |
| # Training | 7698 | 1074 | 709 |  
| # Validation | 938  | 107 | 0 | 

By running the data preprocessing steps in TAT-QA, we provide the processed pickle files in `dataset/data_nhq_triplet`, which can be directly used to run the training code. The three types of data are grouped as triplets. Remember to unzip `dataset/data_nhq_triplet/tagop_roberta_cached_train.pkl.zip`. 

## Testing data:
Apart from the validation set of TAT-QA&HQA, we also create a stress test to evaluate the model's reliance on spurious correlations. 
- stress test: our created stress test data by minimally editing the factual TAT-QA questions to change the answer. There are 921 questions in total. The original json file and processed pickle file are saved in `dataset/stress test`. 

# Trainining & Inference

## Step 1: Train the base model.

Firstly, we train a base model using a simple mix of TAT-QA, TAT-HQA and TAT-NHQA data on `TagOp` structure. Firstly, we covert the training data into a mix of them, saved in `dataset/data_nhq_mix`. 

```bash
python convert_base_model_training_data.py
```
Then, run the following command to train the base model `model_nhq_mix_80e`. 

```bash
PYTHONPATH=$PYTHONPATH:$(pwd):$(pwd)/tag_op python tag_op/trainer.py --data_dir dataset/data_nhq_mix --test_dir dataset/data_nhq_mix --save_dir tag_op/model_nhq_mix_b16_80epoch --batch_size 16 --eval_batch_size 16 --oq_weight 0 --hq_weight 0 --do_finetune 0 --max_epoch 80 --roberta_model roberta.large
```

## Step 2: Fine-tuning on gradient regularization terms.

We fine-tune the base model with the gradient regulariztaion terms. 

```bash
PYTHONPATH=$PYTHONPATH:$(pwd):$(pwd)/tag_op python tag_op/trainer.py --data_dir dataset/data_nhq_triplet --test_dir dataset/data_nhq_triplet --save_dir tag_op/model_ow001_hw001_60epoch --batch_size 8 --eval_batch_size 8 --oq_weight 0.01 --hq_weight 0.01 --do_finetune 1 --model_finetune_from tag_op/model_nhq_mix_b16_80epoch --save_model_from_epoch 10 --max_epoch 60 --learning_rate 5e-5 --bert_learning_rate 1.5e-6 --roberta_model roberta.large
```

## Step 3: Inference and evaluation. 

Use this command to inference the results on the validationset of TAT-QA&HQA, and the stress test. Answers will be saved at `tag_op/results/answer_dev.json`. 

```bash
PYTHONPAfTH=$PYTHONPATH:$(pwd) python tag_op/predictor.py --data_dir dataset/data_nhq_triplet --test_data_dir dataset/[data_nhq_triplet/stress_test] --save_dir tag_op/results --eval_batch_size 16 --model_path tag_op/model_ow001_hw001_60epoch 
```

# Reference
Please kindly add the following citation if you find our work helpful. Thanks!
```bash 
@inproceedings{li-etal-2023-hypothetical,
title = "Hypothetical Training for Robust Machine Reading Comprehension of Tabular Context",
author = "Li, Moxin  and
  Wang, Wenjie  and
  Feng, Fuli  and
  Zhang, Hanwang  and
  Wang, Qifan  and
  Chua, Tat-Seng",
booktitle = "Findings of the Association for Computational Linguistics: ACL 2023",
year = "2023",
pages = "1220--1236",
}
```

# Contact
Kindly contact us at [limoxin@u.nus.edu](mailto:limoxin@u.nus.edu) for any issue. Thank you!

