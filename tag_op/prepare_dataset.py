import os
import pickle
import argparse
from tag_op.data.tatqa_dataset import TagTaTQAReader, TagTaTQATestReader
from transformers.tokenization_roberta import RobertaTokenizer

from transformers import BertTokenizer

parser = argparse.ArgumentParser()
parser.add_argument("--input_path", type=str, default='dataset/stress_test')
parser.add_argument("--output_dir", type=str, default="dataset/stress_test")
parser.add_argument("--passage_length_limit", type=int, default=463)
parser.add_argument("--question_length_limit", type=int, default=46)
parser.add_argument("--encoder", type=str, default="roberta")
parser.add_argument("--mode", type=str, default='dev')
parser.add_argument("--roberta_model", type=str, default='')
parser.add_argument("--data_format", type=str, default="tatqa_dataset_{}.json")

args = parser.parse_args()

if args.encoder == 'roberta':
    tokenizer = RobertaTokenizer.from_pretrained(args.roberta_model)
    sep = '<s>'
elif args.encoder == 'bert':
    tokenizer = BertTokenizer.from_pretrained('bert-large-uncased')
    sep = '[SEP]'


if args.mode == 'test':
    data_reader = TagTaTQATestReader(tokenizer, args.passage_length_limit, args.question_length_limit, sep=sep)
    data_mode = ["test"]
elif args.mode == 'dev':
    data_reader = TagTaTQATestReader(tokenizer, args.passage_length_limit, args.question_length_limit, sep=sep)
    data_mode = ["dev"]
elif args.mode == 'train':
    data_reader = TagTaTQAReader(tokenizer, args.passage_length_limit, args.question_length_limit, sep=sep)
    data_mode = ["train"]

data_format = args.data_format
print(f'==== NOTE ====: encoder:{args.encoder}, mode:{args.mode}')

for dm in data_mode:
    dpath = os.path.join(args.input_path, data_format.format(dm))
    data = data_reader._read(dpath)
    print("skipping number: ", data_reader.skip_count)
    data_reader.skip_count = 0
    print("Save data to {}.".format(os.path.join(args.output_dir, f"tagop_{args.encoder}_cached_{dm}.pkl")))
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    with open(os.path.join(args.output_dir, f"tagop_{args.encoder}_cached_{dm}.pkl"), "wb") as f:
        pickle.dump(data, f)
