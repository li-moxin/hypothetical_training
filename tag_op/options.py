import torch
from argparse import ArgumentParser


def add_data_args(parser: ArgumentParser):
    parser.add_argument("--gpu_num", default=torch.cuda.device_count(), type=int, help="training gpu num.")
    parser.add_argument("--data_dir", default="dataset/data_nhq_mix", type=str, required=True, help="data dir.")
    parser.add_argument("--save_dir", default="", type=str, required=True, help="save dir.")
    parser.add_argument("--log_file", default="train.log", type=str, help="train log file.")

def add_train_args(parser: ArgumentParser):
    parser.add_argument('--gradient_accumulation_steps', type=int, default=4,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--log_per_updates", default=100, type=int, help="log pre update size.")
    parser.add_argument("--max_epoch", default=80, type=int, help="max epoch.") # 80 for step 1, 60 for step 2.
    parser.add_argument("--weight_decay", default=5e-5, type=float, help="weight decay.")
    parser.add_argument("--learning_rate", default=5e-4, type=float, help="learning rate.")
    parser.add_argument("--grad_clipping", default=1.0, type=float, help="gradient clip.")
    parser.add_argument('--warmup', type=float, default=0.06,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument("--warmup_schedule", default="warmup_linear", type=str, help="warmup schedule.")
    parser.add_argument("--optimizer", default="adam", type=str, help="train optimizer.")
    parser.add_argument('--seed', type=int, default=123, help='random seed for data shuffling, embedding init, etc.')
    parser.add_argument('--pre_path', type=str, default=None, help="Load from pre trained.")
    parser.add_argument("--dropout", default=0.1, type=float, help="dropout.")
    parser.add_argument('--batch_size', type=int, default=16, help="batch size.")
    parser.add_argument('--eval_batch_size', type=int, default=16, help="eval batch size.")
    parser.add_argument("--eps", default=1e-6, type=float, help="ema gamma.")

def add_bert_args(parser: ArgumentParser):
    parser.add_argument("--bert_learning_rate", type=float, default=1.5e-5, help="bert learning rate.")
    parser.add_argument("--bert_weight_decay", type=float, default=0.01, help="bert weight decay.")
    parser.add_argument("--roberta_model", type=str, help="robert modle path.", default="roberta.large")

def add_inference_args(parser: ArgumentParser):
    parser.add_argument("--pre_path", type=str, help="Prepath")
    parser.add_argument("--data_mode", type=str, help="inference data mode")
    parser.add_argument("--inf_path", type=str, help="inference data path.")
    parser.add_argument("--dump_path", type=str, help="inference data path.")
    parser.add_argument('--eval_batch_size', type=int, default=16, help="eval batch size.")
