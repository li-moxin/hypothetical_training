import io, requests, zipfile
import os
import json
import argparse
import torch
from datetime import datetime
from tag_op import options
import torch.nn as nn
from pprint import pprint
from tag_op.data.data_util import get_op_1, get_arithmetic_op_index_1, get_op_2, get_arithmetic_op_index_2
from tag_op.data.data_util import get_op_3, get_arithmetic_op_index_3
from tag_op.data.data_util import OPERATOR_CLASSES_
from tag_op.tagop.util import create_logger, set_environment
from tag_op.data.tatqa_batch_gen import TaTQABatchGen, TaTQATestBatchGen
from transformers import RobertaModel, BertModel
from tag_op.tagop.modeling_tagop import TagopModel
from tag_op.tagop.model import TagopFineTuningModel
from pathlib import Path
import copy

parser = argparse.ArgumentParser("TagOp training task.")
options.add_data_args(parser)
options.add_train_args(parser)
options.add_bert_args(parser)
parser.add_argument("--encoder", type=str, default='roberta')
parser.add_argument("--op_mode", type=int, default=0)
parser.add_argument("--ablation_mode", type=int, default=0)
parser.add_argument("--test_data_dir", type=str, default="dataset/data_nhq_mix")
parser.add_argument("--oq_weight", type=float, default=0.01)
parser.add_argument("--hq_weight", type=float, default=0.01)
parser.add_argument("--do_finetune", type=int, default=0) # 0 for step 1, 1 for step 2.
parser.add_argument("--model_finetune_from", type=str, default="tag_op/data_nhq_mix_b16_80epoch")
parser.add_argument("--save_model_from_epoch", type=int, default=0) # 0 for step 1, 10 for step 2,

args = parser.parse_args()
Path(args.save_dir).mkdir(parents=True, exist_ok=True)

args.cuda = args.gpu_num > 0
args_path = os.path.join(args.save_dir, "args.json")
with open(args_path, "w") as f:
    json.dump((vars(args)), f)

args.batch_size = args.batch_size // args.gradient_accumulation_steps

logger = create_logger("Roberta Training", log_file=os.path.join(args.save_dir, args.log_file))

pprint(args)
set_environment(args.seed, args.cuda)

def main():
    best_result = 0.001
    logger.info("Loading data...")

    train_itr = TaTQABatchGen(args, data_mode = "train", encoder=args.encoder)
    dev_itr = TaTQATestBatchGen(args, data_mode="dev", encoder=args.encoder)

    num_train_steps = int(args.max_epoch * len(train_itr) / args.gradient_accumulation_steps)
    logger.info("Num update steps {}!".format(num_train_steps))

    logger.info(f"Build {args.encoder} model.")
    if args.encoder == 'bert':
        bert_model = BertModel.from_pretrained('bert-large-uncased')
    elif args.encoder == 'roberta':
        bert_model = RobertaModel.from_pretrained(args.roberta_model)
    elif args.encoder == 'finbert':
        bert_model = BertModel.from_pretrained(args.finbert_model)
        
    operators = {"SPAN-TEXT": 0, "SPAN-TABLE": 1, "MULTI_SPAN": 2, "COUNT": 3}
    network = TagopModel(
        encoder = bert_model,
        config = bert_model.config,
        bsz = args.batch_size,
        operator_classes = len(operators),
        scale_classes = 5,
        operator_criterion = nn.CrossEntropyLoss(),
    )
    if args.do_finetune:
        network.load_state_dict(torch.load(os.path.join(args.model_finetune_from,"checkpoint_best.pt")))
        print("loading model from", os.path.join(args.model_finetune_from,"checkpoint_best.pt"))
        
    logger.info("Build optimizer etc...")
    model = TagopFineTuningModel(args, network, num_train_steps=num_train_steps)
    
    train_start = datetime.now()
    first = True
    print("GS loss weights", args.oq_weight, args.hq_weight)
    for epoch in range(1, args.max_epoch + 1):
        model.reset()
        if not first:
            train_itr.reset()
        first = False
        logger.info('At epoch {}'.format(epoch))
        for step, batch in enumerate(train_itr):
            model.update(batch, oq_weight = args.oq_weight, hq_weight = args.hq_weight)
            if model.step % (args.log_per_updates * args.gradient_accumulation_steps) == 0 or model.step == 1:
                logger.info("Updates[{0:6}] train loss[{1:.5f}] remaining[{2}].\r\n".format(
                    model.updates, model.train_loss.avg,
                    str((datetime.now() - train_start) / (step + 1) * (num_train_steps - step - 1)).split('.')[0]))
                model.avg_reset()

        model.reset()
        model.avg_reset()
        model.predict(dev_itr)
        metrics = model.get_metrics(logger)
        model.avg_reset()

        if epoch > args.save_model_from_epoch and metrics["em"] > best_result:
            save_prefix = os.path.join(args.save_dir, "checkpoint_best")
            model.save(save_prefix, epoch)
            best_result = metrics["em"]
            logger.info("Best eval EM {} at epoch {}.\r\n".format(best_result, epoch))


if __name__ == "__main__":
    main()
