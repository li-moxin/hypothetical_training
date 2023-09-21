import os
import pickle
import torch
import random
import numpy as np

class TaTQABatchGen(object):
    def __init__(self, args, data_mode, encoder='roberta'):
        dpath =  f"tagop_{encoder}_cached_{data_mode}.pkl"
        self.is_train = data_mode == "train"
        assert self.is_train == True
        
        self.args = args
        with open(os.path.join(args.data_dir, dpath), 'rb') as f:
            print("Load data from {}.".format(os.path.join(args.data_dir, dpath)))
            data = pickle.load(f)
    
        print("loaded %d raw data" % len(data))
        
        if type(data[0]) == list:
            self.input_mode = len(data[0])
        else:
            self.input_mode = 1
        
        self.batch_size = args.batch_size
        
        if self.input_mode == 1:
            print("no paired data")
            self.all_data = self.make_one_instance(data)
            print("Loaded %d single data" % len(self.all_data))
            self.data = TaTQABatchGen.make_batches([self.all_data], self.batch_size)
            
        elif self.input_mode == 2:
            print("paired oq and hq")
            data = [d for d in zip(*data)]
            all_data_oq = self.make_one_instance(data[0])
            all_data_hq = self.make_one_instance(data[1])
            self.all_data = zip(all_data_oq, all_data_hq)
            self.all_data_single, self.all_data_double = [], []
            for line in self.all_data:
                if line[1] is None:
                    self.all_data_single.append(line)
                else:
                    self.all_data_double.append(line)
            print("Loaded", len(self.all_data_single), "single data and", len(self.all_data_double), "double data")
            self.data = TaTQABatchGen.make_batches([self.all_data_single, self.all_data_double], self.batch_size)
            
        elif self.input_mode == 3:
            print("paired oq hq nhq")
            data = [d for d in zip(*data)]
            all_data_oq = self.make_one_instance(data[0])
            all_data_hq = self.make_one_instance(data[1])
            all_data_nhq = self.make_one_instance(data[2])
            self.all_data = zip(all_data_oq, all_data_hq, all_data_nhq)
            self.all_data_single, self.all_data_double, self.all_data_triple = [], [], []
            for line in self.all_data:
                if line[0] is not None and line[1] is None and line[2] is None:
                    self.all_data_single.append(line)
                elif line[0] is not None and line[1] is not None and line[2] is None:
                    self.all_data_double.append(line)
                elif line[0] is not None and line[1] is not None and line[2] is not None:
                    self.all_data_triple.append(line)
                else:
                    print("Error!")
                    assert 0 > 1
            print("Loaded ", len(self.all_data_single), "single data and ", len(self.all_data_double), " double data and ", len(self.all_data_triple), " triple data")
            self.data = TaTQABatchGen.make_batches([self.all_data_single, self.all_data_double, self.all_data_triple], self.batch_size)
        else:
            print("Error!", self.input_mode)
            assert 0 > 1
        print("Loaded ", len(self.data), "batches")
        self.offset = 0

    def make_one_instance(self, data):
        all_data = []
        for item in data:
            if item is None:
                all_data.append(None)
                continue
            input_ids = torch.from_numpy(item["input_ids"])
            attention_mask = torch.from_numpy(item["attention_mask"])
            token_type_ids = torch.from_numpy(item["token_type_ids"])
            paragraph_mask = torch.from_numpy(item["paragraph_mask"])
            table_mask = torch.from_numpy(item["table_mask"])
            paragraph_index = torch.from_numpy(item["paragraph_index"])
            table_cell_index = torch.from_numpy(item["table_cell_index"])
            tag_labels = torch.from_numpy(item["tag_labels"])
            operator_labels = torch.tensor(item["operator_label"])
            # scale_labels = torch.tensor(item["scale_label"])
            gold_answers = item["answer_dict"]
            paragraph_tokens = item["paragraph_tokens"]
            table_cell_tokens = item["table_cell_tokens"]
            question_id = item["question_id"]
            all_data.append((input_ids, attention_mask, token_type_ids, paragraph_mask, table_mask, paragraph_index, table_cell_index, tag_labels, operator_labels, gold_answers, paragraph_tokens, table_cell_tokens, question_id))

        return all_data

    @staticmethod
    def make_batches(data, batch_size):
        batched_data = []
        for item in data:
            random.shuffle(item)
            batched_data.extend([ item[i: i + batch_size] if i + batch_size < len(item) else item[i: ] + item[: i + batch_size - len(item)] for i in range(0, len(item), batch_size)])
        random.shuffle(batched_data)
        return batched_data

    def reset(self):
        if self.input_mode == 1:
            self.data = TaTQABatchGen.make_batches([self.all_data], self.batch_size)
        elif self.input_mode == 2:
            self.data = TaTQABatchGen.make_batches([self.all_data_single, self.all_data_double], self.batch_size)
        elif self.input_mode == 3:
            self.data = TaTQABatchGen.make_batches([self.all_data_single, self.all_data_double, self.all_data_triple], self.batch_size)
        self.offset = 0

    def __len__(self):
        return len(self.data)

    def make_torch_longtensor(self, batch):
        if batch[0] is None:
            return None
        bsz = len(batch)
        input_ids_batch, attention_mask_batch, token_type_ids_batch, paragraph_mask_batch, table_mask_batch, paragraph_index_batch, table_cell_index_batch,\
        tag_labels_batch, operator_labels_batch, gold_answers_batch, paragraph_tokens_batch, table_cell_tokens_batch, question_ids_batch = zip(*batch)
        self.offset += 1
        input_ids = torch.LongTensor(bsz, 512)
        attention_mask = torch.LongTensor(bsz, 512)
        token_type_ids = torch.LongTensor(bsz, 512).fill_(0)
        paragraph_mask = torch.LongTensor(bsz, 512)
        table_mask = torch.LongTensor(bsz, 512)
        paragraph_index = torch.LongTensor(bsz, 512)
        table_cell_index = torch.LongTensor(bsz, 512)
        tag_labels = torch.LongTensor(bsz, 512)
        operator_labels = torch.LongTensor(bsz)
        #scale_labels = torch.LongTensor(bsz)
        paragraph_tokens = []
        table_cell_tokens = []
        gold_answers = []
        question_ids = []
        for i in range(bsz):
            input_ids[i] = input_ids_batch[i]
            attention_mask[i] = attention_mask_batch[i]
            token_type_ids[i] = token_type_ids_batch[i]
            paragraph_mask[i] = paragraph_mask_batch[i]
            table_mask[i] = table_mask_batch[i]
            paragraph_index[i] = paragraph_index_batch[i]
            table_cell_index[i] = table_cell_index_batch[i]
            tag_labels[i] = tag_labels_batch[i]
            operator_labels[i] = operator_labels_batch[i]
            #scale_labels[i] = scale_labels_batch[i]
            paragraph_tokens.append(paragraph_tokens_batch[i])
            table_cell_tokens.append(table_cell_tokens_batch[i])
            gold_answers.append(gold_answers_batch[i])
            question_ids.append(question_ids_batch[i])
        out_batch = {"input_ids": input_ids, "attention_mask": attention_mask, "token_type_ids":token_type_ids,
            "paragraph_mask": paragraph_mask, "tag_labels": tag_labels,
            "operator_labels": operator_labels,
            "gold_answers": gold_answers, "question_ids": question_ids,
            "table_mask": table_mask,
            "paragraph_index": paragraph_index,
            "table_cell_index": table_cell_index,
            "paragraph_tokens": paragraph_tokens,
            "table_cell_tokens": table_cell_tokens,
            }
        if self.args.cuda:
            for k in out_batch.keys():
                if isinstance(out_batch[k], torch.Tensor):
                    out_batch[k] = out_batch[k].cuda()
        return out_batch

    def __iter__(self):
        while self.offset < len(self):
            batch = self.data[self.offset]
            self.offset += 1
            if self.input_mode == 1:
                oq_batch = self.make_torch_longtensor(batch)
                yield {"oq_input_batch": oq_batch}
            elif self.input_mode == 2:
                oq_input_batch, hq_input_batch = [line[0] for line in batch], [line[1] for line in batch]
                oq_input_batch, hq_input_batch = self.make_torch_longtensor(oq_input_batch), self.make_torch_longtensor(hq_input_batch)
                yield {"oq_input_batch": oq_input_batch, "hq_input_batch": hq_input_batch}
            elif self.input_mode == 3:
                oq_input_batch, hq_input_batch, nhq_input_batch = [line[0] for line in batch], [line[1] for line in batch], [line[2] for line in batch]
                oq_input_batch, hq_input_batch, nhq_input_batch = self.make_torch_longtensor(oq_input_batch), self.make_torch_longtensor(hq_input_batch), self.make_torch_longtensor(nhq_input_batch)
                yield {"oq_input_batch": oq_input_batch, "hq_input_batch": hq_input_batch, "nhq_input_batch": nhq_input_batch}

class TaTQATestBatchGen(object):
    def __init__(self, args, data_mode, encoder='roberta'):
        dpath =  f"tagop_{encoder}_cached_{data_mode}.pkl"
        self.is_train = data_mode == "train"
        assert self.is_train == False
        self.args = args
        print(os.path.join(args.test_data_dir, dpath))
        with open(os.path.join(args.test_data_dir, dpath), 'rb') as f:
            print("Load data from {}.".format(dpath))
            data = pickle.load(f)
        print("Loading raw test data ", len(data))
        
        all_data = []
        for item in data:
            input_ids = torch.from_numpy(item["input_ids"])
            attention_mask = torch.from_numpy(item["attention_mask"])
            token_type_ids = torch.from_numpy(item["token_type_ids"])
            paragraph_mask = torch.from_numpy(item["paragraph_mask"])
            table_mask = torch.from_numpy(item["table_mask"])
            paragraph_index = torch.from_numpy(item["paragraph_index"])
            table_cell_index = torch.from_numpy(item["table_cell_index"])
            gold_answers = item["answer_dict"] if "answer_dict" in item else {'answer_type': '', 'answer': '', 'answer_from': '', 'scale': ''}
            paragraph_tokens = item["paragraph_tokens"]
            table_cell_tokens = item["table_cell_tokens"]
            question_id = item["question_id"]
            all_data.append((input_ids, attention_mask, token_type_ids, paragraph_mask, table_mask, paragraph_index,
                             table_cell_index, gold_answers, paragraph_tokens, table_cell_tokens,
                             question_id))
        print("Load data size {}.".format(len(all_data)))
        self.data = TaTQATestBatchGen.make_batches(all_data, args.eval_batch_size)
        print("Loaded", len(self.data), "batches")
        self.offset = 0

    @staticmethod
    def make_batches(data, batch_size=32):
        return [data[i:i + batch_size] for i in range(0, len(data), batch_size)]

    def reset(self):
        if self.is_train:
            indices = list(range(len(self.data)))
            random.shuffle(indices)
            self.data = [self.data[i] for i in indices]
            for i in range(len(self.data)):
                random.shuffle(self.data[i])
        self.offset = 0

    def __len__(self):
        return len(self.data)

    def __iter__(self):
        while self.offset < len(self):
            batch = self.data[self.offset]
            self.offset += 1
            input_ids_batch, attention_mask_batch, token_type_ids_batch, paragraph_mask_batch, table_mask_batch, \
            paragraph_index_batch, table_cell_index_batch, gold_answers_batch, paragraph_tokens_batch, \
            table_cell_tokens_batch, question_ids_batch = zip(*batch)
            bsz = len(batch)
            input_ids = torch.LongTensor(bsz, 512)
            attention_mask = torch.LongTensor(bsz, 512)
            token_type_ids = torch.LongTensor(bsz, 512).fill_(0)
            paragraph_mask = torch.LongTensor(bsz, 512)
            table_mask = torch.LongTensor(bsz, 512)
            paragraph_index = torch.LongTensor(bsz, 512)
            table_cell_index = torch.LongTensor(bsz, 512)

            paragraph_tokens = []
            table_cell_tokens = []
            gold_answers = []
            question_ids = []

            for i in range(bsz):
                input_ids[i] = input_ids_batch[i]
                attention_mask[i] = attention_mask_batch[i]
                token_type_ids[i] = token_type_ids_batch[i]
                paragraph_mask[i] = paragraph_mask_batch[i]
                table_mask[i] = table_mask_batch[i]
                paragraph_index[i] = paragraph_index_batch[i]
                table_cell_index[i] = table_cell_index_batch[i]
               
                paragraph_tokens.append(paragraph_tokens_batch[i])
                table_cell_tokens.append(table_cell_tokens_batch[i])
                gold_answers.append(gold_answers_batch[i])
                question_ids.append(question_ids_batch[i])
            out_batch = {"input_ids": input_ids, "attention_mask": attention_mask, "token_type_ids": token_type_ids,
                         "paragraph_mask": paragraph_mask, "paragraph_index": paragraph_index,
                         "paragraph_tokens": paragraph_tokens, "table_cell_tokens": table_cell_tokens,
                         "gold_answers": gold_answers, "question_ids": question_ids,
                         "table_mask": table_mask, "table_cell_index": table_cell_index,
                         # "paragraph_mapping_content": paragraph_mapping_content,
                         # "table_mapping_content": table_mapping_content,
                         }

            if self.args.cuda:
                for k in out_batch.keys():
                    if isinstance(out_batch[k], torch.Tensor):
                        out_batch[k] = out_batch[k].cuda()

            yield  out_batch
