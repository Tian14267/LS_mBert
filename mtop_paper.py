# coding=utf-8
"""Fine-tuning models for NER and POS tagging."""

from __future__ import absolute_import, division, print_function

import argparse
import glob
import logging
import os
#os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import random

import numpy as np
import torch
from seqeval.metrics import precision_score, recall_score, f1_score
from tensorboardX import SummaryWriter
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data import RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange

from third_party.processors.utils_top_codeswitch import (
    convert_examples_to_features,
    get_intent_labels,
    get_slot_labels,
    SequenceDataset,
    batchify,
    read_examples_from_file,
    get_exact_match,
    ChangeTokensToMutilLanguage
)

from transformers import (
    AdamW,
    WEIGHTS_NAME,
    BertConfig,
    BertTokenizer,
    get_linear_schedule_with_warmup,
)

from third_party.modeling_bert import BertForJointClassification
#from third_party.modeling_bert import BertForJointClassification
from third_party.processors.constants import *

logger = logging.getLogger(__name__)
"""
ALL_MODELS = sum(
    (tuple(conf.pretrained_config_archive_map.keys())
     for conf in (BertConfig,)),
    ()
)
"""
MODEL_CLASSES = {
    "bert": (BertConfig, BertForJointClassification, BertTokenizer),
}


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def train(args, train_dataset, model, tokenizer, labels, pad_token_label_id, lang2id=None):
    """Train the model."""
    if args.local_rank in [-1, 0]:
        tb_writer = SummaryWriter()

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    train_dataloader = DataLoader(
        train_dataset,
        sampler=train_sampler,
        batch_size=args.train_batch_size,
        num_workers=4,
        pin_memory=True,
        collate_fn=batchify,
    )

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": args.weight_decay
        },
        {
            "params": [
                p for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0
        }
    ]
    optimizer = AdamW(
        optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon
    )
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
    )
    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            find_unused_parameters=True
        )

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                args.train_batch_size * args.gradient_accumulation_steps * (
                    torch.distributed.get_world_size() if args.local_rank != -1 else 1))
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    best_score = 0.0
    best_checkpoint = None
    patience = 0
    global_step = 0
    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    train_iterator = trange(int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0])
    set_seed(args)  # Add here for reproductibility (even between python 2 and 3)

    for _ in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])
        for step, batch in enumerate(epoch_iterator):
            model.train()
            batch = tuple(t.to(args.device) if t is not None else None for t in batch)

            inputs = dict()
            inputs['input_ids'] = batch[0]
            inputs['attention_mask'] = batch[1]
            inputs["token_type_ids"] = batch[2] if args.model_type in ["bert"] else None
            inputs["labels"] = batch[3]
            inputs['sequence_labels'] = batch[4]

            if args.use_syntax:
                inputs["dep_tag_ids"] = batch[5]
                inputs["pos_tag_ids"] = batch[6]
                inputs["dist_mat"] = batch[7]
                inputs["tree_depths"] = batch[8]

            outputs = model(**inputs)
            loss = outputs[0]

            if args.n_gpu > 1:
                # mean() to average on multi-gpu parallel training
                loss = loss.mean()
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            tr_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                if args.fp16:
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1

                if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    # Log metrics
                    if args.local_rank == -1 and args.evaluate_during_training:
                        # Only evaluate on single GPU otherwise metrics may not average well
                        results, _ = evaluate(
                            args, model, tokenizer, labels, pad_token_label_id, mode="dev",
                            lang=args.train_langs, lang2id=lang2id
                        )
                        for key, value in results.items():
                            tb_writer.add_scalar("eval_{}".format(key), value, global_step)
                    tb_writer.add_scalar("lr", scheduler.get_lr()[0], global_step)
                    tb_writer.add_scalar("loss", (tr_loss - logging_loss) / args.logging_steps, global_step)
                    logging_loss = tr_loss

                if args.local_rank in [-1, 0] and args.save_steps > 0 and global_step % args.save_steps == 0:
                    if args.save_only_best_checkpoint:
                        result, _ = evaluate(
                            args, model, tokenizer, labels, pad_token_label_id, mode="dev",
                            prefix=global_step, lang=args.train_langs, lang2id=lang2id
                        )
                        if result["exact_match"] > best_score:
                            logger.info(
                                "result['exact_match']={} > best_score={}".format(result["exact_match"], best_score)
                            )
                            best_score = result["exact_match"]
                            # Save the best model checkpoint
                            output_dir = os.path.join(args.output_dir, "checkpoint-best")
                            best_checkpoint = output_dir
                            if not os.path.exists(output_dir):
                                os.makedirs(output_dir)
                            # Take care of distributed/parallel training
                            model_to_save = model.module if hasattr(model, "module") else model
                            model_to_save.save_pretrained(output_dir)
                            torch.save(args, os.path.join(output_dir, "training_args.bin"))
                            logger.info("Saving the best model checkpoint to %s", output_dir)
                            logger.info("Reset patience to 0")
                            patience = 0
                        else:
                            patience += 1
                            logger.info("Hit patience={}".format(patience))
                            if args.eval_patience > 0 and patience > args.eval_patience:
                                logger.info("early stop! patience={}".format(patience))
                                epoch_iterator.close()
                                train_iterator.close()
                                if args.local_rank in [-1, 0]:
                                    tb_writer.close()
                                return global_step, tr_loss / global_step
                    else:
                        # Save model checkpoint
                        output_dir = os.path.join(args.output_dir, "checkpoint-{}".format(global_step))
                        if not os.path.exists(output_dir):
                            os.makedirs(output_dir)
                        # Take care of distributed/parallel training
                        model_to_save = model.module if hasattr(model, "module") else model
                        model_to_save.save_pretrained(output_dir)
                        torch.save(args, os.path.join(output_dir, "training_args.bin"))
                        logger.info("Saving model checkpoint to %s", output_dir)

            if args.max_steps > 0 and global_step > args.max_steps:
                epoch_iterator.close()
                break
        if args.max_steps > 0 and global_step > args.max_steps:
            train_iterator.close()
            break

    if args.local_rank in [-1, 0]:
        tb_writer.close()

    return global_step, tr_loss / global_step


def evaluate(args, model, tokenizer, labels, pad_token_label_id, mode, prefix="", lang="en", lang2id=None,
             print_result=True):
    eval_dataset = load_and_cache_examples(
        args, tokenizer, labels, pad_token_label_id, mode=mode, lang=lang, lang2id=lang2id
    )

    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    # Note that DistributedSampler samples randomly
    eval_sampler = SequentialSampler(eval_dataset) if args.local_rank == -1 else DistributedSampler(eval_dataset)
    eval_dataloader = DataLoader(
        eval_dataset,
        sampler=eval_sampler,
        batch_size=args.eval_batch_size,
        num_workers=4,
        pin_memory=True,
        collate_fn=batchify,
    )

    # multi-gpu evaluate
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Eval!
    logger.info("***** Running evaluation %s in %s *****" % (prefix, lang))
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0

    intent_preds = None
    slot_preds = None
    out_intent_label_ids = None
    out_slot_labels_ids = None

    model.eval()
    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        batch = tuple(t.to(args.device) if t is not None else None for t in batch)

        with torch.no_grad():
            inputs = dict()
            inputs['input_ids'] = batch[0]
            inputs['attention_mask'] = batch[1]
            inputs["token_type_ids"] = batch[2] if args.model_type in ["bert"] else None
            inputs["labels"] = batch[3]
            inputs['sequence_labels'] = batch[4]

            if args.use_syntax:
                inputs["dep_tag_ids"] = batch[5]
                inputs["pos_tag_ids"] = batch[6]
                inputs["dist_mat"] = batch[7]
                inputs["tree_depths"] = batch[8]

            outputs = model(**inputs)
            tmp_eval_loss, (intent_logits, slot_logits) = outputs[:2]

            if args.n_gpu > 1:
                # mean() to average on multi-gpu parallel evaluating
                tmp_eval_loss = tmp_eval_loss.mean()

            eval_loss += tmp_eval_loss.item()
        nb_eval_steps += 1

        # Intent prediction
        if intent_preds is None:
            intent_preds = intent_logits.detach().cpu().numpy()
            out_intent_label_ids = inputs["labels"].detach().cpu().numpy()
        else:
            intent_preds = np.append(intent_preds, intent_logits.detach().cpu().numpy(), axis=0)
            out_intent_label_ids = np.append(out_intent_label_ids, inputs["labels"].detach().cpu().numpy(), axis=0)

        # Slot prediction
        if slot_preds is None:
            if args.use_crf:
                # decode() in `torchcrf` returns list with best index directly
                slot_preds = np.array(model.crf.decode(slot_logits))
            else:
                slot_preds = slot_logits.detach().cpu().numpy()
            out_slot_labels_ids = inputs["sequence_labels"].detach().cpu().numpy()
        else:
            if args.use_crf:
                slot_preds = np.append(slot_preds, np.array(model.crf.decode(slot_logits)), axis=0)
            else:
                slot_preds = np.append(slot_preds, slot_logits.detach().cpu().numpy(), axis=0)
            out_slot_labels_ids = np.append(
                out_slot_labels_ids, inputs["sequence_labels"].detach().cpu().numpy(), axis=0
            )

    if nb_eval_steps == 0:
        results = {k: 0 for k in ["loss", "acc", "precision", "recall", "f1", "exact_match"]}
        slot_preds_list = []
        intent_preds_list = []
    else:
        eval_loss = eval_loss / nb_eval_steps
        intent_preds = np.argmax(intent_preds, axis=1)
        out_intent_label_ids = np.squeeze(out_intent_label_ids, axis=1)
        if not args.use_crf:
            slot_preds = np.argmax(slot_preds, axis=2)

        intent_label_lst, slot_label_lst = labels
        intent_preds_list = [intent_label_lst[ip] for ip in intent_preds]

        slot_label_map = {i: label for i, label in enumerate(slot_label_lst)}
        out_slot_label_list = [[] for _ in range(out_slot_labels_ids.shape[0])]
        slot_preds_list = [[] for _ in range(out_slot_labels_ids.shape[0])]

        for i in range(out_slot_labels_ids.shape[0]):
            for j in range(out_slot_labels_ids.shape[1]):
                if out_slot_labels_ids[i, j] != pad_token_label_id:
                    out_slot_label_list[i].append(slot_label_map[out_slot_labels_ids[i][j]])
                    slot_preds_list[i].append(slot_label_map[slot_preds[i][j]])

        results = {
            "loss": eval_loss,
            "acc": (intent_preds == out_intent_label_ids).mean(),
            "precision": precision_score(out_slot_label_list, slot_preds_list),
            "recall": recall_score(out_slot_label_list, slot_preds_list),
            "f1": f1_score(out_slot_label_list, slot_preds_list),
            "exact_match": get_exact_match(
                intent_preds, out_intent_label_ids, slot_preds_list, out_slot_label_list
            )
        }

    if print_result:
        logger.info("***** Evaluation result %s in %s *****" % (prefix, lang))
        for key in sorted(results.keys()):
            logger.info("  %s = %s", key, str(results[key]))

    return results, (intent_preds_list, slot_preds_list)


def load_and_cache_examples(
        args, tokenizer, labels, pad_token_label_id,
        mode, lang, lang2id=None, few_shot=-1
):
    # Make sure only the first process in distributed training process
    # the dataset, and the others will use the cache
    if args.local_rank not in [-1, 0] and not evaluate:
        torch.distributed.barrier()

    #############################
    if mode == "train":
        CT = ChangeTokensToMutilLanguage(args.covert_rate)

    else:
        CT = None
    #############################

    langs = lang.split(',')
    # Load data features from cache or dataset file
    cached_features_file = os.path.join(
        args.output_dir, "cached_{}_{}_{}_{}_{}".format(
            mode, '-'.join(langs),
            list(filter(None, args.model_name_or_path.split("/"))).pop(),
            str(args.task_name),
            str(args.max_seq_length)
        )
    )
    if os.path.exists(cached_features_file) and not args.overwrite_cache:
        logger.info("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
    else:
        logger.info("all languages = {}".format(lang))
        features = []
        suffix = args.model_name_or_path
        if os.path.isdir(suffix):
            suffix = list(filter(None, suffix.split("/"))).pop()
        for lg in langs:
            # data_file = os.path.join(args.data_dir, lg, "{}.{}".format(mode, suffix))
            data_file = os.path.join(args.data_dir, "{}-{}.jsonl".format(mode, lg))
            logger.info("Creating features from dataset file at {} in language {}".format(data_file, lg))
            examples = read_examples_from_file(data_file, lg, lang2id)
            features_lg = convert_examples_to_features(
                examples,
                labels,
                args.max_seq_length,
                tokenizer,
                cls_token_segment_id=0,
                sep_token_extra=bool(args.model_type in ["roberta", "xlm-roberta"]),
                pad_on_left=False,
                pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
                pad_token_segment_id=0,
                pad_token_label_id=pad_token_label_id,
                lang=lg,
                use_syntax=args.use_syntax,
                CT=CT
            )
            features.extend(features_lg)
        if args.local_rank in [-1, 0]:
            logger.info(
                "Saving features into cached file {}, len(features)={}".format(cached_features_file, len(features)))
            torch.save(features, cached_features_file)

    # Make sure only the first process in distributed training process
    # the dataset, and the others will use the cache
    if args.local_rank == 0 and not evaluate:
        torch.distributed.barrier()

    if few_shot > 0 and mode == 'train':
        logger.info("Original no. of examples = {}".format(len(features)))
        features = features[: few_shot]
        logger.info('Using few-shot learning on {} examples'.format(len(features)))

    # Convert to Tensors and build dataset
    # all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    # all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    # all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    # all_label_ids = torch.tensor([f.label_ids for f in features], dtype=torch.long)

    dataset = SequenceDataset(features)
    return dataset


def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--data_dir", default="./download/mtop_udpipe_processed", type=str,
                        help="The input data dir. Should contain the training files for the NER/POS task.")
    parser.add_argument("--model_type", default="bert", type=str,
                        help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()))
    parser.add_argument("--model_name_or_path", default="bert-base-multilingual-cased", type=str,
                        help="Path to pre-trained model or shortcut name selected in the list: ")
    parser.add_argument("--output_dir", default="./outputs/mtop/syntax_codeswitch_0825", type=str,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--task_name", default="mtop", type=str,
                        help="The task name.")
    parser.add_argument("--covert_rate", default=0.5, type=float, help="转换比例")
    parser.add_argument("--target_lang", default="es", type=str, help="目标语言")
    ## Other parameters
    parser.add_argument("--intent_labels", default="./download/mtop_udpipe_processed/intent_label.txt", type=str,
                        help="Path to a file containing all labels. If not specified, NER/POS labels are used.")
    parser.add_argument("--slot_labels", default="./download/mtop_udpipe_processed/slot_label.txt", type=str,
                        help="Path to a file containing all labels. If not specified, NER/POS labels are used.")
    parser.add_argument("--config_name", default="", type=str,
                        help="Pretrained config name or path if not the same as model_name")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Pretrained tokenizer name or path if not the same as model_name")
    parser.add_argument("--cache_dir", default=None, type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")
    parser.add_argument("--max_seq_length", default=96, type=int,
                        help="The maximum total input sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")
    parser.add_argument("--do_train", action="store_true",
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action="store_true",
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_predict", action="store_true",
                        help="Whether to run predictions on the test set.")
    parser.add_argument("--do_test", action="store_true",
                        help="Whether to run predictions on the test set.")
    parser.add_argument("--do_predict_dev", action="store_true",
                        help="Whether to run predictions on the dev set.")
    parser.add_argument("--init_checkpoint", default=None, type=str,
                        help="initial checkpoint for train/predict")
    parser.add_argument("--evaluate_during_training", action="store_true",
                        help="Whether to run evaluation during training at each logging step.")
    parser.add_argument("--do_lower_case", action="store_true",
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--few_shot", default=-1, type=int,
                        help="num of few-shot exampes")

    parser.add_argument("--per_gpu_train_batch_size", default=32, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--per_gpu_eval_batch_size", default=32, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=2e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=10.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")

    parser.add_argument("--logging_steps", type=int, default=50,
                        help="Log every X updates steps.")
    parser.add_argument("--save_steps", type=int, default=200,
                        help="Save checkpoint every X updates steps.")
    parser.add_argument("--save_only_best_checkpoint", default=True,
                        help="Save only the best checkpoint during training")
    parser.add_argument("--eval_all_checkpoints", default=True,
                        help="Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number")
    parser.add_argument("--no_cuda", action="store_true",
                        help="Avoid using CUDA when available")
    parser.add_argument("--overwrite_output_dir", default=True,
                        help="Overwrite the content of the output directory")
    parser.add_argument("--overwrite_cache", action="store_true",
                        help="Overwrite the cached training and evaluation sets")
    parser.add_argument("--seed", type=int, default=1111,
                        help="random seed for initialization")

    parser.add_argument("--use_crf", action="store_true", help="Whether to use CRF")
    parser.add_argument(
        "--use_structural_loss",
        type=bool,
        default=True,
        help="Whether to use structural loss",
    )
    parser.add_argument(
        "--struct_loss_coeff",
        default=1.0, type=float,
        help="Multiplying factor for the structural loss",
    )
    parser.add_argument(
        "--use_dependency_tag",
        type=bool,
        default=False,
        help="Whether to use structural distance instead of linear distance",
    )
    parser.add_argument(
        "--use_pos_tag",
        type=bool,
        default=True,
        help="Whether to use structural distance instead of linear distance",
    )
    parser.add_argument(
        "--syntactic_layers",
        type=str, default='0,1,2,3,4,5,6,7,8,9,10,11',
        help="comma separated layer indices for syntax fusion",
    )
    parser.add_argument(
        "--num_syntactic_heads",
        default=2, type=int,
        help="Number of syntactic heads",
    )
    parser.add_argument(
        "--use_syntax",
        type=bool,
        default=True,
        help="Whether to use syntax-based modeling",
    )
    parser.add_argument(
        "--freeze_word_embeddings",
        action="store_true",
        help="Whether to freeze word embeddings",
    )
    parser.add_argument(
        "--freeze_gat",
        action="store_true",
        help="Freeze the graph attention networks",
    )
    parser.add_argument(
        "--max_syntactic_distance",
        default=1, type=int,
        help="Max distance to consider during graph attention",
    )
    parser.add_argument(
        "--num_gat_layer",
        default=4, type=int,
        help="Number of layers in Graph Attention Networks (GAT)",
    )
    parser.add_argument(
        "--num_gat_head",
        default=4, type=int,
        help="Number of attention heads in Graph Attention Networks (GAT)",
    )
    parser.add_argument(
        "--batch_normalize",
        action="store_true",
        help="Apply batch normalization to <s> representation",
    )
    parser.add_argument(
        "--pretrained_gat",
        type=str, default=None,
        help="Path of the pretrained GAT model",
    )
    parser.add_argument(
        "--slot_loss_coef",
        default=1.0, type=int,
        help="Coefficient of the slot tagging loss",
    )

    parser.add_argument("--fp16", action="store_true",
                        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
    parser.add_argument("--fp16_opt_level", type=str, default="O1",
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="For distributed training: local_rank")
    parser.add_argument("--server_ip", type=str, default="", help="For distant debugging.")
    parser.add_argument("--server_port", type=str, default="", help="For distant debugging.")
    parser.add_argument("--predict_langs", type=str, default="en,es,fr,de,hi", help="prediction languages")
    parser.add_argument("--train_langs", default="en", type=str,
                        help="The languages in the training sets.")

    parser.add_argument("--log_file", type=str, default="", help="log file")
    parser.add_argument("--eval_patience", type=int, default=-1,
                        help="wait N times of decreasing dev score before early stop during training")
    args = parser.parse_args()

    args.log_file = os.path.join(args.output_dir, "train.log")
    os.makedirs(args.output_dir, exist_ok=True)

    if os.path.exists(args.output_dir) and os.listdir(
            args.output_dir) and args.do_train and not args.overwrite_output_dir:
        raise ValueError(
            "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
                args.output_dir))

    # Setup distant debugging if needed
    if args.server_ip and args.server_port:
        import ptvsd
        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:
        # Initializes the distributed backend which sychronizes nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl")
        args.n_gpu = 1
    args.device = device

    # Setup logging
    logging.basicConfig(handlers=[logging.FileHandler(args.log_file), logging.StreamHandler()],
                        format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    logging.info("Input args: %r" % args)
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
                   args.local_rank, device, args.n_gpu, bool(args.local_rank != -1), args.fp16)

    # Set seed
    set_seed(args)

    # Prepare NER/POS task
    intent_labels = get_intent_labels(args.intent_labels)
    slot_labels = get_slot_labels(args.slot_labels)
    num_intent_labels = len(intent_labels)
    num_slot_labels = len(slot_labels)
    # Use cross entropy ignore index as padding label id
    # so that only real label ids contribute to the loss later
    pad_token_label_id = CrossEntropyLoss().ignore_index

    # Load pretrained model and tokenizer
    # Make sure only the first process in distributed training loads model/vocab
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()

    args.model_type = args.model_type.lower()
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    config = config_class.from_pretrained(
        args.config_name if args.config_name else args.model_name_or_path,
        cache_dir=args.cache_dir if args.cache_dir else None
    )

    ####################################
    config.dep_tag_vocab_size = len(DEPTAG_SYMBOLS) + NUM_SPECIAL_TOKENS
    config.pos_tag_vocab_size = len(POS_SYMBOLS) + NUM_SPECIAL_TOKENS
    config.use_dependency_tag = args.use_dependency_tag
    config.use_pos_tag = args.use_pos_tag
    config.use_structural_loss = args.use_structural_loss
    config.struct_loss_coeff = args.struct_loss_coeff
    config.num_labels = num_intent_labels
    config.num_slot_labels = num_slot_labels
    config.slot_loss_coef = args.slot_loss_coef
    config.use_crf = args.use_crf
    config.num_syntactic_heads = args.num_syntactic_heads
    config.syntactic_layers = args.syntactic_layers
    config.max_syntactic_distance = args.max_syntactic_distance
    config.use_syntax = args.use_syntax
    config.batch_normalize = args.batch_normalize
    config.num_gat_layer = args.num_gat_layer
    config.num_gat_head = args.num_gat_head
    ####################################

    tokenizer = tokenizer_class.from_pretrained(
        args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
        do_lower_case=args.do_lower_case,
        cache_dir=args.cache_dir if args.cache_dir else None
    )

    if args.init_checkpoint:
        logger.info("loading from init_checkpoint={}".format(args.init_checkpoint))
        model = model_class.from_pretrained(
            args.init_checkpoint,
            config=config,
            cache_dir=args.init_checkpoint
        )
    else:
        logger.info("loading from cached model = {}".format(args.model_name_or_path))
        model = model_class.from_pretrained(
            args.model_name_or_path,
            from_tf=bool(".ckpt" in args.model_name_or_path),
            config=config,
            cache_dir=args.cache_dir if args.cache_dir else None
        )
    lang2id = config.lang2id if args.model_type == "xlm" else None
    logger.info("Using lang2id = {}".format(lang2id))

    if args.freeze_word_embeddings:
        model.bert.embeddings.word_embeddings.weight.requires_grad = False

    if args.use_syntax:
        if args.pretrained_gat:
            state_dict = torch.load(
                args.pretrained_gat, map_location=lambda storage, loc: storage
            )
            model.bert.encoder.gat_model.load_state_dict(state_dict)
            logger.info(" the gat model states are loaded from %s", args.pretrained_gat)

        if args.freeze_gat:
            for p in model.bert.encoder.gat_model.parameters():
                p.requires_grad = False

    # Make sure only the first process in distributed training loads model/vocab
    if args.local_rank == 0:
        torch.distributed.barrier()
    model.to(args.device)
    logger.info("Training/evaluation parameters %s", args)

    # Training
    if args.do_train:
        train_dataset = load_and_cache_examples(
            args, tokenizer, (intent_labels, slot_labels), pad_token_label_id, mode="train",
            lang=args.train_langs, lang2id=lang2id, few_shot=args.few_shot
        )
        global_step, tr_loss = train(
            args, train_dataset, model, tokenizer, (intent_labels, slot_labels), pad_token_label_id, lang2id
        )
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)

    # Saving best-practices: if you use default names for the model,
    # you can reload it using from_pretrained()
    if args.do_train and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        # Create output directory if needed
        if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(args.output_dir)

        # Save model, configuration and tokenizer using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        # Take care of distributed/parallel training
        logger.info("Saving model checkpoint to %s", args.output_dir)
        model_to_save = model.module if hasattr(model, "module") else model
        model_to_save.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)

        # Good practice: save your training arguments together with the model
        torch.save(args, os.path.join(args.output_dir, "training_args.bin"))

    # Initialization for evaluation
    results = {}
    if args.init_checkpoint:
        best_checkpoint = args.init_checkpoint
    elif os.path.exists(os.path.join(args.output_dir, 'checkpoint-best')):
        best_checkpoint = os.path.join(args.output_dir, 'checkpoint-best')
    else:
        best_checkpoint = args.output_dir
    best_f1 = 0

    # Evaluation
    if args.do_eval and args.local_rank in [-1, 0]:
        tokenizer = tokenizer_class.from_pretrained(args.output_dir, do_lower_case=args.do_lower_case)
        checkpoints = [args.output_dir]
        if args.eval_all_checkpoints:
            checkpoints = list(
                os.path.dirname(c) for c in sorted(glob.glob(args.output_dir + "/**/" + WEIGHTS_NAME, recursive=True)))
            logging.getLogger("pytorch_transformers.modeling_utils").setLevel(logging.WARN)
        logger.info("Evaluate the following checkpoints: %s", checkpoints)

        for checkpoint in checkpoints:
            global_step = checkpoint.split("-")[-1] if len(checkpoints) > 1 else ""
            model = model_class.from_pretrained(checkpoint)
            model.to(args.device)
            result, _ = evaluate(
                args, model, tokenizer, (intent_labels, slot_labels), pad_token_label_id, mode="dev",
                prefix=global_step, lang=args.train_langs, lang2id=lang2id
            )
            if result["f1"] > best_f1:
                best_checkpoint = checkpoint
                best_f1 = result["f1"]
            if global_step:
                result = {"{}_{}".format(global_step, k): v for k, v in result.items()}
            results.update(result)

        output_eval_file = os.path.join(args.output_dir, "eval_results.txt")
        with open(output_eval_file, "w") as writer:
            for key in sorted(results.keys()):
                writer.write("{} = {}\n".format(key, str(results[key])))
            writer.write("best checkpoint = {}, best f1 = {}\n".format(best_checkpoint, best_f1))

    # Prediction
    if args.do_predict and args.local_rank in [-1, 0]:
        logger.info("Loading the best checkpoint from {}\n".format(best_checkpoint))
        tokenizer = tokenizer_class.from_pretrained(args.output_dir, do_lower_case=args.do_lower_case)
        model = model_class.from_pretrained(best_checkpoint)
        model.to(args.device)

        output_test_results_file = os.path.join(args.output_dir, "test_results.txt")
        suffix = args.model_name_or_path
        if os.path.isdir(suffix):
            suffix = list(filter(None, suffix.split("/"))).pop()
        with open(output_test_results_file, "a") as result_writer:
            for lang in args.predict_langs.split(','):
                # if not os.path.exists(os.path.join(args.data_dir, lang, 'test.{}'.format(suffix))):
                if not os.path.exists(os.path.join(args.data_dir, "test-{}.jsonl".format(lang))):
                    logger.info("Language {} does not exist".format(lang))
                    continue
                result, predictions = evaluate(
                    args, model, tokenizer, (intent_labels, slot_labels), pad_token_label_id, mode="test",
                    lang=lang, lang2id=lang2id
                )

                # Save results
                result_writer.write("=====================\nlanguage={}\n".format(lang))
                for key in sorted(result.keys()):
                    result_writer.write("{} = {}\n".format(key, str(result[key])))

    # test
    if args.do_test and args.local_rank in [-1, 0]:
        tokenizer = tokenizer_class.from_pretrained(args.model_name_or_path, do_lower_case=args.do_lower_case)
        model = model_class.from_pretrained(args.model_name_or_path)
        model.to(args.device)

        output_test_results_file = os.path.join(args.output_dir, "test_results.txt")
        suffix = args.model_name_or_path
        if os.path.isdir(suffix):
            suffix = list(filter(None, suffix.split("/"))).pop()
        with open(output_test_results_file, "a") as result_writer:
            for lang in args.predict_langs.split(','):
                # if not os.path.exists(os.path.join(args.data_dir, lang, 'test.{}'.format(suffix))):
                if not os.path.exists(os.path.join(args.data_dir, "test-{}.jsonl".format(lang))):
                    logger.info("Language {} does not exist".format(lang))
                    continue
                result, predictions = evaluate(
                    args, model, tokenizer, (intent_labels, slot_labels), pad_token_label_id, mode="test",
                    lang=lang, lang2id=lang2id
                )

                # Save results
                result_writer.write("=====================\nlanguage={}\n".format(lang))
                for key in sorted(result.keys()):
                    result_writer.write("{} = {}\n".format(key, str(result[key])))


def save_predictions(args, predictions, output_file, text_file, idx_file, output_word_prediction=False):
    # Save predictions
    with open(text_file, "r") as text_reader, open(idx_file, "r") as idx_reader:
        text = text_reader.readlines()
        index = idx_reader.readlines()
        assert len(text) == len(index)

    # Sanity check on the predictions
    with open(output_file, "w") as writer:
        example_id = 0
        prev_id = int(index[0])
        for line, idx in zip(text, index):
            if line == "" or line == "\n":
                example_id += 1
            else:
                cur_id = int(idx)
                output_line = '\n' if cur_id != prev_id else ''
                if output_word_prediction:
                    output_line += line.split()[0] + '\t'
                output_line += predictions[example_id].pop(0) + '\n'
                writer.write(output_line)
                prev_id = cur_id


if __name__ == "__main__":
    main()
