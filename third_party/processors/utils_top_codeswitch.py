# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors,
# The HuggingFace Inc. team, and The XTREME Benchmark Authors.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Utility functions for NER/POS tagging tasks."""

from __future__ import absolute_import, division, print_function
from tqdm import tqdm
import logging
import os
import torch
import json
import copy
import random
from random import sample
from io import open
from torch.utils.data import Dataset
from third_party.processors.tree import *
from third_party.processors.constants import *

logger = logging.getLogger(__name__)


class InputExample(object):
    """A single training/test example for token classification."""

    def __init__(
            self,
            guid,
            words,
            intent_label,
            slot_labels,
            heads=None,
            dep_tags=None,
            pos_tags=None
    ):
        """Constructs a InputExample.

        Args:
          guid: Unique id for the example.
          words: list. The words of the sequence.
          labels: (Optional) list. The labels for each word of the sequence. This should be
          specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.words = words
        self.intent_label = intent_label
        self.slot_labels = slot_labels
        self.heads = heads
        self.dep_tags = dep_tags
        self.pos_tags = pos_tags


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(
            self,
            input_ids,
            input_mask,
            segment_ids,
            intent_label_id,
            slot_label_ids,
            dep_tag_ids=None,
            pos_tag_ids=None,
            root=None,
            trunc_token_ids=None,
            sep_token_indices=None,
            depths=None,
    ):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.intent_label_id = intent_label_id
        self.slot_label_ids = slot_label_ids
        self.root = root
        self.trunc_token_ids = trunc_token_ids
        self.sep_token_indices = sep_token_indices
        self.dep_tag_ids = dep_tag_ids
        self.pos_tag_ids = pos_tag_ids
        self.depths = depths


def read_examples_from_file(file_path, lang, lang2id=None):
    if not os.path.exists(file_path):
        logger.info("[Warning] file {} not exists".format(file_path))
        return []

    guid_index = 1
    examples = []

    # {
    #     "tokens": ["Has", "Angelika", "Kratzer", "video", "messaged", "me", "?"],
    #     "head": [5, 4, 4, 5, 0, 5, 5],
    #     "slot_labels": ["O", "B-CONTACT", "I-CONTACT", "B-TYPE_CONTENT", "O", "B-RECIPIENT", "O"],
    #     "intent_label": "GET_MESSAGE"
    # }
    with open(file_path, encoding="utf-8") as f:
        for line in f:
            ex = json.loads(line.strip())
            examples.append(
                InputExample(
                    guid="%s-%d".format(lang, guid_index),
                    words=ex['tokens'],
                    intent_label=ex['intent_label'],
                    slot_labels=ex['slot_labels'],
                    heads=ex['head'],
                    dep_tags=[tag.split(':')[0] if ':' in tag else tag \
                              for tag in ex['deptag']],
                    pos_tags=ex['postag']
                )
            )

    return examples


###########################################################################################################

class ChangeTokensToMutilLanguage(object):
    def __init__(self, covert_rate):
        #####  读取 多语言 map
        file_path = "./muse_dict_data/mutillang_map/"
        self.covert_rate = covert_rate
        all_file_list = []
        for path, dir_list, file_list in os.walk(file_path):
            for file_name in file_list:
                if ".json" in file_name:
                    all_file_list.append(os.path.join(path, file_name))

        self.all_language_map = {}
        for one_file in tqdm(all_file_list):
            file_name = one_file.split("/")[-1].split(".")[0]
            self.all_language_map[file_name] = {}
            with open(one_file, "r", encoding="utf-8") as f:
                lines = f.readlines()
                for line in lines:
                    line = line.replace("\n", "")
                    try:
                        line_json = json.loads(line)
                    except:
                        print(line)
                        pass
                    assert len(list(line_json.keys())) == 1
                    token = list(line_json.keys())[0]
                    self.all_language_map[file_name][token] = line_json[token]
            f.close()

        self.language_used()

    def language_used(self):
        self.lang_used = {
            "Indo-European": ["en", "bg", "de", "el", "es", "fr", "hi", "ru", "ur", "pt"],
            "Afro-Asiatic": ["ar"],
            "Altaic": ["tr"], "Austro-Asiatic": ["vi"], "Sino-Tibetan": ["zh"],
            "mtop_ex": ["en", "de", "es", "fr", "hi"]
        }

    def change_tokenfrom_map(self, original_token, covert_rate=None):
        #####  筛选出需要替换的token，当前默认替换为同一种语言
        #####  默认替换比例
        if covert_rate == None:
            covert_rate = self.covert_rate

        covert_num = int(len(original_token) * covert_rate)
        # covert_num = len(original_token)
        ###  随机挑选出目标语言
        lang_list = self.lang_used["Indo-European"]
        # print("####  lang_list: ",lang_list)
        if "en" in lang_list:
            lang_list.remove("en")
        #############################
        covert_token = copy.deepcopy(original_token)

        ori_token_ids = [i for i in range(len(original_token))]
        covert_token_ids = random.sample(ori_token_ids, covert_num)
        for i in covert_token_ids:
            #########   每一个token都是不同的语言
            choice_lang = random.choice(lang_list)
            while "en-" + choice_lang not in self.all_language_map.keys():
                choice_lang = random.choice(lang_list)
            lang_choose_map = self.all_language_map["en-" + choice_lang]
            choose_token = covert_token[i].lower()
            if choose_token in lang_choose_map.keys():
                covert_token[i] = lang_choose_map[choose_token][0]
            else:
                pass

        return covert_token


class ChangeTokensToMutilLanguage_TargetLang(object):
    def __init__(self, train_lang, target_lang, covert_rate):
        self.covert_rate = covert_rate
        self.train_lang = train_lang
        self.target_lang = target_lang
        print("#### Token  转换率：", self.covert_rate)
        print("#### 目标语言：", self.target_lang)
        #####  读取 多语言 map
        all_file_list = []
        file_path = os.path.join(os.getcwd(), "muse_dict_data/mutillang_map")
        for path, dir_list, file_list in os.walk(file_path):
            for file_name in file_list:
                if ".json" in file_name:
                    all_file_list.append(os.path.join(path, file_name))

        self.all_language_map = {}
        self.all_target_langs = []
        for one_file in tqdm(all_file_list):
            file_name = one_file.split("/")[-1].split(".")[0]
            self.all_language_map[file_name] = {}
            with open(one_file, "r", encoding="utf-8") as f:
                lines = f.readlines()
                for line in lines:
                    line = line.replace("\n", "")
                    try:
                        line_json = json.loads(line)
                    except:
                        print(line)
                        pass
                    assert len(list(line_json.keys())) == 1
                    token = list(line_json.keys())[0]
                    self.all_language_map[file_name][token] = line_json[token]
            f.close()
            other_lang = file_name.split("-")[1]
            if len(other_lang) < 2:
                continue
            self.all_target_langs.append(other_lang)

        self.language_used()
        self.all_target_langs = list(set(self.all_target_langs))
        print("###   all_target_langs: ", self.all_target_langs)

    def language_used(self):
        self.lang_used = {
            "Indo-European": ["en", "bg", "de", "el", "es", "fr", "hi", "ru", "ur", "pt"],
            "Afro-Asiatic": ["ar"],
            "Altaic": ["tr"], "Austro-Asiatic": ["vi"], "Sino-Tibetan": ["zh"]}

    def statistic_token_trans(self, tokens, lang_list):
        lang_choose_map_list = []
        for one_lang in lang_list:
            lang_choose_map = self.all_language_map["en-" + one_lang]
            lang_choose_map_list.append(lang_choose_map)

        all_tokens_trans_dict = {}
        for one_token in tokens:
            #####  针对每个token，遍历所有map，找出所有能够替换的map
            for one_map in lang_choose_map_list:
                if one_token in one_map.keys():
                    if one_token not in all_tokens_trans_dict.keys():
                        all_tokens_trans_dict[one_token] = one_map[one_token]
                    else:
                        all_tokens_trans_dict[one_token] = list(
                            set(all_tokens_trans_dict[one_token] + one_map[one_token]))

        ###################################
        replaceable_rate = len(all_tokens_trans_dict.keys()) / len(tokens)
        return all_tokens_trans_dict, replaceable_rate

    def change_tokenfrom_map(self, original_token):
        #####  筛选出需要替换的token，当前默认替换为同一种语言
        #####  默认替换比例
        covert_rate = self.covert_rate
        ###  随机挑选出目标语言
        lang_list = self.lang_used["Indo-European"]
        # lang_list = [self.target_lang]
        # lang_list = list(set(self.all_target_langs) - set(self.lang_used["Indo-European"]))
        # print("###  lang_list:",lang_list)
        # lang_list = self.all_target_langs

        if self.train_lang in lang_list:
            lang_list.remove(self.train_lang)
        #############################   找出 token中，能够被替换的token，并根据替换比例筛选token   #######################
        all_tokens_trans_dict, replaceable_rate = self.statistic_token_trans(original_token, lang_list)
        all_tokens_trans_list = list(all_tokens_trans_dict.keys())
        if all_tokens_trans_list:
            if replaceable_rate > covert_rate:
                if int(len(all_tokens_trans_list) * covert_rate) == len(all_tokens_trans_list):
                    covert_token_num = int(len(all_tokens_trans_list) * covert_rate)
                else:
                    covert_token_num = int(len(all_tokens_trans_list) * covert_rate) + 1
            else:
                covert_token_num = len(all_tokens_trans_list)
            replaceable_token_ids = [i for i in range(len(all_tokens_trans_list))]
            covert_token_ids = sorted(random.sample(replaceable_token_ids, covert_token_num))
            replaceable_token = [all_tokens_trans_list[one] for one in covert_token_ids]
            ############################################################################################################
            covert_token = copy.deepcopy(original_token)

            for i, one_token in enumerate(covert_token):
                #########   遍历每一个token，判断是否在 replaceable_token中
                if one_token in replaceable_token:
                    one_covert_token = list(set(all_tokens_trans_dict[one_token]))[0]
                    covert_token[i] = one_covert_token
        else:
            covert_token = original_token
        return covert_token


class ChangeTokensToMutilLanguage_TrainLang(object):
    def __init__(self, train_lang, covert_rate):
        self.train_lang = train_lang
        self.covert_rate = covert_rate
        print("#### Token  转换率：", self.covert_rate)
        #####  读取 多语言 map
        all_file_list = []
        code_switch_path = os.path.join(os.getcwd(), "muse_dict_data/mutillang_map")
        for path, dir_list, file_list in os.walk(code_switch_path):
            for file_name in file_list:
                if ".json" in file_name:
                    if self.train_lang in file_name:
                        all_file_list.append(os.path.join(path, file_name))

        if len(all_file_list) < 1:
            print("#######  语言映射文件有误！")
            exit()

        self.all_language_map = {}
        for one_file in tqdm(all_file_list):
            file_name = one_file.split("/")[-1].split(".")[0]
            # self.all_language_map[file_name] = {}

            if self.train_lang == file_name.split("-")[0]:  ###  train_lang > other_Lang
                with open(one_file, "r", encoding="utf-8") as f:
                    lines = f.readlines()
                    for line in lines:
                        line = line.replace("\n", "")
                        try:
                            line_json = json.loads(line)
                        except:
                            # print(line)
                            pass
                        assert len(list(line_json.keys())) == 1
                        token = list(line_json.keys())[0]
                        # self.all_language_map[file_name][token] = line_json[token]
                        if token not in self.all_language_map.keys():
                            self.all_language_map[token] = {}
                            self.all_language_map[token][file_name] = line_json[token]
                        else:
                            self.all_language_map[token][file_name] = line_json[token]
            else:  ###   other_Lang > train_lang
                other_lang = file_name.split("-")[0]
                new_file_name = file_name.split("-")[1] + "-" + other_lang
                with open(one_file, "r", encoding="utf-8") as f:
                    lines = f.readlines()
                    for line in lines:
                        line = line.replace("\n", "")
                        try:
                            line_json = json.loads(line)
                        except:
                            # print(line)
                            pass
                        assert len(list(line_json.keys())) == 1
                        ##########
                        new_line_json_list = []
                        if len(line_json.values()) > 1:
                            new_line_json = {}
                            key = list(line_json.keys())[0]
                            for one_value in line_json[key]:
                                new_line_json[one_value] = [key]
                                new_line_json_list.append(new_line_json)
                        else:
                            new_line_json = {}
                            key = list(line_json.keys())[0]
                            new_line_json[line_json[key][0]] = [key]
                            new_line_json_list.append(new_line_json)
                        #################################################
                        for one_line_json in new_line_json_list:
                            token = list(one_line_json.keys())[0]
                            # self.all_language_map[file_name][token] = line_json[token]
                            if token not in self.all_language_map.keys():
                                self.all_language_map[token] = {}
                                self.all_language_map[token][new_file_name] = one_line_json[token]
                            else:
                                self.all_language_map[token][new_file_name] = one_line_json[token]

            f.close()

        self.language_used()

    def language_used(self):
        self.lang_used = {
            # "Indo-European":["en","bg","de","el","es","fr","hi","ru","ur"],
            "Indo-European": ["en", "bg", "de", "el", "es", "fr", "hi", "ru", "ur"],
            "Afro-Asiatic": ["ar"],
            "Altaic": ["tr"], "Austro-Asiatic": ["vi"], "Sino-Tibetan": ["zh"],
            "mtop_ex": ["en", "de", "es", "fr", "hi"]
        }

    def change_tokenfrom_map(self, original_token):
        original_token_temp = [one.lower() for one in original_token]
        original_token = original_token_temp
        #####  筛选出需要替换的token，一句话中的token替换多种不同语言
        #####  默认替换比例
        covert_num = int(len(original_token) * self.covert_rate)
        ###  随机挑选出目标语言
        lang_list = self.lang_used["mtop_ex"]
        if self.train_lang in lang_list:
            lang_list.remove(self.train_lang)
        #############################
        ##  找出该句子中，能够被替换的所有token（被替换的语言必须包含在目标语言中）
        token_language_map = {}
        can_be_covert_token = []  ###  能够被替换的token; 例：[2,4,7,9,15]
        for j, one_token in enumerate(original_token):
            if one_token in self.all_language_map.keys():
                token_language_map[one_token] = {}
                for one_lang in self.all_language_map[one_token]:
                    if one_lang.split("-")[0] == self.train_lang:
                        change_lang = one_lang.split("-")[1]
                        if change_lang in lang_list:  ###  该词的map中，能替换的语言中，在所筛选的语言中的数据
                            if j not in can_be_covert_token:
                                can_be_covert_token.append(j)
                            if one_lang not in token_language_map[one_token].keys():  ###  避免重复添加
                                token_language_map[one_token][one_lang] = self.all_language_map[one_token][one_lang]

        ###########
        covert_token = copy.deepcopy(original_token)
        if covert_num < len(can_be_covert_token):
            covert_token_ids = sample(can_be_covert_token, covert_num)  ###  token的索引
        else:
            covert_token_ids = can_be_covert_token

        for i in covert_token_ids:
            #########   每一个token都是不同的语言
            token = covert_token[i]
            token_lang_list = list(token_language_map[token].keys())
            choice_lang = random.choice(token_lang_list)

            lang_choose_list = token_language_map[token][choice_lang]
            random.shuffle(lang_choose_list)
            covert_token[i] = lang_choose_list[0]

        return covert_token


class ChangeTokensToMutilLanguage_target_lang(object):
    def __init__(self, train_lang, target_lang, covert_rate):
        self.train_lang = train_lang
        self.target_lang = target_lang
        ####
        # print("###  设置的 target_lang 不生效。重设 target_lang：",["ar","tr","vi","zh"])
        # self.target_lang = ["ar","tr","vi","zh"]
        #####
        self.covert_rate = covert_rate
        print("#### Token  转换率：", self.covert_rate)
        print("#### 目标语言：", self.target_lang)
        #####  读取 多语言 map
        all_file_list = []
        code_switch_path = os.path.join(os.getcwd(), "muse_dict_data/mutillang_map")
        for path, dir_list, file_list in os.walk(code_switch_path):
            for file_name in file_list:
                if ".json" in file_name:
                    all_file_list.append(os.path.join(path, file_name))

        if len(all_file_list) < 1:
            print("#######  语言映射文件有误！")
            exit()
        self.all_language_map = {}
        for one_file in tqdm(all_file_list):
            file_name = one_file.split("/")[-1].split(".")[0]
            # self.all_language_map[file_name] = {}
            with open(one_file, "r", encoding="utf-8") as f:
                lines = f.readlines()
                for line in lines:
                    line = line.replace("\n", "")
                    try:
                        line_json = json.loads(line)
                    except:
                        # print(line)
                        pass
                    assert len(list(line_json.keys())) == 1
                    token = list(line_json.keys())[0]
                    # self.all_language_map[file_name][token] = line_json[token]
                    if token not in self.all_language_map.keys():
                        self.all_language_map[token] = {}
                        self.all_language_map[token][file_name] = line_json[token]
                    else:
                        self.all_language_map[token][file_name] = line_json[token]

            f.close()

        self.language_used()

    def language_used(self):
        self.lang_used = {
            # "Indo-European":["en","bg","de","el","es","fr","hi","ru","ur"],
            "Indo-European": ["bg", "de", "el", "es", "fr", "hi", "ru", "ur"],
            "Afro-Asiatic": ["ar"],
            "Altaic": ["tr"], "Austro-Asiatic": ["vi"], "Sino-Tibetan": ["zh"]}

    def change_tokenfrom_map(self, original_token):
        original_token_temp = [one.lower() for one in original_token]
        original_token = original_token_temp
        #####  筛选出需要替换的token，一句话中的token替换多种不同语言
        #####  默认替换比例
        covert_num = int(len(original_token) * self.covert_rate)
        ###  随机挑选出目标语言
        # lang_list = self.lang_used["Indo-European"]

        lang_list = [self.target_lang]

        if self.train_lang in lang_list:
            lang_list.remove(self.train_lang)
        #############################
        ##  找出该句子中，能够被替换的所有token（被替换的语言必须包含在目标语言中）
        token_language_map = {}
        can_be_covert_token = []  ###  能够被替换的token; 例：[2,4,7,9,15]
        for j, one_token in enumerate(original_token):
            if one_token in self.all_language_map.keys():
                token_language_map[one_token] = {}
                for one_lang in self.all_language_map[one_token]:
                    if one_lang.split("-")[0] == self.train_lang:
                        change_lang = one_lang.split("-")[1]
                        if change_lang in lang_list:  ###  该词的map中，能替换的语言中，在所筛选的语言中的数据
                            if j not in can_be_covert_token:
                                can_be_covert_token.append(j)
                            if one_lang not in token_language_map[one_token].keys():  ###  避免重复添加
                                token_language_map[one_token][one_lang] = self.all_language_map[one_token][one_lang]

        ###########
        covert_token = copy.deepcopy(original_token)
        if covert_num < len(can_be_covert_token):
            covert_token_ids = sample(can_be_covert_token, covert_num)  ###  token的索引
        else:
            covert_token_ids = can_be_covert_token

        for i in covert_token_ids:
            #########   每一个token都是不同的语言
            token = covert_token[i]
            token_lang_list = list(token_language_map[token].keys())
            choice_lang = random.choice(token_lang_list)

            lang_choose_list = token_language_map[token][choice_lang]
            random.shuffle(lang_choose_list)
            covert_token[i] = lang_choose_list[0]

        return covert_token


#########################################################


###########################################################################################################

def process_sentence(
        token_list,
        head_list,
        label_list,
        dep_tag_list,
        pos_tag_list,
        tokenizer,
        label_map,
        pad_token_label_id
):
    """
    When a token gets split into multiple word pieces,
    we make all the pieces (except the first) children of the first piece.
    However, only the first piece acts as the node that contains
    the dependent tokens as the children.
    """
    assert len(token_list) == len(head_list) == len(label_list) \
           == len(dep_tag_list) == len(pos_tag_list)

    text_tokens = []
    text_deptags = []
    text_postags = []
    # My name is Wa ##si Ah ##mad
    # 0  1    2  3  3    4  4
    sub_tok_to_orig_index = []
    # My name is Wa ##si Ah ##mad
    # 0  1    2  3       5
    old_index_to_new_index = []
    # My name is Wa ##si Ah ##mad
    # 1  1    1  1  0    1  0
    first_wpiece_indicator = []
    offset = 0
    labels = []
    for i, (token, label) in enumerate(zip(token_list, label_list)):
        word_tokens = tokenizer.tokenize(token)
        if len(token) != 0 and len(word_tokens) == 0:
            word_tokens = [tokenizer.unk_token]
        old_index_to_new_index.append(offset)  # word piece index
        offset += len(word_tokens)
        for j, word_token in enumerate(word_tokens):
            first_wpiece_indicator += [1] if j == 0 else [0]
            labels += [label_map[label]] if j == 0 else [pad_token_label_id]
            text_tokens.append(word_token)
            sub_tok_to_orig_index.append(i)
            text_deptags.append(dep_tag_list[i])
            text_postags.append(pos_tag_list[i])

    assert len(text_tokens) == len(sub_tok_to_orig_index), \
        "{} != {}".format(len(text_tokens), len(sub_tok_to_orig_index))
    assert len(text_tokens) == len(first_wpiece_indicator)

    text_heads = []
    head_idx = -1
    assert max(head_list) <= len(head_list), (max(head_list), len(head_list))
    # iterating over the word pieces to adjust heads
    for i, orig_idx in enumerate(sub_tok_to_orig_index):
        # orig_idx: index of the original word (the word-piece belong to)
        head = head_list[orig_idx]
        if head == 0:  # root
            # if root word is split into multiple pieces,
            # we make the first piece as the root node
            # and all the other word pieces as the child of the root node
            if head_idx == -1:
                head_idx = i + 1
                text_heads.append(0)
            else:
                text_heads.append(head_idx)
        else:
            if first_wpiece_indicator[i] == 1:
                # head indices start from 1, so subtracting 1
                head = old_index_to_new_index[head - 1]
                text_heads.append(head + 1)
            else:
                # word-piece of a token (except the first)
                # so, we make the first piece the parent of all other word pieces
                head = old_index_to_new_index[orig_idx]
                text_heads.append(head + 1)

    assert len(text_tokens) == len(text_heads), \
        "{} != {}".format(len(text_tokens), len(text_heads))

    return text_tokens, text_heads, labels, text_deptags, text_postags


def convert_examples_to_features(
        examples,
        label_list,
        max_seq_length,
        tokenizer,
        cls_token_segment_id=0,
        sep_token_extra=False,
        pad_on_left=False,
        pad_token=0,
        pad_token_segment_id=0,
        pad_token_label_id=-1,
        sequence_a_segment_id=0,
        mask_padding_with_zero=True,
        lang="en",
        use_syntax=False,
        CT=None
):
    """Loads a data file into a list of `InputBatch`s
    `cls_token_at_end` define the location of the CLS token:
      - False (Default, BERT/XLM pattern): [CLS] + A + [SEP] + B + [SEP]
      - True (XLNet/GPT pattern): A + [SEP] + B + [SEP] + [CLS]
    `cls_token_segment_id` define the segment id associated to the CLS token (0 for BERT, 2 for XLNet)
    """

    intent_label_list, slot_label_list = label_list
    intent_label_map = {label: i for i, label in enumerate(intent_label_list)}
    slot_label_map = {label: i for i, label in enumerate(slot_label_list)}
    special_tokens_count = 3 if sep_token_extra else 2

    features = []
    over_length_examples = 0
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            logger.info("Writing example %d of %d", ex_index, len(examples))
        #################################################################
        if CT:
            example.words = CT.change_tokenfrom_map(example.words)

        #################################################################
        tokens, heads, slot_label_ids, dep_tags, pos_tags = process_sentence(
            example.words,
            example.heads,
            example.slot_labels,
            example.dep_tags,
            example.pos_tags,
            tokenizer,
            slot_label_map,
            pad_token_label_id
        )

        orig_text_len = len(tokens)
        if 0 not in heads:
            # exit()
            continue
        root_idx = heads.index(0)
        text_offset = 1  # text_a follows <s>
        # So, we add 1 to head indices
        heads = np.add(heads, text_offset).tolist()
        # HEAD(<text_a> root) = index of <s> (1-based)
        heads[root_idx] = 1

        if len(tokens) > max_seq_length - special_tokens_count:
            # assert False  # we already truncated sequence
            # print("truncate token", len(tokens), max_seq_length, special_tokens_count)
            # tokens = tokens[: (max_seq_length - special_tokens_count)]
            # label_ids = label_ids[: (max_seq_length - special_tokens_count)]
            over_length_examples += 1
            continue

        tokens += [tokenizer.sep_token]
        dep_tags += [tokenizer.sep_token]
        pos_tags += [tokenizer.sep_token]
        slot_label_ids += [pad_token_label_id]
        if sep_token_extra:
            # roberta uses an extra separator b/w pairs of sentences
            tokens += [tokenizer.sep_token]
            dep_tags += [tokenizer.sep_token]
            pos_tags += [tokenizer.sep_token]
            slot_label_ids += [pad_token_label_id]
        segment_ids = [sequence_a_segment_id] * len(tokens)

        # cls_token_at_begining
        tokens = [tokenizer.cls_token] + tokens
        dep_tags = [tokenizer.cls_token] + dep_tags
        pos_tags = [tokenizer.cls_token] + pos_tags
        slot_label_ids = [pad_token_label_id] + slot_label_ids
        segment_ids = [cls_token_segment_id] + segment_ids

        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding_length = max_seq_length - len(input_ids)
        if pad_on_left:
            input_ids = ([pad_token] * padding_length) + input_ids
            input_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + input_mask
            segment_ids = ([pad_token_segment_id] * padding_length) + segment_ids
            slot_label_ids = ([pad_token_label_id] * padding_length) + slot_label_ids
        else:
            input_ids += [pad_token] * padding_length
            input_mask += [0 if mask_padding_with_zero else 1] * padding_length
            segment_ids += [pad_token_segment_id] * padding_length
            slot_label_ids += [pad_token_label_id] * padding_length

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        assert len(slot_label_ids) == max_seq_length

        # if ex_index < 5:
        #     logger.info("*** Example ***")
        #     logger.info("guid: %s", example.guid)
        #     logger.info("tokens: %s", " ".join([str(x) for x in tokens]))
        #     logger.info("input_ids: %s", " ".join([str(x) for x in input_ids]))
        #     logger.info("input_mask: %s", " ".join([str(x) for x in input_mask]))
        #     logger.info("segment_ids: %s", " ".join([str(x) for x in segment_ids]))
        #     logger.info("label_ids: %s", " ".join([str(x) for x in label_ids]))
        #     logger.info("langs: {}".format(langs))

        intent_label_id = intent_label_map[example.intent_label]

        one_ex_features = InputFeatures(
            input_ids=input_ids,
            input_mask=input_mask,
            segment_ids=segment_ids,
            intent_label_id=intent_label_id,
            slot_label_ids=slot_label_ids,
        )

        if use_syntax:
            #####################################################
            # prepare the UPOS and DEPENDENCY tag tensors
            #####################################################
            dep_tag_ids = deptag_to_id(dep_tags, tokenizer=str(type(tokenizer)))
            pos_tag_ids = upos_to_id(pos_tags, tokenizer=str(type(tokenizer)))

            if pad_on_left:
                dep_tag_ids = ([0] * padding_length) + dep_tag_ids
                pos_tag_ids = ([0] * padding_length) + pos_tag_ids
            else:
                dep_tag_ids += [0] * padding_length
                pos_tag_ids += [0] * padding_length

            assert len(input_ids) == len(dep_tag_ids)
            assert len(input_ids) == len(pos_tag_ids)
            assert len(dep_tag_ids) == max_seq_length
            assert len(pos_tag_ids) == max_seq_length

            one_ex_features.tag_ids = pos_tag_ids
            one_ex_features.dep_tag_ids = dep_tag_ids

            #####################################################
            # form the tree structure using head information
            #####################################################
            heads = [0] + heads + [1, 1] if sep_token_extra else [0] + heads + [1]
            assert len(tokens) == len(heads)
            root, nodes = head_to_tree(heads, tokens)
            #assert len(heads) == root.size()
            if len(heads) != root.size():
                continue
            sep_token_indices = [i for i, x in enumerate(tokens) if x == tokenizer.sep_token]
            depths = [nodes[i].depth() for i in range(len(nodes))]
            depths = np.asarray(depths, dtype=np.int32)

            one_ex_features.root = root
            one_ex_features.depths = depths
            one_ex_features.sep_token_indices = sep_token_indices
            one_ex_features.pos_tag_ids = pos_tag_ids

        features.append(one_ex_features)
        
    if over_length_examples > 0:
        logger.info('{} examples are discarded due to exceeding maximum length'.format(over_length_examples))
    return features


def get_intent_labels(path):
    with open(path, "r") as f:
        labels = f.read().splitlines()
    return labels


def get_slot_labels(path):
    with open(path, "r") as f:
        labels = f.read().splitlines()
    if "O" not in labels:
        labels = ["O"] + labels
    return labels


class SequenceDataset(Dataset):
    def __init__(self, features):
        self.features = features

    def __len__(self):
        return len(self.features)

    def __getitem__(self, index):
        """Generates one sample of data"""
        feature = self.features[index]
        input_ids = torch.tensor(feature.input_ids, dtype=torch.long)
        intent_label_id = torch.tensor([feature.intent_label_id], dtype=torch.long)
        slot_label_ids = torch.tensor(feature.slot_label_ids, dtype=torch.long)
        attention_mask = torch.tensor(feature.input_mask, dtype=torch.long)
        token_type_ids = torch.tensor(feature.segment_ids, dtype=torch.long)

        dist_matrix = None
        depths = None
        dep_tag_ids = None
        pos_tag_ids = None
        if feature.root is not None:
            dep_tag_ids = torch.tensor(feature.dep_tag_ids, dtype=torch.long)
            pos_tag_ids = torch.tensor(feature.pos_tag_ids, dtype=torch.long)
            dist_matrix = root_to_dist_mat(feature.root)
            if feature.trunc_token_ids is not None:
                dist_matrix = np.delete(dist_matrix, feature.trunc_token_ids, 0)  # delete rows
                dist_matrix = np.delete(dist_matrix, feature.trunc_token_ids, 1)  # delete columns
            dist_matrix = torch.tensor(dist_matrix, dtype=torch.long)  # seq_len x seq_len x max-path-len

        if feature.depths is not None:
            depths = feature.depths
            if feature.trunc_token_ids is not None:
                depths = np.delete(depths, feature.trunc_token_ids, 0)
            depths = torch.tensor(depths, dtype=torch.long)  # seq_len

        return [
            input_ids,
            attention_mask,
            token_type_ids,
            intent_label_id,
            slot_label_ids,
            dep_tag_ids,
            pos_tag_ids,
            dist_matrix,
            depths,
        ]


def batchify(batch):
    """Receives a batch of SequencePairDataset examples"""
    input_ids = torch.stack([data[0] for data in batch], dim=0)
    attention_mask = torch.stack([data[1] for data in batch], dim=0)
    token_type_ids = torch.stack([data[2] for data in batch], dim=0)
    intent_labels = torch.stack([data[3] for data in batch], dim=0)
    slot_labels = torch.stack([data[4] for data in batch], dim=0)

    dist_matrix = None
    depths = None
    dep_tag_ids = None
    pos_tag_ids = None

    if batch[0][5] is not None:
        dep_tag_ids = torch.stack([data[5] for data in batch], dim=0)

    if batch[0][6] is not None:
        pos_tag_ids = torch.stack([data[6] for data in batch], dim=0)

    if batch[0][7] is not None:
        dist_matrix = torch.full(
            (len(batch), input_ids.size(1), input_ids.size(1)), 99999, dtype=torch.long
        )
        for i, data in enumerate(batch):
            slen, slen = data[7].size()
            dist_matrix[i, :slen, :slen] = data[7]

    if batch[0][8] is not None:
        depths = torch.full(
            (len(batch), input_ids.size(1)), 99999, dtype=torch.long
        )
        for i, data in enumerate(batch):
            slen = data[8].size(0)
            depths[i, :slen] = data[8]

    return [
        input_ids,
        attention_mask,
        token_type_ids,
        intent_labels,
        slot_labels,
        dep_tag_ids,
        pos_tag_ids,
        dist_matrix,
        depths
    ]


def get_exact_match(intent_preds, intent_labels, slot_preds, slot_labels):
    """For the cases that intent and all the slots are correct (in one sentence)"""
    # Get the intent comparison result
    intent_result = (intent_preds == intent_labels)

    # Get the slot comparision result
    slot_result = []
    for preds, labels in zip(slot_preds, slot_labels):
        assert len(preds) == len(labels)
        one_sent_result = True
        for p, l in zip(preds, labels):
            if p != l:
                one_sent_result = False
                break
        slot_result.append(one_sent_result)
    slot_result = np.array(slot_result)

    exact_match_acc = np.multiply(intent_result, slot_result).mean()
    return exact_match_acc
