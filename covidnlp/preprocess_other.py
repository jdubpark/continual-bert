import argparse
import gc
import glob
import itertools
import json
import logging
import math
import os
import pandas as pd
import re
from shutil import copyfile
import subprocess
import time
import xml.etree.ElementTree as ET
from datetime import datetime
from multiprocess import Pool
from tqdm import tqdm

import pandas as pd
import torch
from transformers import BertTokenizer


parser = argparse.ArgumentParser()

parser.add_argument('--root_dir', '-D', dest='root_dir')
parser.add_argument('--skip_format', action='store_true')
parser.add_argument('--skip_tokenize', action='store_true')
parser.add_argument('--skip_shard', action='store_true')
parser.add_argument('--skip_bert', action='store_true')

parser.add_argument('--n_cpus', default=1, type=int)
parser.add_argument('--n_top', default=10, help='Top n sentences')
parser.add_argument('--oracle_mode', default='combination', type=str,
                    help='How to generate oracle summaries, greedy or combination, combination will generate more accurate oracles but take much longer time.')
parser.add_argument('--test', dest='is_test', action='store_true')
parser.add_argument('--use_bert_basic_tokenizer', action='store_true')
parser.add_argument('--shard_size', default=5000, type=int)

parser.add_argument('--min_nsents', default=3, type=int)
parser.add_argument('--max_nsents', default=1000, type=int)
parser.add_argument('--min_src_ntokens_per_sent', default=5, type=int)
parser.add_argument('--max_src_ntokens_per_sent', default=200, type=int)
parser.add_argument('--min_tgt_ntokens', default=5, type=int)
parser.add_argument('--max_tgt_ntokens', default=500, type=int)


REMAP = {"-lrb-": "(", "-rrb-": ")", "-lcb-": "{", "-rcb-": "}",
         "-lsb-": "[", "-rsb-": "]", "``": '"', "''": '"'}


def clean(x):
    return re.sub(
        r"-lrb-|-rrb-|-lcb-|-rcb-|-lsb-|-rsb-|``|''",
        lambda m: REMAP.get(m.group()), x)


def _get_ngrams(n, text):
    """Calcualtes n-grams.
    Args:
      n: which n-grams to calculate
      text: An array of tokens
    Returns:
      A set of n-grams
    """
    ngram_set = set()
    text_length = len(text)
    max_index_ngram_start = text_length - n
    for i in range(max_index_ngram_start + 1):
        ngram_set.add(tuple(text[i:i + n]))
    return ngram_set


def _get_word_ngrams(n, sentences):
    """Calculates word n-grams for multiple sentences.
    """
    assert len(sentences) > 0
    assert n > 0

    # words = _split_into_words(sentences)

    words = sum(sentences, [])
    # words = [w for w in words if w not in stopwords]
    return _get_ngrams(n, words)


def load_json(f_main, f_abs):
    source = []
    tgt = []
    flag = False

    with open(f_main, 'r') as f:
        json_main = json.load(f)
    with open(f_abs, 'r') as f:
        json_abs = json.load(f)

    src_sent_tokens = [
        list(t['word'].lower() for t in sent['tokens'])
        for sent in json_main['sentences']]
    tgt_sent_tokens = [
        list(t['word'].lower() for t in sent['tokens'])
        for sent in json_abs['sentences']]

    src = [clean(' '.join(tokens)).split() for tokens in src_sent_tokens]
    tgt = [clean(' '.join(tokens)).split() for tokens in tgt_sent_tokens]
    return src, tgt


def cal_rouge(evaluated_ngrams, reference_ngrams):
    reference_count = len(reference_ngrams)
    evaluated_count = len(evaluated_ngrams)

    overlapping_ngrams = evaluated_ngrams.intersection(reference_ngrams)
    overlapping_count = len(overlapping_ngrams)

    if evaluated_count == 0:
        precision = 0.0
    else:
        precision = overlapping_count / evaluated_count

    if reference_count == 0:
        recall = 0.0
    else:
        recall = overlapping_count / reference_count

    f1_score = 2.0 * ((precision * recall) / (precision + recall + 1e-8))
    return {"f": f1_score, "p": precision, "r": recall}


def _rouge_clean(s):
    return re.sub(r'[^a-zA-Z0-9 ]', '', s)


def combination_selection(doc_sent_list, abstract_sent_list, summary_size):
    max_rouge = 0.0
    max_idx = (0, 0)
    abstract = sum(abstract_sent_list, [])
    abstract = _rouge_clean(' '.join(abstract)).split()
    sents = [_rouge_clean(' '.join(s)).split() for s in doc_sent_list]
    evaluated_1grams = [_get_word_ngrams(1, [sent]) for sent in sents]
    reference_1grams = _get_word_ngrams(1, [abstract])
    evaluated_2grams = [_get_word_ngrams(2, [sent]) for sent in sents]
    reference_2grams = _get_word_ngrams(2, [abstract])

    impossible_sents = []
    for s in range(summary_size + 1):
        combinations = itertools.combinations([i for i in range(len(sents)) if i not in impossible_sents], s + 1)

        for c in combinations:
            candidates_1 = [evaluated_1grams[idx] for idx in c]
            candidates_1 = set.union(*map(set, candidates_1))
            candidates_2 = [evaluated_2grams[idx] for idx in c]
            candidates_2 = set.union(*map(set, candidates_2))
            rouge_1 = cal_rouge(candidates_1, reference_1grams)['f']
            rouge_2 = cal_rouge(candidates_2, reference_2grams)['f']

            rouge_score = rouge_1 + rouge_2
            if s == 0 and rouge_score == 0:
                impossible_sents.append(c[0])
            if rouge_score > max_rouge:
                max_idx = c
                max_rouge = rouge_score

    return sorted(list(max_idx))


def greedy_selection(doc_sent_list, abstract_sent_list, summary_size):
    max_rouge = 0.0
    abstract = sum(abstract_sent_list, [])
    abstract = _rouge_clean(' '.join(abstract)).split()
    sents = [_rouge_clean(' '.join(s)).split() for s in doc_sent_list]
    evaluated_1grams = [_get_word_ngrams(1, [sent]) for sent in sents]
    reference_1grams = _get_word_ngrams(1, [abstract])
    evaluated_2grams = [_get_word_ngrams(2, [sent]) for sent in sents]
    reference_2grams = _get_word_ngrams(2, [abstract])

    selected = []
    for s in range(summary_size):
    # for s in range(len(abstract_sent_list))
        cur_max_rouge = max_rouge
        cur_id = -1
        for i in range(len(sents)):
            if (i in selected):
                continue
            c = selected + [i]
            candidates_1 = [evaluated_1grams[idx] for idx in c]
            candidates_1 = set.union(*map(set, candidates_1))
            candidates_2 = [evaluated_2grams[idx] for idx in c]
            candidates_2 = set.union(*map(set, candidates_2))
            rouge_1 = cal_rouge(candidates_1, reference_1grams)['f']
            rouge_2 = cal_rouge(candidates_2, reference_2grams)['f']
            rouge_score = rouge_1 + rouge_2
            if rouge_score > cur_max_rouge:
                cur_max_rouge = rouge_score
                cur_id = i
        if (cur_id == -1):
            return selected
        selected.append(cur_id)
        max_rouge = cur_max_rouge

    return sorted(selected)


class BertData():
    def __init__(self, args):
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
        self.min_src_ntokens_per_sent = args.min_src_ntokens_per_sent
        self.max_src_ntokens_per_sent = args.max_src_ntokens_per_sent
        self.min_nsents = args.min_nsents
        self.max_nsents = args.max_nsents
        self.min_tgt_ntokens = args.min_tgt_ntokens
        self.max_tgt_ntokens = args.max_tgt_ntokens

        self.sep_token = '[SEP]'
        self.cls_token = '[CLS]'
        self.pad_token = '[PAD]'
        self.tgt_bos = '[unused0]'
        self.tgt_eos = '[unused1]'
        self.tgt_sent_split = '[unused2]'
        self.sep_vid = self.tokenizer.vocab[self.sep_token]
        self.cls_vid = self.tokenizer.vocab[self.cls_token]
        self.pad_vid = self.tokenizer.vocab[self.pad_token]

    def preprocess(self, src, tgt, sent_labels, use_bert_basic_tokenizer=False, is_test=False):
        if not is_test and len(src) == 0:
            return None

        original_src_txt = [' '.join(s) for s in src]

        idxs = [i for i, s in enumerate(src) if len(s) > self.min_src_ntokens_per_sent]

        _sent_labels = [0] * len(src)
        for l in sent_labels:
            _sent_labels[l] = 1

        src = [src[i][:self.max_src_ntokens_per_sent] for i in idxs]
        sent_labels = [_sent_labels[i] for i in idxs]
        src = src[:self.max_nsents]
        sent_labels = sent_labels[:self.max_nsents]

        if not is_test and len(src) < self.min_nsents:
            return None

        src_txt = [' '.join(sent) for sent in src]
        text = ' {} {} '.format(self.sep_token, self.cls_token).join(src_txt)

        src_subtokens = self.tokenizer.tokenize(text)

        src_subtokens = [self.cls_token] + src_subtokens + [self.sep_token]
        src_subtoken_idxs = self.tokenizer.convert_tokens_to_ids(src_subtokens)
        _segs = [-1] + [i for i, t in enumerate(src_subtoken_idxs) if t == self.sep_vid]
        segs = [_segs[i] - _segs[i - 1] for i in range(1, len(_segs))]
        segments_ids = []
        for i, s in enumerate(segs):
            if (i % 2 == 0):
                segments_ids += s * [0]
            else:
                segments_ids += s * [1]
        cls_ids = [i for i, t in enumerate(src_subtoken_idxs) if t == self.cls_vid]
        sent_labels = sent_labels[:len(cls_ids)]

        tgt_subtokens_str = '[unused0] ' + ' [unused2] '.join(
            [' '.join(self.tokenizer.tokenize(' '.join(tt), use_bert_basic_tokenizer=use_bert_basic_tokenizer)) for tt in tgt]) + ' [unused1]'
        tgt_subtoken = tgt_subtokens_str.split()[:self.max_tgt_ntokens]
        if not is_test and len(tgt_subtoken) < self.min_tgt_ntokens:
            return None

        tgt_subtoken_idxs = self.tokenizer.convert_tokens_to_ids(tgt_subtoken)

        tgt_txt = '<q>'.join([' '.join(tt) for tt in tgt])
        src_txt = [original_src_txt[i] for i in idxs]

        return src_subtoken_idxs, sent_labels, tgt_subtoken_idxs, segments_ids, cls_ids, src_txt, tgt_txt


def format_to_lines(params):
    f_main, f_abs, args = params
    source, tgt = load_json(f_main, f_abs)
    return {'src': source, 'tgt': tgt, 'fname': f_main}


def format_to_bert(params):
    json_f, i, args = params

    with open(json_f, 'r') as f:
        json_data = json.load(f)

    bert = BertData(args)

    dataset = []
    with tqdm(total=len(json_data), position=i+1) as spbar:
        for data in json_data:
            src, tgt, fname = data['src'], data['tgt'], data['fname']

            sent_labels = greedy_selection(src[:args.max_nsents], tgt, args.n_top)
            # sent_labels = combination_selection(src[:args.max_nsents], tgt, args.n_top)
            b_data = bert.preprocess(src, tgt, sent_labels,
                                     use_bert_basic_tokenizer=args.use_bert_basic_tokenizer,
                                     is_test=args.is_test)

            if b_data is None:
                continue
            src_subtoken_idxs, sent_labels, tgt_subtoken_idxs, segments_ids, cls_ids, src_txt, tgt_txt = b_data
            b_data_dict = {'src': src_subtoken_idxs, 'tgt': tgt_subtoken_idxs,
                           'src_sent_labels': sent_labels, 'segs': segments_ids, 'clss': cls_ids,
                           'src_txt': src_txt, 'tgt_txt': tgt_txt, 'fname': fname}
            dataset.append(b_data_dict)
            spbar.update()
        spbar.close()

    return dataset


def convert_to_shard_data(post_dir, shard_dir, args):
    shard_count = 0

    corpora = sorted([os.path.join(post_dir, f) for f in os.listdir(post_dir)
                      if not f.startswith('.') and not f.endswith('.abs.txt.json')])
    args_list = []
    for f_main in corpora:
        f_abs_name = '{}.abs.txt.json'.format(os.path.basename(f_main).split('.')[0])
        f_abs = os.path.join(post_dir, f_abs_name)
        args_list.append((f_main, f_abs, args))

    start = time.time()
    print('... (4) Packing tokenized data into shards...')
    print('Converting files count: {}'.format(len(corpora)))

    shard_count = 0
    dataset = []
    t_len = math.ceil(len(corpora) / args.shard_size)
    # imap executes in sync multiprocess manner
    # use array and shard_size to save the flow of ordered data
    with Pool(args.n_cpus) as pool:
        with tqdm(total=t_len) as pbar:
            with tqdm(total=args.shard_size) as spbar:
                for i, data in enumerate(pool.imap(format_to_lines, args_list)):
                    dataset.append(data)
                    spbar.update()
                    if i != 0 and i % args.shard_size == 0:
                        fpath = os.path.join(shard_dir, 'shard.{}.json'.format(shard_count))
                        with open(fpath, 'w') as f:
                            f.write(json.dumps(dataset))
                        dataset = []
                        shard_count += 1
                        pbar.update()
                        spbar.reset()
                        # gc.collect()
                spbar.close()
            pbar.close()

        if len(dataset) > 0:
            fpath = os.path.join(shard_dir, 'shard.{}.json'.format(shard_count))
            print('last shard {} saved'.format(shard_count))
            with open(fpath, 'w') as f:
                f.write(json.dumps(dataset))
            dataset = []
            shard_count += 1

    end = time.time()
    print('... Ending (4), time elapsed {}'.format(end-start))


def convert_to_bert_data(shard_dir, save_dir, args):
    start = time.time()
    print('... (5) Converting data to special BERT data... this will take a while')

    corpora = [os.path.join(shard_dir, f) for f in os.listdir(shard_dir)
               if not f.startswith('.') and f.endswith('.json')]
    args_list = []
    i = 0
    for f in corpora:
        args_list.append((f, i, args))
        i += 1

    with Pool(args.n_cpus) as pool:
        with tqdm(total=len(corpora)) as pbar:
            for i, dataset in enumerate(pool.imap(format_to_bert, args_list)):
                save_path = os.path.join(save_dir, '{}.pt'.format(i))
                torch.save(dataset, save_path)
                dataset = []
                gc.collect()
                pbar.update()
            pbar.close()

    end = time.time()
    print('... Ending (5), time elapsed {}'.format(end-start))


if __name__ == '__main__':
    args = parser.parse_args()

    # dirname = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.abspath(args.root_dir)
    save_dir = os.path.join(root_dir, 'bert')
    shard_dir = os.path.join(root_dir, 'shard')
    data_dir = os.path.join(root_dir, 'top1000_complete')
    txt_dir = os.path.join(root_dir, 'txt')
    post_dir = os.path.join(root_dir, 'post')

    print('... Saving final special BERT data to {}'.format(save_dir))

    """
    Tokenize raw PMC json
    """

    if not args.skip_format:
        print('... Loading data from {}'.format(data_dir))

        dirs = os.listdir(data_dir)
        print('Total files: {}'.format(len(dirs)))
        for di in tqdm(dirs):
            xml_dir = os.path.join(data_dir, di, 'Documents_xml')
            xml_file = os.path.join(xml_dir, os.listdir(xml_dir)[0])
            fname = xml_file.split('/')[-1].split('.')[0]
            abs_file = os.path.join(data_dir, di, 'summary', f'{fname}.gold.txt')

            # print(xml_file)
            # print(abs_file)
            # print(fname)
            with open(xml_file) as f:
                root = ET.fromstringlist(['<root>', f.read(), '</root>'])
                paper = root.findall('PAPER')[0]

            sents = []
            try:
                if paper.find('SECTION') is not None:
                    for section in paper.iter('SECTION'):
                        sents.append(' '.join([sent.text for sent in section.iter('S')]))
                else:
                    sents.append(' '.join([sent.text for sent in root.iter('S')]))
            except TypeError:
                continue
            # print(sents)
            sents = '\n'.join(sents)
            if len(sents) == 0:
                continue

            tpath = os.path.join(txt_dir, '{}.txt'.format(fname))
            tpath_abs = os.path.join(txt_dir, '{}.abs.txt'.format(fname))
            with open(tpath, 'w') as f:
                f.write(sents)

            if not os.path.exists(tpath_abs):
                copyfile(abs_file, tpath_abs)

    """
    Tokenize text using StanfordCoreNLP
    """

    if not args.skip_tokenize:
        start = time.time()
        print('... (2) Printing mapping of text files...')

        with open('mapping_for_corenlp.txt', 'w') as fi:
            for fname in os.listdir(txt_dir):
                fpath = os.path.join(txt_dir, fname)
                fi.write('{}\n'.format(fpath))

        end = time.time()
        print('... Ending (2), time elapsed {}'.format(end-start))

        start = time.time()
        print('... (3) Tokenizing text files, both text and abstract... this will take some time')

        command = [
            'java', 'edu.stanford.nlp.pipeline.StanfordCoreNLP' ,'-annotators',
            'tokenize,ssplit', '-ssplit.newlineIsSentenceBreak', 'always',
            '-filelist', 'mapping_for_corenlp.txt', '-outputFormat',
            'json', '-outputDirectory', post_dir, '-quiet', '-prettyPrint false 2&>1 >/dev/null']
        subprocess.call(command)

        end = time.time()
        print('... Ending (3), time elapsed {}'.format(end-start))

        os.remove("mapping_for_corenlp.txt")

    """
    Format tokenized json to special BERT data for summarization
    """

    if not args.skip_shard:
        convert_to_shard_data(post_dir, shard_dir, args)

    if not args.skip_bert:
        convert_to_bert_data(shard_dir, save_dir, args)
