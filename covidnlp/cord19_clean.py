import argparse
import json
import os
import re
from tqdm import tqdm

import csv
import pandas as pd


parser = argparse.ArgumentParser()

parser.add_argument('-R', '--root', dest='root_dir', default='cord-19/cord-19-june-10',
                    help='Directory to CORD-19 downloaded')


def clean_json(json_dict):
    #
    # how about bib? they also indicate what the paper is about in general
    #
    title = json_dict['metadata']['title']
    body = json_dict['body_text']
    text = []

    for p in body:
        if p['section'] == 'Pre-publication history':
            continue
        p_text = p['text'].strip()
        p_text = re.sub('\[[\d\s,]+?\]', '', p_text)
        p_text = re.sub('\(Table \d+?\)', '', p_text)
        p_text = re.sub('\(Fig. \d+?\)', '', p_text)
        text.append(p_text)

    return {'title': title, 'text': text}


if __name__ == '__main__':
    args = parser.parse_args()

    # dirname = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.abspath(args.root_dir)
    meta_path = os.path.join(root_dir, 'metadata.csv')
    pmc_path = os.path.join(root_dir, 'document_parses', 'pmc_json')
    post_path = os.path.join(root_dir, 'document_parses', 'post_json')

    pmc_files = []

    with open(meta_path, 'r') as f:
        df = pd.read_csv(meta_path, sep=',', error_bad_lines=False, index_col=False, dtype='unicode')
        # skip papers without abstract
        df = df[df.abstract.astype(bool)]

    # pandas with tqdm requires manual update for now
    df_len = df.shape[0]
    # df_len = 10

    if not os.path.isdir(post_path):
        raise ValueError('{} is not a directory'.format(post_path))

    ppath = os.path.join(post_path, 'PMC.csv')
    write_head = False
    with open(ppath, 'w') as f:
        w = csv.writer(f)

        with tqdm(total=df_len) as pbar:
            for i, row in df.iterrows():
                if i >= df_len:
                    break
                pbar.update(1)

                fpath = os.path.join(pmc_path, '{}.xml.json'.format(row['pmcid']))
                if not os.path.isfile(fpath):
                    continue
                with open(fpath, 'r') as fi:
                    json_dict = json.load(fi)
                    dict = clean_json(json_dict)
                    dict['abstract'] = row['abstract']

                    if not write_head:
                        w.writerow(dict.keys())
                        write_head = True
                    w.writerow(dict.values())



    # print('Total completed: \t{}'.format(len(pmc_files))) # 50818 for Jun 10
