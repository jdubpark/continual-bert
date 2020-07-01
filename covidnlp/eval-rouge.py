import argparse
import os
import pyrouge
import re
import random
import shutil
import time


parser = argparse.ArgumentParser(description='Test')
parser.add_argument('--candidate', '-C', dest='cand_dir', help='Directory containing candidate text')
parser.add_argument('--gold', '-G', dest='gold_dir', help='Directory containing gold text')
parser.add_argument('--dump_dir', '-O', dest='dump_dir', help='Root directory to dump all logs and checkpoints')
parser.add_argument('--rouge_dir', '-R', dest='rouge_dir', default='../pyrouge-git/tools/ROUGE-1.5.5/',
                    help='Root directory to ROUGE-1.5.5')


def test_rouge(cand_file, gold_file, temp_dir, rouge_dir):
    with open(cand_file, 'r') as f:
        candidates = [line.strip() for line in f.readlines()[0].split(' . ')]
    with open(gold_file, 'r') as f:
        references = [line.strip() for line in f.readlines()]

    clen = len(candidates)
    rlen = len(references)

    if clen > rlen:
        candidates = candidates[:rlen]
        clen = len(candidates)

    print(f'# of sentences in candidate file: {clen}')
    print(f'# of sentences in reference file: {rlen}')
    # assert clen == rlen

    current_time = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime())
    tmp_dir = os.path.join(temp_dir, "rouge-tmp-{}".format(current_time))
    if not os.path.isdir(tmp_dir):
        os.mkdir(tmp_dir)
        os.mkdir(tmp_dir + "/candidate")
        os.mkdir(tmp_dir + "/reference")
    try:
        for i in range(clen):
            # print(i, references[i])
            if len(references[i]) < 1:
                continue
            with open(tmp_dir + "/candidate/cand.{}.txt".format(i), "w",
                      encoding="utf-8") as f:
                f.write(candidates[i])
            with open(tmp_dir + "/reference/ref.{}.txt".format(i), "w",
                      encoding="utf-8") as f:
                f.write(references[i])
        r = pyrouge.Rouge155(rouge_dir)
        r.model_dir = tmp_dir + "/reference/"
        r.system_dir = tmp_dir + "/candidate/"
        r.model_filename_pattern = 'ref.#ID#.txt'
        r.system_filename_pattern = r'cand.(\d+).txt'
        rouge_results = r.convert_and_evaluate()
        # print(rouge_results)
        results_dict = r.output_to_dict(rouge_results)
    finally:
        pass
        if os.path.isdir(tmp_dir):
            shutil.rmtree(tmp_dir)
    return results_dict


def rouge_results(results_dict):
    # ">> ROUGE-F(1/2/3/l): {:.2f}/{:.2f}/{:.2f}\nROUGE-R(1/2/3/l): {:.2f}/{:.2f}/{:.2f}\n"
    return (
        results_dict["rouge_1_f_score"] * 100,
        results_dict["rouge_2_f_score"] * 100,
        results_dict["rouge_l_f_score"] * 100,
        results_dict["rouge_1_recall"] * 100,
        results_dict["rouge_2_recall"] * 100,
        results_dict["rouge_l_recall"] * 100
    )


if __name__ == '__main__':
    args = parser.parse_args()

    args.cand_dir = os.path.abspath(args.cand_dir)
    args.gold_dir = os.path.abspath(args.gold_dir)
    args.dump_dir = os.path.abspath(args.dump_dir)
    args.rouge_dir = os.path.abspath(args.rouge_dir)
    temp_dir = os.path.join(args.dump_dir, 'tmp')
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)

    i = 0
    acc = {'f1': 0, 'f2': 0, 'fl': 0, 'r1': 0, 'r2': 0, 'rl': 0}
    files = os.listdir(args.cand_dir)
    random.shuffle(files)
    for f in files:
        i += 1
        fname = f.split('.')[0]
        cand_file = os.path.join(args.cand_dir, f)
        gold_file = os.path.join(args.gold_dir, f'{fname}.abs.txt')
        rouges = test_rouge(cand_file, gold_file, temp_dir, args.rouge_dir)
        f1, f2, fl, r1, r2, rl = rouge_results(rouges)
        acc['f1'] += f1
        acc['f2'] += f2
        acc['fl'] += fl
        acc['r1'] += r1
        acc['r2'] += r2
        acc['rl'] += rl

    print('Rouge F1', acc['f1']/i)
    print('Rouge F2', acc['f2']/i)
    print('Rouge Fl', acc['fl']/i)
    print('Rouge R1', acc['r1']/i)
    print('Rouge R2', acc['r2']/i)
    print('Rouge Rl', acc['rl']/i)
