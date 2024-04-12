from nltk.tokenize import sent_tokenize
from tools import utils
from tqdm import tqdm
import argparse
def Calculate_Rouge(hyps_path, refs_path):
    hyps =[]
    with open(hyps_path, 'r') as f:
        for line in tqdm(f):
            hyp = line.replace('<q>', '')
            hyp = hyp.split('.')
            hyp = [hyp[i].strip() for i in range(len(hyp))]
            hyps.append(' .\n'.join(hyp))
    refs = []
    with open(refs_path, 'r') as f:
        for line in tqdm(f):
            ref = line.replace('<q>', '')
            ref = ref.split('.')
            ref = [ref[i].strip() for i in range(len(ref))]
            refs.append('\n'.join(ref))
    scores_all = utils.rouge_corpus(refs, hyps)

    res = "Rouge1:\n\tp:%.6f, r:%.6f, f:%.6f\n" % (scores_all['rouge1']['precision'], scores_all['rouge1']['recall'], scores_all['rouge1']['fmeasure']) \
                + "Rouge2:\n\tp:%.6f, r:%.6f, f:%.6f\n" % (scores_all['rouge2']['precision'], scores_all['rouge2']['recall'], scores_all['rouge2']['fmeasure']) \
                    + "Rougel:\n\tp:%.6f, r:%.6f, f:%.6f\n" % (scores_all['rougeLsum']['precision'], scores_all['rougeLsum']['recall'], scores_all['rougeLsum']['fmeasure'])
    print(res)
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Calculate Rouge Score')
    parser.add_argument('--hyps_path', type=str, default='output/test_final_preds.candidate')
    parser.add_argument('--refs_path', type=str, default="output/test_final_preds.gold")
    args = parser.parse_args()
    Calculate_Rouge(args.refs_path, args.hyps_path)