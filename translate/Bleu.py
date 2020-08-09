from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from seutil import LoggingUtils, BashUtils, IOUtils
import sys

class Bleu:

    #logger = LoggingUtils.get_logger(__name__, LoggingUtils.DEBUG if Environment.is_debug else LoggingUtils.INFO)

    @classmethod
    def compute_bleu(cls, references:str, hypotheses: str, exp: str) -> int:
        with open(references, 'r') as fr, open(hypotheses, 'r') as fh:
            refs = fr.readlines()
            hyps = fh.readlines()
        bleu_4_sentence_scores = []
        for ref, hyp in zip(refs, hyps):
            if len(hyp.strip().split()) < 2:
                bleu_4_sentence_scores.append(0)
            else:
                bleu_4_sentence_scores.append(sentence_bleu([ref.strip().split()], hyp.strip().split(),
                                                        smoothing_function=SmoothingFunction().method2,
                                                        auto_reweigh=True))
        score = 100 * sum(bleu_4_sentence_scores) / float(len(bleu_4_sentence_scores))
        print(score)
        bleu_result_file = f"{exp}/bleu.json"
        result = {"Bleu": score}
        IOUtils.dump(bleu_result_file, result)
        return score

if __name__ == "__main__":
    ref_file = sys.argv[1]
    hyp_file = sys.argv[2]
    exp_dir = sys.argv[3]
    Bleu.compute_bleu(ref_file, hyp_file, exp_dir)
