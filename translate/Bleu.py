from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from seutil import LoggingUtils, BashUtils, IOUtils


class Bleu:


    @classmethod
    def compute_bleu(cls, references, hypotheses):
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
        results = {"Bleu": score}
        IOUtils.dump()

        return
