from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from seutil import LoggingUtils, BashUtils, IOUtils
from csevo.Environment import Environment


class Bleu:

    logger = LoggingUtils.get_logger(__name__, LoggingUtils.DEBUG if Environment.is_debug else LoggingUtils.INFO)

    @classmethod
    def compute_bleu(cls, references, hypotheses):
        with open(references, 'r') as fr, open(hypotheses, 'r') as fh:
            refs = fr.readlines()
            hyps = fh.readlines()
        bleu_4_sentence_scores = []
        for ref, hyp in zip(refs, hyps):
            bleu_4_sentence_scores.append(sentence_bleu([ref.strip().split()], hyp.strip().split(),
                                                        smoothing_function=SmoothingFunction().method2,
                                                        auto_reweigh=True))
        score = 100 * sum(bleu_4_sentence_scores) / float(len(bleu_4_sentence_scores))
        cls.logger.info(f"BLEU-4 score is: {score}")
        return
