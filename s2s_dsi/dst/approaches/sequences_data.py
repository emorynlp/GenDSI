import ezpyz as ez
from dst.data.results import Results
from dataclasses import dataclass
import math
import typing as T


@dataclass
class Sequence:
    seq_input: str = None
    seq_label: str = None
    seq_output: str = None
    seq_logits: list[float] = None


seqslike: T.TypeAlias = T.Union[
    str,
    tuple[str, str],
    list[str],
    list[tuple[str, str]],
    list[Sequence],
    'SequencesData'
]
class SequencesData(ez.Data, list[Sequence]):
    def __init__(self, seqs:seqslike=None, file:ez.filelike=None):
        if seqs is None:
            list.__init__(self)
        elif isinstance(seqs, list) and seqs and isinstance(seqs[0], Sequence):
            list.__init__(self, seqs)
        elif isinstance(seqs, list) and seqs and isinstance(seqs[0], str):
            list.__init__(self, [Sequence(seq) for seq in seqs])
        elif isinstance(seqs, list):
            list.__init__(self, [Sequence(*pair) for pair in seqs])
        elif isinstance(seqs, tuple):
            list.__init__(self, [Sequence(*seqs)])
        elif isinstance(seqs, str):
            list.__init__(self, [Sequence(seqs)])
        super().__init__(_file=file)

    def perplexity(self, results:'SequenceResults'):
        n = 0
        lnP = 0.0
        for seq in self:
            logit_seq = seq.seq_logits
            if isinstance(logit_seq, float):
                lnP += logit_seq
                n += 1
            else:
                lnP += sum(logit_seq)
                n += len(logit_seq)
        lnPP = lnP / n
        pp = math.exp(-lnPP)
        results.perplexity = pp

    def exact_match(self, results:'SequenceResults'):
        matches = 0
        total = 0
        for seq in self:
            if seq.seq_label == seq.seq_output:
                matches += 1
            total += 1
        results.exact_match = matches / total

    def __str__(self):
        return f"{type(self).__name__}({len(self)} sequences{f' from {self.file}' if self.file else ''})"
    __repr__ = __str__


@dataclass
class SequenceResults(Results):
    epoch: int = None
    loss: float = None
    perplexity: float = None
    exact_match: float = None


class Hyperparameters(ez.Data, list[dict[str, str]]):

    def __init__(self, file:ez.filelike=None):
        list.__init__(self)
        super().__init__(_file=file)

    def record(self, model, **kwargs):
        hyperparameters = {}
        for key, value in vars(model).items():
            hyperparameters[key] = repr(value)
        hyperparameters.update({k: repr(v) for k, v in kwargs.items()})
        self.append(hyperparameters)

    def display(self):
        display = 'Hyperparameters\n===============\n' + '\n'.join(
            f"{key}: {str(value)[:30] + ('...' if len(str(value)) > 30 else '')}" for key, value in self[-1].items()
        ) if self else 'Empty Hyperparameters'
        return display


