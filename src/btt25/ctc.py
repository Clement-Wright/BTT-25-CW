"""CTC contracts for integer frame paths and unpadded reference targets."""

from operator import index


def _labels(sequence):
    labels = [index(label) for label in sequence]
    if any(label < 0 for label in labels):
        raise ValueError("CTC labels must be nonnegative integers")
    return labels


def greedy_ctc(path, blank_id=0, output_length=None) -> list[int]:
    """Collapse frame repeats, then remove blanks, within the valid prefix.

    This accepts frame labels after argmax, never an already collapsed beam
    prefix. Blanks separate repeated output labels: [1, 0, 1] becomes [1, 1].
    """
    frames = list(path)
    blank_id = index(blank_id)
    if blank_id < 0:
        raise ValueError("blank_id must be nonnegative")
    length = len(frames) if output_length is None else index(output_length)
    if not 0 <= length <= len(frames):
        raise ValueError("output_length must be within the stored frame path")
    result = []
    previous = None
    for label in _labels(frames[:length]):
        if label != previous and label != blank_id:
            result.append(label)
        previous = label
    return result


def ctc_minimum_frames(target) -> int:
    """Return U plus adjacent repeated pairs, rejecting blank in real targets."""
    labels = _labels(target)
    if 0 in labels:
        raise ValueError("Actual CTC targets must not contain blank ID 0")
    return len(labels) + sum(left == right for left, right in zip(labels, labels[1:]))
