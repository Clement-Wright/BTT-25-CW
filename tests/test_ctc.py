import numpy as np
import pytest

from btt25.ctc import ctc_minimum_frames, greedy_ctc


@pytest.mark.parametrize(
    ("path", "expected"),
    [
        ([1, 1, 0, 2, 2], [1, 2]),
        ([1, 0, 1], [1, 1]),
        ([0, 0, 0], []),
        ([], []),
    ],
)
def test_greedy_collapses_frame_repeats_before_removing_blank(path, expected):
    assert greedy_ctc(path) == expected


def test_greedy_uses_only_the_valid_output_prefix():
    assert greedy_ctc(np.array([1, 0, 1, 2, 2]), output_length=3) == [1, 1]
    assert greedy_ctc([1, 0, 1, -100], output_length=3) == [1, 1]
    assert greedy_ctc([1, 2], output_length=0) == []
    assert greedy_ctc([4, 4, 9, 4], blank_id=9) == [4, 4]


@pytest.mark.parametrize("length", [-1, 4, 1.5])
def test_greedy_rejects_invalid_output_lengths(length):
    with pytest.raises((ValueError, TypeError)):
        greedy_ctc([1, 0, 1], output_length=length)


@pytest.mark.parametrize("path", [[-1], [1.5], [[1, 2]]])
def test_greedy_rejects_noninteger_or_negative_frame_labels(path):
    with pytest.raises((ValueError, TypeError)):
        greedy_ctc(path)


@pytest.mark.parametrize(
    ("target", "minimum"),
    [([], 0), ([1, 2, 1], 3), ([1, 1, 2], 4), ([1, 1, 1], 5)],
)
def test_ctc_feasibility_counts_adjacent_repeated_targets(target, minimum):
    assert ctc_minimum_frames(target) == minimum


@pytest.mark.parametrize("target", [[0], [-1], [2.5]])
def test_ctc_feasibility_rejects_blank_or_invalid_actual_targets(target):
    with pytest.raises((ValueError, TypeError)):
        ctc_minimum_frames(target)
