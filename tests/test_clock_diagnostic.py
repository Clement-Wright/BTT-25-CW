import numpy as np
import pytest

from btt25.clock_diagnostic import clock_views
from btt25.ctc import greedy_ctc


@pytest.mark.parametrize("fine_length", [7, 8])
def test_repeat_keeps_the_phone_path_and_even_view_keeps_the_anchor_clock(fine_length):
    slow = np.eye(41, dtype=np.float32)[[1, 0, 1, 40]]
    fast = np.arange(fine_length * 41, dtype=np.float32).reshape(fine_length, 41)
    views = clock_views(slow, fast)
    held = views["repeat80_on40"]
    assert len(held) == fine_length
    assert greedy_ctc(held.argmax(-1).tolist()) == [1, 1, 40]
    np.testing.assert_equal(held[::2], slow)
    np.testing.assert_equal(views["fast40_on80"], fast[::2])


def test_clock_mismatch_is_explicit():
    with pytest.raises(ValueError, match="anchor counts"):
        clock_views(np.zeros((3, 41)), np.zeros((8, 41)))
