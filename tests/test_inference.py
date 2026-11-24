import numpy as np
import pytest
from btt25.inference import validate_cache_metadata


def test_logit_cache_mismatch_rejected():
    actual = {"checkpoint_hash": "a", "contract_hash": "b", "split_hash": "c", "dtype": "float32"}
    validate_cache_metadata(actual, actual)
    with pytest.raises(ValueError, match="checkpoint_hash"):
        validate_cache_metadata({**actual, "checkpoint_hash": "other"}, actual)
