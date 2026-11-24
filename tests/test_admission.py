import pytest
from btt25.data import write_json
from btt25.training import admit


def fixture_profile(tmp_path):
    write_json(tmp_path/"profile.json",{"sustained_seconds":901,"step_p95_seconds":.2,"checkpoint_median_seconds":1})
    write_json(tmp_path/"resources.json",{"stop_reason":None,"peak_tree_rss_bytes":1_000_000_000,"minimum_system_available_bytes":4_000_000_000,"peak_device_used_bytes":9_500_000_000,"hardware":{"gpu":[{"total_vram_bytes":10_000_000_000}]}})


def test_vram_margin_is_independent_admission_requirement(tmp_path):
    fixture_profile(tmp_path)
    with pytest.raises(ValueError,match="VRAM"):
        admit(tmp_path,10,100)
