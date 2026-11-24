import json
from btt25.resources import ResourceMonitor


def test_caught_cooperative_stop_is_not_recorded_as_complete(tmp_path,monkeypatch):
    import btt25.resources as resources
    monkeypatch.setattr(resources.torch.cuda,'is_available',lambda:False)
    with ResourceMonitor(tmp_path/'resources.json') as monitor:
        monitor.reason='requested safe stop'
    result=json.loads((tmp_path/'resources.json').read_text())
    assert result['outcome']=='budget_stopped'


def test_unrelated_failure_is_failed(tmp_path,monkeypatch):
    import btt25.resources as resources
    monkeypatch.setattr(resources.torch.cuda,'is_available',lambda:False)
    try:
        with ResourceMonitor(tmp_path/'resources.json'):
            raise RuntimeError('broken checkpoint')
    except RuntimeError: pass
    assert json.loads((tmp_path/'resources.json').read_text())['outcome']=='failed'
