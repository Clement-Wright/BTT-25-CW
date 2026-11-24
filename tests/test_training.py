import torch
import pytest
from btt25.model import OfficialGRU
from btt25.training import configure_trainable, train_step, atomic_checkpoint, restore_optimizer


def make_model():
    return OfficialGRU(512, 8, 2, 41, 2, 14, 4, 0, 0).eval()


def test_head_and_session_gradients_are_real_and_backbone_is_fixed():
    torch.manual_seed(23)
    model = make_model()
    names = configure_trainable(model, "session_bias")
    assert "day_biases.0" in names and "out.weight" in names
    assert not model.gru.weight_ih_l0.requires_grad
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=1e-3)
    before = model.day_biases[0].detach().clone()
    batch = {"features": torch.randn(1, 80, 512), "day_index": 0, "phoneme_ids": [1,1,40]}
    result = train_step(model, batch, optimizer, "session_bias")
    assert result["loss"] > 0
    assert torch.isfinite(model.day_biases[0].grad).all()
    assert model.day_biases[0].grad.abs().sum() > 0
    assert not torch.equal(before, model.day_biases[0])
    assert model.gru.weight_ih_l0.grad is None


def test_infeasible_repeated_ctc_not_zeroed():
    model = make_model()
    configure_trainable(model, "head")
    opt = torch.optim.Adam(model.out.parameters())
    with pytest.raises(ValueError, match="CTC"):
        train_step(model, {"features": torch.randn(1, 24, 512), "day_index": 0, "phoneme_ids": [1,1,1]}, opt, "head")


def test_atomic_checkpoint_and_optimizer_rebind(tmp_path):
    model = make_model()
    configure_trainable(model, "head")
    opt = torch.optim.Adam(model.out.parameters(), lr=1e-3)
    train_step(model, {"features": torch.randn(1,80,512), "day_index": 0, "phoneme_ids": [1,2,40]}, opt, "head")
    path = tmp_path / "last.pt"
    atomic_checkpoint(path, {"model_state_dict": model.state_dict(), "optimizer_state_dict": opt.state_dict(), "step": 1})
    state = torch.load(path, weights_only=True)
    other = make_model()
    other.load_state_dict(state["model_state_dict"], strict=True)
    configure_trainable(other, "head")
    restored = restore_optimizer(other, state, lr=1e-3)
    assert {id(p) for g in restored.param_groups for p in g["params"]} == {id(p) for p in other.parameters() if p.requires_grad}
    for a,b in zip(model.parameters(),other.parameters()):
        torch.testing.assert_close(a,b)


@pytest.mark.skipif(not torch.cuda.is_available(),reason="CUDA regression requires an actual GPU")
def test_input_adapter_backward_works_on_actual_cudnn():
    model=make_model().cuda()
    configure_trainable(model,"session_bias")
    opt=torch.optim.Adam([p for p in model.parameters() if p.requires_grad],lr=1e-4)
    result=train_step(model,{"features":torch.randn(1,80,512,device="cuda"),"day_index":0,"phoneme_ids":[1,1,40]},opt,"session_bias")
    assert result["loss"]>0
    assert model.gru.dropout==0
    assert model.day_biases[0].grad.abs().sum()>0


@pytest.mark.parametrize("evidence", ["run_manifest.json", "checkpoints/last.pt", "learning.jsonl", "config.resolved.yaml"])
def test_train_rejects_existing_evidence_before_writing(tmp_path, evidence):
    from btt25.training import train
    path = tmp_path / evidence
    path.parent.mkdir(parents=True, exist_ok=True)
    original = b'{"state":"budget_stopped","step":12}'
    path.write_bytes(original)
    with pytest.raises(ValueError, match="Existing training directory"):
        train({}, tmp_path, "head", 3, tmp_path / "unused-pilot")
    assert path.read_bytes() == original
    assert not (tmp_path / "admission.json").exists()


@pytest.mark.parametrize("missing", ["run_manifest.json", "checkpoints/last.pt"])
def test_resume_requires_previous_manifest_and_checkpoint(tmp_path, missing):
    import json
    from btt25.training import train
    (tmp_path / "checkpoints").mkdir()
    (tmp_path / "checkpoints/last.pt").write_bytes(b"old checkpoint")
    (tmp_path / "run_manifest.json").write_text(json.dumps({"state": "budget_stopped"}))
    (tmp_path / missing).unlink()
    with pytest.raises(ValueError, match="Resume requires"):
        train({}, tmp_path, "head", 3, tmp_path / "unused-pilot", resume=True)
    assert not (tmp_path / "admission.json").exists()


@pytest.mark.parametrize("state", ["complete", "training_complete_inference_pending", "diagnostic_complete"])
def test_resume_does_not_rewrite_finished_run(tmp_path, state):
    import json
    from btt25.training import train
    (tmp_path / "checkpoints").mkdir()
    (tmp_path / "checkpoints/last.pt").write_bytes(b"completed checkpoint")
    manifest = tmp_path / "run_manifest.json"
    original = json.dumps({"state": state}).encode()
    manifest.write_bytes(original)
    with pytest.raises(ValueError, match="interrupted training"):
        train({}, tmp_path, "head", 3, tmp_path / "unused-pilot", resume=True)
    assert manifest.read_bytes() == original
    assert not (tmp_path / "attempts").exists()


def test_resume_preserves_each_attempt_and_actual_parent_checkpoint(tmp_path, monkeypatch):
    import json
    import time
    import numpy as np
    from btt25 import training
    from btt25.assets import sha256

    class FixtureMonitor:
        def __init__(self, output, **kwargs):
            self.output = output
            self.start = time.monotonic()
        def __enter__(self):
            return self
        def __exit__(self, *args):
            self.output.write_text(json.dumps({"fixture": str(self.output)}))
            self.output.with_suffix(".samples.jsonl").write_text('{"fixture":true}\n')
        def check(self):
            pass

    def fixture_encoder(path, *args):
        model = make_model()
        model.contract = {"fixture": True}
        if path != "official-checkpoint":
            model.load_state_dict(torch.load(path, weights_only=True)["model_state_dict"])
        return model, {}

    records = [{"trial_id": "train1", "split": "train", "session_id": "s1", "day_index": 0}]
    data = tmp_path / "data.json"
    data.write_text(json.dumps(records))
    run = tmp_path / "run"
    config = {"data_manifest": str(data), "split_manifest": "fixture", "checkpoint": "official-checkpoint",
              "checkpoint_hash": "official-source-hash", "contract_hash": "contract", "model_args": "fixture",
              "device": "cpu", "calibration_per_session": 1}
    monkeypatch.setattr(training, "admit", lambda *args, **kwargs: {"admitted": True})
    monkeypatch.setattr(training, "load_split_manifest", lambda *args: {})
    monkeypatch.setattr(training, "load_encoder", fixture_encoder)
    monkeypatch.setattr(training, "ResourceMonitor", FixtureMonitor)
    monkeypatch.setattr(training, "read_trial", lambda *args, **kwargs: {"features": np.zeros((80, 512), dtype=np.float32), "phoneme_ids": [1, 2, 40]})
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    first = training.train(config, run, "head", 3, "fixture-pilot", stop_after_updates=1)
    assert first["state"] == "budget_stopped" and first["step"] == 1
    original_manifest = (run / "run_manifest.json").read_bytes()
    original_resources = (run / "resources.json").read_bytes()
    first_checkpoint = sha256(run / "checkpoints/last.pt")

    second = training.train(config, run, "head", 3, "fixture-pilot", resume=True, stop_after_updates=1)
    assert second["state"] == "budget_stopped" and second["step"] == 2
    first_archive = run / second["previous_attempt"]
    assert (first_archive / "run_manifest.json").read_bytes() == original_manifest
    assert (first_archive / "resources.json").read_bytes() == original_resources
    assert second["parent_checkpoint_hash"] == first_checkpoint
    second_manifest = (run / "run_manifest.json").read_bytes()
    second_resources = (run / "resources-resume.json").read_bytes()
    second_checkpoint = sha256(run / "checkpoints/last.pt")

    third = training.train(config, run, "head", 3, "fixture-pilot", resume=True)
    assert third["state"] == "training_complete_inference_pending" and third["step"] == 3
    second_archive = run / third["previous_attempt"]
    assert second_archive != first_archive
    assert (second_archive / "run_manifest.json").read_bytes() == second_manifest
    assert (second_archive / "resources-resume.json").read_bytes() == second_resources
    assert (first_archive / "run_manifest.json").read_bytes() == original_manifest
    assert third["attempt"] == 3
    assert third["source_checkpoint_hash"] == "official-source-hash"
    assert third["parent_checkpoint_hash"] == second_checkpoint
    assert [json.loads(line)["step"] for line in (run / "learning.jsonl").read_text().splitlines()] == [1, 2, 3]

    uninterrupted = tmp_path / "uninterrupted"
    training.train(config, uninterrupted, "head", 3, "fixture-pilot")
    resumed_state = torch.load(run / "checkpoints/last.pt", weights_only=True)
    direct_state = torch.load(uninterrupted / "checkpoints/last.pt", weights_only=True)
    for name, tensor in direct_state["model_state_dict"].items():
        torch.testing.assert_close(resumed_state["model_state_dict"][name], tensor, rtol=0, atol=0)
    assert resumed_state["optimizer_state_dict"]["param_groups"] == direct_state["optimizer_state_dict"]["param_groups"]
    for parameter, values in direct_state["optimizer_state_dict"]["state"].items():
        for name, tensor in values.items():
            torch.testing.assert_close(resumed_state["optimizer_state_dict"]["state"][parameter][name], tensor, rtol=0, atol=0)
