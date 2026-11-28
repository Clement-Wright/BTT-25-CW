"""Synthetic checks on the scalp data reader, cue boundaries, and fixed classifier."""
import numpy as np
import pytest

from btt25.scalp_eeg import decode_i24, fit, inner_trials, predict, spectral_features


def test_biosemi_signed_little_endian():
    a=np.array([[0,0,0],[1,0,0],[255,255,127],[0,0,128],[255,255,255]],dtype="u1")
    assert decode_i24(a).tolist()==[0,1,8388607,-8388608,-1]


def test_only_inner_speech_and_no_visual_cue_bins():
    events=[(5,21),(10,31),(20,44),(30,45),(32,16),(40,22)]
    for i in range(4):
        cue=4000+i*10000
        events.extend([(cue-512,42),(cue,31+i),(cue+512,44),(cue+3072,45)])
    trials=inner_trials(events)
    assert [r["class"] for r in trials]==list(range(4))
    assert all(r["precue"]+2048<r["cue"] and r["action"]>=r["cue"]+512 for r in trials)
    with pytest.raises(ValueError,match="Ambiguous"):
        inner_trials(events[:-1])


def test_window_power_does_not_include_other_trials():
    t=np.arange(2048)/1024
    sinusoid=np.tile(np.sin(2*np.pi*10*t),(128,1))
    powers=spectral_features(sinusoid).reshape(4,128)
    assert np.argmax(powers[:,0])==1
    x=np.tile(np.eye(4),(8,1))
    y=np.tile(np.arange(4),8)
    model=fit(x,y)
    assert [predict(row,model) for row in np.eye(4)]==[0,1,2,3]
