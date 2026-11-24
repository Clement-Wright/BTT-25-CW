from btt25.evaluation import corpus_metrics,evaluate_predictions


def test_failed_word_search_keeps_valid_encoder_per():
    row={"reference_text":"hi","text":"","reference_phoneme_ids":[1,1],"phoneme_ids":[1,1],"status":"decoder_error","phoneme_status":"ok"}
    metric=corpus_metrics([row])
    assert metric["wer"]==1
    assert metric["per"]==0
    assert metric["failed_count"]==1


def test_evaluator_preserves_explicit_stage_status():
    records=[{"trial_id":k,"split":"val","official_order_index":i} for i,k in enumerate(["a","b"])]
    split={"dev_tune":["a"],"dev_report":["b"],"manifest_hash":"fixture"}
    pred=[{"trial_id":"a","text":"","phoneme_ids":[1,1],"status":"decoder_error","phoneme_status":"ok"}]
    result=evaluate_predictions(pred,records,split,lambda r:{"reference_text":"hi","phoneme_ids":[1,1]})
    assert result["metrics"]["per"]==0
    assert result["metrics"]["wer"]==1
