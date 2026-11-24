from btt25.data import attach_corpus_metadata


def test_metadata_maps_by_explicit_session_block_only(tmp_path):
    path=tmp_path/"blocks.csv"
    path.write_text("Date,Block number,Corpus\n2023-08-11,2,Random\n",encoding="utf-8")
    records=[{"trial_id":"x","session_id":"t15.2023.08.11","block_id":2,"corpus_id":None}, {"trial_id":"y","session_id":"t15.2023.08.11","block_id":3,"corpus_id":None}]
    result=attach_corpus_metadata(records,path)
    assert result[0]["corpus_id"]=="Random"
    assert result[1]["corpus_id"] is None
    assert records[0]["corpus_id"] is None
