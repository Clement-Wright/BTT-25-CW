"""Fit a compact 3-gram on normalized official training transcripts only."""
import argparse
import json
from pathlib import Path
import subprocess
import time

from btt25.assets import sha256
from btt25.data import object_hash, reference_only, write_json
from btt25.evaluation import normalize_text
from btt25.resources import ResourceMonitor


def training_sentences(records, reference_loader=reference_only):
    seen = set()
    for row in sorted((r for r in records if r['split']=='train'), key=lambda r:r['official_order_index']):
        trial_id = row['trial_id']
        if trial_id in seen:
            raise ValueError(f'Duplicate training ID: {trial_id}')
        seen.add(trial_id)
        text = normalize_text(reference_loader(row)['reference_text'])
        if text:
            yield trial_id, text


def vocabulary_from_arpa(path):
    vocab=[]
    reading=False
    with Path(path).open(encoding='utf-8') as src:
        for line in src:
            line=line.strip()
            if line==r'\1-grams:':
                reading=True
            elif reading and line.startswith('\\'):
                break
            elif reading and line:
                word=line.split()[1]
                if word not in ('<unk>','<s>','</s>'):
                    vocab.append(word)
    if not vocab or len(vocab)!=len(set(vocab)):
        raise ValueError('ARPA unigram vocabulary is empty or duplicated')
    return sorted(vocab)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-manifest',type=Path,required=True)
    parser.add_argument('--kenlm-bin',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args()
    output=args.output_dir.resolve()
    if (output/'manifest.json').exists():
        raise ValueError('Existing trained language model already has evidence')
    output.mkdir(parents=True,exist_ok=True)
    work=output/'temporary'
    work.mkdir(exist_ok=True)
    lmplz, build_binary=(args.kenlm_bin.resolve()/name for name in ('lmplz','build_binary'))
    records=json.loads(args.data_manifest.read_text(encoding='utf-8'))
    train_count=sum(r['split']=='train' for r in records)
    if not train_count:
        raise ValueError('Official training partition is unavailable')
    text_file, arpa, model=(output/name for name in ('train-normalized.txt','train-3gram.arpa','train-3gram.binary'))
    estimator=[str(lmplz),'-o','3','-S','1G','-T',str(work),'--discount_fallback']
    packer=[str(build_binary),'trie',str(arpa),str(model)+'.partial']
    with ResourceMonitor(output/'resources.json',max_seconds=900) as monitor:
        started=time.monotonic()
        pairs=list(training_sentences(records))
        ids=[trial_id for trial_id,_ in pairs]
        text_file.write_bytes(''.join(text+'\n' for _,text in pairs).encode('utf-8'))
        monitor.check()
        prep_seconds=time.monotonic()-started
        with text_file.open('rb') as src, arpa.open('wb') as dst, (output/'lmplz.log').open('wb') as log:
            begin=time.monotonic()
            subprocess.run(estimator,stdin=src,stdout=dst,stderr=log,check=True,timeout=420)
            estimation_seconds=time.monotonic()-begin
        monitor.check()
        with (output/'build_binary.log').open('wb') as log:
            begin=time.monotonic()
            subprocess.run(packer,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=300)
            pack_seconds=time.monotonic()-begin
        Path(str(model)+'.partial').replace(model)
        monitor.check()
    vocab=vocabulary_from_arpa(arpa)
    vocabulary=output/'train-vocabulary.txt'
    vocabulary.write_bytes(('\n'.join(vocab)+'\n').encode('utf-8'))
    result={'source_split':'official train only','training_trials':train_count,'used_trials':len(ids),
            'training_id_hash':object_hash(ids),'data_manifest_sha256':sha256(args.data_manifest),
            'order':3,'algorithm':'KenLM modified Kneser-Ney; discount fallback enabled',
            'normalization':'btt25.evaluation.normalize_text','text_sha256':sha256(text_file),
            'arpa_sha256':sha256(arpa),'binary_sha256':sha256(model),'vocabulary_sha256':sha256(vocabulary),
            'vocabulary_words':len(vocab),'word_tokens':sum(len(text.split()) for _,text in pairs),
            'lmplz_sha256':sha256(lmplz),'build_binary_sha256':sha256(build_binary),
            'estimator_memory':'1G','binary_format':'trie','seconds':{'prepare':prep_seconds,'estimate':estimation_seconds,'pack':pack_seconds}}
    result['manifest_hash']=object_hash(result)
    write_json(output/'manifest.json',result)
    print(json.dumps(result))


if __name__=='__main__':
    main()
