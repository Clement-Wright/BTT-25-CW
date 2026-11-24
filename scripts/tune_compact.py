"""Decode one fixed validation logit cache and select a compact LM weight on dev_tune."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

import yaml

from btt25.assets import sha256
from btt25.data import object_hash, write_json
from btt25.inference import load_config


def checked(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def run(*command):
    subprocess.run(list(map(str,command)),check=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True)
    parser.add_argument('--logit-run',required=True)
    parser.add_argument('--decoder-python',required=True)
    parser.add_argument('--decoder-config',default='configs/compact_decoder.yaml')
    parser.add_argument('--lexicon',required=True)
    parser.add_argument('--kenlm-model')
    parser.add_argument('--lexicon-only',action='store_true')
    parser.add_argument('--run-dir',type=Path,required=True)
    args=parser.parse_args()
    if bool(args.kenlm_model)==args.lexicon_only:
        raise ValueError('Choose an explicit KenLM model or the named lexicon-only fallback')
    config=load_config(args.config)
    settings=yaml.safe_load(Path(args.decoder_config).read_text(encoding='utf-8'))
    mode='kenlm_3gram' if args.kenlm_model else 'lexicon_only'
    weights=settings['dev_tune_lm_weights'] if args.kenlm_model else [0.0]
    state=checked(Path(args.logit_run)/'run_manifest.json')
    if state['state']!='complete' or state.get('references_loaded') is not False:
        raise ValueError('Compact tuning needs a complete reference-free validation cache')
    output=args.run_dir
    output.mkdir(parents=True,exist_ok=True)
    selected_path=output/'selection.json'
    if selected_path.exists():
        raise ValueError('A compact configuration has already been selected in this run')
    evidence=[]
    for weight in weights:
        root=output/f'weight-{weight:g}'
        decode, evaluate=root/'decode',root/'dev_tune'
        command=[args.decoder_python,'scripts/decode_worker.py','--logit-run',args.logit_run,
                 '--lexicon',args.lexicon,'--output-dir',decode,'--manifest',config['data_manifest'],
                 '--beam',settings['beam'],'--nbest',settings['nbest'],'--beam-threshold',settings['beam_threshold']]
        if args.kenlm_model:
            command+=['--kenlm-model',args.kenlm_model,'--lm-weight',weight]
        run(*command)
        run(sys.executable,'-m','btt25.cli','evaluate','--config',args.config,'--predictions',decode/'predictions.jsonl',
            '--output-dir',evaluate,'--score-split','dev_tune')
        metrics=checked(evaluate/'metrics.json')
        coverage=checked(decode/'run_manifest.json')
        if coverage['state']!='complete' or coverage['saved_trials']!=state['saved_trials']:
            raise ValueError('Word candidate cache is no longer complete')
        evidence.append({'weight':weight,'candidate_run':str(decode),'tune_run':str(evaluate),'word':metrics['word'],
                         'wer':metrics['wer'],'per':metrics['per'],'coverage':coverage['saved_trials'],
                         'cache_manifest_sha256':sha256(decode/'cache_manifest.json'),
                         'candidates_sha256':sha256(decode/'candidates.jsonl'),
                         'predictions_sha256':sha256(decode/'predictions.jsonl')})
    selected=min(evidence,key=lambda row:(row['word']['errors'],row['weight']!=settings['initial_lm_weight'],row['weight']))
    resolved={'settings':settings,'score_split':'dev_tune','scoring_split_hash':metrics['split_manifest_hash'],
              'lexicon_sha256':sha256(args.lexicon),'lexicon':args.lexicon,'mode':mode,
              'kenlm_model':args.kenlm_model,'kenlm_sha256':sha256(args.kenlm_model) if args.kenlm_model else None,
              'input_logits_run':args.logit_run,'checkpoint_hash':config['checkpoint_hash'],
              'selected':selected,'trials':evidence,'report_references_read':False}
    resolved['manifest_hash']=object_hash(resolved)
    write_json(selected_path,resolved)
    print(json.dumps(resolved))


if __name__=='__main__':
    main()
