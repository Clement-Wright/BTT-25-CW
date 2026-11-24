"""Run the registered compact search, confidence tuning and matched calibration stages."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

import yaml

from btt25.assets import sha256
from btt25.data import object_hash
from btt25.inference import load_config


def invoke(*command):
    command=list(map(str,command))
    print(json.dumps({'command':command}),flush=True)
    subprocess.run(command,check=True)


def resolve_learning(config_path, target):
    config=load_config(config_path)
    scientific=yaml.safe_load(Path('configs/calibration.yaml').read_text(encoding='utf-8'))
    target.parent.mkdir(parents=True,exist_ok=True)
    resolved={**config,**scientific}
    text=yaml.safe_dump(resolved)
    if target.exists() and target.read_text(encoding='utf-8')!=text:
        raise ValueError('Calibration control configuration changed')
    target.write_text(text,encoding='utf-8')
    return scientific


def read_selection(path):
    value=json.loads(Path(path).read_text(encoding='utf-8'))
    if value['manifest_hash']!=object_hash({k:v for k,v in value.items() if k!='manifest_hash'}):
        raise ValueError('Compact decoder selection changed')
    chosen=value['selected']
    for name,key in [('predictions.jsonl','predictions_sha256'),('candidates.jsonl','candidates_sha256'),
                     ('cache_manifest.json','cache_manifest_sha256')]:
        if sha256(Path(chosen['candidate_run'])/name)!=chosen[key]:
            raise ValueError(f'Selected compact cache changed: {name}')
    if value['kenlm_model'] and sha256(value['kenlm_model'])!=value['kenlm_sha256']:
        raise ValueError('Selected KenLM binary changed')
    if sha256(value['lexicon'])!=value['lexicon_sha256']:
        raise ValueError('Selected lexicon changed')
    return value


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['compact','confidence-tune','calibration','test'])
    parser.add_argument('--config',default='configs/local.yaml')
    parser.add_argument('--decoder-python',required=True)
    parser.add_argument('--kenlm-bin')
    parser.add_argument('--lexicon-only',action='store_true')
    parser.add_argument('--logit-run',default='runs/baseline-val-a100')
    parser.add_argument('--lexicon',default='assets/local/lexicon.txt')
    parser.add_argument('--scorer-dir',default='assets/local/distilgpt2')
    parser.add_argument('--selection',default='runs/compact-a100/selection.json')
    parser.add_argument('--mode',choices=['head','session_bias'])
    parser.add_argument('--profile-dir')
    parser.add_argument('--run-dir',type=Path,required=True)
    args=parser.parse_args()
    config=load_config(args.config)
    root=args.run_dir
    if args.stage=='compact':
        if bool(args.kenlm_bin)==args.lexicon_only:
            raise ValueError('Choose an installed estimator or the explicit lexicon-only fallback')
        extra=['--lexicon-only']
        if args.kenlm_bin:
            # Validate the actual native API with a real tiny ARPA model before large decoding.
            invoke(args.decoder_python,'-c','from flashlight.lib.text.decoder.kenlm import KenLM; print(KenLM.__name__)')
            invoke(args.decoder_python,'-m','unittest','discover','-s','tests','-p','test_decoder_worker.py','-v')
            lm=root/'lm'
            if not (lm/'manifest.json').exists():
                invoke(sys.executable,'scripts/fit_kenlm.py','--data-manifest',config['data_manifest'],
                       '--kenlm-bin',args.kenlm_bin,'--output-dir',lm)
            extra=['--kenlm-model',lm/'train-3gram.binary']
        invoke(sys.executable,'scripts/tune_compact.py','--config',args.config,'--logit-run',args.logit_run,
               '--decoder-python',args.decoder_python,'--lexicon',args.lexicon,'--run-dir',root,*extra)
    else:
        selected=read_selection(args.selection)
        extra=[] if not selected['kenlm_model'] else ['--kenlm-model',selected['kenlm_model'],
                                                      '--lm-weight',selected['selected']['weight']]
        if args.stage=='confidence-tune':
            root.mkdir(parents=True,exist_ok=True)
            settings=yaml.safe_load(Path('configs/confidence_lm.yaml').read_text(encoding='utf-8'))
            resolved={**config,**settings,'compact_decoder':selected['mode'],'allow_cpu_fallback':True}
            resolved_path=root/'config.resolved.yaml'
            resolved_path.write_text(yaml.safe_dump(resolved),encoding='utf-8')
            invoke(sys.executable,'-m','btt25.study_a','tune','--config',resolved_path,'--candidate-run',
                   selected['selected']['candidate_run'],'--scorer-dir',args.scorer_dir,'--output-dir',root)
        elif args.stage=='calibration':
            if not args.mode or not args.profile_dir:
                raise ValueError('Calibration requires a mode and its measured pilot')
            path=root/'calibration.resolved.yaml'
            scientific=resolve_learning(args.config,path)
            invoke(sys.executable,'-m','btt25.pipeline','--config',path,'--run-dir',root,
                   '--decoder-python',args.decoder_python,'--lexicon',args.lexicon,'--mode',args.mode,
                   '--profile-dir',args.profile_dir,'--updates',scientific['updates'],
                   '--target-seconds','3600','--hard-seconds','3720',*extra)
        else:
            infer,decode=root/'infer',root/'decode'
            invoke(sys.executable,'-m','btt25.cli','infer','--config',args.config,'--split','test','--run-dir',infer)
            invoke(args.decoder_python,'scripts/decode_worker.py','--logit-run',infer,'--output-dir',decode,
                   '--lexicon',args.lexicon,'--manifest',config['data_manifest'],'--beam','50','--nbest','10',*extra)
            invoke(sys.executable,'-m','btt25.cli','export','--predictions',decode/'predictions.jsonl',
                   '--manifest',config['data_manifest'],'--output',root/'submission.csv')


if __name__=='__main__':
    main()
