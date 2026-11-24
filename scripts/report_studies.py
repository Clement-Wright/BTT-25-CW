"""Seal the two studies without labels and produce scored evidence after the seal."""
import argparse
import csv
import json
from pathlib import Path
import random

from btt25.assets import sha256
from btt25.comparison import _load_run, compare_runs
from btt25.data import attach_corpus_metadata, load_split_manifest, object_hash, reference_only, write_json
from btt25.diagnostics import fixed_examples, write_diagnostics
from btt25.evaluation import corpus_metrics, edit_counts, evaluate_predictions, normalize_text
from btt25.freeze import seal, validate_score_inputs
from btt25.inference import load_config, read_jsonl
from btt25.reporting import write_evaluation
from btt25.resources import ResourceMonitor
from btt25.study_a import report as report_a
from btt25.training import calibration_manifest
from run_studies import read_selection


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_csv(path, rows):
    with path.open('w',encoding='utf-8',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def example_manifest(records, split):
    choices=set(split['dev_report'])
    ordered=sorted((r for r in records if r['trial_id'] in choices),key=lambda r:(r['input_length'],r['trial_id']))
    ids=list(dict.fromkeys(ordered[len(ordered)*i//5]['trial_id'] for i in (0,1,2,3,4)))
    random_corpus=sorted(r['trial_id'] for r in ordered if (r.get('corpus_id') or '').casefold()=='random' and r['trial_id'] not in ids)
    if random_corpus:
        ids.append(random.Random(20261116).choice(random_corpus))
    return {'ids':ids,'ids_hash':object_hash(ids),'split_hash':split['manifest_hash'],
            'rule':'Five equally spaced raw input-length quantiles and one seeded Random corpus trial when present'}


def complete_calibration(path, mode):
    """Check the small provenance chain without loading weights or labels."""
    path=Path(path)
    state=read(path/'pipeline_manifest.json')
    checkpoint=path/'train/checkpoints/last.pt'
    train=read(path/'train/run_manifest.json')
    training=path/'train/config.resolved.yaml'
    executed=load_config(training)
    digest=sha256(checkpoint)
    if (state.get('state')!='complete' or train.get('mode')!=mode or executed.get('mode')!=mode
            or train.get('state')!='training_complete_inference_pending' or train.get('diagnostic') is not False
            or digest!=train.get('checkpoint_hash') or train.get('step',0)<=0
            or train['step']!=executed.get('updates') or train.get('source_checkpoint_hash')!=executed.get('checkpoint_hash')
            or sha256(executed['checkpoint'])!=executed.get('checkpoint_hash')):
        raise ValueError(f'Incomplete {mode} calibration')
    records=read(executed['data_manifest'])
    split=load_split_manifest(executed['split_manifest'],records)
    calibration=path/'train/calibration_manifest.json'
    allowed=calibration_manifest(records,executed['per_session'],executed['seed'])
    if read(calibration)!=allowed:
        raise ValueError('Calibration IDs differ from the fixed allowed official train selection')
    trainables=path/'train/trainable_parameters.json'
    params=read(trainables)
    names=params.get('names',[])
    head={'out.weight','out.bias'}
    if (not head.issubset(names) or len(names)!=len(set(names)) or params.get('count',0)<=0
            or any(name not in head and not (mode=='session_bias' and name.startswith('day_biases.')) for name in names)
            or (mode=='session_bias' and not any(name.startswith('day_biases.') for name in names))
            or params.get('backprop_through_frozen_gru')!=(mode=='session_bias')
            or params.get('gru_training_mode')!=(mode=='session_bias')
            or (mode=='session_bias' and params.get('gru_dropout')!=0)):
        raise ValueError('Trainable parameters do not match the declared calibration condition')
    inference_path=path/'train/inference.yaml'
    inference=load_config(inference_path)
    if Path(inference['checkpoint']).resolve()!=checkpoint.resolve() or inference.get('checkpoint_hash')!=digest:
        raise ValueError('Calibration inference does not use its final checkpoint')
    for key in ('data_manifest','split_manifest','model_args'):
        if sha256(inference[key])!=sha256(executed[key]):
            raise ValueError(f'Calibration inference changed {key}')
    infer_cache_path=path/'infer/cache_manifest.json'
    neural=read(infer_cache_path)
    val={row['trial_id']:row for row in records if row['split']=='val'}
    if (neural.get('checkpoint_hash')!=digest or neural.get('split_hash')!=split['manifest_hash']
            or neural.get('manifest_hash')!=object_hash(records) or neural.get('contract_hash')!=inference.get('contract_hash')
            or neural.get('partition')!='val' or neural.get('dtype')!='float32'
            or neural.get('references_loaded') is not False or neural.get('expected_trials')!=len(val)):
        raise ValueError('Calibration encoder cache has different checkpoint/split provenance')
    infer_state_path=path/'infer/run_manifest.json'
    decode_state_path=path/'decode/run_manifest.json'
    for status_path in (infer_state_path,decode_state_path):
        status=read(status_path)
        if (status.get('state')!='complete' or status.get('references_loaded') is not False
                or status.get('expected_trials')!=len(val) or status.get('saved_trials')!=len(val)):
            raise ValueError('Calibration inference/decoding is incomplete or not reference-free')
    if read(infer_state_path).get('checkpoint_hash')!=digest:
        raise ValueError('Calibration inference state names a different checkpoint')
    cache_path=path/'decode/cache_manifest.json'
    cache=read(cache_path)
    if cache.get('manifest_hash')!=object_hash({key:value for key,value in cache.items() if key!='manifest_hash'}):
        raise ValueError('Calibration decoder cache hash changed')
    source_predictions=path/'infer/predictions.jsonl'
    if (cache.get('source_cache_manifest')!=neural or cache.get('references_loaded') is not False
            or Path(cache.get('source_predictions_path','')).resolve()!=source_predictions.resolve()
            or cache.get('source_predictions_sha256')!=sha256(source_predictions)
            or cache.get('data_manifest_sha256')!=sha256(executed['data_manifest'])
            or read(decode_state_path).get('decoder_cache_hash')!=cache['manifest_hash']):
        raise ValueError('Calibration decoded predictions have different source provenance')
    source_rows=read_jsonl(source_predictions)
    decoded_rows=read_jsonl(path/'decode/predictions.jsonl')
    for rows in (source_rows,decoded_rows):
        if len(rows)!=len(val) or {row['trial_id'] for row in rows}!=set(val):
            raise ValueError('Calibration prediction coverage is incomplete or duplicated')
    source_map={row['trial_id']:row for row in source_rows}
    for row in decoded_rows:
        trial=row['trial_id']
        original=source_map[trial]
        if (original.get('status')!='ok' or row.get('phoneme_status')!='ok'
                or row.get('decoder_cache_hash')!=cache['manifest_hash']
                or row.get('logit_sha256')!=original.get('logit_sha256')
                or row.get('phoneme_ids')!=original.get('phoneme_ids')
                or any(value.get('official_order_index')!=val[trial]['official_order_index']
                       or value.get('split')!='val' for value in (row,original))):
            raise ValueError('Calibration decoded phonemes or trial identity differ from encoder predictions')
    return {'root':path,'checkpoint':checkpoint,'source':train['source_checkpoint_hash'],
            'steps':train['step'],'predictions':path/'decode/predictions.jsonl','config':inference_path,
            'training':training,'calibration':calibration,'trainables':trainables,'train_state':path/'train/run_manifest.json',
            'infer_cache':infer_cache_path,'infer_state':infer_state_path,'infer_predictions':source_predictions,
            'decode_cache':cache_path,'decode_state':decode_state_path,
            'trainable_count':params['count'],'calibration_count':len(allowed['trial_ids']),
            'executed':executed,'decoder':cache}


def _decoder_protocol(cache):
    # Compare actual settings/assets; manifest mode names are presentation labels.
    return {key:cache.get(key) for key in ('decoder','lexicon_sha256','tokens_hash','unigrams_sha256','kenlm_model_sha256')}


def _calibrations(args, config, records, split, base):
    conditions={name:complete_calibration(directory,mode) for name,directory,mode in
                [('B1',args.head,'head'),('B2',args.bias,'session_bias')]}
    b1,b2=conditions.values()
    settings=('updates','lr','seed','per_session','regularization','backbone_dropout','selection')
    if (b1['steps']!=b2['steps'] or sha256(b1['calibration'])!=sha256(b2['calibration'])
            or any(b1['executed'].get(key)!=b2['executed'].get(key) for key in settings)):
        raise ValueError('Head and session calibration must share the matched update protocol and allowed IDs')
    compact=_decoder_protocol(read(Path(base)/'cache_manifest.json'))
    for row in conditions.values():
        if (row['source']!=config['checkpoint_hash'] or row['executed'].get('contract_hash')!=config.get('contract_hash')
                or any(sha256(row['executed'][key])!=sha256(config[key]) for key in ('data_manifest','split_manifest','model_args'))):
            raise ValueError('Calibration did not start from the locked baseline/data configuration')
        if _decoder_protocol(row['decoder'])!=compact:
            raise ValueError('Calibration decoder settings/assets differ from the selected compact baseline')
    return conditions


def prepare_freeze(args, config, selected, records, split):
    root=args.run_dir
    registry_path=root/'artifact_registry.json'
    freeze_path=root/'joint_freeze.json'
    if freeze_path.exists():
        raise ValueError('Joint freeze already exists')
    root.mkdir(parents=True,exist_ok=True)
    examples_path=root/'example_ids.json'
    write_json(examples_path,example_manifest(records,split))
    base=Path(selected['selected']['candidate_run'])
    conditions=_calibrations(args,config,records,split,base)
    assets={'predictions':str(base/'predictions.jsonl'),'candidates':str(base/'candidates.jsonl'),
            'cache':str(base/'cache_manifest.json'),'lexicon':selected['lexicon'],'official_manifest':config['data_manifest'],
            'model_args':config['model_args'],'lexical_code':'scripts/decode_worker.py','scorer':'src/btt25/rescoring.py',
            'deployment':'src/btt25/study_a.py','normalization':'src/btt25/evaluation.py','scoring':'scripts/report_studies.py',
            'training_code':'src/btt25/training.py','encoder_code':'src/btt25/model.py'}
    if config.get('corpus_metadata'):
        assets['corpus_metadata']=config['corpus_metadata']
    if args.effective_lexicon:
        assets['reachable_word_lexicon']=args.effective_lexicon
    if selected['kenlm_model']:
        model=Path(selected['kenlm_model'])
        assets.update(kenlm_binary=str(model),kenlm_manifest=str(model.parent/'manifest.json'),
                      lm_vocabulary=str(model.parent/'train-vocabulary.txt'))
    registry={'schema_version':1,'split_manifest':config['split_manifest'],'artifacts':{
        'baseline':{'kind':'files','config':{'encoder':args.config,'compact':args.selection,'examples':str(examples_path)},
                    'checkpoint':{'weights':config['checkpoint']},'assets':assets},
        'studies':{'confidence_lm':{'kind':'confidence_lm','config':str(args.confidence/'config.resolved.yaml'),
                                  'candidate_run':str(base),'scorer_dir':args.scorer_dir,
                                  'selected_config':str(args.confidence/'selected_config.json')},
                   'session_calibration':{'kind':'files','config':{f'{name}-{key}':str(row[key]) for name,row in conditions.items() for key in ('config','training')},
                       'checkpoint':{name:str(row['checkpoint']) for name,row in conditions.items()},
                       'assets':{f'{name}-{key}':str(row[key]) for name,row in conditions.items()
                                 for key in ('predictions','calibration','trainables','train_state','infer_cache','infer_state',
                                             'infer_predictions','decode_cache','decode_state')}}}}}
    write_json(registry_path,registry)
    rule=('Compact weight selected by summed dev_tune word edits. Confidence lambda selected on grouped gate-fit '
          'with gate-selection as a check. Median gate and three reference-free length-matched random seeds fixed. '
          'Calibration uses the final fixed matched update count from the same allowed official training IDs. '
          'Examples selected on features and corpus metadata before labels. All conditions retained.')
    return seal(registry_path,freeze_path,rule)


def measured_resources(root):
    root=Path(root)
    parts={}
    for stage in ('train','infer','decode','evaluate'):
        directory=root/stage
        candidates=[directory/name for name in ('resources.json','resources-resume.json')]
        candidates.extend(directory.glob('attempts/*/resources*.json'))
        for path in candidates:
            if path.is_file():
                parts[str(path.relative_to(root))]=read(path)
    for path in (root/'logs').glob('*-attempt-*/resources*.json'):
        parts[str(path.relative_to(root))]=read(path)
    pipeline=read(root/'pipeline_manifest.json')
    elapsed={stage:sum(attempt.get('elapsed_seconds',0) for attempt in value.get('attempts',[]))
             for stage,value in pipeline.get('stages',{}).items()}
    peaks=[r.get('peak_tree_rss_bytes',r.get('peak_rss_bytes',0)) for r in parts.values()]
    return {'elapsed_seconds':sum(elapsed.values()),'stage_elapsed_seconds':elapsed,
            'latest_supervisor_elapsed_seconds':read(root/'supervisor.json')['elapsed_seconds'],
            'peak_tree_rss_bytes':max(peaks,default=0),
            'peak_device_used_bytes':max((r.get('peak_device_used_bytes',0) for r in parts.values()),default=0),
            'scope':'Sum of recorded sequential pipeline attempt wall times, including retries; excludes queue/outage time. '
                    'Peaks include archived attempt measurements. Attempts without recorded durations are not imputed.',
            'stages':parts}


def _reuse_report(directory, predictions, records, split, frozen):
    """Use authorized scored rows only when outputs and all traces still agree."""
    metrics,rows,edits,_=_load_run(directory)
    trace={'score_split':'dev_report','split_manifest_hash':split['manifest_hash'],
           'freeze_manifest_hash':frozen['manifest_hash']}
    if any(metrics.get(key)!=value for key,value in trace.items()) or set(rows)!=set(split['dev_report']):
        raise ValueError('Cached confidence report has a different selection/freeze trace')
    live={row['trial_id']:row for row in predictions}
    metadata={row['trial_id']:row for row in records}
    for trial,row in rows.items():
        prediction=live[trial]
        status=prediction.get('status','ok')
        phone_status=prediction.get('phoneme_status',status)
        if (row['status']!=status or row.get('phoneme_status',row['status'])!=phone_status
                or row['text']!=(prediction.get('text','') if status=='ok' else '')
                or row['phoneme_ids']!=(prediction.get('phoneme_ids',[]) if phone_status=='ok' else [])
                or any(row.get(key)!=metadata[trial].get(key)
                       for key in ('official_order_index','session_id','block_id','corpus_id','input_length'))):
            raise ValueError('Cached confidence prediction differs from live deployment output/metadata')
        expected={'word':edit_counts(normalize_text(row['reference_text']).split(),normalize_text(row['text']).split()),
                  'phoneme':edit_counts(row['reference_phoneme_ids'],row['phoneme_ids'])}
        if any(edits[trial][kind]!=counts for kind,counts in expected.items()):
            raise ValueError('Cached confidence edits differ from the saved scored outputs')
    if any(metrics.get(key)!=value for key,value in corpus_metrics(rows.values()).items()):
        raise ValueError('Cached confidence metrics differ from the saved scored outputs')
    return {'metrics':metrics,'scored_rows':list(rows.values()),'per_trial':list(edits.values())}


def summarize(args, config, selected, records, split, frozen, registry):
    root=args.run_dir
    base=Path(selected['selected']['candidate_run'])
    # Both conditions and the baseline configuration are bound before any labels are loaded.
    expected=validate_score_inputs(registry,frozen,base/'predictions.jsonl',args.config)
    b_paths=_calibrations(args,config,records,split,base)
    for value in b_paths.values():
        validate_score_inputs(registry,frozen,value['predictions'],value['config'])
    a_dir=root/'A'
    a_summary=read(a_dir/'study_summary.json')
    required={'A0','A1','A2','A3_seed20261116','A3_seed20261117','A3_seed20261118'}
    if (a_summary['freeze_manifest_hash']!=frozen['manifest_hash'] or a_summary['score_split']!='dev_report'
            or a_summary.get('split_hash')!=split['manifest_hash'] or set(a_summary['conditions'])!=required
            or a_summary.get('selected_config_hash')!=expected['studies']['confidence_lm']['config_hash']
            or object_hash(a_summary.get('source_hashes'))!=expected['studies']['confidence_lm']['assets_hash']):
        raise ValueError('Missing compatible confidence deployment report')
    devices={'A0':read(base/'resources.json')}
    for key in a_summary['conditions']:
        if key!='A0': devices[key]=read(a_dir/key/'resources.json')
    devices.update({name:measured_resources(value['root']) for name,value in b_paths.items()})
    sources={key:a_dir/key/'full_predictions.jsonl' for key in a_summary['conditions']}
    for name,path in sources.items():
        if sha256(path)!=a_summary['deployed_predictions_sha256'][name]:
            raise ValueError(f'Live deployment predictions changed for {name}')
    sources.update({name:value['predictions'] for name,value in b_paths.items()})
    expected_ids={row['trial_id'] for row in records if row['split']=='val'}
    predictions={name:read_jsonl(path) for name,path in sources.items()}
    for name,rows in predictions.items():
        if len(rows)!=len(expected_ids) or {row['trial_id'] for row in rows}!=expected_ids:
            raise ValueError(f'Incomplete full validation predictions for {name}')
    # Verify every A report before B opens any new reporting references.
    reused={name:_reuse_report(a_dir/name,predictions[name],records,split,frozen) for name in required}
    examples=read(root/'example_ids.json')
    if (examples.get('ids_hash')!=object_hash(examples['ids']) or examples.get('split_hash')!=split['manifest_hash']
            or len(examples['ids'])!=len(set(examples['ids'])) or not set(examples['ids']).issubset(split['dev_report'])):
        raise ValueError('Frozen example IDs differ from the report allowlist')
    league={}
    summary_rows=[]
    lm_vocab=Path(selected['kenlm_model']).parent/'train-vocabulary.txt' if selected['kenlm_model'] else None
    for score_split in ('dev_report','official_val'):
        paths={}
        league[score_split]={}
        with ResourceMonitor(root/f'eval-{score_split}-resources.json') as monitor:
            references=({row['trial_id']:{'reference_text':row['reference_text'],'phoneme_ids':row['reference_phoneme_ids']}
                         for row in reused['A0']['scored_rows']} if score_split=='dev_report' else {})
            def load_once(row):
                monitor.check()
                if row['trial_id'] not in references:
                    references[row['trial_id']]=reference_only(row)
                return references[row['trial_id']]
            for name,rows in predictions.items():
                reuse=score_split=='dev_report' and name in reused
                directory=a_dir/name if reuse else root/score_split/name
                result=reused[name] if reuse else evaluate_predictions(rows,records,split,load_once,score_split,frozen,expected)
                provenance={'condition':name,'predictions_sha256':sha256(sources[name]),
                            'checkpoint_hash':config['checkpoint_hash'] if name not in b_paths else sha256(b_paths[name]['checkpoint'])}
                if not reuse:
                    write_evaluation(result,directory,provenance)
                write_json(directory/'resources.json',devices[name])
                diagnostic=write_diagnostics(directory,args.effective_lexicon or selected['lexicon'],lm_vocabulary=lm_vocab)
                metrics=result['metrics']
                league[score_split][name]={'metrics':metrics,'overall':diagnostic['overall'],
                    'random_corpora':diagnostic['unusual_random_corpora'],'performance':devices[name]}
                run=devices[name]
                gate=a_summary['conditions'].get(name,{})
                call_counts=gate.get('partition_calls',{}).get('dev_report',{}) if score_split=='dev_report' else gate
                calibration=b_paths.get(name,{})
                timing_scope=('sequential calibration, inference, compact decode and dev_tune evaluation attempts' if name in b_paths
                              else 'shared compact source decoder' if name=='A0'
                              else 'incremental LM deployment; shared compact decoding and one-time scorer setup recorded separately')
                summary_rows.append({'score_split':score_split,'condition':name,'evaluated_count':metrics['evaluated_count'],
                    'checkpoint_hash':provenance['checkpoint_hash'],'predictions_sha256':provenance['predictions_sha256'],
                    'freeze_hash':frozen['manifest_hash'],'per':metrics['per'],'wer':metrics['wer'],
                    'word_edits':metrics['word']['errors'],'word_denominator':metrics['word']['reference_length'],
                    'phoneme_edits':metrics['phoneme']['errors'],'phoneme_denominator':metrics['phoneme']['reference_length'],
                    'additional_lm_calls_on_full_validation':gate.get('lm_calls',''),'p95_incremental_lm_seconds':gate.get('called_latency_p95_seconds',''),
                    'lm_calls_scored_selection':call_counts.get('lm_calls',0),
                    'lm_call_length_bins_scored_selection':json.dumps(call_counts.get('call_bins',{}),sort_keys=True),
                    'calibration_updates':calibration.get('steps',0),'calibration_trial_count':calibration.get('calibration_count',0),
                    'trainable_parameters':calibration.get('trainable_count',0),'source_checkpoint_hash':config['checkpoint_hash'],
                    'timing_scope':timing_scope,'stage_elapsed_seconds':json.dumps(run.get('stage_elapsed_seconds',{}),sort_keys=True),
                    'shared_compact_trial_decode_seconds':a_summary['shared_compact_decode'].get('sum_trial_decode_seconds'),
                    'one_time_scorer_setup_seconds':a_summary['scorer_setup_seconds'],
                    'elapsed_seconds':run.get('elapsed_seconds'),'peak_rss_bytes':run.get('peak_tree_rss_bytes',run.get('peak_rss_bytes')),
                    'peak_device_used_bytes':run.get('peak_device_used_bytes'),'status':'complete'})
                paths[name]=directory
        if score_split=='dev_report':
            fixed_examples({name:paths[name] for name in ('A0','A1','A2','B1','B2')},examples['ids'],root)
            pairs=[('A0','A1'),('A0','A2'),('A1','A2'),('A0','B1'),('A0','B2'),('B1','B2')]
            pairs += [(name,'A2') for name in paths if name.startswith('A3_seed')]
            league['comparisons']={f'{b}-minus-{a}':compare_runs(paths[a],paths[b],root/'pairs'/f'{b}-minus-{a}') for a,b in pairs}
    write_csv(root/'experiments.csv',summary_rows)
    write_json(root/'complete.json',{'freeze_hash':frozen['manifest_hash'],'compact_selection':selected,'confidence':a_summary,
                                   'results':league,'rows':len(summary_rows),'state':'complete'})


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['seal','deploy','summarize'])
    parser.add_argument('--config',default='configs/local.yaml')
    parser.add_argument('--selection',default='runs/compact-a100/selection.json')
    parser.add_argument('--confidence',type=Path,default=Path('runs/confidence-tune-a100'))
    parser.add_argument('--head',default='runs/calibration-head-a100')
    parser.add_argument('--bias',default='runs/calibration-bias-a100')
    parser.add_argument('--scorer-dir',default='assets/local/distilgpt2')
    parser.add_argument('--effective-lexicon',help='Trie-retained pronunciations from audit_trie.py for OOV diagnostics')
    parser.add_argument('--run-dir',type=Path,default=Path('runs/locked-a100'))
    args=parser.parse_args()
    config=load_config(args.config)
    records=read(config['data_manifest'])
    split=load_split_manifest(config['split_manifest'],records)
    if config.get('corpus_metadata'):
        records=attach_corpus_metadata(records,config['corpus_metadata'])
    selected=read_selection(args.selection)
    if args.stage=='seal':
        print(json.dumps(prepare_freeze(args,config,selected,records,split)))
        return
    frozen=read(args.run_dir/'joint_freeze.json')
    registry=read(args.run_dir/'artifact_registry.json')
    if registry['artifacts']['baseline']['assets'].get('reachable_word_lexicon') != args.effective_lexicon:
        raise ValueError('Reporting requires the same reachable-word lexicon used at seal')
    if args.stage=='deploy':
        report_a(args.confidence/'config.resolved.yaml',selected['selected']['candidate_run'],args.run_dir/'A',args.scorer_dir,
                 args.confidence/'selected_config.json',frozen,registry,score_split='dev_report')
    else:
        summarize(args,config,selected,records,split,frozen,registry)


if __name__=='__main__':
    main()
