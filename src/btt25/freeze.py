"""Seal both studies and recompute real artifact hashes before reference access."""
from datetime import datetime, timezone
import json
from pathlib import Path

from .assets import sha256
from .data import object_hash, write_json
from .evaluation import _validate_freeze


def _read(value):
    return value if isinstance(value,dict) else json.loads(Path(value).read_text(encoding='utf-8'))


def _registry(value):
    result=_read(value)
    if result.get('schema_version')!=1 or not isinstance(result.get('artifacts'),dict):
        raise ValueError('Reporting requires a live artifact registry, not supplied expected hash values')
    return result


def _group(group):
    if group.get('kind')=='confidence_lm':
        from .study_a import study_hashes
        return study_hashes(group['config'],group['candidate_run'],group['scorer_dir'],group['selected_config'])
    if group.get('kind')!='files':
        raise ValueError('Unsupported frozen artifact provider')
    result={}
    for kind in ('config','checkpoint','assets'):
        files=group.get(kind)
        if not isinstance(files,dict) or not files:
            raise ValueError(f'Frozen group requires nonempty {kind} file mapping')
        digests={name:sha256(path) for name,path in sorted(files.items())}
        result[kind+'_hash']=next(iter(digests.values())) if kind=='checkpoint' and len(digests)==1 else object_hash(digests)
    return result


def current_hashes(registry):
    """Compute expected values from current bytes, never from saved hash claims.

    All registry paths resolve from the command's working directory. Logical
    file names enter group hashes; machine-specific absolute paths do not.
    """
    registry=_registry(registry)
    split=_read(registry['split_manifest'])
    if object_hash({k:v for k,v in split.items() if k!='manifest_hash'})!=split.get('manifest_hash'):
        raise ValueError('Split manifest content hash mismatch')
    artifacts=registry['artifacts']
    return {'baseline':_group(artifacts['baseline']),
            'studies':{name:_group(group) for name,group in artifacts['studies'].items()},
            'split_hash':split['manifest_hash']}


def seal(registry, output, selection_rules):
    """Write an immutable joint declaration after tuning, before report scoring."""
    output=Path(output)
    if output.exists():
        raise ValueError('Joint freeze already exists; preserve the original declaration')
    registry=_registry(registry)
    expected=current_hashes(registry)
    freeze={**expected,'selection_rules':selection_rules,'created_at':datetime.now(timezone.utc).isoformat(),
            'registry_hash':object_hash(registry),'reporting_references_opened':False}
    freeze['manifest_hash']=object_hash(freeze)
    _validate_freeze(freeze,expected,expected['split_hash'])
    write_json(output,freeze)
    return freeze


def validate_score_inputs(registry, freeze, predictions, config):
    """Check the entire joint freeze and the requested source before labels."""
    registry=_registry(registry)
    freeze=_read(freeze)
    if object_hash({k:v for k,v in freeze.items() if k!='manifest_hash'})!=freeze.get('manifest_hash'):
        raise ValueError('Joint freeze content hash mismatch')
    if freeze.get('registry_hash')!=object_hash(registry):
        raise ValueError('Frozen artifact registry changed')
    expected=current_hashes(registry)
    _validate_freeze(freeze,expected,expected['split_hash'])
    groups=[registry['artifacts']['baseline'],*registry['artifacts']['studies'].values()]
    target=Path(predictions).resolve()
    cfg=Path(config).resolve()
    for group in groups:
        if group.get('kind')!='files': continue
        if target in {Path(p).resolve() for p in group['assets'].values()} and cfg in {Path(p).resolve() for p in group['config'].values()}:
            return expected
    raise ValueError('Requested predictions/config are not bound together in the frozen artifact registry')
