"""Conservative lineage, exposure screening, and split-before-derivation contracts.
A clean registry lookup is necessary, never proof of unexposed upstream training.
"""
from __future__ import annotations
from collections import Counter
import csv
import json
from pathlib import Path
import numpy as np
from .common import ContractError, digest, read_json, check_hash

CATEGORIES = ('Voice-only','Music-only','Mixed','Stress')
EXPOSURES = ('head_train','head_selection','historical_development')
HASH_FIELDS = ('audio_sha256','pcm_sha256','native_pcm_sha256','pcm16k_sha256','source_sha256')


def _valid(v):
    return v is not None and str(v).strip().lower() not in ('','none','null','nan','unknown','[]')


def entity_tokens(entity: dict) -> set[str]:
    tokens=set()
    for field in HASH_FIELDS:
        if _valid(entity.get(field)): tokens.add('hash:'+str(entity[field]))
    aliases={'parent_id':'parent','speaker':'speaker','speaker_id':'speaker','track':'track','track_id':'track','component_id':'component','group_id':'component'}
    for key,prefix in aliases.items():
        if _valid(entity.get(key)): tokens.add(prefix+':'+str(entity[key]))
    return tokens


def lineage_tokens(row: dict) -> set[str]:
    tokens=entity_tokens(row)|{'id:'+str(row['ID'])}
    for parent in row.get('parents',[]):
        tokens |= entity_tokens(parent)
        if parent.get('parent_id'): tokens.add('id:'+str(parent['parent_id']))
    return tokens


def connected_groups(rows: list[dict], mode='parent') -> list[str]:
    """Transitive parent/speaker/track/hash groups, including BOTH mixture parents.
    Optional source grouping intentionally also connects all parent source domains.
    """
    if mode not in ('parent','source','generator'): raise ContractError('Unknown group mode')
    parents=list(range(len(rows)))
    def find(x):
        while parents[x]!=x:
            parents[x]=parents[parents[x]];x=parents[x]
        return x
    seen={}
    for i,row in enumerate(rows):
        tokens=lineage_tokens(row)
        if mode!='parent':
            key='source' if mode=='source' else 'generator_family'
            tokens|={mode+':'+str(p[key]) for p in row['parents'] if _valid(p.get(key)) and p[key]!='real'}
        for token in sorted(tokens):
            if token in seen:
                a,b=find(i),find(seen[token]);parents[max(a,b)]=min(a,b)
            else: seen[token]=i
    members={}
    for i,row in enumerate(rows):members.setdefault(find(i),[]).append(str(row['ID']))
    names={g:digest(sorted(ids))[:24] for g,ids in members.items()}
    return [names[find(i)] for i in range(len(rows))]


def registry_from_input(root: str | Path) -> set[str]:
    """Screen selected 9504 AND known historical catalog exposure, not CSV name."""
    root=Path(root); tokens=set()
    check_hash(root/'INPUT_MANIFEST.json','6d8d81c074c92981b8f6466c0517956e0fa99029f5ca75ea802c105687a01bfd')
    pins={e['path']:e['sha256'] for e in read_json(root/'INPUT_MANIFEST.json')['files']}
    for name in ('data/selected_list.csv','data/runtime_manifest.csv','data/catalog_list.csv'):
        check_hash(root/name,pins[name])
    import torch
    for route in ('music','voice'):
        for seed in (41,42,43):
            name=f'fixtures/heads/{route}_{seed}.pt'
            check_hash(root/name,pins[name])
            checkpoint=torch.load(root/name,map_location='cpu',weights_only=True)
            for identifier in checkpoint['train_ids']:
                tokens.update(('id:'+identifier,'parent:'+identifier))
    for filename in ('selected_list.csv','runtime_manifest.csv','catalog_list.csv'):
        with (root/'data'/filename).open(newline='',encoding='utf8') as stream:
            for row in csv.DictReader(stream):
                historical=str(row.get('historical_development_exposure','')).lower() in ('1','true','yes')
                selected=str(row.get('selected','')).lower() in ('1','true','yes')
                if filename=='catalog_list.csv' and not(historical or selected): continue
                tokens|=lineage_tokens(row)
                value=row.get('parent_ids','')
                if value:
                    try:
                        parsed=json.loads(value)
                        if isinstance(parsed,list): tokens|={'id:'+str(p) for p in parsed}|{'parent:'+str(p) for p in parsed}
                    except json.JSONDecodeError:
                        tokens.add('parent:'+value)
    return tokens


def validate_manifest(rows: list[dict], registry: set[str], expected_split: str | None = None, allow_synthetic=False) -> dict:
    if not rows or len({r.get('ID') for r in rows})!=len(rows):
        raise ContractError('Empty manifest or duplicate IDs')
    if any(r.get('synthetic',False) for r in rows) and not allow_synthetic:
        raise ContractError('Synthetic fixture is prohibited from production training/lock')
    for row in rows:
        if not isinstance(row['ID'],str) or not row['ID'].strip(): raise ContractError('Invalid ID')
        if row.get('split') not in ('meta_dev','locked_ood'): raise ContractError('Old val/test cannot be relabeled as new meta data')
        if expected_split and row['split']!=expected_split: raise ContractError('Unexpected split')
        if row.get('category') not in CATEGORIES: raise ContractError('Require one exclusive category')
        lineage=row.get('parents')
        if not isinstance(lineage,list) or not lineage: raise ContractError('Missing original parent lineage')
        if row['category']=='Mixed' and len(lineage)!=2: raise ContractError('Mixed requires exactly two original parents')
        if len(lineage)>2: raise ContractError('Initial experiment allows at most two parents')
        if len({p.get('parent_id') for p in lineage})!=len(lineage): raise ContractError('Duplicate or unspecified parents')
        for parent in lineage:
            required=('parent_id','audio_sha256','source','generator_family')
            if any(not _valid(parent.get(k)) for k in required): raise ContractError('Incomplete parent provenance')
            if len(str(parent['audio_sha256']))!=64: raise ContractError('Parent audio SHA-256 required')
        if len(str(row.get('audio_sha256','')))!=64: raise ContractError('Rendered audio SHA-256 required')
        exposure=row.get('base_exposure',{})
        if any(exposure.get(k)!='no' for k in EXPOSURES):
            raise ContractError(f"Downstream base exposure not cleared: {row['ID']}")
        if exposure.get('upstream') not in ('no','yes','unknown') or not exposure.get('evidence'):
            raise ContractError('Require exposure evidence; upstream absence cannot be inferred')
        if sorted(exposure.get('reviewed_parent_ids',[]))!=sorted(p['parent_id'] for p in lineage):
            raise ContractError('Exposure review must explicitly cover every original parent')
        validate_label_consistency(row)
        # Evidence declaration must cover all original ancestors, not just this derivative.
        if row.get('purpose','').startswith('diagnostic'): raise ContractError('Diagnostic data cannot train the stack')
        hits=lineage_tokens(row)&registry
        if hits: raise ContractError(f'Exposure registry match: {sorted(hits)[:4]}')
        if len(lineage)==2:
            if row.get('composition') not in ('RR','FR','RF','FF','unknown'):
                raise ContractError('Mixed composition must be voice/music RR,FR,RF,FF or unknown')
    groups=connected_groups(rows)
    splits={}
    for g,row in zip(groups,rows,strict=True): splits.setdefault(g,set()).add(row['split'])
    if any(len(s)>1 for s in splits.values()): raise ContractError('Connected parent/speaker/track/hash group crosses split')
    return counts(rows,groups)


def validate_ood_domains(dev: list[dict], ood: list[dict]) -> None:
    for key in ('source','generator_family'):
        sets=[{p[key] for r in part for p in r['parents'] if p[key]!='real'} for part in (dev,ood)]
        shared=sets[0]&sets[1]
        if shared: raise ContractError(f'locked OOD {key} not disjoint: {sorted(shared)}')


def counts(rows: list[dict], groups=None) -> dict:
    parents=[p for r in rows for p in r.get('parents',[])]
    def unique(key): return len({p[key] for p in parents if _valid(p.get(key))})
    return {'files':len(rows),'connected_groups':len(set(groups if groups is not None else connected_groups(rows))),
            'independent_parent_ids':unique('parent_id'),'unique_parent_hashes':unique('audio_sha256'),
            'speakers_recorded':unique('speaker'),'tracks_recorded':unique('track'),
            'generator_families':len({p['generator_family'] for p in parents if p.get('generator_family') not in (None,'real','unknown','')}),
            'sources':unique('source'),'category_counts':dict(Counter(r['category'] for r in rows)),
            'split_counts':dict(Counter(r['split'] for r in rows)),
            'upstream_exposure':dict(Counter(r['base_exposure'].get('upstream','unknown') for r in rows)),
            'caution':'Distinct identifiers/hashes do not prove independent originals; unresolved speaker/track identities reduce audit strength.'}


def assign_parent_splits(parents: list[dict], reserved_sources: list[str], reserved_generators: list[str]) -> list[dict]:
    """Assign ORIGINALS first; derivatives must inherit all parent assignments.
    No automatic random 80/20 forcing. An explicit OOD domain list is required.
    """
    if not reserved_sources or not reserved_generators:
        raise ContractError('Predeclare held-out source domains and fake generator families')
    rows=[{'ID':p['parent_id'],'parents':[p],**p} for p in parents]
    groups=connected_groups(rows); assignment={}
    for g,p in zip(groups,parents,strict=True):
        is_ood=p['source'] in reserved_sources
        if p['generator_family']!='real' and (p['generator_family'] in reserved_generators)!=is_ood:
            raise ContractError('Source/generator OOD reservations conflict; revise acquisition before mixing')
        split='locked_ood' if is_ood else 'meta_dev'
        if g in assignment and assignment[g]!=split: raise ContractError('Reserved domain shares a connected original with dev')
        assignment[g]=split
    return [{**p,'split':assignment[g],'parent_group':g} for p,g in zip(parents,groups,strict=True)]


def derivative_split(parent_ids: list[str], assignments: list[dict]) -> str:
    byid={p['parent_id']:p for p in assignments}
    if not parent_ids or len(set(parent_ids))!=len(parent_ids) or any(p not in byid for p in parent_ids):
        raise ContractError('Unknown/duplicate derivative parents')
    splits={byid[p]['split'] for p in parent_ids}
    if len(splits)!=1: raise ContractError('Cannot mix parents across splits')
    return next(iter(splits))


def validate_label_consistency(row: dict) -> None:
    from .evaluation import known_label
    names=('FILE_FAKE','VOICE_FAKE','MUSIC_FAKE','VOICE_PRESENT','MUSIC_PRESENT')
    if any(n not in row or n+'_mask' not in row for n in names):
        raise ContractError('All five labels and explicit masks are required, including unknowns')
    labels={n:known_label(row,n) for n in names}
    expected_presence={'Voice-only':(1,0),'Music-only':(0,1),'Mixed':(1,1)}.get(row['category'])
    if expected_presence is not None:
        actual=(labels['VOICE_PRESENT'],labels['MUSIC_PRESENT'])
        if actual!=expected_presence: raise ContractError('Category and known presence labels disagree')
    composition=row.get('composition')
    if composition in ('RR','FR','RF','FF'):
        for b,letter in zip(('VOICE','MUSIC'),composition,strict=True):
            if labels[b+'_PRESENT']!=1 or (labels[b+'_FAKE'] is not None and labels[b+'_FAKE']!=int(letter=='F')):
                raise ContractError('Composition and component labels disagree')
    active=[labels[b+'_FAKE'] for b in ('VOICE','MUSIC') if labels[b+'_PRESENT']==1]
    presence_known=labels['VOICE_PRESENT'] is not None and labels['MUSIC_PRESENT'] is not None
    inferred=1 if any(v==1 for v in active) else (0 if presence_known and all(v==0 for v in active) else None)
    if inferred is not None and labels['FILE_FAKE'] is not None and labels['FILE_FAKE']!=inferred:
        raise ContractError('File label conflicts with known present-component labels')
