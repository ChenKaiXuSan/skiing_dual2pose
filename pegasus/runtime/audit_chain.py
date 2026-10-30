"""Read-only checks of prepared commands and migrated baseline/metadata inputs."""
import argparse
import importlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys


def main():
    p=argparse.ArgumentParser();p.add_argument('--plan',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False)
    plan=json.loads((a.plan/'plan.json').read_text());repo=Path(plan['repo']);root=Path(plan['root']);report={'status':'running','checks':{},'errors':[],'full_rerun_started':False}
    def check(name,fn):
        try:report['checks'][name]=fn()
        except Exception as exc:report['errors'].append({'check':name,'error':repr(exc)})
        (a.output/'report.json').write_text(json.dumps(report,indent=2))
    def paths():
        missing=set();legacy=set();seen=set()
        for job in plan['jobs']:
            text=json.dumps(job)
            legacy.update(re.findall(r'/home/[^\s"\\,}]+',text))
            for raw in re.findall(r'/work/[^\s"\\,};]+',text):
                if raw.endswith(':'):raw=raw[:-1]
                path=Path(raw)
                if path.is_relative_to(root):continue
                # Manifest CSV globs select future outputs; all external inputs
                # must be concrete present files/directories or explicit globs.
                seen.add(raw)
                if not path.exists() and not any(path.parent.glob(path.name)):missing.add(raw)
        if missing or legacy:raise ValueError({'missing':sorted(missing),'workstation_paths':sorted(legacy)})
        return {'external_paths':len(seen),'legacy_paths':0}
    check('command_paths',paths)
    def modules():
        names=set()
        for job in plan['jobs']:
            for cmd in job['commands']:
                if '-m' in cmd:names.add(cmd[cmd.index('-m')+1])
        for name in sorted(names):importlib.import_module(name)
        return sorted(names)
    check('entrypoint_imports',modules)
    from dual2pose.runtime_paths import input_path
    from dual2pose.eval.main_baseline_data import sha256
    def raw_indexes():
        results={}
        for ds,splits in [('unity',['train','val','test']),('ski',['train','test'])]:
            for split in splits:
                path=Path(plan['context']['raw_cache'])/ds/split/'manifest.json';m=json.loads(path.read_text());index=input_path(m['index'])
                if sha256(index)!=m['index_sha256']:raise ValueError(str(index))
                for name in m['files']:
                    if not (path.parent/name).is_file():raise FileNotFoundError(path.parent/name)
                results[f'{ds}/{split}']={'index':str(index),'samples':m['sample_count'],'index_sha256':m['index_sha256']}
        return results
    check('raw_cache_indexes',raw_indexes)
    def frontends():
        from dual2pose.experiments.workflow_training import frontend_manifests
        from dual2pose.eval.frontend_manifest import FrontEndManifest
        result={}
        for name,splits in frontend_manifests(repo).items():
            for split,path in splits.items():
                m=FrontEndManifest.load(path);result[f'{name}/{split}']=len(m.entries)
        base=repo/'logs/ivc_mmsports_extension/image_occlusion/inference'
        for pattern in ('random','distal','temporal'):
            for ratio in ('0p5','1'):
                path=base/pattern/ratio/'frontend_manifest.json';m=FrontEndManifest.load(path);result[f'rgb/{pattern}/{ratio}']=len(m.entries)
        return result
    check('frontend_manifest_paths',frontends)
    def c07():
        base=repo/'logs/ivc_review_revision_20260916/occluded_geometry_full/frontend';n=0
        for f in base.glob('*/frontend_manifest.json'):
            m=json.loads(f.read_text())
            for entry in m['entries']:
                if not input_path(entry['path']).exists():raise FileNotFoundError(input_path(entry['path']))
                n+=1
        if n==0:raise ValueError('Missing C07 inputs')
        return {'streams':n}
    check('c07_manifest_paths',c07)
    def official():
        result={}
        for name in ('DeciWatch','STRIDE','google-research'):
            directory=repo/'.cache/external_baselines'/name
            commit=subprocess.check_output(['git','-C',str(directory),'rev-parse','HEAD'],text=True).strip()
            dirty=subprocess.check_output(['git','-C',str(directory),'status','--porcelain'],text=True).strip()
            if dirty:raise ValueError(name+' official checkout is modified')
            result[name]=commit
        return result
    check('official_baseline_sources',official)
    report['status']='passed' if not report['errors'] else 'failed';(a.output/'report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2));raise SystemExit(bool(report['errors']))


if __name__=='__main__':main()
