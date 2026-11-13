"""Placeholder, intentionally prohibits running an unverified experimental policy."""
import argparse,json,pathlib,sys
p=argparse.ArgumentParser();p.add_argument('--strategy',choices='ABC',required=True);p.add_argument('--anchor-json',required=True);p.add_argument('--output',required=True);a=p.parse_args()
record={'strategy':a.strategy,'anchor':json.loads(a.anchor_json),'status':'BLOCKED_NOT_IMPLEMENTED','reason':'Real forward, writeback, read provenance and policy integration are not yet validated.'}
pathlib.Path(a.output,'manifest.json').write_text(json.dumps(record,indent=2));print(json.dumps(record));sys.exit(2)
