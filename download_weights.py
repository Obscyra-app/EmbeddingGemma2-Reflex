"""Download the pinned public release and verify every file before using it."""
import argparse,hashlib,json
from pathlib import Path
REPO='Nurymanau/EmbeddingGemma2-Reflex'
REVISION='17cfbac9be0f36f051da598a4c7a045e8f249aba'
MANIFEST='MANIFEST.json'

def sha(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for block in iter(lambda:f.read(8*1024**2),b''):h.update(block)
 return h.hexdigest()

def main():
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=Path('weights'));p.add_argument('--verify-only',action='store_true');a=p.parse_args()
 frozen=json.loads((Path(__file__).parent/'MODEL_MANIFEST.json').read_text())
 if not a.verify_only:
  from huggingface_hub import snapshot_download
  snapshot_download(REPO,revision=REVISION,local_dir=a.output,allow_patterns=sorted(frozen['files'])+[MANIFEST],token=False)
 for name,info in frozen['files'].items():
  f=a.output/name
  if not f.is_file() or sha(f)!=info['sha256']:raise RuntimeError('Missing or mismatched file: '+name)
 print('Verified',len(frozen['files']),'files from',REPO,'at',REVISION)
if __name__=='__main__':main()
