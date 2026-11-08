#!/usr/bin/env bash
set -euo pipefail
export CUBLAS_WORKSPACE_CONFIG=:4096:8
python - <<'PY'
import torch
if torch.cuda.device_count() < 2:
    raise SystemExit('This experiment requires two CUDA GPUs')
print({'torch':torch.__version__, 'gpus':[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]})
PY
python -m src.validate_distributed --device cuda --precision float64 \
  --epochs 3 --comparison numerical --output reports/ddp_gpu_reference
python -m src.validate_distributed --device cuda --precision float32 \
  --epochs 30 --comparison ranking --repetitions 3 --output reports/ddp_gpu_benchmark
python - <<'PY'
import json
from pathlib import Path
for folder in ['ddp_gpu_reference','ddp_gpu_benchmark']:
    result=json.loads((Path('reports')/folder/'comparison.json').read_text())
    print('DDP_RESULT_JSON '+json.dumps(result),flush=True)
PY
python - <<'PY'
import base64,hashlib,io,zipfile
from pathlib import Path
buffer=io.BytesIO()
with zipfile.ZipFile(buffer,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=9) as archive:
    for folder in ['ddp_gpu_reference','ddp_gpu_benchmark']:
        for path in sorted((Path('reports')/folder).rglob('*')):
            if path.is_file() and path.name!='launcher.log':
                archive.write(path,str(path))
raw=buffer.getvalue()
encoded=base64.b64encode(raw).decode()
chunks=[encoded[i:i+4096] for i in range(0,len(encoded),4096)]
print(f'DDP_ARCHIVE_SHA256 {hashlib.sha256(raw).hexdigest()} {len(raw)} {len(chunks)}',flush=True)
for index,chunk in enumerate(chunks):
    print(f'DDP_ARCHIVE_CHUNK {index} {chunk}',flush=True)
PY
