import argparse
import gzip
import hashlib
import json
import urllib.request
from pathlib import Path

URL = 'https://ftp.ncbi.nlm.nih.gov/genomes/all/GCF/000/005/845/GCF_000005845.2_ASM584v2/GCF_000005845.2_ASM584v2_genomic.fna.gz'

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default='data/ecoli.fna')
    args = parser.parse_args()
    with urllib.request.urlopen(URL, timeout=60) as response:
        content = gzip.decompress(response.read())
    if not content.startswith(b'>'):
        raise ValueError('Download did not contain FASTA data')
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    path.with_suffix('.source.json').write_text(json.dumps({'url': URL, 'sha256': hashlib.sha256(content).hexdigest(), 'assembly': 'GCF_000005845.2'}, indent=2)+'\n')
    print(str(path))
