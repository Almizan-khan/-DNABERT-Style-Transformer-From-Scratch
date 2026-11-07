import argparse
import json
import random
from pathlib import Path
from train import run

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default='reports/smoke-benchmark.json')
    args = parser.parse_args()
    rng = random.Random(42)
    path = Path('outputs/fixture.fna')
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('>synthetic_seed_42\n'+''.join(rng.choices('ACGT', k=16384))+'\n')
    report = run(str(path), 'outputs/smoke', epochs=2, window=64, hidden=32, heads=4, layers=1, vocab_size=64, batch_size=8, max_batches=8)
    report['scope'] = 'tiny CPU training/evaluation smoke experiment on synthetic random DNA; not a biological benchmark or DNABERT reproduction'
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report))
