import argparse
import hashlib
import json
import math
import platform
import random
import time
from pathlib import Path
import torch
from model import DNABERT2ForMLM
from data import read_fasta, split_windows, train_tokenizer, encode_batches, mask_tokens

def evaluate(model, batches, vocab_size, device, seed, prior):
    generator = torch.Generator().manual_seed(seed)
    total, baseline, correct, count = 0.0, 0.0, 0, 0
    model.eval()
    with torch.inference_mode():
        for ids in batches:
            masked, labels = mask_tokens(ids, vocab_size, generator)
            selection = labels != -100
            amount = selection.sum().item()
            logits = model(masked.to(device), (ids != 0).to(device)).cpu()
            total += torch.nn.functional.cross_entropy(logits.reshape(-1, vocab_size), labels.reshape(-1), ignore_index=-100, reduction='sum').item()
            baseline -= prior[labels[selection]].log().sum().item()
            correct += (logits.argmax(-1)[selection] == labels[selection]).sum().item()
            count += amount
    if count == 0:
        raise ValueError('Evaluation produced no prediction targets')
    return {'masked_token_loss': total/count, 'perplexity': math.exp(min(total/count, 50)),
            'masked_token_accuracy': correct/count, 'masked_tokens': count, 'unigram_baseline_loss': baseline/count}

def run(fasta, output, epochs=5, window=512, hidden=256, heads=8, layers=6, vocab_size=4096, batch_size=8, seed=42, device='cpu', max_batches=0):
    if epochs < 1 or batch_size < 1 or hidden % heads:
        raise ValueError('Invalid training configuration')
    torch.manual_seed(seed)
    torch.set_num_threads(2)
    rng = random.Random(seed)
    splits, excluded = split_windows(read_fasta(fasta), window)
    tokenizer = train_tokenizer([r['sequence'] for r in splits['train']], vocab_size)
    vocab_size = tokenizer.get_vocab_size()
    config = {'v': vocab_size, 'h': hidden, 'heads': heads, 'inter': hidden*4, 'layers': layers, 'drop': 0.1}
    model = DNABERT2ForMLM(**config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.01)
    counts = torch.ones(vocab_size)
    for ids in encode_batches(splits['train'], tokenizer, batch_size):
        counts += torch.bincount(ids[ids >= 5], minlength=vocab_size)
    counts[:5] = 0
    prior = counts/counts.sum()
    destination = Path(output)
    destination.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(destination/'tokenizer.json'))
    manifest = {k: [{a: b for a, b in row.items() if a != 'sequence'} for row in rows] for k, rows in splits.items()}
    (destination/'split_manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    validation_batches = list(encode_batches(splits['validation'], tokenizer, batch_size))
    history, best = [], float('inf')
    initial = evaluate(model, validation_batches, vocab_size, device, seed+1, prior)
    mask_generator = torch.Generator().manual_seed(seed)
    started = time.perf_counter()
    for epoch in range(epochs):
        order = list(splits['train'])
        rng.shuffle(order)
        model.train()
        loss_sum, mask_count = 0.0, 0
        for index, ids in enumerate(encode_batches(order, tokenizer, batch_size)):
            if max_batches and index >= max_batches:
                break
            masked, labels = mask_tokens(ids, vocab_size, mask_generator)
            optimizer.zero_grad()
            logits = model(masked.to(device), (ids != 0).to(device))
            loss = torch.nn.functional.cross_entropy(logits.reshape(-1, vocab_size), labels.to(device).reshape(-1), ignore_index=-100)
            if not torch.isfinite(loss):
                raise ValueError('Non-finite training loss')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            amount = (labels != -100).sum().item()
            loss_sum += loss.item()*amount
            mask_count += amount
        validation = evaluate(model, validation_batches, vocab_size, device, seed+1, prior)
        history.append({'epoch': epoch+1, 'train_loss': loss_sum/mask_count, 'validation': validation})
        if validation['masked_token_loss'] < best:
            best = validation['masked_token_loss']
            torch.save({'model_state': model.state_dict(), 'config': config}, destination/'model.pt')
    model.load_state_dict(torch.load(destination/'model.pt', map_location=device, weights_only=True)['model_state'])
    test = evaluate(model, list(encode_batches(splits['test'], tokenizer, batch_size)), vocab_size, device, seed+2, prior)
    report = {'task': 'DNA BPE masked-language modeling', 'data_file': str(fasta), 'data_sha256': hashlib.sha256(Path(fasta).read_bytes()).hexdigest(),
              'split': '70/15/15 contiguous genomic regions per record, one-window gaps, exact/reverse-complement duplicates removed',
              'examples': {k: len(v) for k, v in splits.items()}, 'excluded_windows': excluded,
              'config': config, 'training': {'epochs': epochs, 'batch_size': batch_size, 'learning_rate': 5e-4, 'seed': seed, 'window': window, 'max_batches': max_batches},
              'initial_validation': initial, 'history': history, 'test': test, 'training_seconds': time.perf_counter()-started,
              'parameters': sum(p.numel() for p in model.parameters()),
              'limitations': 'Exact-match filtering does not remove all homologous regions. MLM metrics do not establish downstream biological utility.',
              'environment': {'torch': torch.__version__, 'python': platform.python_version(), 'platform': platform.platform(), 'device': device}}
    (destination/'metrics.json').write_text(json.dumps(report, indent=2)+'\n')
    return report

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--fasta', required=True)
    parser.add_argument('--output', default='outputs/dna-mlm')
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--window', type=int, default=512)
    parser.add_argument('--hidden', type=int, default=256)
    parser.add_argument('--heads', type=int, default=8)
    parser.add_argument('--layers', type=int, default=6)
    parser.add_argument('--vocab-size', type=int, default=4096)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--max-batches', type=int, default=0)
    args = parser.parse_args()
    print(json.dumps(run(**vars(args))))
