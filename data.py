import re
import torch
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel

SPECIAL = ['[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]']

def read_fasta(path):
    records, name, sequence = [], None, []
    with open(path) as stream:
        for line in stream:
            line = line.strip()
            if not line:
                continue
            if line.startswith('>'):
                if name is not None:
                    records.append((name, ''.join(sequence).upper()))
                name, sequence = line[1:].split()[0], []
            else:
                if name is None:
                    raise ValueError('FASTA must start with a record header')
                sequence.append(line)
        if name is not None:
            records.append((name, ''.join(sequence).upper()))
    if not records:
        raise ValueError('No FASTA records found')
    return records

def reverse_complement(sequence):
    return sequence.translate(str.maketrans('ACGT', 'TGCA'))[::-1]

def split_windows(records, window=512):
    if window < 8:
        raise ValueError('window must be at least 8')
    result = {'train': [], 'validation': [], 'test': []}
    seen, removed = set(), 0
    for name, sequence in records:
        first, second = int(len(sequence)*0.70), int(len(sequence)*0.85)
        regions = {'train': (0, first), 'validation': (first+window, second), 'test': (second+window, len(sequence))}
        for split, (begin, end) in regions.items():
            for start in range(begin, end-window+1, window):
                text = sequence[start:start+window]
                if re.fullmatch('[ACGT]+', text) is None:
                    removed += 1
                    continue
                key = min(text, reverse_complement(text))
                if key in seen:
                    removed += 1
                    continue
                seen.add(key)
                result[split].append({'record': name, 'start': start, 'end': start+window, 'sequence': text})
    if any(not rows for rows in result.values()):
        raise ValueError('Each genomic-region split needs windows after gap, ambiguity and duplicate filtering. Supply longer sequences.')
    return result, removed

def train_tokenizer(sequences, vocab_size=4096):
    tokenizer = Tokenizer(BPE(unk_token='[UNK]'))
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
    tokenizer.train_from_iterator(sequences, BpeTrainer(vocab_size=vocab_size, min_frequency=2, special_tokens=SPECIAL, show_progress=False))
    return tokenizer

def encode_batches(rows, tokenizer, batch_size):
    encoded = [torch.tensor(tokenizer.encode(row['sequence']).ids, dtype=torch.long) for row in rows]
    for start in range(0, len(encoded), batch_size):
        yield torch.nn.utils.rnn.pad_sequence(encoded[start:start+batch_size], batch_first=True, padding_value=0)

def mask_tokens(ids, vocab_size, generator, probability=0.15):
    if not 0 < probability <= 1 or vocab_size <= 5:
        raise ValueError('Invalid masking probability or vocabulary')
    eligible = ids >= 5
    chosen = (torch.rand(ids.shape, generator=generator) < probability) & eligible
    for i in range(len(ids)):
        positions = eligible[i].nonzero().flatten()
        if len(positions) and not chosen[i].any():
            chosen[i, positions[torch.randint(len(positions), (1,), generator=generator)]] = True
    labels = ids.clone()
    labels[~chosen] = -100
    masked = ids.clone()
    draws = torch.rand(ids.shape, generator=generator)
    masked[chosen & (draws < 0.8)] = 4
    random_positions = chosen & (draws >= 0.8) & (draws < 0.9)
    random_ids = torch.randint(5, vocab_size, ids.shape, generator=generator)
    masked[random_positions] = random_ids[random_positions]
    return masked, labels
