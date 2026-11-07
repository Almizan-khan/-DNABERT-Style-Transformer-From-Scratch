import random
import unittest
import torch
from data import split_windows, reverse_complement, mask_tokens, train_tokenizer
from model import DNABERT2ForMLM

class DNATests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        torch.set_num_threads(2)

    def test_windows_have_disjoint_genomic_coordinates(self):
        rng = random.Random(42)
        splits, _ = split_windows([('contig', ''.join(rng.choices('ACGT', k=8192)))], 64)
        regions = {name: set(i for row in rows for i in range(row['start'], row['end'])) for name, rows in splits.items()}
        self.assertFalse(regions['train'] & regions['validation'])
        self.assertFalse(regions['train'] & regions['test'])
        self.assertFalse(regions['test'] & regions['validation'])

    def test_masking_ignores_special_tokens_and_is_reproducible(self):
        ids = torch.tensor([[2, 5, 6, 3, 0]])
        a, labels = mask_tokens(ids, 10, torch.Generator().manual_seed(42))
        b, _ = mask_tokens(ids, 10, torch.Generator().manual_seed(42))
        torch.testing.assert_close(a, b)
        self.assertTrue((labels[ids < 5] == -100).all())
        self.assertGreater((labels != -100).sum().item(), 0)

    def test_padding_does_not_change_valid_token_logits(self):
        model = DNABERT2ForMLM(16, 16, 4, 32, 1, 0).eval()
        a = model(torch.tensor([[5, 6]]), torch.ones(1, 2))
        b = model(torch.tensor([[5, 6, 0]]), torch.tensor([[1, 1, 0]]))
        torch.testing.assert_close(a, b[:, :2], atol=1e-6, rtol=1e-5)

    def test_finite_masked_loss_and_gradient(self):
        model = DNABERT2ForMLM(16, 16, 4, 32, 1, 0)
        ids = torch.tensor([[5, 6, 7, 8]])
        masked, labels = mask_tokens(ids, 16, torch.Generator().manual_seed(1))
        loss = torch.nn.functional.cross_entropy(model(masked, torch.ones_like(ids)).reshape(-1, 16), labels.reshape(-1), ignore_index=-100)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(model.embedding.token_embedding.weight.grad.abs().sum().item(), 0)

    def test_all_padding_is_rejected(self):
        with self.assertRaises(ValueError):
            DNABERT2ForMLM(16, 16, 4, 32, 1, 0)(torch.zeros(1, 4, dtype=torch.long), torch.zeros(1, 4))

    def test_tokenizer_special_mapping_is_stable(self):
        tokenizer = train_tokenizer(['ACGTACGT', 'TGCATGCA'], 16)
        self.assertEqual(tokenizer.token_to_id('[PAD]'), 0)
        self.assertEqual(tokenizer.token_to_id('[MASK]'), 4)
        self.assertEqual(reverse_complement('ACG'), 'CGT')
