import hashlib
import json
import shutil
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip('torch')
from src.config import ARTIFACTS_DIR, REPORTS_DIR
from src.neural import seed_cpu
from src.run_transformer import BASE, DEFAULT_OUTPUT, verify
from src.transformer_reranker import ContentCrossEncoder, TransformerConfig, TransformerReranker, tokenize


@pytest.mark.parametrize('settings', [{'width': 31}, {'heads': 0}, {'epochs': True}, {'max_tokens': 0},
                                      {'vocabulary_size': 1}, {'learning_rate': float('nan')}])
def test_invalid_transformer_settings_are_rejected(settings):
    with pytest.raises(ValueError):
        TransformerConfig(**settings)


def test_text_hashing_is_deterministic_bounded_and_preserves_padding():
    config = TransformerConfig(max_tokens=8)
    assert tokenize('Bitcoin 安全', config) == tokenize('BITCOIN 安全', config)
    assert tokenize('', config) == [0]*8
    ids = tokenize(' '.join(['crypto']*100), config)
    assert len(ids) == 8 and all(0 < value < config.vocabulary_size for value in ids)


def test_padding_embedding_does_not_affect_valid_output_and_layers_differ():
    seed_cpu(42)
    config = TransformerConfig(max_tokens=8)
    network = ContentCrossEncoder(10, config).eval()
    users = torch.randn(2, 10)
    tokens = torch.tensor([tokenize('bitcoin', config), tokenize('', config)])
    with torch.inference_mode():
        before = network(users, tokens)
        network.embedding.weight[0].fill_(1234)
        after = network(users, tokens)
    torch.testing.assert_close(before, after, atol=1e-6, rtol=0)
    assert torch.isfinite(after).all()
    assert not torch.equal(network.layers[0].linear1.weight, network.layers[1].linear1.weight)


@pytest.fixture(scope='module')
def model():
    seed_cpu(42)
    return TransformerReranker.load(DEFAULT_OUTPUT/'transformer.pt', BASE)


def test_actual_training_examples_match_frozen_baseline_and_temporal_windows(model):
    audit = model.training_audit
    reference = pd.read_csv(ARTIFACTS_DIR/'neural/training_examples.csv')
    assert audit['indices'] == reference.index[reference.ranker_candidate].tolist()
    windows = np.asarray(audit['windows'])
    stamps = np.asarray(audit['timestamps_ns'])
    ends = np.asarray(audit['label_ends_ns'])
    bounds = np.array([pd.Timestamp(value).value for value in audit['boundaries']])
    assert set(windows) == {2, 3}
    assert (stamps > bounds[windows-1]).all() and (ends <= bounds[windows]).all()
    assert (ends > stamps).all()
    history = pd.DataFrame(model.history)
    assert history.validation_loss.idxmin() == model.training_summary['selected_epoch']
    assert model.training_summary['parameters_changed']


def test_reranking_preserves_candidates_seen_filter_and_chunk_equivalence(model):
    initial = model.config
    try:
        for user in ['U0001', 'new-user']:
            scores = model.rank_scores(user)
            candidates = model.system.candidate_indices(user)
            assert set(np.flatnonzero(np.isfinite(scores))) == set(candidates)
            assert not set(candidates) & model.system.seen.get(user, set())
            model.config = replace(initial, inference_batch_size=1)
            single = model.rank_scores(user)
            np.testing.assert_allclose(scores[candidates], single[candidates], atol=1e-6, rtol=1e-5)
            model.config = initial
            assert set(model.recommend(user, k=100).content_id) <= set(model.candidate_ids(user))
            assert model.recommend(user, k=0).empty
            with pytest.raises(ValueError):
                model.recommend(user, k=-1)
    finally:
        model.config = initial


def test_checkpoint_requires_identical_frozen_retrieval_weights(tmp_path):
    base = tmp_path/'base.pt'
    base.write_bytes(BASE.read_bytes()+b'changed')
    with pytest.raises(ValueError, match='Frozen retrieval checkpoint differs'):
        TransformerReranker.load(DEFAULT_OUTPUT/'transformer.pt', base)


def test_actual_transformer_exports_verify():
    assert verify(DEFAULT_OUTPUT) == {'files':12, 'recommendation_users':81, 'benchmark_requests':800}


@pytest.mark.parametrize('mutation', ['curve', 'quality', 'performance', 'missing_request'])
def test_modified_exports_fail_even_after_rehashing(tmp_path, mutation):
    output = tmp_path/'report'
    shutil.copytree(DEFAULT_OUTPUT, output)
    if mutation == 'curve':
        path = output/'training_history.csv'
        frame = pd.read_csv(path)
        frame.loc[0,'train_loss'] += 0.25
        frame.to_csv(path,index=False)
    elif mutation == 'quality':
        path = output/'quality_deltas.json'
        record = json.loads(path.read_text())
        record['mean_paired_delta'] += 0.1
        path.write_text(json.dumps(record))
    elif mutation == 'performance':
        path = output/'performance.csv'
        frame = pd.read_csv(path)
        frame.loc[0,'p95_ms'] += 10
        frame.to_csv(path,index=False)
    else:
        path = output/'latency_samples.csv'
        pd.read_csv(path).iloc[1:].to_csv(path,index=False)
    hashes = json.loads((output/'manifest.json').read_text())
    hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    (output/'manifest.json').write_text(json.dumps(hashes))
    with pytest.raises((ValueError, AssertionError)):
        verify(output)
