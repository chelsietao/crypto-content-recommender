from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch
import json
import shutil

import numpy as np
import pandas as pd
import pytest

pytest.importorskip('torch')
pytest.importorskip('fastapi')
pytest.importorskip('httpx')
from fastapi.testclient import TestClient
from src.config import ARTIFACTS_DIR
from src.neural import NeuralSystem
from src.serve_neural import create_app, DEFAULT_CHECKPOINT, MODEL_NAMES


@pytest.fixture(scope='module')
def service():
    with patch.object(NeuralSystem, 'load', wraps=NeuralSystem.load) as loader:
        app = create_app()
        with TestClient(app) as client:
            yield app, client, loader


def test_service_loads_once_and_exposes_model_identity(service):
    app, client, loader = service
    assert client.get('/health').json() == {'status': 'ok'}
    ready = client.get('/ready').json()
    assert ready['status'] == 'ready'
    assert ready['device'] == 'cpu'
    assert ready['torch_threads'] == ready['workers'] == ready['inference_slots'] == 1
    assert ready['catalog_size'] == 199
    assert len(ready['checkpoint_sha256']) == 64
    for name in MODEL_NAMES:
        assert client.post('/recommend', json={'user_id': 'U0001', 'model': name}).status_code == 200
    assert loader.call_count == 1


def test_http_results_match_all_saved_users_and_scores(service):
    _, client, _ = service
    saved = pd.read_csv(ARTIFACTS_DIR / 'neural' / 'recommendations.csv')
    for (model, user), rows in saved.groupby(['model', 'user_id']):
        response = client.post('/recommend', json={'user_id': user, 'model': model})
        assert response.status_code == 200
        items = response.json()['items']
        rows = rows.sort_values('rank')
        assert [row['content_id'] for row in items] == rows.content_id.tolist()
        np.testing.assert_allclose([row['score'] for row in items], rows.score, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize('change', [
    {'user_id': ''}, {'user_id': '  '}, {'user_id': 'x' * 129}, {'user_id': 123},
    {'model': 'missing'}, {'k': 0}, {'k': -1}, {'k': 51}, {'k': True},
    {'k': '10'}, {'k': 1.5}, {'exclude_seen': 'false'}, {'other': True},
])
def test_invalid_requests_rejected_before_inference(service, change):
    app, client, _ = service
    system = app.state.runtime[0]
    with patch.object(system, 'retrieval_scores', wraps=system.retrieval_scores) as calls:
        response = client.post('/recommend', json={'user_id': 'U0001', **change})
    assert response.status_code == 422
    assert calls.call_count == 0


def test_new_user_k_and_seen_controls_preserve_model_state(service):
    app, client, _ = service
    system, models, _ = app.state.runtime
    original_seen = {key: set(value) for key, value in system.seen.items()}
    for name in MODEL_NAMES:
        for user in ['U0001', 'new-user']:
            for exclude in [True, False]:
                result = client.post('/recommend', json={
                    'user_id': user, 'model': name, 'k': 50, 'exclude_seen': exclude}).json()
                expected = models[name].recommend(user, k=50, exclude_seen=exclude)
                assert [row['content_id'] for row in result['items']] == expected.content_id.tolist()
                assert len(result['items']) == min(50, len(expected))
    assert system.seen == original_seen


def test_concurrent_requests_return_consistent_results(service):
    _, client, loader = service
    expected = client.post('/recommend', json={'user_id': 'U0001'}).json()
    def request(_):
        response = client.post('/recommend', json={'user_id': 'U0001'})
        assert response.status_code == 200
        return response.json()
    with ThreadPoolExecutor(max_workers=8) as pool:
        assert all(value == expected for value in pool.map(request, range(40)))
    assert loader.call_count == 1


def test_not_ready_and_checkpoint_hash_mismatch(tmp_path):
    client = TestClient(create_app())
    assert client.get('/ready').status_code == 503
    assert client.post('/recommend', json={'user_id': 'U0001'}).status_code == 503
    client.close()
    checkpoint = tmp_path / DEFAULT_CHECKPOINT.name
    shutil.copy2(DEFAULT_CHECKPOINT, checkpoint)
    (tmp_path / 'manifest.json').write_text(json.dumps({checkpoint.name: '0' * 64}))
    with pytest.raises(ValueError, match='hash does not match'):
        with TestClient(create_app(checkpoint)):
            pass


@pytest.mark.parametrize('capacity, expected_status', [(1, 429), (2, 503)])
def test_full_or_timed_out_queue_rejects_and_recovers(capacity, expected_status):
    import threading
    started, release = threading.Event(), threading.Event()
    app = create_app(max_inflight=capacity, queue_timeout=0.05)
    with TestClient(app) as client:
        model = app.state.runtime[1][MODEL_NAMES[0]]
        original = model.recommend
        def blocked(*args, **kwargs):
            started.set()
            assert release.wait(5), 'Test did not release inference'
            return original(*args, **kwargs)
        with ThreadPoolExecutor(max_workers=1) as pool:
            with patch.object(model, 'recommend', side_effect=blocked):
                running = pool.submit(client.post, '/recommend', json={'user_id': 'U0001'})
                try:
                    assert started.wait(3)
                    response = client.post('/recommend', json={'user_id': 'U0002'})
                    assert response.status_code == expected_status
                    assert response.headers['Retry-After'] == '1'
                    assert client.get('/health').status_code == 200
                    assert client.get('/ready').json()['max_inflight'] == capacity
                finally:
                    release.set()
                assert running.result(timeout=3).status_code == 200
        assert client.post('/recommend', json={'user_id': 'U0002'}).status_code == 200


def test_failed_inference_releases_all_capacity():
    app = create_app(max_inflight=1, queue_timeout=0.05)
    with TestClient(app, raise_server_exceptions=False) as client:
        model = app.state.runtime[1][MODEL_NAMES[0]]
        with patch.object(model, 'recommend', side_effect=RuntimeError('injected inference failure')):
            assert client.post('/recommend', json={'user_id': 'U0001'}).status_code == 500
        assert client.post('/recommend', json={'user_id': 'U0001'}).status_code == 200


@pytest.mark.parametrize('settings', [
    {'max_inflight': 0}, {'max_inflight': 33}, {'max_inflight': True}, {'max_inflight': 1.5},
    {'queue_timeout': 0}, {'queue_timeout': -1}, {'queue_timeout': float('nan')},
    {'queue_timeout': float('inf')}, {'queue_timeout': True},
])
def test_invalid_queue_settings_rejected(settings):
    with pytest.raises(ValueError):
        create_app(**settings)


def test_transformer_http_matches_saved_users_and_exposes_both_checkpoint_hashes(service):
    from src.config import REPORTS_DIR
    _,client,loader=service
    loaded_before_requests=loader.call_count
    saved=pd.read_csv(REPORTS_DIR/'transformer_reranking/recommendations.csv')
    ready=client.get('/ready').json()
    assert len(ready['models'])==3 and len(ready['transformer_checkpoint_sha256'])==64
    for user,rows in saved.groupby('user_id'):
        response=client.post('/recommend',json={'user_id':user,'model':MODEL_NAMES[2]})
        assert response.status_code==200
        payload=response.json()
        assert payload['checkpoint_sha256']==ready['checkpoint_sha256']
        assert payload['reranker_checkpoint_sha256']==ready['transformer_checkpoint_sha256']
        assert [item['content_id'] for item in payload['items']]==rows.sort_values('rank').content_id.tolist()
        np.testing.assert_allclose([item['score'] for item in payload['items']],rows.sort_values('rank').score,atol=1e-6,rtol=1e-5)
    assert loader.call_count==loaded_before_requests
    models=service[0].state.runtime[1]
    assert models[MODEL_NAMES[2]].system is models[MODEL_NAMES[0]].system


def test_transformer_can_be_disabled_and_failure_is_rejected_before_inference():
    with TestClient(create_app(transformer_checkpoint=None)) as client:
        assert len(client.get('/ready').json()['models'])==2
        assert client.post('/recommend',json={'user_id':'new-user','model':MODEL_NAMES[2]}).status_code==422
        assert client.post('/recommend',json={'user_id':'new-user'}).status_code==200


def test_changed_transformer_manifest_fails_closed(tmp_path):
    from src.serve_neural import DEFAULT_TRANSFORMER
    target=tmp_path/'transformer.pt'
    target.write_bytes(DEFAULT_TRANSFORMER.read_bytes())
    (tmp_path/'manifest.json').write_text(json.dumps({'transformer.pt':'0'*64}))
    with pytest.raises(ValueError,match='hash does not match'):
        with TestClient(create_app(transformer_checkpoint=target)):
            pass
