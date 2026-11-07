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
