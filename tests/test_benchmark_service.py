import copy

import numpy as np
import pytest

pytest.importorskip('torch')
pytest.importorskip('fastapi')
httpx = pytest.importorskip('httpx')
from src.benchmark_service import validate_response, summarize

CUTOFF = '2026-03-08T04:48:00+00:00'
MODEL = 'v7_pytorch_two_tower'
PAYLOAD = {'user_id': 'U0001', 'model': MODEL, 'checkpoint_sha256': 'checkpoint',
    'training_cutoff': CUTOFF, 'items': [
        {'content_id': 'a', 'title': 'A', 'topic_tags': 'bitcoin', 'score': 0.5},
        {'content_id': 'b', 'title': 'B', 'topic_tags': 'ethereum', 'score': 0.5}]}


def check(payload, status=200):
    response = httpx.Response(status, json=payload, request=httpx.Request('POST', 'http://localhost/recommend'))
    validate_response(response, (['a', 'b'], np.array([0.5, 0.5])), 'U0001', MODEL, 'checkpoint', CUTOFF)


def test_benchmark_accepts_complete_matching_response():
    check(PAYLOAD)


@pytest.mark.parametrize('mutation', ['array_score', 'string_score', 'bool_score', 'missing_title',
    'extra_field', 'wrong_cutoff', 'wrong_user', 'wrong_model', 'wrong_checkpoint', 'wrong_order',
    'wrong_score', 'missing_item', 'status_201'])
def test_benchmark_rejects_schema_identity_and_result_errors(mutation):
    payload = copy.deepcopy(PAYLOAD)
    status = 200
    if mutation == 'array_score':
        for item in payload['items']:
            item['score'] = [0.5]
    elif mutation == 'string_score':
        payload['items'][0]['score'] = '0.5'
    elif mutation == 'bool_score':
        payload['items'][0]['score'] = True
    elif mutation == 'missing_title':
        del payload['items'][0]['title']
    elif mutation == 'extra_field':
        payload['other'] = True
    elif mutation == 'wrong_cutoff':
        payload['training_cutoff'] = 'invalid'
    elif mutation == 'wrong_user':
        payload['user_id'] = 'U0002'
    elif mutation == 'wrong_model':
        payload['model'] = 'v8_pytorch_neural_ranker'
    elif mutation == 'wrong_checkpoint':
        payload['checkpoint_sha256'] = 'other'
    elif mutation == 'wrong_order':
        payload['items'].reverse()
    elif mutation == 'wrong_score':
        payload['items'][0]['score'] = 0.6
    elif mutation == 'missing_item':
        payload['items'].pop()
    else:
        status = 201
    with pytest.raises((ValueError, AssertionError)):
        check(payload, status)


def test_rejected_requests_count_as_errors_in_summary():
    records = [{'latency_ms': 1, 'error': ''}, {'latency_ms': 2, 'error': 'HTTPStatusError'},
               {'latency_ms': 4, 'error': 'ValidationError'}]
    result = summarize(records, 0.5)
    assert result['requests'] == 3
    assert result['successful_requests'] == 1
    assert result['errors'] == 2
    assert result['error_rate'] == 2 / 3
    assert result['successful_requests_per_second'] == 2
    assert result['p50_ms'] == result['p95_ms'] == result['p99_ms'] == 1
