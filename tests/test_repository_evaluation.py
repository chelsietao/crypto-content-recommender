import hashlib
import json
import shutil

import pandas as pd
import pytest

pytest.importorskip('torch')
from src.config import REPORTS_DIR
from src.evaluate_repository import OUTPUT, paired_intervals, verify
from src.portfolio_exports import load_portfolio


def test_nine_model_pipeline_recomputes_every_metric_and_recommendation():
    assert verify()=={'models':9,'evaluated_users':30,'saved_recommendation_groups':243}
    tables=load_portfolio(OUTPUT)
    expected=pd.read_csv(REPORTS_DIR/'REPOSITORY_MODEL_METRICS.csv')
    pd.testing.assert_frame_equal(tables['model_metrics.csv'],expected,check_exact=False,atol=1e-10,rtol=1e-10)
    v6=tables['quality_intervals.csv'].query("model == baseline")
    assert len(v6)==2
    assert v6[['mean_delta','low_95','high_95']].eq(0).all().all()
    protocols=tables['candidate_protocols.csv']
    assert protocols[protocols.model.str.startswith(('v7_','v8_','v9_'))].max_candidates.le(50).all()


def test_paired_quality_requires_every_user_and_rejects_duplicate_model_user():
    details=load_portfolio(OUTPUT)['evaluation_details.csv']
    with pytest.raises(ValueError):
        paired_intervals(details.iloc[1:])
    with pytest.raises(ValueError):
        paired_intervals(pd.concat([details,details.iloc[:1]]))


@pytest.mark.parametrize('mutation',['target','interval','candidate','missing_recommendation'])
def test_recomputed_evaluation_rejects_rehashed_wrong_exports(tmp_path,mutation):
    output=tmp_path/'evaluation'
    shutil.copytree(OUTPUT,output)
    if mutation=='target':
        name='evaluation_details.csv'
        frame=pd.read_csv(output/name)
        frame.loc[0,'target_content_id']='wrong-content'
    elif mutation=='interval':
        name='quality_intervals.csv'
        frame=pd.read_csv(output/name)
        frame.loc[0,'mean_delta']+=0.2
    elif mutation=='candidate':
        name='candidate_protocols.csv'
        frame=pd.read_csv(output/name)
        frame.loc[0,'max_candidates']+=1
    else:
        name='recommendations.csv'
        frame=pd.read_csv(output/name).query("user_id != 'new-user'")
    frame.to_csv(output/name,index=False)
    hashes=json.loads((output/'manifest.json').read_text())
    hashes[name]=hashlib.sha256((output/name).read_bytes()).hexdigest()
    (output/'manifest.json').write_text(json.dumps(hashes))
    with pytest.raises((ValueError,AssertionError)):
        verify(output)
