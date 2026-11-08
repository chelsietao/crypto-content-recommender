import json
import shutil
import subprocess
import sys

import pytest
from streamlit.testing.v1 import AppTest
from src.config import PROJECT_ROOT,REPORTS_DIR
from src.portfolio_exports import load_portfolio

OUTPUT=REPORTS_DIR/'repository_evaluation'


def test_display_reader_has_no_torch_import_or_weight_loading():
    command="from src.portfolio_exports import load_portfolio; import sys; load_portfolio(sys.argv[1]); assert 'torch' not in sys.modules"
    result=subprocess.run([sys.executable,'-c',command,str(OUTPUT)],cwd=PROJECT_ROOT,capture_output=True,text=True,timeout=20)
    assert result.returncode==0,result.stderr


@pytest.mark.parametrize('mutation',['missing_manifest','wrong_hash','missing_required_file'])
def test_display_rejects_incomplete_or_changed_exports(tmp_path,mutation):
    folder=tmp_path/'report'
    shutil.copytree(OUTPUT,folder)
    if mutation=='missing_manifest':
        (folder/'manifest.json').unlink()
    elif mutation=='wrong_hash':
        with (folder/'model_metrics.csv').open('a') as handle:
            handle.write('changed')
    else:
        hashes=json.loads((folder/'manifest.json').read_text())
        hashes.pop('recommendations.csv')
        (folder/'manifest.json').write_text(json.dumps(hashes))
    with pytest.raises((ValueError,OSError)):
        load_portfolio(folder)


def test_streamlit_shows_nine_models_and_v9_cold_user_recommendations():
    app=AppTest.from_file(str(PROJECT_ROOT/'app.py')).run(timeout=60)
    assert not app.exception
    assert 'Saved Model Results' in [tab.label for tab in app.tabs]
    app.selectbox(key='saved_model').select('v9_transformer_content_reranker')
    app.selectbox(key='saved_user').select('new-user')
    app.run(timeout=60)
    assert not app.exception and not app.error
    tables=[element.value for element in app.dataframe]
    assert any('model' in table and len(table)==9 and 'v9_transformer_content_reranker' in set(table.model) for table in tables)
    expected=load_portfolio(OUTPUT)['recommendations.csv'].query("model == 'v9_transformer_content_reranker' and user_id == 'new-user'")
    matches=[table for table in tables if set(['rank','content_id','title','topic_tags','score'])==set(table.columns)]
    assert len(matches)==1
    assert matches[0].content_id.tolist()==expected.content_id.tolist()
