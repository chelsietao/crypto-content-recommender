"""Read hash-verified display tables without importing PyTorch or loading weights."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

FILES={'model_metrics.csv','evaluation_details.csv','recommendations.csv','candidate_protocols.csv',
       'quality_intervals.csv','run_summary.json'}


def load_portfolio(output):
    output=Path(output)
    manifest=json.loads((output/'manifest.json').read_text())
    if not isinstance(manifest,dict) or set(manifest)!=FILES:
        raise ValueError('Portfolio manifest must cover all six display exports')
    tables={}
    for name,digest in manifest.items():
        if hashlib.sha256((output/name).read_bytes()).hexdigest()!=digest:
            raise ValueError(f'Portfolio export hash mismatch: {name}')
        tables[name]=json.loads((output/name).read_text()) if name.endswith('.json') else pd.read_csv(output/name)
    summary=tables['run_summary.json']
    expected=set(summary['models'])
    metrics=tables['model_metrics.csv']
    if len(expected)!=9 or set(metrics.model)!=expected or not metrics.model.is_unique:
        raise ValueError('Portfolio must cover nine distinct model metrics')
    details=tables['evaluation_details.csv']
    targets=details[['user_id','target_content_id']].drop_duplicates()
    if (not targets.user_id.is_unique or len(targets)!=summary['evaluated_users']
            or details.duplicated(['model','user_id']).any()
            or set(zip(details.model,details.user_id))!={(name,user) for name in expected for user in targets.user_id}):
        raise ValueError('Portfolio evaluation target coverage is incomplete')
    recommendations=tables['recommendations.csv']
    neural={name for name in expected if name.startswith(('v7_','v8_','v9_'))}
    users=set(recommendations.user_id)
    if (len(users)!=summary['recommendation_users'] or 'new-user' not in users
            or set(zip(recommendations.model,recommendations.user_id))!={(name,user) for name in neural for user in users}
            or recommendations.duplicated(['model','user_id','rank']).any()
            or recommendations.duplicated(['model','user_id','content_id']).any()
            or not np.isfinite(recommendations.score).all()):
        raise ValueError('Portfolio recommendations are incomplete or invalid')
    return tables
