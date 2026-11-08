"""Rebuild a nine-model evaluation and display exports from verified checkpoints."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.checkpoint_io import verified_checkpoint_hash
from src.config import ARTIFACTS_DIR, REPORTS_DIR
from src.data_pipeline import build_dataset
from src.evaluation import chronological_holdout, evaluate_all
from src.neural import NeuralSystem, seed_cpu
from src.recommenders import build_all_models
from src.transformer_reranker import TransformerReranker

BASE = ARTIFACTS_DIR/'neural/neural_recommender.pt'
TRANSFORMER = REPORTS_DIR/'transformer_reranking/transformer.pt'
OUTPUT = REPORTS_DIR/'repository_evaluation'
FILES = {'model_metrics.csv','evaluation_details.csv','recommendations.csv','candidate_protocols.csv',
         'quality_intervals.csv','run_summary.json'}


def load_models():
    seed_cpu(42)
    bundle = build_dataset(save=False)
    split = chronological_holdout(bundle.interactions,bundle.contents)
    models = build_all_models(split.contents,bundle.users,split.train)
    base_hash, transformer_hash = verified_checkpoint_hash(BASE), verified_checkpoint_hash(TRANSFORMER)
    system = NeuralSystem.load(BASE)
    # Bind the frozen checkpoint to the same dataset and chronological cutoff.
    for actual,expected in ((system.contents,split.contents),(system.interactions,split.train)):
        pd.testing.assert_frame_equal(actual.reset_index(drop=True),expected.reset_index(drop=True),check_dtype=False)
    models.update(system.recommenders())
    transformer = TransformerReranker.from_checkpoint(TRANSFORMER,system,base_hash)
    models[transformer.name] = transformer
    return bundle,split,models,{'base':base_hash,'transformer':transformer_hash}


def paired_intervals(details, baseline='v6_diversity_reranking'):
    rows=[]
    for metric in ('hit_at_k','ndcg_at_k'):
        pivot=details.pivot(index='user_id',columns='model',values=metric).sort_index()
        if baseline not in pivot or pivot.empty or not np.isfinite(pivot.to_numpy()).all():
            raise ValueError('Paired quality evaluation requires every model/user target')
        draws=np.random.default_rng(42).integers(0,len(pivot),size=(2000,len(pivot)))
        for name in sorted(pivot):
            delta=(pivot[name]-pivot[baseline]).to_numpy()
            low,high=np.percentile(delta[draws].mean(axis=1),[2.5,97.5])
            rows.append({'model':name,'baseline':baseline,'metric':metric,'users':len(delta),
                'mean_delta':float(delta.mean()),'low_95':float(low),'high_95':float(high),
                'bootstrap_repetitions':2000,'seed':42})
    return pd.DataFrame(rows)


def compute():
    bundle,split,models,hashes=load_models()
    metrics,details=evaluate_all(models,split.test)
    protocols=[]
    recommendations=[]
    for name,model in models.items():
        counts=[len(model.candidate_ids(user)) for user in split.test.user_id]
        protocols.append({'model':name,'candidate_protocol':'two_tower_top_50' if name.startswith(('v7_','v8_','v9_')) else 'baseline_model_defined',
            'min_candidates':min(counts),'mean_candidates':float(np.mean(counts)),'max_candidates':max(counts)})
        if name.startswith(('v7_','v8_','v9_')):
            for user in [*bundle.users.user_id,'new-user']:
                result=model.recommend(user)
                result=result[['content_id','title','topic_tags','score','model']].copy()
                result.insert(0,'user_id',user)
                result.insert(1,'rank',range(1,len(result)+1))
                recommendations.append(result)
    summary={'format_version':1,'models':list(models),'known_users':len(bundle.users),
        'recommendation_users':len(bundle.users)+1,'evaluated_users':len(split.test),
        'catalog_size':len(split.contents),'training_cutoff':split.cutoff.isoformat(),
        'checkpoint_sha256':hashes,'k':10,'bootstrap_baseline':'v6_diversity_reranking',
        'target_sha256':hashlib.sha256(split.test[['user_id','content_id']].sort_values('user_id').to_csv(index=False).encode()).hexdigest()}
    frames={'model_metrics.csv':metrics,'evaluation_details.csv':details,
        'candidate_protocols.csv':pd.DataFrame(protocols),'quality_intervals.csv':paired_intervals(details),
        'recommendations.csv':pd.concat(recommendations,ignore_index=True)}
    return frames,summary


def run(output):
    output=Path(output)
    output.mkdir(parents=True,exist_ok=True)
    if any(output.iterdir()):
        raise ValueError('Use an empty evaluation output directory')
    frames,summary=compute()
    for name,frame in frames.items():
        frame.to_csv(output/name,index=False)
    (output/'run_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    (output/'manifest.json').write_text(json.dumps({name:hashlib.sha256((output/name).read_bytes()).hexdigest()
                                                   for name in sorted(FILES)},indent=2)+'\n')
    return verify(output)


def verify(output=OUTPUT):
    from src.portfolio_exports import load_portfolio
    saved=load_portfolio(output)
    frames,summary=compute()
    if saved['run_summary.json'] != summary:
        raise ValueError('Repository evaluation summary differs from checkpoints and dataset')
    for name,frame in frames.items():
        pd.testing.assert_frame_equal(saved[name],frame,check_dtype=False,check_exact=False,atol=1e-10,rtol=1e-10)
    return {'models':len(summary['models']),'evaluated_users':summary['evaluated_users'],
            'saved_recommendation_groups':saved['recommendations.csv'].groupby(['model','user_id']).ngroups}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,default=OUTPUT)
    parser.add_argument('--verify-only',action='store_true')
    args=parser.parse_args()
    print(json.dumps(verify(args.output) if args.verify_only else run(args.output)))


if __name__=='__main__':
    main()
