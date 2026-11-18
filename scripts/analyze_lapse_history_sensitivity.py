# Recomputes Table 11 and Supplement S5 (lapse-history sensitivity analysis) from saved outputs.
# Usage: python scripts/analyze_lapse_history_sensitivity.py --main <notebook-01 SAVE_DIR> --sens <lapse_history_sensitivity folder> [--out s5_results.csv]
# Tests: paired, two-sided exact Wilcoxon signed-rank tests on the 10 fold-level AUCs, Holm-adjusted across the
# 24 configurations within each task and variant. With 10 non-zero, untied differences the minimum exact
# two-sided P is 2/2**10 = 0.001953125, which gives 0.046875 (reported as .047) after Holm adjustment over 24.
import argparse, json, glob, os, numpy as np, pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score
ap=argparse.ArgumentParser(); ap.add_argument('--main',required=True); ap.add_argument('--sens',required=True); ap.add_argument('--out',default='s5_results.csv')
A=ap.parse_args(); M=A.main; S=A.sens
names=sorted(os.path.basename(f)[:-5] for f in glob.glob(M+'/*.json') if 'fold_assignment' not in f and os.path.exists(f'{S}/no_prior/'+os.path.basename(f)))
LAB={'LR':'Logistic regression','RF':'Random forest','XGB':'XGBoost','SVM':'SVM','LSTM':'LSTM','GRU':'GRU','GRUD':'GRU-D','TCN':'TCN','GRUTCN':'GRU-TCN','TFM':'Transformer','SSLGRU':'SSL-GRU','SSLGRUD':'SSL-GRU-D'}
TR={'LR','RF','XGB','SVM'}
def preds(d,name,task):
    fs=sorted(glob.glob(f'{d}/predictions/{name}__{task}__fold*.csv')); assert len(fs)==10,(d,name,task,len(fs))
    return [pd.read_csv(f,dtype={'participant_id':str}) for f in fs]
def pp(P):
    P=pd.concat(P); a=[roc_auc_score(g.y_true,g.y_prob) for _,g in P.groupby('participant_id') if 0<g.y_true.sum()<len(g)]
    return float(np.median(a)), len(a)
def holm(p):
    p=np.asarray(p,float); o=np.argsort(p); adj=np.empty(len(p)); r_=0
    for r,i in enumerate(o): r_=max(r_,min(1,(len(p)-r)*p[i])); adj[i]=r_
    return adj
rows=[]; foldcheck=0
for name in names:
    base,fs=name.rsplit('_',1)
    full=json.load(open(f'{M}/{name}.json'))['fold_aucs']
    for task in ('classify','next'):
        fa=np.array(full[task],float)
        Pm=preds(M,name,task)
        r={'model':LAB[base],'family':'traditional' if base in TR else 'sequence','features':'original + engineered' if fs=='exp' else 'original','task':task,
           'full':fa.mean()}
        r['full_pp'],r['n_pp']=pp(Pm)
        for v in ('no_prior','no_history'):
            d=json.load(open(f'{S}/{v}/{name}.json'))['folds']; va=np.array([d[str(k)][task] for k in range(10)],float)
            Pv=preds(f'{S}/{v}',name,task)
            # same test participants per fold?
            for k in range(10):
                if set(Pm[k].participant_id)!=set(Pv[k].participant_id): foldcheck+=1
            dd=va-fa
            r[v]=va.mean(); r[v+'_delta']=dd.mean(); r[v+'_p']=stats.wilcoxon(dd, method='exact').pvalue if np.any(dd!=0) else 1.0
            r[v+'_pp'],_=pp(Pv)
        rows.append(r)
R=pd.DataFrame(rows)
for v in ('no_prior','no_history'):
    for t in ('classify','next'):
        m=R.task==t; R.loc[m,v+'_p_holm']=holm(R.loc[m,v+'_p'])
print('fold participant mismatches:',foldcheck)
R.to_csv(A.out,index=False)
pd.set_option('display.width',260); pd.set_option('display.float_format','{:.3f}'.format)
for t in ('classify','next'):
    print('\n==',t); print(R[R.task==t].sort_values(['features','full'],ascending=[True,False])[['model','features','full','no_prior','no_prior_delta','no_prior_p_holm','no_history','no_history_delta','no_history_p_holm','full_pp','no_prior_pp','no_history_pp','n_pp']].to_string(index=False))
print()
for v in ('full','no_prior','no_history'):
    for t in ('classify','next'):
        s=R[R.task==t][v].sort_values(ascending=False)
        sub=R[R.task==t].pivot(index='model',columns='features',values=v); g=sub['original + engineered']-sub['original']
        trad=g[g.index.isin([LAB[x] for x in TR])]; seq=g.drop(trad.index)
        ppc='full_pp' if v=='full' else v+'_pp'
        print(f'{v:10s} {t:8s} range {s.min():.3f}-{s.max():.3f}; 15-best spread {s.iloc[0]-s.iloc[14]:.3f}; eng gain trad {trad.min():+.3f}..{trad.max():+.3f} seq {seq.min():+.3f}..{seq.max():+.3f}; pp {R[R.task==t][ppc].min():.3f}-{R[R.task==t][ppc].max():.3f}; pooled-pp {(R[R.task==t][v]-R[R.task==t][ppc]).min():.3f}..{(R[R.task==t][v]-R[R.task==t][ppc]).max():.3f}')
for v in ('no_prior','no_history'):
    for t in ('classify','next'):
        for fsx in ('original','original + engineered'):
            m=(R.task==t)&(R.features==fsx); x=R[m]
            print(f'{v} {t} {fsx}: delta {x[v+"_delta"].min():+.3f}..{x[v+"_delta"].max():+.3f}; holm<.05 {int((x[v+"_p_holm"]<.05).sum())}/{len(x)}; pp change {(x[v+"_pp"]-x["full_pp"]).min():+.3f}..{(x[v+"_pp"]-x["full_pp"]).max():+.3f}')
