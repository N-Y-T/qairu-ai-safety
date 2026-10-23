"""ART/SHAP on the site's 8-dimensional cosine retrieval function, NOT on LLM weights."""
import json,importlib.metadata as md,hashlib
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
from art.estimators.classification import BlackBoxClassifier
from art.attacks.evasion import HopSkipJump
import shap
OUT=Path(__file__).resolve().parents[1]/'public/evaluations'
cfg=json.loads((OUT/'retriever.json').read_text())
D=np.array([d['features'] for d in cfg['documents']],dtype=np.float32)
ids=[d['id'] for d in cfg['documents']]
def predict(x):
 x=np.asarray(x,dtype=np.float32)
 norm=np.linalg.norm(x,axis=1,keepdims=True)
 return (x@D.T)/np.maximum(norm*np.sqrt(2),1e-12)
for p in cfg['probes']:
 scores=predict([p['vector']])[0];order=sorted(range(4),key=lambda i:(-float(scores[i]),ids[i]));got=[{'id':ids[i],'score':float(scores[i])} for i in order if scores[i]>0][:2]
 assert [r['id'] for r in got]==[r['id'] for r in p['retrieved']]
 for a,b in zip(got,p['retrieved']):assert abs(a['score']-b['score'])<1e-6
print('retriever parity: 6 probes pass')
np.random.seed(42)
count=0
def counted(x):
 global count
 count+=len(x);return predict(x)
model=BlackBoxClassifier(predict_fn=counted,input_shape=(8,),nb_classes=4,clip_values=(0.,1.))
attack=HopSkipJump(classifier=model,targeted=False,max_iter=12,max_eval=200,init_eval=40,init_size=50,verbose=False)
x=D.copy();adv=attack.generate(x=x);before=np.argmax(predict(x),axis=1);after=np.argmax(predict(adv),axis=1)
art={'tool':'IBM ART','version':md.version('adversarial-robustness-toolbox'),'timestamp_utc':datetime.now(timezone.utc).isoformat(),'scope':'Continuous feature-space decision-boundary test of the cosine retriever. Fractional features may not correspond to realizable text. No LLM calls or text jailbreak; no fixed perturbation-budget claim.','algorithm':'HopSkipJump','config':{'max_iter':12,'max_eval':200,'init_eval':40,'init_size':50,'clip_values':[0,1],'numpy_global_seed':42,'note':'ART internal RNG may not be controlled by the global seed; saved adversarial vectors are the evidence.'},'parity_probes':6,'predict_queries':count,'rows':[{'original_id':ids[int(a)],'adversarial_id':ids[int(b)],'changed':bool(a!=b),'l2':float(np.linalg.norm(z-v)),'x':v.tolist(),'x_adv':z.tolist()} for a,b,v,z in zip(before,after,x,adv)]}
(OUT/'art.json').write_text(json.dumps(art,indent=2));print('ART',sum(r['changed'] for r in art['rows']),'/4 label flips')
explainer=shap.ExactExplainer(predict,np.zeros((1,8),dtype=np.float32),feature_names=cfg['features'])
ex=explainer(x);error=float(np.max(np.abs(np.asarray(ex.values).sum(axis=1)+np.asarray(ex.base_values)-predict(x))))
result={'tool':'SHAP','version':md.version('shap'),'timestamp_utc':datetime.now(timezone.utc).isoformat(),'scope':'Exact Shapley feature contributions to cosine document scores; zero-vector background. Explains retrieval only, not Qwen reasoning or security.','explainer':'ExactExplainer','features':cfg['features'],'output_ids':ids,'base_values':np.asarray(ex.base_values).tolist(),'values':np.asarray(ex.values).tolist(),'inputs':x.tolist(),'predictions':predict(x).tolist(),'max_additivity_error':error}
(OUT/'shap.json').write_text(json.dumps(result,indent=2));print('SHAP additivity error',error)
