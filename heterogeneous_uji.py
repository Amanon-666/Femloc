"""AISTATS 2025 task-specific ridge UJI adaptation; proprietary author module loaded privately."""
import argparse
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import time

import numpy as np
import torch

import adkf_uji as common

AUTHOR_FILE = Path('/home/panyushuo/projects/panyushuo/external-private/meta-linear-2025/model_meta_linear.py')
OFFICIAL_URL = 'https://www.kecl.ntt.co.jp/as/members/iwata/code_meta_linear.tar.gz'
HIDDEN, BLOCKS, HEADS, LR = 128, 3, 4, 1e-4
METHODS = ['full', 'flatten', 'shuffle', 'scalar_ridge']
CHECKS = [300, 1000]
SCALAR_GRID = [1e-3, 1e-2, .1, 1., 10., 100.]


def make_model(seed):
    spec = importlib.util.spec_from_file_location('private_meta_linear_2025', AUTHOR_FILE)
    author = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(author)
    torch.manual_seed(seed)
    return author.MetaLinear(n_hidden=HIDDEN, n_layer=BLOCKS, n_head=HEADS,
                             use_mu=0, normalize=0, dropout=0., use_y=1,
                             use_mask=1, single_beta=0).double()


def active_columns(sx):
    # Local AP correspondence remains intact; Query never chooses columns.
    return torch.any(sx != 0, dim=0)


def generated_lambda(model, sx, sy):
    """Paper Eqs 1--8 via unchanged public author submodules; no Query input."""
    values = torch.cat([sx, sy], dim=1)
    types = torch.cat([torch.zeros_like(sx), torch.ones_like(sy)], dim=1)
    z = model.f_z(model.f_xym(torch.stack([values, types], dim=-1)))
    ap = model.f_mub(z[:, :sx.shape[1]].mean(0))[:, 1]
    bias = model.f_a1(model.f_a0(z.mean(0)).mean(0))[1:2]
    return torch.cat([ap, bias]).exp().clamp(min=1e-3)


def coefficients(sx, sy, lam):
    x = torch.cat([sx, torch.ones(len(sx), 1)], dim=1)
    precision_inv = lam.reciprocal()
    # Author code follows sum SSE, despite Eq9's printed 1/N.
    dual = torch.linalg.solve(torch.eye(len(sx)) + (x * precision_inv) @ x.T, sy)
    return precision_inv[:, None] * (x.T @ dual)


def predict(model, sx, sy, qx, method='full', scalar=1., permutation_seed=12345):
    active = active_columns(sx)
    s, q = sx[:, active], qx[:, active]
    if method == 'scalar_ridge':
        lam = torch.full((s.shape[1]+1,), scalar)
    else:
        lam = generated_lambda(model, s, sy)
        if method == 'flatten' and len(lam)>1:
            lam = torch.cat([lam[:-1].log().mean().exp().expand(len(lam)-1), lam[-1:]])
        elif method == 'shuffle' and len(lam)>1:
            rng = np.random.default_rng(permutation_seed)
            order = torch.tensor(rng.permutation(len(lam)-1))
            lam = torch.cat([lam[:-1][order], lam[-1:]])
    if not torch.isfinite(lam).all():
        raise FloatingPointError(f"Nonfinite lambda in {method}")
    w = coefficients(s, sy, lam)
    q = torch.cat([q, torch.ones(len(q), 1)], dim=1)
    return q @ w, lam, w


def source_names(floors, building):
    source = [n for n in floors if not n.startswith(f'B{building}') and len(common.eligible(floors[n]))>=30]
    dev = [max([n for n in source if n.startswith(f'B{b}')], key=lambda n:int(n.split('F')[1])) for b in range(3) if b!=building]
    return source, [n for n in source if n not in dev], dev


def protocol(building, seed, floors):
    source, train, dev = source_names(floors, building)
    return dict(building=building,seed=seed,source=source,train=train,dev=dev,
                source_scale=common.coordinate_scale(floors,source),hidden=HIDDEN,
                matrix_blocks=BLOCKS,attention_stages=2*BLOCKS,heads=HEADS,
                learning_rate=LR,checks=CHECKS,lambda_floor=1e-3,use_mu=0,
                source_model_url=OFFICIAL_URL,author_module_sha256=hashlib.sha256(AUTHOR_FILE.read_bytes()).hexdigest(),
                adapter_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                scalar_grid=SCALAR_GRID,feature_encoding='fixed 100->0; otherwise (RSS+105)/105; no X centering',
                active_AP='only nonzero columns in current Support; Query-only AP coefficient is zero',
                response='Support coordinate mean removed; source-only global RMS scale; 2 labels as target-type tokens',
                ridge='sum SSE; positive feature and bias lambdas; zero coefficient prior mean',
                controls='same trained checkpoint flatten/shuffle AP lambdas, preserve predicted bias lambda',
                target_manifest='previous frozen ADKF manifest, physical Query positions separated')


def evaluate(model, method, names, floors, valid, scale, draws=2, scalar=1., manifests=None):
    rows=[]
    if manifests is None:
        manifest=[]
        for name in names:
            f=floors[name];sp,qp=common.fixed_queries(f,771)
            qr=np.concatenate([f['pools'][j] for j in qp])
            for draw in range(draws):
                rng=np.random.default_rng(12000+draw);order=rng.permutation(sp)
                scans={int(j):rng.choice(f['pools'][j],3,False) for j in order[:max(common.KS)]}
                max_s=np.concatenate([scans[int(j)] for j in order[:max(common.KS)]])
                mask=~np.any(np.all(valid[name]['xy'][:,None,:]==f['xy'][max_s][None,:,:],axis=2),axis=1)
                for k in common.KS:
                    s=np.concatenate([scans[int(j)] for j in order[:k]])
                    manifest.append(dict(floor=name,draw=draw,k=k,support=f['ids'][s].tolist(),internal_query=f['ids'][qr].tolist(),official_query=valid[name]['ids'][mask].tolist()))
    else:
        manifest=manifests
    model.eval()
    with torch.no_grad():
        for entry in manifest:
            name=entry['floor'];f,v=floors[name],valid[name]
            fi={int(r):i for i,r in enumerate(f['ids'])};vi={int(r):i for i,r in enumerate(v['ids'])}
            s=np.array([fi[r] for r in entry['support']]);qi=np.array([fi[r] for r in entry['internal_query']]);qv=np.array([vi[r] for r in entry['official_query']])
            center=f['xy'][s].mean(0);sx=common.tensor(f['x'][s]);sy=common.tensor((f['xy'][s]-center)/scale)
            qx=common.tensor(np.vstack([f['x'][qi],v['x'][qv]]))
            yp,lam,w=predict(model,sx,sy,qx,method,scalar,12345+entry['draw'])
            yy=np.vstack([f['xy'][qi],v['xy'][qv]]);err=np.linalg.norm(yp.numpy()*scale+center-yy,axis=1)
            for split,sl in [('internal',slice(0,len(qi))),('official',slice(len(qi),None))]:
                rows.append(dict(floor=name,k=entry['k'],draw=entry['draw'],method=method,split=split,mde=float(err[sl].mean()),nquery=len(err[sl]),nactive=len(lam)-1,bias_lambda=float(lam[-1]),lambda_min=float(lam.min()),lambda_max=float(lam.max()),floor_fraction=float((lam<=1e-3).double().mean())))
    return rows,manifest


def train(floors,names,scale,seed,checks,out,dev=None,valid=None):
    model=make_model(seed);optimizer=torch.optim.Adam(model.parameters(),lr=LR)
    rng=np.random.default_rng(seed);start=time.time()
    for step in range(1,max(checks)+1):
        task=common.episode(floors[names[int(rng.integers(len(names)))]],rng,scale,augment=True)
        sx,sy,qx,qy=task[:4]
        model.train();optimizer.zero_grad()
        yp,lam,w=predict(model,sx,sy,qx)
        loss=(yp-qy).square().mean()
        assert torch.isfinite(loss)
        loss.backward()
        if not all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None):
            raise FloatingPointError(f"Nonfinite source gradient at step {step}")
        gradient_norm=torch.sqrt(sum(p.grad.square().sum() for p in model.parameters() if p.grad is not None)).detach()
        optimizer.step()
        if step%25==0:
            print(json.dumps(dict(step=step,seconds=round(time.time()-start,2),source_loss=float(loss.detach()),nactive=len(lam)-1,lambda_floor_fraction=float((lam<=1e-3).double().mean()),gradient_norm=float(gradient_norm))),flush=True)
        if step in checks:
            # Proprietary model-derived checkpoints remain private/internal; not added to git.
            torch.save(copy.deepcopy(model.state_dict()),out/f'checkpoint_{step}.pt')
            if dev is not None:
                scores=[]
                for method in METHODS[:-1]:
                    rows,_=evaluate(model,method,dev,floors,valid,scale)
                    scores.append(dict(method=method,step=step,mde=float(np.mean([r['mde'] for r in rows]))))
                (out/f'dev_{step}.json').write_text(json.dumps(scores,indent=2))
                print('dev',scores,flush=True)
    return model


def preflight(floors,spec,out):
    model=make_model(spec['seed']);model.eval();rng=np.random.default_rng(888)
    sx,sy,qx,qy,*_=common.episode(floors[spec['train'][0]],rng,spec['source_scale'],k=3,augment=True)
    active=active_columns(sx);s,q=sx[:,active],qx[:,active]
    checks={}
    for ny in [1,2]:
        yp,lam,w=predict(model,sx,sy[:,:ny],qx)
        original_loss=model(s,sy[:,:ny],q,qy[:,:ny])[0]
        our_loss=(yp-qy[:,:ny]).square().mean()
        checks[f'author_loss_difference_y{ny}']=float((original_loss-our_loss).abs())
        assert checks[f'author_loss_difference_y{ny}']<1e-10
    yp,lam,w=predict(model,sx,sy,qx)
    perm=torch.tensor(rng.permutation(sx.shape[1]))
    yp_perm,lp,_=predict(model,sx[:,perm],sy,qx[:,perm])
    checks['AP_rename_prediction_max_difference']=float((yp-yp_perm).abs().max());assert checks['AP_rename_prediction_max_difference']<1e-10
    sample_order=torch.tensor(rng.permutation(len(sx)))
    ys,ls,_=predict(model,sx[sample_order],sy[sample_order],qx)
    checks['Support_row_permutation_max_difference']=float((yp-ys).abs().max());assert checks['Support_row_permutation_max_difference']<1e-10
    zero=torch.zeros_like(sx)
    yp0,l0,w0=predict(model,zero,sy,qx)
    checks['all_missing_source_center_max_relative_prediction']=float(yp0.abs().max());assert checks['all_missing_source_center_max_relative_prediction']<1e-10
    new_s=torch.cat([sx,torch.zeros(len(sx),1)],dim=1);new_q=torch.cat([qx,torch.ones(len(qx),1)],dim=1)
    yp_new,_,_=predict(model,new_s,sy,new_q)
    checks['query_only_AP_prediction_difference']=float((yp_new-yp).abs().max());assert checks['query_only_AP_prediction_difference']<1e-10
    ypzero,_,_=predict(model,sx,torch.zeros_like(sy),qx)
    checks['zero_residual_max_prediction']=float(ypzero.abs().max());assert checks['zero_residual_max_prediction']==0
    # Finite difference on a nonconstant positive lambda, not on clamped values.
    s1=torch.cat([s,torch.ones(len(s),1)],dim=1);q1=torch.cat([q,torch.ones(len(q),1)],dim=1)
    ll=lam.detach().log().requires_grad_();val=(q1@coefficients(s,sy,ll.exp())-qy).square().mean();g=torch.autograd.grad(val,ll)[0];j=int(g.abs().argmax());eps=1e-5
    v0=ll.detach().clone();v1=v0.clone();v0[j]-=eps;v1[j]+=eps
    fd=(((q1@coefficients(s,sy,v1.exp())-qy).square().mean()-(q1@coefficients(s,sy,v0.exp())-qy).square().mean())/(2*eps)).item()
    checks['lambda_gradient_analytic']=float(g[j]);checks['lambda_gradient_fd']=fd;checks['lambda_gradient_relative_error']=abs(fd-float(g[j]))/(abs(float(g[j]))+1e-12);assert checks['lambda_gradient_relative_error']<1e-4
    # Identical Support columns remain indistinguishable under a shared equivariant policy.
    dup_s=torch.cat([s[:, :1], s[:, :1]],dim=1)
    dup_q=torch.tensor([[.2,.8],[.8,.2]])
    dup_pred,dup_lam,dup_w=predict(model,dup_s,sy,dup_q)
    checks['identical_column_lambda_difference']=float((dup_lam[0]-dup_lam[1]).abs().detach())
    checks['identical_column_coefficient_difference']=float((dup_w[0]-dup_w[1]).abs().max().detach())
    checks['identical_S_AP_swapped_Q_difference']=float((dup_pred[0]-dup_pred[1]).abs().max().detach())
    assert max(checks[k] for k in ['identical_column_lambda_difference','identical_column_coefficient_difference','identical_S_AP_swapped_Q_difference'])<1e-10
    toy_s=torch.tensor([[0.,.2],[.2,0.]])
    toy_y=torch.tensor([[-1.],[1.]])
    toy_q=torch.tensor([[.8,0.,1.]])
    toy_w=coefficients(toy_s,toy_y,torch.full((3,),1e-3))
    checks['linear_ridge_extrapolation']=float((toy_q@toy_w).item())
    assert checks['linear_ridge_extrapolation']>1.
    one_pred,one_lam,one_w=predict(model,sx[:1],torch.zeros(1,2),qx)
    checks['single_anchor_centeredY_prediction']=float(one_pred.abs().max().detach())
    assert checks['single_anchor_centeredY_prediction']==0.
    checks['n_parameters']=sum(p.numel() for p in model.parameters());checks['source_preflight_only']=True
    (out/'preflight.json').write_text(json.dumps(checks,indent=2));print(checks,flush=True)
    train(floors,spec['train'],spec['source_scale'],spec['seed'],[25],out)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--building',type=int,required=True);parser.add_argument('--seed',type=int,default=0);parser.add_argument('--mode',choices=['preflight','run'],default='preflight');args=parser.parse_args()
    floors=common.load(common.DATA/'trainingData.csv');valid=common.load(common.DATA/'validationData.csv')
    out=Path(__file__).parent/'outputs'/'heterogeneous_meta'/f'B{args.building}_seed{args.seed}';out.mkdir(parents=True,exist_ok=True)
    spec=protocol(args.building,args.seed,floors);(out/'protocol.json').write_text(json.dumps(spec,indent=2));print(spec,flush=True)
    if args.mode=='preflight':preflight(floors,spec,out);return
    train(floors,spec['train'],spec['source_scale'],args.seed,CHECKS,out,spec['dev'],valid)
    full_step=min(CHECKS,key=lambda step:next(r['mde'] for r in json.loads((out/f'dev_{step}.json').read_text()) if r['method']=='full'))
    choices={method:full_step for method in METHODS[:-1]}
    devmodel=make_model(args.seed);scalar_scores={}
    for scalar in SCALAR_GRID:
        rr,_=evaluate(devmodel,'scalar_ridge',spec['dev'],floors,valid,spec['source_scale'],scalar=scalar)
        scalar_scores[scalar]=float(np.mean([r['mde'] for r in rr]))
    scalar=min(scalar_scores,key=scalar_scores.get)
    (out/'selected.json').write_text(json.dumps(dict(checkpoints=choices,scalar_ridge=scalar,scalar_source_scores=scalar_scores),indent=2))
    final=out/'final';final.mkdir(exist_ok=True);train(floors,spec['source'],spec['source_scale'],args.seed,sorted(set(choices.values())),final)
    manifests=json.loads((Path('/home/panyushuo/projects/panyushuo/DKT-ADKF-20261013/outputs/adkf_lobo')/f'B{args.building}_seed{args.seed}'/'manifest.json').read_text())
    rows=[]
    for method in METHODS:
        model=make_model(args.seed)
        if method!='scalar_ridge':model.load_state_dict(torch.load(final/f'checkpoint_{choices[method]}.pt',weights_only=True))
        rr,_=evaluate(model,method,[],floors,valid,spec['source_scale'],scalar=scalar,manifests=manifests);rows.extend(rr)
    (out/'manifest.json').write_text(json.dumps(manifests));(out/'results.json').write_text(json.dumps(rows,indent=2));print('DONE',choices,scalar,flush=True)

if __name__=='__main__':main()
