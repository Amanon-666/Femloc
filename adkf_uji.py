"""UJI LOBO adaptation of ADKF-IFT; exact author JVP hypergradient, source-only selection."""
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
from scipy.optimize import minimize
import torch
from torch import nn
from torch.func import functional_call

sys.path.insert(0, str(Path(__file__).parent))
from dkt.data import load, coordinate_scale
from third_party.adkf.cauchy_hypergradient_jvp import cauchy_hypergradient_jvp

torch.set_default_dtype(torch.float64)
torch.set_num_threads(2)
DATA = Path('/home/panyushuo/projects/panyushuo/UJIIndoorLoc/data/raw/UJIndoorLoc')
KS = [3, 5, 10, 20]
SCANS, QTRAIN = 3, 30
NOISE_MODE, PRIOR_STD = .01, .25
SIGNAL_PRIOR = True
SIGNAL_PRIOR_STD = 1.
CHECKS = [300, 1000]


def tensor(a):
    return torch.as_tensor(a, dtype=torch.float64)


def d2(a, b):
    # cdist's backward has no second derivative; squared differences support exact IFT.
    return (a[:, None] - b[None, :]).square().sum(-1)


def kernel(a, b, p):
    return p[0].exp() * (-d2(a, b) / (2 * (2*p[1]).exp())).exp()


def lognormal_nll(logvalue, mode, sd):
    return .5*((logvalue-(mode+sd*sd))/sd).square() + logvalue + math.log(sd)


def support_loss(z, y, p, loc):
    K = kernel(z, z, p) + (p[2].exp()+1e-8)*torch.eye(len(z))
    L = torch.linalg.cholesky(K)
    val = .5*(y @ torch.cholesky_solve(y[:, None], L)[:, 0]) + L.diag().log().sum() + .5*len(y)*math.log(2*math.pi)
    if loc is not None:
        val = val + lognormal_nll(p[1], loc, PRIOR_STD) + lognormal_nll(p[2], math.log(NOISE_MODE), PRIOR_STD)
        if SIGNAL_PRIOR:
            val = val + lognormal_nll(p[0], 0., SIGNAL_PRIOR_STD)
    return val / len(y)


def posterior(s, y, q, p, full=False):
    K = kernel(s, s, p) + (p[2].exp()+1e-8)*torch.eye(len(s))
    L = torch.linalg.cholesky(K)
    cross = kernel(q, s, p)
    mean = cross @ torch.cholesky_solve(y[:, None], L)[:, 0]
    if full:
        cov = kernel(q, q, p) - cross @ torch.cholesky_solve(cross.T, L)
        return mean, cov + (p[2].exp()+1e-8)*torch.eye(len(q))
    var = p[0].exp()+p[2].exp() - (cross*torch.cholesky_solve(cross.T, L).T).sum(-1)
    return mean, var


def fit(z, y, prior=True, fixed_loc=None):
    z = z.detach()
    dist = d2(z, z)
    loc = (.5*dist[dist>0].median()).sqrt().log().detach() if fixed_loc is None else fixed_loc
    initial = np.array([0., loc.item(), math.log(NOISE_MODE)])
    def objective(v):
        p = tensor(v).requires_grad_()
        value = support_loss(z, y, p, loc if prior else None)
        grad = torch.autograd.grad(value, p)[0]
        return value.item(), grad.detach().numpy()
    fitted = minimize(objective, initial, method='L-BFGS-B', jac=True,
                       options={'maxiter':100, 'gtol':1e-7, 'ftol':1e-12, 'maxls':40})
    p = tensor(fitted.x).requires_grad_()
    return p, loc, {'success':bool(fitted.success), 'grad_max':float(np.abs(fitted.jac).max()), 'niter':int(fitted.nit), 'message':str(fitted.message)}


class Model(nn.Module):
    def __init__(self, learned):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(520,40), nn.ReLU(), nn.Linear(40,40), nn.ReLU()) if learned else nn.Identity()
        self.gp = nn.Parameter(torch.tensor([[0.,0.,math.log(NOISE_MODE)],[0.,0.,math.log(NOISE_MODE)]]))

    def forward(self, x):
        return self.net(x)


def eligible(f):
    return np.array([i for i,p in enumerate(f['pools']) if len(p)>=SCANS])


def episode(f, rng, scale, k=None, query_positions=None, augment=False):
    k = int(rng.choice(KS)) if k is None else k
    chosen = rng.choice(eligible(f),k,False)
    s = np.concatenate([rng.choice(f['pools'][j],SCANS,False) for j in chosen])
    others = np.setdiff1d(np.arange(len(f['pools'])), chosen) if query_positions is None else query_positions
    assert not np.intersect1d(chosen,others).size
    q = np.array([rng.choice(f['pools'][j]) for j in rng.choice(others,min(QTRAIN,len(others)),False)])
    o = f['xy'][s].mean(0)
    sy, qy = (f['xy'][s]-o)/scale, (f['xy'][q]-o)/scale
    if augment:
        t = rng.uniform(0,2*np.pi)
        R = np.array([[np.cos(t),-np.sin(t)],[np.sin(t),np.cos(t)]]) @ np.diag([1.,rng.choice([-1.,1.])])
        sy,qy = sy @ R.T,qy @ R.T
    return tensor(f['x'][s]),tensor(sy),tensor(f['x'][q]),tensor(qy),s,q,o


def predict(md, sx, sy, qx, adapt):
    with torch.no_grad():
        s,q = md(sx),md(qx)
    means, vars, diagnostics = [],[],[]
    for a in range(2):
        if adapt:
            p,_,diag = fit(s,sy[:,a]); diagnostics.append(diag)
            if diag['grad_max']>1e-4:
                raise RuntimeError(f'Target GP not stationary: {diag}')
        else:
            p = md.gp[a].detach()
        with torch.no_grad():
            m,v = posterior(s,sy[:,a],q,p)
        means.append(m); vars.append(v)
    return torch.stack(means,-1).numpy(), torch.stack(vars,-1).numpy(), diagnostics


def adkf_step(md, task):
    sx,sy,qx,qy = task[:4]
    params = tuple(md.net.parameters())
    names = tuple(n for n,p in md.net.named_parameters())
    with torch.no_grad():
        s = md(sx)
    fits = [fit(s,sy[:,a]) for a in range(2)]
    inner = tuple(v[0] for v in fits)
    locs = [v[1] for v in fits]
    def embed(pp,x):
        return functional_call(md.net,dict(zip(names,pp)),(x,))
    def f_inner(po,pi):
        z = embed(po,sx)
        return sum(support_loss(z,sy[:,a],pi[a],locs[a]) for a in range(2))/2
    def f_outer(po,pi):
        zs,zq = embed(po,sx),embed(po,qx)
        out = 0
        for a in range(2):
            m,c = posterior(zs,sy[:,a],zq,pi[a],full=True)
            out = out-torch.distributions.MultivariateNormal(m,covariance_matrix=c).log_prob(qy[:,a])
        return out / (2*len(qx))
    for v in fits:
        if v[2]['grad_max']>1e-4:
            raise RuntimeError(f"Inner optimization not stationary: {v[2]}")
    h = torch.autograd.functional.hessian(lambda *pi:f_inner(params,pi),inner)
    H = torch.cat([torch.cat(r,dim=1) for r in h],dim=0)
    eigen = torch.linalg.eigvalsh(H).detach()
    if eigen.min()<=1e-8:
        raise RuntimeError(f"IFT Hessian is not strictly positive/invertible: {eigen.tolist()}")
    for v in fits:
        v[2]['hessian_min']=float(eigen.min())
    loss = cauchy_hypergradient_jvp(f_outer,f_inner,params,inner,'cpu')
    return loss, fits, f_inner


def fixed_queries(f, seed):
    rng = np.random.default_rng(seed)
    allpos = np.arange(len(f['pools']))
    qp = rng.choice(allpos,max(1,int(.3*len(allpos))),False)
    sp = np.setdiff1d(eligible(f),qp)
    assert len(sp)>=max(KS)
    return sp, qp


def evaluate(md, kind, names, floors, valid, scale, draws, manifest=None):
    rows = []
    for name in names:
        f=floors[name]
        sp,qp = fixed_queries(f,771)
        qr = np.concatenate([f['pools'][j] for j in qp])
        v = valid[name]
        # Official Query rows at any Support coordinate are excluded explicitly.
        for draw in range(draws):
            rng=np.random.default_rng(12000+draw)
            order=rng.permutation(sp)
            scans={int(j):rng.choice(f['pools'][j],SCANS,False) for j in order[:max(KS)]}
            # Common official Query for all K/methods in this draw.
            max_s=np.concatenate([scans[int(j)] for j in order[:max(KS)]])
            mask=~np.any(np.all(v['xy'][:,None,:]==f['xy'][max_s][None,:,:],axis=2),axis=1)
            for k in KS:
                selected=order[:k]
                s=np.concatenate([scans[int(j)] for j in selected])
                o=f['xy'][s].mean(0)
                qx=np.vstack([f['x'][qr],v['x'][mask]])
                yy=np.vstack([f['xy'][qr],v['xy'][mask]])
                m,var,diags=predict(md,tensor(f['x'][s]),tensor((f['xy'][s]-o)/scale),tensor(qx),kind not in ('dkt','raw_source'))
                pred=m*scale+o
                err=np.linalg.norm(pred-yy,axis=1)
                cov=((pred-yy)**2/(var*scale**2)).sum(-1)<=4.605170185988091
                for part,sl in [('internal',slice(0,len(qr))),('official',slice(len(qr),None))]:
                    rows.append(dict(floor=name,draw=draw,k=k,method=kind,split=part,mde=float(err[sl].mean()),coverage90=float(cov[sl].mean()),nquery=len(err[sl]),fit=diags))
                if manifest is not None:
                    manifest.append(dict(floor=name,draw=draw,k=k,support=f['ids'][s].tolist(),internal_query=f['ids'][qr].tolist(),official_query=v['ids'][mask].tolist()))
    return rows


def train_models(floors,names,scale,seed,checks,out,devnames=None,valid=None):
    torch.manual_seed(seed)
    dkt=Model(True); adkf=copy.deepcopy(dkt); raw=Model(False)
    optd=torch.optim.Adam(dkt.parameters(),lr=1e-3)
    opta=torch.optim.Adam(adkf.net.parameters(),lr=1e-3)
    optr=torch.optim.Adam(raw.parameters(),lr=1e-3)
    rng=np.random.default_rng(seed)
    states={}; start=time.time()
    for step in range(1,max(checks)+1):
        name=names[int(rng.integers(len(names)))]
        task=episode(floors[name],rng,scale,augment=True)
        sx,sy,qx,qy=task[:4]
        optd.zero_grad()
        z=dkt(torch.cat([sx,qx])); y=torch.cat([sy,qy])
        loss=sum(support_loss(z,y[:,a],dkt.gp[a],None) for a in range(2))/2
        loss.backward(); optd.step()
        optr.zero_grad(); rz=raw(torch.cat([sx,qx])); rloss=sum(support_loss(rz,y[:,a],raw.gp[a],None) for a in range(2))/2; rloss.backward(); optr.step()
        opta.zero_grad()
        aloss,fits,_=adkf_step(adkf,task)
        assert torch.isfinite(aloss)
        assert all(torch.isfinite(p.grad).all() for p in adkf.net.parameters())
        opta.step()
        if step%25==0:
            print(json.dumps(dict(step=step,seconds=round(time.time()-start,1),dkt_loss=float(loss.detach()),adkf_loss=float(aloss.detach()),inner=fits[0][2])),flush=True)
        if step in checks:
            states[step]={'dkt':copy.deepcopy(dkt.state_dict()),'adkf_ift':copy.deepcopy(adkf.state_dict()),'raw_source':copy.deepcopy(raw.state_dict())}
            torch.save(states[step],out/f'checkpoint_{step}.pt')
            if devnames:
                dev=[]
                for key,md in [('dkt',dkt),('adkf_ift',adkf),('raw_source',raw)]:
                    r=evaluate(md,key,devnames,floors,valid,scale,2)
                    dev.append(dict(method=key,step=step,mde=float(np.mean([x['mde'] for x in r]))))
                (out/f'dev_{step}.json').write_text(json.dumps(dev,indent=2))
                print('dev',dev,flush=True)
    return states


def main():
    p=argparse.ArgumentParser(); p.add_argument('--building',type=int,required=True);p.add_argument('--seed',type=int,default=0);p.add_argument('--mode',choices=['preflight','run'],default='preflight');a=p.parse_args()
    floors=load(DATA/'trainingData.csv');valid=load(DATA/'validationData.csv')
    targets=[n for n in floors if n.startswith(f'B{a.building}') and n in valid and len(eligible(floors[n]))>=30]
    sources=[n for n in floors if not n.startswith(f'B{a.building}') and len(eligible(floors[n]))>=30]
    devs=[max([n for n in sources if n.startswith(f'B{b}')],key=lambda n:int(n.split('F')[1])) for b in range(3) if b!=a.building]
    train=[n for n in sources if n not in devs]
    scale=coordinate_scale(floors,sources)
    out=Path(__file__).parent/'outputs'/'adkf_lobo'/f'B{a.building}_seed{a.seed}';out.mkdir(parents=True,exist_ok=True)
    spec=dict(building=a.building,seed=a.seed,sources=sources,train=train,dev=devs,target=targets,source_coordinate_scale=scale,ks=KS,scans=SCANS,checks=CHECKS,noise_prior_mode_normalized=NOISE_MODE,signal_prior_mode_normalized=1.,signal_prior_logstd=SIGNAL_PRIOR_STD,noise_std_mode_meters=scale*np.sqrt(NOISE_MODE),fixed_internal_query_fraction=.3,torch_threads=2,source_code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (out/'protocol.json').write_text(json.dumps(spec,indent=2));print(spec,flush=True)
    if a.mode=='preflight':
        torch.manual_seed(a.seed);md=Model(True);rng=np.random.default_rng(4321)
        audits=[]
        for k in KS:
            task=episode(floors[train[0]],rng,scale,k=k,augment=True)
            loss,fits,inner=adkf_step(md,task)
            h=torch.autograd.functional.hessian(lambda *pp:inner(tuple(md.net.parameters()),pp),tuple(x[0] for x in fits))
            H=torch.cat([torch.cat(r,dim=1) for r in h],dim=0)
            audits.append(dict(k=k,loss=float(loss.detach()),fit=[v[2] for v in fits],hessian_eig=torch.linalg.eigvalsh(H).tolist(),grad_norm=float(torch.sqrt(sum(p.grad.square().sum() for p in md.net.parameters())))))
        # Finite difference of the same fixed-context inner objective used by IFT.
        task=episode(floors[train[0]],rng,scale,k=3,augment=True)
        loss,fits,_=adkf_step(md,task)
        directions=[p.grad.detach().clone() for p in md.net.parameters()]
        norm=torch.sqrt(sum(v.square().sum() for v in directions))
        directions=[v/norm for v in directions]
        base=[p.detach().clone() for p in md.net.parameters()]
        def value(sign,epsilon):
            with torch.no_grad():
                for p,b,v in zip(md.net.parameters(),base,directions):p.copy_(b+sign*epsilon*v)
                zs,zq=md(task[0]),md(task[2])
            total=0.
            for axis in range(2):
                gp,_,_=fit(zs,task[1][:,axis],fixed_loc=fits[axis][1])
                m,c=posterior(zs,task[1][:,axis],zq,gp,full=True)
                total=total-torch.distributions.MultivariateNormal(m,covariance_matrix=c).log_prob(task[3][:,axis])
            return float(total.detach())/(2*len(task[2]))
        fd=[]
        for epsilon in [1e-4,1e-5]:
            difference=(value(1,epsilon)-value(-1,epsilon))/(2*epsilon)
            fd.append(dict(epsilon=epsilon,analytic=float(norm),finite_difference=difference,relative_error=abs(difference-float(norm))/float(norm)))
        with torch.no_grad():
            for p,b in zip(md.net.parameters(),base):p.copy_(b)
        assert max(r['relative_error'] for r in fd)<.02,fd
        audit=dict(inner_audits=audits,hypergradient_finite_difference=fd)
        (out/'source_preflight.json').write_text(json.dumps(audit,indent=2));print(audit,flush=True);return
    devstates=train_models(floors,train,scale,a.seed,CHECKS,out,devs,valid)
    choice={}
    for key in ['dkt','adkf_ift','raw_source']:
        values={step:next(r['mde'] for r in json.loads((out/f'dev_{step}.json').read_text()) if r['method']==key) for step in CHECKS}
        choice[key]=min(values,key=values.get)
    (out/'selected.json').write_text(json.dumps(choice,indent=2))
    # Refit source representations on every allowed source floor, keeping chosen steps.
    finaldir=out/'final';finaldir.mkdir(exist_ok=True)
    states=train_models(floors,sources,scale,a.seed,sorted(set(choice.values())),finaldir)
    dk=Model(True);dk.load_state_dict(states[choice['dkt']]['dkt'])
    ak=Model(True);ak.load_state_dict(states[choice['adkf_ift']]['adkf_ift'])
    rw=Model(False);rw.load_state_dict(states[choice['raw_source']]['raw_source'])
    manifest=[];rows=[]
    for kind,md in [('raw_refit',Model(False)),('raw_source',rw),('dkt',dk),('dkt_refit',dk),('adkf_ift',ak)]:
        rows+=evaluate(md,kind,targets,floors,valid,scale,10,manifest if kind=='raw_refit' else None)
    (out/'manifest.json').write_text(json.dumps(manifest));(out/'results.json').write_text(json.dumps(rows,indent=2));print('DONE',choice,flush=True)

if __name__=='__main__':main()
