import argparse,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def savefig(fig,out):
    fig.tight_layout()
    fig.savefig(out/'preview.png',dpi=160,bbox_inches='tight')
    plt.close(fig)

def register(reference,moving):
    a,b=np.asarray(reference,float),np.asarray(moving,float)
    if a.ndim!=2 or a.shape!=b.shape or min(a.shape)<8 or not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)) or np.std(a)<1e-12 or np.std(b)<1e-12:raise ValueError('Need matching nonconstant finite 2D images.')
    cross=np.fft.fft2(a-a.mean())*np.conj(np.fft.fft2(b-b.mean()));cross/=np.maximum(abs(cross),1e-12)
    corr=np.fft.ifft2(cross).real;idx=np.array(np.unravel_index(np.argmax(corr),corr.shape));shape=np.array(a.shape);shift=np.where(idx>shape//2,idx-shape,idx)
    return tuple(int(v) for v in shift),np.roll(b,shift,axis=(0,1))
def phantom(n=128):
    y,x=np.mgrid[:n,:n];a=np.zeros((n,n))
    for cy,cx,s,amp in [(30,35,4,1),(82,73,9,.7),(47,96,5,.9),(100,22,6,.5)]:a+=amp*np.exp(-((x-cx)**2+(y-cy)**2)/(2*s*s))
    return a
def example(out,seed):
    rng=np.random.default_rng(seed);a=phantom();b=np.roll(a,(7,-11),(0,1))+rng.normal(0,.002,a.shape);shift,aligned=register(a,b)
    fig,ax=plt.subplots(1,4,figsize=(12,3.5))
    for axis,img,title in zip(ax,[a,b,aligned,abs(aligned-a)],['Reference','Shifted + noise','Aligned','Absolute residual']):
        im=axis.imshow(img,cmap='magma');axis.set_title(title);axis.set_axis_off();fig.colorbar(im,ax=axis,fraction=.046,pad=.04)
    savefig(fig,out);np.savez(out/'images.npz',reference=a,moving=b,aligned=aligned)
    return {'applied_shift_yx':[7,-11],'estimated_correction_yx':list(shift),'rmse_before':float(np.sqrt(np.mean((a-b)**2))),'rmse_after':float(np.sqrt(np.mean((a-aligned)**2)))}

def run(out='results',seed=42):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    metrics=example(out,seed);metrics.update(seed=seed,data='synthetic')
    (out/'metrics.json').write_text(json.dumps(metrics,indent=2));return metrics
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',default='results');p.add_argument('--seed',type=int,default=42);a=p.parse_args();print(json.dumps(run(a.out,a.seed),indent=2))
