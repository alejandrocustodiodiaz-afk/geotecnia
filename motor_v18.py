"""Motor v18: diagnóstico reproducible de celdas y conectividad.
No calcula cuñas cinemáticamente liberables ni factores de seguridad.
"""
import numpy as np
from scipy import ndimage as ndi
from scipy.spatial import ConvexHull
from scipy.optimize import linprog


def normal(dd, dip, az=0):
    a,b=np.deg2rad([(dd-az)%360,dip])
    return np.array([np.sin(a)*np.sin(b),np.cos(a)*np.sin(b),np.cos(b)])


def planes(systems, nplanes, H, az):
    out=[]
    for j,s in enumerate(systems,1):
        n=normal(s['dd'],s['dip'],az)
        base=n@np.array([s['xx'],s['yy'],H])
        for k in range(nplanes):
            level=base+(k-(nplanes-1)/2)*s['spacing']
            out.append((f'J{j}.{k+1}',n,float(level)))
    return out


def signature_at(p, plane_list):
    return tuple(int(np.dot(n,p)>=d) for _,n,d in plane_list)


def rock_mask(X,Z,W,H):
    r=W/2; spring=max(0,H-r)
    void=(np.abs(X)<=r)&((Z<=spring)|((Z>spring)&(X*X+(Z-spring)**2<=r*r)))
    return ~void


def grid_diagnostic(block, labels, xs,ys,zs, systems,nplanes,W,H,L,az, resolution=80):
    """Reconstructs the plane cell containing the selected voxel block.
    Explicitly reports when the cell contains multiple original labels.
    Uses same physical coordinates for all comparisons, no centroid-only matching.
    """
    pl=planes(systems,nplanes,H,az)
    ident=int(block['id'])
    idx=np.argwhere(labels==ident)
    if not len(idx):return {'status':'missing'}
    # The signature must be identical across a valid original region.
    samples=idx[np.linspace(0,len(idx)-1,min(len(idx),60),dtype=int)]
    sigs={signature_at(np.array([xs[i],ys[j],zs[k]]),pl) for i,j,k in samples}
    if len(sigs)!=1:
        return {'status':'inconsistent_original_signatures','signatures':len(sigs)}
    sig=next(iter(sigs))
    # Limit the integration box to the original region plus a conservative margin.
    # This is a diagnostic local window, NOT the exact complete cell.
    lo=idx.min(axis=0);hi=idx.max(axis=0)
    span0=np.array([xs[1]-xs[0],ys[1]-ys[0],zs[1]-zs[0]])
    margin=np.maximum(3*span0,0.75)
    domainlo=np.array([xs[0],ys[0],zs[0]])
    domainhi=np.array([xs[-1],ys[-1],zs[-1]])
    low=np.maximum(domainlo,np.array([xs[lo[0]],ys[lo[1]],zs[lo[2]]])-margin)
    high=np.minimum(domainhi,np.array([xs[hi[0]],ys[hi[1]],zs[hi[2]]])+margin)
    step=float(max(high-low)/int(resolution))
    dims=np.maximum(3,np.ceil((high-low)/step).astype(int))
    ax=[low[i]+(np.arange(dims[i])+.5)*step for i in range(3)]
    X,Y,Z=np.meshgrid(*ax,indexing='ij',sparse=True)
    mask=rock_mask(X,Z,W,H)
    mask=np.broadcast_to(mask,tuple(dims)).copy()
    for bit,(_,n,d) in zip(sig,pl):
        signed=n[0]*X+n[1]*Y+n[2]*Z-d
        mask &= signed>=-1e-10 if bit else signed < -1e-10
    cc,num=ndi.label(mask,structure=ndi.generate_binary_structure(3,1))
    # Compare every candidate component against all original block labels on the same grid.
    ii=np.rint((ax[0]-xs[0])/span0[0]).astype(int)
    jj=np.rint((ax[1]-ys[0])/span0[1]).astype(int)
    kk=np.rint((ax[2]-zs[0])/span0[2]).astype(int)
    validx=(ii>=0)&(ii<len(xs));validy=(jj>=0)&(jj<len(ys));validz=(kk>=0)&(kk<len(zs))
    original=np.zeros(tuple(dims),dtype=np.int32)
    original[np.ix_(validx,validy,validz)]=labels[np.ix_(np.clip(ii[validx],0,len(xs)-1),np.clip(jj[validy],0,len(ys)-1),np.clip(kk[validz],0,len(zs)-1))]
    counts=[]
    for c in range(1,num+1):
        component=cc==c
        if component.sum()<2:continue
        unique,numbers=np.unique(original[component],return_counts=True)
        matched={int(u):int(v) for u,v in zip(unique,numbers) if u>0 and v>0}
        overlap=matched.get(ident,0)
        counts.append({'component':c,'volume_m3':float(component.sum()*step**3),
                       'overlap_selected_m3':float(overlap*step**3),
                       'purity_pct':100*overlap/max(1,int(component.sum())),
                       'other_blocks':', '.join('B'+str(u) for u in matched if u!=ident),
                       'mask':component})
    if not counts:return {'status':'no_components'}
    counts.sort(key=lambda d:d['overlap_selected_m3'],reverse=True)
    chosen=counts[0]
    original_selected=original==ident
    intersection=chosen['mask']&original_selected
    union=chosen['mask']|original_selected
    coverage=100*intersection.sum()/max(1,original_selected.sum())
    iou=100*intersection.sum()/max(1,union.sum())
    return {'status':'ok','signature':sig,'planes':pl,'component_count':len(counts),
            'components':counts,'chosen':chosen,'coverage_pct':coverage,'iou_pct':iou,
            'original_resampled_m3':float(original_selected.sum()*step**3),
            'selected_voxel_m3':float(block['volume']),'origin':low,'step':step,
            'grid':tuple(dims),'original_mask':original_selected,'intersection_mask':intersection,
            'window_touches':bool(np.any(chosen['mask'][0]) or np.any(chosen['mask'][-1]) or np.any(chosen['mask'][:,0]) or np.any(chosen['mask'][:,-1]) or np.any(chosen['mask'][:,:,0]) or np.any(chosen['mask'][:,:,-1]))}


def synthetic_tests():
    """Simple numerical invariants independent of the application scenario."""
    n=normal(0,90,0)
    assert np.allclose(n,[0,1,0],atol=1e-12)
    a=np.array([-.5,.5]);z=np.array([1.,4.])
    assert not rock_mask(a[0],z[0],5.8,5.8)
    assert rock_mask(4.,3.,5.8,5.8)
    assert rock_mask(0.,6.,5.8,5.8)
    return True
