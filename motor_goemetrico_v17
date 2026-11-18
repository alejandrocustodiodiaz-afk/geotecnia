"""Motor experimental v17: intersección convexa de planos y aproximación tangente local de excavación.
No calcula liberación cinemática, persistencia finita ni FS.
"""
import numpy as np
from scipy.optimize import linprog
from scipy.spatial import HalfspaceIntersection, ConvexHull, QhullError


def normal_local(dd, dip, az=0):
    a, b = np.deg2rad([(dd-az) % 360, dip])
    return np.array([np.sin(a)*np.sin(b), np.cos(a)*np.sin(b), np.cos(b)])


def cell_halfspaces(systems, nplanes, W, H, L, az, sample):
    """Halfspaces a*x+b*y+c*z+d <= 0 containing sample."""
    halfspaces=[]
    for fam in systems:
        n=normal_local(fam['dd'],fam['dip'],az)
        base=float(n @ np.array([fam['xx'],fam['yy'],H]))
        for k in range(nplanes):
            level=base+(k-(nplanes-1)/2)*fam['spacing']
            sign=1. if n @ sample >= level else -1.
            # If sign positive, require n*x>=level => -n*x+level<=0.
            halfspaces.append(np.r_[-sign*n,sign*level])
    margin=max(2.,W*.5); r=W/2
    bounds=[(-r-margin,r+margin),(0,L),(0,H+margin)]
    for i,(lo,hi) in enumerate(bounds):
        unit=np.eye(3)[i]
        halfspaces.extend([np.r_[-unit,lo],np.r_[unit,-hi]])
    return np.array(halfspaces,float)


def local_tunnel_halfspace(contact,W,H):
    """Local tangent supporting plane to the horseshoe; not exact curved tunnel clipping."""
    r=W/2.; spring=max(0.,H-r)
    x,z=float(contact[0]),float(contact[2])
    if z<=spring:
        sign=1. if x>=0 else -1.
        return np.array([-sign,0.,0.,r]) # outside sidewall
    radial=np.array([x,z-spring]); norm=np.linalg.norm(radial)
    if norm<1.e-10:return None
    nx,nz=radial/norm
    # nx*x+nz*(z-spring)>=r
    return np.array([-nx,0.,-nz,r+nz*spring])


def convex_polyhedron(halfspaces):
    """Find strict interior with LP, then exact plane-plane vertices and convex hull."""
    A=halfspaces[:,:3];d=halfspaces[:,3]
    norms=np.linalg.norm(A,axis=1)
    res=linprog([0,0,0,-1],A_ub=np.column_stack([A,norms]),b_ub=-d,
                bounds=[(None,None)]*4,method='highs')
    if not res.success or res.x[3]<1.e-7:
        return {'status':'empty_or_degenerate','detail':res.message}
    try:
        hs=HalfspaceIntersection(halfspaces,res.x[:3],qhull_options='Qx')
        vertices=np.unique(np.round(hs.intersections,10),axis=0)
        hull=ConvexHull(vertices)
    except (QhullError,ValueError) as exc:
        return {'status':'geometry_error','detail':str(exc)}
    return {'status':'ok','vertices':vertices,'faces':hull.simplices,'volume':float(hull.volume),
            'area':float(hull.area),'interior':res.x[:3], 'clearance':float(res.x[3])}


def reconstruct_local_block(block,systems,nplanes,W,H,L,az):
    sample=np.asarray(block['center'],float)
    hs=cell_halfspaces(systems,nplanes,W,H,L,az,sample)
    tunnel=local_tunnel_halfspace(block['contact'],W,H)
    if tunnel is not None:hs=np.vstack([hs,tunnel])
    result=convex_polyhedron(hs)
    result['method']='planar_halfspaces_plus_local_tunnel_tangent'
    result['approximation']='La bóveda curva se sustituye por un plano tangente local; no es un recorte exacto.'
    return result
