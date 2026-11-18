"""Motor v17.1 experimental: celdas de discontinuidades + integración de bóveda curva.
La superficie de excavación se evalúa con su ecuación curva exacta; el volumen se
estima numéricamente mediante muestreo cuasi-Monte Carlo, NO es volumen exacto.
No modela persistencia finita, liberación cinemática ni factor de seguridad.
"""
import numpy as np
from scipy.optimize import linprog
from scipy.spatial import HalfspaceIntersection, ConvexHull, QhullError
from scipy.stats import qmc


def normal_local(dd, dip, az=0):
    a, b = np.deg2rad([(dd-az) % 360, dip])
    return np.array([np.sin(a)*np.sin(b), np.cos(a)*np.sin(b), np.cos(b)])


def cell_halfspaces(systems, nplanes, W, H, L, az, sample):
    """Semiespacios de la celda que contiene sample; límites de dominio incluidos."""
    sample = np.asarray(sample, float)
    halfspaces, names = [], []
    for j, fam in enumerate(systems, 1):
        n = normal_local(fam['dd'], fam['dip'], az)
        base = float(n @ np.array([fam['xx'], fam['yy'], H]))
        levels = [base+(k-(nplanes-1)/2)*fam['spacing'] for k in range(nplanes)]
        # Solo las dos discontinuidades adyacentes pueden delimitar esta celda.
        # Evita planos redundantes y permite atribuir correctamente las caras.
        proj = float(n @ sample)
        below = [(k, lev) for k, lev in enumerate(levels) if lev <= proj]
        above = [(k, lev) for k, lev in enumerate(levels) if lev > proj]
        if below:
            k, lev = max(below, key=lambda t: t[1])
            halfspaces.append(np.r_[-n, lev]); names.append(f'J{j} plano {k+1} (inferior)')
        if above:
            k, lev = min(above, key=lambda t: t[1])
            halfspaces.append(np.r_[n, -lev]); names.append(f'J{j} plano {k+1} (superior)')
    r = W/2; margin = max(2., W*.5)
    for i, (lo, hi), name in zip(range(3), [(-r-margin,r+margin),(0,L),(0,H+margin)], ['X','Y','Z']):
        unit = np.eye(3)[i]
        halfspaces.extend([np.r_[-unit,lo], np.r_[unit,-hi]])
        names.extend([f'Dominio {name} mínimo', f'Dominio {name} máximo'])
    return np.asarray(halfspaces, float), names


def convex_polyhedron(halfspaces):
    A = halfspaces[:,:3]; d = halfspaces[:,3]
    norms = np.linalg.norm(A, axis=1)
    res = linprog([0,0,0,-1], A_ub=np.column_stack([A,norms]), b_ub=-d,
                  bounds=[(None,None)]*4, method='highs')
    if not res.success or res.x[3] < 1.e-8:
        return {'status':'empty_or_degenerate','detail':res.message}
    try:
        hs = HalfspaceIntersection(halfspaces, res.x[:3], qhull_options='Qx')
        vertices = np.unique(np.round(hs.intersections, 10), axis=0)
        hull = ConvexHull(vertices)
    except (QhullError, ValueError) as exc:
        return {'status':'geometry_error','detail':str(exc)}
    return {'status':'ok','vertices':vertices,'faces':hull.simplices,
            'volume':float(hull.volume),'area':float(hull.area),
            'interior':res.x[:3], 'clearance':float(res.x[3])}


def outside_horseshoe(points, W, H):
    """Puntos de roca, fuera del hueco de herradura extruido (solera incluida)."""
    x, z = points[:,0], points[:,2]
    r = W/2.; spring = max(0., H-r)
    in_void = (np.abs(x) <= r) & ((z <= spring) | (x*x+(z-spring)**2 <= r*r))
    return ~in_void


def _tetra_sampling(vertices, faces, interior, n, seed):
    """Muestreo uniforme de un poliedro convexo, triangulado desde un punto interior."""
    tri = vertices[faces]
    edges = tri[:,1:,:] - tri[:,:1,:]
    cross = np.cross(edges[:,0,:], edges[:,1,:])
    vol = np.abs(np.einsum('ij,ij->i',cross,tri[:,0,:]-interior))/6.
    good = vol > 1.e-13
    tri, vol = tri[good], vol[good]
    if len(vol) == 0: raise ValueError('No hay tetraedros de volumen positivo')
    cum = np.cumsum(vol)/vol.sum()
    m = int(np.log2(n))
    u = qmc.Sobol(d=5, scramble=True, seed=seed).random_base2(m)
    ids = np.searchsorted(cum, u[:,0], side='right')
    ids = np.minimum(ids, len(tri)-1)
    weights = -np.log(np.clip(u[:,1:], 1.e-14, 1.))
    weights /= weights.sum(axis=1,keepdims=True)
    pts = (weights[:,0,None]*interior[None,:]
           + np.einsum('ni,nij->nj', weights[:,1:], tri[ids]))
    return pts


def _face_diagnostics(vertices, faces, halfspaces, names, tolerance=1.e-5):
    face_info=[]
    for i,(h,name) in enumerate(zip(halfspaces,names)):
        dist = np.abs(vertices @ h[:3]+h[3])/max(np.linalg.norm(h[:3]),1.e-12)
        count = int(np.count_nonzero(dist < tolerance))
        if count >= 3:
            face_info.append({'Plano delimitante':name,'Vértices sobre plano':count,
                              'Tipo':'Límite artificial' if name.startswith('Dominio') else 'Discontinuidad'})
    return face_info


def reconstruct_curved_block(block, systems, nplanes, W, H, L, az,
                             powers=(12,14,16), repeats=4):
    """Volumen de la celda convexa FUERA de la herradura curva, estimado con Sobol.
    Una celda puede contener varias componentes: no garantiza correspondencia con
    una componente voxelizada, ni continuidad o liberación cinemática.
    """
    sample=np.asarray(block['center'],float)
    hs,names=cell_halfspaces(systems,nplanes,W,H,L,az,sample)
    poly=convex_polyhedron(hs)
    if poly['status'] != 'ok': return poly
    vertices,faces=poly['vertices'],poly['faces']
    estimates=[]; progress=[]
    for power in powers:
        vals=[]
        for rep in range(repeats):
            pts=_tetra_sampling(vertices,faces,poly['interior'],2**power,seed=7919+power*41+rep)
            vals.append(float(poly['volume']*outside_horseshoe(pts,W,H).mean()))
        progress.append({'Muestras por réplica':2**power,'Volumen medio (m³)':float(np.mean(vals)),
                         'Desviación entre réplicas (m³)':float(np.std(vals,ddof=1)) if repeats>1 else np.nan})
        estimates=vals
    volume=float(np.mean(estimates))
    sd=float(np.std(estimates,ddof=1)) if repeats>1 else float('nan')
    return {**poly, 'status':'ok', 'volume_convex_cell':poly['volume'],
            'volume_curved_estimate':volume,'replicate_sd':sd,
            'relative_difference':float(volume/block['volume']-1) if block['volume'] else None,
            'progress':progress,'faces_diagnostic':_face_diagnostics(vertices,faces,hs,names),
            'note':'Volumen numérico de TODA la porción exterior de la celda; no necesariamente la misma componente voxelizada.'}
