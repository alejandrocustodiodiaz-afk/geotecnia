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


def connected_curved_components(block, systems, nplanes, W, H, L, az,
                                resolutions=(48, 72, 96), connectivity=1):
    """Experimental grid-based connected components of rock within a plane cell.

    Identifies the component containing (or nearest to) the selected voxel-block
    centroid. This is a diagnostic, NOT an exact mesh or a validated wedge volume.
    """
    from scipy import ndimage as ndi
    sample = np.asarray(block['center'], dtype=float)
    hs, names = cell_halfspaces(systems, nplanes, W, H, L, az, sample)
    poly = convex_polyhedron(hs)
    if poly['status'] != 'ok':
        return poly
    low = poly['vertices'].min(axis=0)
    high = poly['vertices'].max(axis=0)
    span = high - low
    if np.any(span <= 1e-9):
        return {'status': 'degenerate_bounds'}
    results = []
    last_mask = None
    last_origin = None
    last_step = None
    last_components = []
    for resolution in resolutions:
        # Isotropic cells; the longest box dimension has 'resolution' cells.
        step = float(np.max(span) / int(resolution))
        shape = np.maximum(3, np.ceil(span / step).astype(int))
        axes = [low[i] + (np.arange(shape[i]) + .5) * step for i in range(3)]
        xx, yy, zz = np.meshgrid(*axes, indexing='ij', sparse=True)
        mask = np.ones(tuple(shape), dtype=bool)
        # Evaluate each halfspace without constructing a large (N,3) point array.
        for h in hs:
            mask &= (h[0]*xx + h[1]*yy + h[2]*zz + h[3] <= 1e-9)
        r = W / 2.
        spring = max(0., H-r)
        void = (np.abs(xx) <= r) & ((zz <= spring) | (xx*xx+(zz-spring)**2 <= r*r))
        mask &= ~void
        structure = ndi.generate_binary_structure(3, connectivity)
        labels, n_components = ndi.label(mask, structure=structure)
        counts = np.bincount(labels.ravel(), minlength=n_components+1)
        counts[0] = 0
        # Associate the selected voxel centroid to one connected component.
        # If centroid falls on a grid boundary, select nearest occupied cell,
        # but report that the association was inferred rather than exact.
        center_index = np.rint((sample - low) / step - .5).astype(int)
        center_index = np.clip(center_index, 0, shape-1)
        selected_label = int(labels[tuple(center_index)])
        association = 'centroid_inside_component'
        distance = 0.
        if selected_label == 0:
            candidates = np.argwhere(labels > 0)
            if len(candidates):
                deltas = (candidates + .5) * step + low - sample
                nearest = int(np.argmin(np.einsum('ij,ij->i', deltas, deltas)))
                cell = candidates[nearest]
                distance = float(np.linalg.norm(deltas[nearest]))
                selected_label = int(labels[tuple(cell)])
                association = 'nearest_component_inferred'
            else:
                association = 'no_rock_cells'
        component_count = int(counts[selected_label]) if selected_label else 0
        selected_volume = component_count * step**3
        total_volume = int(mask.sum()) * step**3
        components = sorted(({'label':int(i), 'volume_m3':float(counts[i]*step**3),
                              'cell_count':int(counts[i])} for i in range(1,n_components+1)),
                            key=lambda item: -item['volume_m3'])
        results.append({'Resolución':int(resolution), 'Celdas X/Y/Z': '×'.join(map(str,shape)),
                        'Componentes':int(n_components),
                        'Volumen componente (m³)':float(selected_volume),
                        'Volumen total (m³)':float(total_volume),
                        'Asociación':association,
                        'Distancia centro (m)':distance})
        last_mask = (labels == selected_label) if selected_label else np.zeros_like(mask)
        last_origin = low.copy()
        last_step = step
        last_components = components
    return {'status':'ok', 'progress':results, 'components':last_components,
            'selected_component_volume':results[-1]['Volumen componente (m³)'],
            'total_volume':results[-1]['Volumen total (m³)'],
            'association':results[-1]['Asociación'],
            'association_distance':results[-1]['Distancia centro (m)'],
            'component_mask':last_mask, 'grid_origin':last_origin,
            'grid_step':last_step, 'halfspace_names':names,
            'warning':'Discretización experimental; las componentes pueden fusionarse o desaparecer al cambiar resolución.'}
def connected_curved_components(block, systems, nplanes, W, H, L, az,
                                resolutions=(48, 72, 96), connectivity=1):
    """Experimental grid-based connected components of rock within a plane cell.

    Identifies the component containing (or nearest to) the selected voxel-block
    centroid. This is a diagnostic, NOT an exact mesh or a validated wedge volume.
    """
    from scipy import ndimage as ndi
    sample = np.asarray(block['center'], dtype=float)
    hs, names = cell_halfspaces(systems, nplanes, W, H, L, az, sample)
    poly = convex_polyhedron(hs)
    if poly['status'] != 'ok':
        return poly
    low = poly['vertices'].min(axis=0)
    high = poly['vertices'].max(axis=0)
    span = high - low
    if np.any(span <= 1e-9):
        return {'status': 'degenerate_bounds'}
    results = []
    last_mask = None
    last_origin = None
    last_step = None
    last_components = []
    for resolution in resolutions:
        # Isotropic cells; the longest box dimension has 'resolution' cells.
        step = float(np.max(span) / int(resolution))
        shape = np.maximum(3, np.ceil(span / step).astype(int))
        axes = [low[i] + (np.arange(shape[i]) + .5) * step for i in range(3)]
        xx, yy, zz = np.meshgrid(*axes, indexing='ij', sparse=True)
        mask = np.ones(tuple(shape), dtype=bool)
        # Evaluate each halfspace without constructing a large (N,3) point array.
        for h in hs:
            mask &= (h[0]*xx + h[1]*yy + h[2]*zz + h[3] <= 1e-9)
        r = W / 2.
        spring = max(0., H-r)
        void = (np.abs(xx) <= r) & ((zz <= spring) | (xx*xx+(zz-spring)**2 <= r*r))
        mask &= ~void
        structure = ndi.generate_binary_structure(3, connectivity)
        labels, n_components = ndi.label(mask, structure=structure)
        counts = np.bincount(labels.ravel(), minlength=n_components+1)
        counts[0] = 0
        # Associate the selected voxel centroid to one connected component.
        # If centroid falls on a grid boundary, select nearest occupied cell,
        # but report that the association was inferred rather than exact.
        center_index = np.rint((sample - low) / step - .5).astype(int)
        center_index = np.clip(center_index, 0, shape-1)
        selected_label = int(labels[tuple(center_index)])
        association = 'centroid_inside_component'
        distance = 0.
        if selected_label == 0:
            candidates = np.argwhere(labels > 0)
            if len(candidates):
                deltas = (candidates + .5) * step + low - sample
                nearest = int(np.argmin(np.einsum('ij,ij->i', deltas, deltas)))
                cell = candidates[nearest]
                distance = float(np.linalg.norm(deltas[nearest]))
                selected_label = int(labels[tuple(cell)])
                association = 'nearest_component_inferred'
            else:
                association = 'no_rock_cells'
        component_count = int(counts[selected_label]) if selected_label else 0
        selected_volume = component_count * step**3
        total_volume = int(mask.sum()) * step**3
        components = sorted(({'label':int(i), 'volume_m3':float(counts[i]*step**3),
                              'cell_count':int(counts[i])} for i in range(1,n_components+1)),
                            key=lambda item: -item['volume_m3'])
        results.append({'Resolución':int(resolution), 'Celdas X/Y/Z': '×'.join(map(str,shape)),
                        'Componentes':int(n_components),
                        'Volumen componente (m³)':float(selected_volume),
                        'Volumen total (m³)':float(total_volume),
                        'Asociación':association,
                        'Distancia centro (m)':distance})
        last_mask = (labels == selected_label) if selected_label else np.zeros_like(mask)
        last_origin = low.copy()
        last_step = step
        last_components = components
    return {'status':'ok', 'progress':results, 'components':last_components,
            'selected_component_volume':results[-1]['Volumen componente (m³)'],
            'total_volume':results[-1]['Volumen total (m³)'],
            'association':results[-1]['Asociación'],
            'association_distance':results[-1]['Distancia centro (m)'],
            'component_mask':last_mask, 'grid_origin':last_origin,
            'grid_step':last_step, 'halfspace_names':names,
            'warning':'Discretización experimental; las componentes pueden fusionarse o desaparecer al cambiar resolución.'}



def compare_spatial_components(block, blocks, original_labels, xs, ys, zs,
                               systems, nplanes, W, H, L, az,
                               resolutions=(48, 72, 96)):
    """Compare reconstructed connected components to original voxel labels.

    Nearest-neighbour resampling is a diagnostic, not geometric validation.
    Returns overlap table and flags ambiguous many-to-one assignments.
    """
    from scipy import ndimage as ndi
    result = connected_curved_components(block, systems, nplanes, W, H, L, az,
                                         resolutions=resolutions)
    if result.get('status') != 'ok':
        return result
    hs, _ = cell_halfspaces(systems, nplanes, W, H, L, az, block['center'])
    poly = convex_polyhedron(hs)
    low = poly['vertices'].min(axis=0)
    high = poly['vertices'].max(axis=0)
    step = float(np.max(high-low) / int(resolutions[-1]))
    shape = np.maximum(3, np.ceil((high-low)/step).astype(int))
    ax = [low[i]+(np.arange(shape[i])+.5)*step for i in range(3)]
    xx, yy, zz = np.meshgrid(*ax, indexing='ij', sparse=True)
    mask = np.ones(tuple(shape), dtype=bool)
    for h in hs:
        mask &= (h[0]*xx+h[1]*yy+h[2]*zz+h[3] <= 1e-9)
    r=W/2.; spring=max(0.,H-r)
    mask &= ~((np.abs(xx)<=r) & ((zz<=spring) | (xx*xx+(zz-spring)**2<=r*r)))
    components, n = ndi.label(mask, ndi.generate_binary_structure(3,1))
    # Original voxel labels at the reconstructed cell centres. Out-of-domain = 0.
    original_axes=[np.asarray(a) for a in (xs,ys,zs)]
    indices=[]; valid=[]
    for a,src in zip(ax,original_axes):
        di=float(src[1]-src[0]); f=(a-src[0])/di
        indices.append(np.clip(np.rint(f).astype(int),0,len(src)-1))
        valid.append((f>=-.5)&(f<=len(src)-.5))
    old=original_labels[np.ix_(*indices)].copy()
    old[~(valid[0][:,None,None]&valid[1][None,:,None]&valid[2][None,None,:])]=0
    old[~mask]=0
    relevant=np.unique(old[old>0]); rows=[]
    counts=np.bincount(components.ravel(),minlength=n+1)
    for bid in relevant:
        sub=(old==bid)
        overlaps=np.bincount(components[sub],minlength=n+1)
        overlaps[0]=0
        cid=int(np.argmax(overlaps))
        overlap=int(overlaps[cid])
        original_count=int(sub.sum())
        component_count=int(counts[cid]) if cid else 0
        rows.append({'Bloque':f'B{bid}', 'Componente':cid,
                     'Intersección (m³)':overlap*step**3,
                     'Volumen original remuestreado (m³)':original_count*step**3,
                     'Volumen componente (m³)':component_count*step**3,
                     'Cobertura del bloque (%)':100*overlap/max(original_count,1),
                     'Pureza de componente (%)':100*overlap/max(component_count,1),
                     'Celdas coincidentes':overlap})
    # Blocks outside the sampled cell are not compared; do not infer their absence.
    mapping={row['Bloque']:row['Componente'] for row in rows if row['Componente']}
    selected_key=f"B{block['id']}"
    selected_row=next((row for row in rows if row['Bloque']==selected_key),None)
    collisions=[key for key,cid in mapping.items()
                if selected_row and cid==selected_row['Componente'] and key!=selected_key]
    result.update({'overlap_table':rows,'selected_overlap':selected_row,
                   'collisions':collisions,'compared_block_count':len(relevant),
                   'mapping_validated':False})
    return result



def reconstruct_labeled_block(block, original_labels, xs, ys, zs, systems,
                              nplanes, W, H, L, az, factors=(1, 2, 3)):
    """Refine one ORIGINAL labeled region, preserving its voxel identity.

    Original label is a topological prior; plane and curved excavation boundaries
    are re-evaluated at finer sample centers. This is NOT an independent exact
    polyhedron, and cannot resolve a missed or incorrectly split original region.
    """
    import numpy as np
    from scipy import ndimage as ndi
    bid = int(block['id'])
    inds = np.argwhere(original_labels == bid)
    if not len(inds):
        return {'status':'missing_label'}
    axes0 = [np.asarray(a,float) for a in (xs,ys,zs)]
    steps0 = np.array([abs(a[1]-a[0]) for a in axes0])
    lo = np.maximum(inds.min(axis=0)-1, 0)
    hi = np.minimum(inds.max(axis=0)+1, np.array(original_labels.shape)-1)
    # Only evaluate a padded neighborhood of the selected block.
    # The original labels are the ownership mask; no other B can be included.
    local = original_labels[lo[0]:hi[0]+1,lo[1]:hi[1]+1,lo[2]:hi[2]+1]
    if local.size > 1200000:
        return {'status':'too_large','detail':'Reduce el dominio o la resolución inicial'}
    sample = np.asarray(block['center'],float)
    hs,names = cell_halfspaces(systems,nplanes,W,H,L,az,sample)
    progress=[]; last=None
    for factor in factors:
        factor=int(factor)
        if factor<1 or factor>5: raise ValueError('Factor fuera de rango')
        # Subdivide each original voxel into factor^3 cells, with the same physical extent.
        orig_index=[np.repeat(np.arange(lo[d],hi[d]+1),factor) for d in range(3)]
        fine_axes=[axes0[d][orig_index[d]] +
                   ((np.tile(np.arange(factor),hi[d]-lo[d]+1)+.5)/factor-.5)*steps0[d]
                   for d in range(3)]
        x,y,z=np.meshgrid(*fine_axes,indexing='ij',sparse=True)
        owned=(local==bid).repeat(factor,axis=0).repeat(factor,axis=1).repeat(factor,axis=2)
        # Planar constraints use the same cell containing the selected block center.
        for h in hs:
            owned &= (h[0]*x+h[1]*y+h[2]*z+h[3] <= 1.e-9)
        r=W/2.; spring=max(0.,H-r)
        inside=(np.abs(x)<=r)&((z<=spring)|((x*x+(z-spring)**2)<=r*r))
        owned &= ~inside
        n=int(owned.sum())
        vol=n*float(np.prod(steps0))/factor**3
        cc,ncomp=ndi.label(owned,ndi.generate_binary_structure(3,1))
        counts=np.bincount(cc.ravel(),minlength=ncomp+1)
        if len(counts): counts[0]=0
        largest=int(np.max(counts)) if ncomp else 0
        progress.append({'Subdivisión por eje':factor,
                         'Celdas de bloque':n,
                         'Volumen individual estimado (m³)':float(vol),
                         'Componentes internas':int(ncomp),
                         'Fracción componente mayor (%)':float(100*largest/max(n,1))})
        last={'mask':owned,'origin':np.array([a[0] for a in fine_axes]),
              'spacing':steps0/factor,'components':int(ncomp)}
    return {'status':'ok','block_id':bid,'progress':progress,
            'individual_volume':progress[-1]['Volumen individual estimado (m³)'],
            'original_volume':float(block['volume']),
            'difference_pct':100*(progress[-1]['Volumen individual estimado (m³)']/block['volume']-1) if block['volume'] else None,
            'mask':last['mask'],'origin':last['origin'],'spacing':last['spacing'],
            'halfspace_names':names,
            'validated':False,
            'method':'Subdivisión de etiquetas originales y reevaluación de límites geométricos',
            'warning':'Volumen experimental condicionado a la segmentación original; no equivale a reconstrucción exacta independiente.'}


def diagnose_independent_block(block, blocks, original_labels, xs, ys, zs,
                               systems, nplanes, W, H, L, az,
                               resolutions=(36, 54, 72)):
    """Independent plane-cell segmentation with spatial label correspondence.

    The reconstructed geometry does not use original labels as constraints.
    Original labels are sampled ONLY for comparison and visual diagnostics.
    This is still a discretized, infinite-persistence, non-kinematic model.
    """
    import numpy as np
    from scipy import ndimage as ndi
    hs, names = cell_halfspaces(systems, nplanes, W, H, L, az, block['center'])
    poly = convex_polyhedron(hs)
    if poly.get('status') != 'ok':
        return poly
    low = np.asarray(poly['vertices']).min(axis=0)
    high = np.asarray(poly['vertices']).max(axis=0)
    span = high-low
    if np.min(span) <= 1.e-10:
        return {'status': 'degenerate_bounds'}
    history=[]
    final=None
    source_axes=[np.asarray(a,float) for a in (xs,ys,zs)]
    for resolution in resolutions:
        step=float(np.max(span)/int(resolution))
        shape=np.maximum(3,np.ceil(span/step).astype(int))
        if int(np.prod(shape)) > 2500000:
            return {'status':'too_large'}
        axes=[low[d]+(np.arange(shape[d])+.5)*step for d in range(3)]
        xx,yy,zz=np.meshgrid(*axes,indexing='ij',sparse=True)
        mask=np.ones(tuple(shape),dtype=bool)
        for h in hs:
            mask &= h[0]*xx+h[1]*yy+h[2]*zz+h[3] <= 1.e-9
        r=W/2.; spring=max(0.,H-r)
        mask &= ~((np.abs(xx)<=r)&((zz<=spring)|(xx*xx+(zz-spring)**2<=r*r)))
        cc,ncomp=ndi.label(mask,ndi.generate_binary_structure(3,1))
        # Original labels are NOT used to construct the independent components.
        idx=[]; valid=[]
        for a,src in zip(axes,source_axes):
            f=(a-src[0])/(src[1]-src[0])
            idx.append(np.clip(np.rint(f).astype(int),0,len(src)-1))
            valid.append((f>=-.5)&(f<=len(src)-.5))
        original=original_labels[np.ix_(*idx)].copy()
        in_domain=(valid[0][:,None,None]&valid[1][None,:,None]&valid[2][None,None,:])
        original[~in_domain]=0
        selected_original=(original==int(block['id']))
        counts=np.bincount(cc.ravel(),minlength=ncomp+1)
        overlaps=np.bincount(cc[selected_original],minlength=ncomp+1)
        overlaps[0]=0
        chosen=int(np.argmax(overlaps)) if ncomp else 0
        matching=int(overlaps[chosen]) if chosen else 0
        candidate=(cc==chosen)&mask if chosen else np.zeros_like(mask)
        selected_inside=selected_original&mask
        original_count=int(selected_original.sum())
        candidate_count=int(counts[chosen]) if chosen else 0
        coverage=100*matching/max(original_count,1)
        purity=100*matching/max(candidate_count,1)
        other_ids=np.unique(original[candidate])
        collisions=[]
        for bid in other_ids:
            if bid<=0 or bid==block['id']:continue
            ct=int(np.count_nonzero(candidate&(original==bid)))
            if ct:collisions.append({'Bloque':f'B{int(bid)}','Solape (m³)':ct*step**3,
                                     'Porcentaje componente (%)':100*ct/max(candidate_count,1)})
        history.append({'Resolución':int(resolution),'Componentes':int(ncomp),
                        'Volumen candidato (m³)':candidate_count*step**3,
                        'Volumen B original remuestreado (m³)':original_count*step**3,
                        'Cobertura B (%)':coverage,'Pureza componente (%)':purity,
                        'Otros bloques en componente':len(collisions)})
        final={'candidate_mask':candidate,'original_mask':selected_original,
               'original_inside_mask':selected_inside,'origin':low.copy(),'step':step,
               'collisions':collisions,'component_id':chosen,
               'component_count':ncomp,'matching_cells':matching,
               'coverage_pct':coverage,'purity_pct':purity,
               'candidate_volume':candidate_count*step**3,
               'original_resampled_volume':original_count*step**3}
    vols=[row['Volumen candidato (m³)'] for row in history]
    convergence=100*abs(vols[-1]-vols[-2])/max(vols[-1],1.e-12) if len(vols)>1 else None
    ambiguous=bool(final['collisions']) or final['purity_pct']<90 or final['coverage_pct']<90
    return {'status':'ok','progress':history,'convergence_pct':convergence,
            'ambiguous':ambiguous,'halfspace_names':names,**final,
            'validated':False,'method':'Segmentación espacial sin etiquetas; comparación posterior con etiquetas originales'}
