"""Motor geotécnico unificado de Geotecnia Mina.

Geometría discretizada por firmas de planos y componentes conectadas.
Los volúmenes son estimaciones numéricas preliminares; no se modela
persistencia finita ni se valida liberación cinemática o factor de seguridad.
"""
import json
import numpy as np
from scipy import ndimage as ndi

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


def grid_diagnostic(block, labels, xs, ys, zs, systems, nplanes, W, H, L, az, resolution=80):
    """Auditoría por propiedad exclusiva de la etiqueta original.

    No atribuye a B el volumen de una componente reconstruida que contenga
    otros bloques. La reconstrucción independiente se conserva solo como
    comprobación de discrepancias, no como geometría ni volumen de B.
    """
    ident = int(block['id'])
    idx = np.argwhere(labels == ident)
    if not len(idx):
        return {'status': 'missing'}
    pl = planes(systems, nplanes, H, az)
    original_sig = int(block.get('signature', -1))
    # Verificación exhaustiva: no basta con 60 puntos o con el centroide.
    for bit, (_, n, d) in enumerate(pl):
        coords = np.column_stack((xs[idx[:, 0]], ys[idx[:, 1]], zs[idx[:, 2]]))
        actual = (coords @ n >= d)
        if np.any(actual != bool(original_sig & (1 << bit))):
            return {'status': 'inconsistent_original_signatures', 'plane': bit + 1}
    step0 = np.array([xs[1]-xs[0], ys[1]-ys[0], zs[1]-zs[0]], float)
    margin = np.maximum(3*step0, 0.75)
    low = np.maximum([xs[0], ys[0], zs[0]],
                     np.array([xs[idx[:,0].min()], ys[idx[:,1].min()], zs[idx[:,2].min()]])-margin)
    high = np.minimum([xs[-1], ys[-1], zs[-1]],
                      np.array([xs[idx[:,0].max()], ys[idx[:,1].max()], zs[idx[:,2].max()]])+margin)
    step = float(max(high-low)/int(resolution))
    dims = np.maximum(3, np.ceil((high-low)/step).astype(int))
    if int(np.prod(dims)) > 3000000:
        return {'status': 'too_large'}
    ax = [low[i]+(np.arange(dims[i])+.5)*step for i in range(3)]
    X,Y,Z = np.meshgrid(*ax,indexing='ij',sparse=True)
    reconstructed = np.broadcast_to(rock_mask(X,Z,W,H),tuple(dims)).copy()
    for bit,(_,n,d) in enumerate(pl):
        signed = n[0]*X+n[1]*Y+n[2]*Z-d
        reconstructed &= signed >= 0 if (original_sig & (1 << bit)) else signed < 0
    # Propietario de cada punto según la única segmentación de referencia.
    ix = np.rint((ax[0]-xs[0])/step0[0]).astype(int)
    iy = np.rint((ax[1]-ys[0])/step0[1]).astype(int)
    iz = np.rint((ax[2]-zs[0])/step0[2]).astype(int)
    valid = ((ix[:,None,None]>=0)&(ix[:,None,None]<len(xs)) &
             (iy[None,:,None]>=0)&(iy[None,:,None]<len(ys)) &
             (iz[None,None,:]>=0)&(iz[None,None,:]<len(zs)))
    owner = labels[np.ix_(np.clip(ix,0,len(xs)-1),np.clip(iy,0,len(ys)-1),np.clip(iz,0,len(zs)-1))].copy()
    owner[~valid]=0
    original_mask = owner==ident
    owned = reconstructed & original_mask
    structure = ndi.generate_binary_structure(3,1)
    components,ncomp = ndi.label(owned,structure=structure)
    records=[]
    for c in range(1,ncomp+1):
        cm=components==c
        if not cm.any(): continue
        records.append({'component':c,'volume_m3':float(cm.sum()*step**3),
                        'overlap_selected_m3':float(cm.sum()*step**3),
                        'purity_pct':100.,'other_blocks':'','mask':cm})
    records.sort(key=lambda x:-x['volume_m3'])
    if not records:return {'status':'no_components'}
    chosen=records[0]
    # La cobertura evalúa la pérdida por recorte de planos y herradura.
    coverage=100*owned.sum()/max(1,original_mask.sum())
    # Fracción de la componente independiente que pertenece a B (diagnóstico).
    independent,ni=ndi.label(reconstructed,structure=structure)
    overlap=np.bincount(independent[owned],minlength=ni+1)
    overlap[0]=0
    cid=int(np.argmax(overlap)) if ni else 0
    candidate=independent==cid if cid else np.zeros_like(owned)
    foreign_ids=np.unique(owner[candidate & (owner>0) & (owner!=ident)])
    foreign=[f'B{int(x)}' for x in foreign_ids]
    purity=100*float((candidate & owned).sum())/max(1,int(candidate.sum()))
    touches=bool(any(np.any(v) for v in (owned[0],owned[-1],owned[:,0],owned[:,-1],owned[:,:,0],owned[:,:,-1])))
    return {'status':'ok','signature':original_sig,'planes':pl,
            'component_count':len(records),'components':records,'chosen':chosen,
            'coverage_pct':coverage,'iou_pct':100*owned.sum()/max(1,(original_mask|owned).sum()),
            'original_resampled_m3':float(original_mask.sum()*step**3),
            'owned_volume_m3':float(owned.sum()*step**3),
            'selected_voxel_m3':float(block['volume']),
            'origin':low,'step':step,'grid':tuple(dims),
            'original_mask':original_mask,'intersection_mask':owned,
            'owned_mask':owned,'independent_mask':candidate,
            'foreign_blocks':foreign,'independent_purity_pct':purity,
            'window_touches':touches,
            'note':'Volumen de submuestreo condicionado a etiqueta original, no volumen exacto independiente.'}


def synthetic_tests():
    """Simple numerical invariants independent of the application scenario."""
    n=normal(0,90,0)
    assert np.allclose(n,[0,1,0],atol=1e-12)
    a=np.array([-.5,.5]);z=np.array([1.,4.])
    assert not rock_mask(a[0],z[0],5.8,5.8)
    assert rock_mask(4.,3.,5.8,5.8)
    assert rock_mask(0.,6.,5.8,5.8)
    return True

def compute_regions(W, H, L, az, res, nplanes, systems_json):
    """Motor único: celdas de semiespacios + componentes conexas 6-vecinos.

    Los IDs B son locales a la resolución. Los volúmenes se estiman mediante
    integración subvoxel 2x2x2 condicionada a la firma de cada celda. La
    persistencia es infinita; no hay liberación cinemática ni FS.
    """
    systems = json.loads(systems_json)
    r = W / 2.0
    spring = max(0.0, H-r)
    margin = max(2.0, W*0.5)
    nx = int(res)
    ny = max(30, min(105, round(res*L/(W+2*margin))))
    nz = max(35, min(85, round(res*(H+margin)/(W+2*margin))))
    xs = np.linspace(-r-margin, r+margin, nx)
    ys = np.linspace(0, L, ny)
    zs = np.linspace(0, H+margin, nz)
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing='ij', sparse=True)
    planes_list = planes(systems, nplanes, H, az)
    if len(planes_list) > 30:
        raise ValueError('Se admiten como máximo 30 planos en el motor de firmas')
    signature = np.zeros((nx, ny, nz), dtype=np.uint32)
    for bit, (_, n, d) in enumerate(planes_list):
        signature |= ((n[0]*X+n[1]*Y+n[2]*Z >= d).astype(np.uint32) << bit)
    rock = np.broadcast_to(rock_mask(X, Z, W, H), signature.shape).copy()
    structure = ndi.generate_binary_structure(3, 1)
    tunnel = ~rock
    adjacent = rock & ndi.binary_dilation(tunnel, structure=structure)
    labels = np.zeros(signature.shape, dtype=np.int16)
    blocks = []
    dx, dy, dz = (float(a[1]-a[0]) for a in (xs, ys, zs))
    cell_volume = dx*dy*dz
    ident = 0
    # Cada firma define una celda de discontinuidades. La conectividad se
    # comprueba DENTRO de esa firma, nunca mediante proximidad de centroides.
    for sig in np.unique(signature[adjacent]):
        component_labels, _ = ndi.label(rock & (signature == sig), structure)
        for component in np.unique(component_labels[adjacent]):
            if component == 0:
                continue
            reg = component_labels == component
            count = int(reg.sum())
            if count < 5:
                continue
            touching = reg & adjacent
            roof = int((touching & (Z > spring)).sum())
            left = int((touching & (Z <= spring) & (X < 0)).sum())
            right = int((touching & (Z <= spring) & (X >= 0)).sum())
            zone = ['Bóveda', 'Hastial izq.', 'Hastial der.'][int(np.argmax([roof,left,right]))]
            opened = bool(reg[0].any() or reg[-1].any() or reg[:,0].any()
                          or reg[:,-1].any() or reg[:,:,0].any() or reg[:,:,-1].any())
            ident += 1
            if ident > np.iinfo(labels.dtype).max:
                raise ValueError('Demasiadas regiones para etiquetas int16')
            labels[reg] = ident
            coords = np.argwhere(reg)
            cc = np.argwhere(touching)
            center = [float(xs[0]+coords[:,0].mean()*dx),
                      float(ys[0]+coords[:,1].mean()*dy),
                      float(zs[0]+coords[:,2].mean()*dz)]
            contact = [float(xs[0]+cc[:,0].mean()*dx),
                       float(ys[0]+cc[:,1].mean()*dy),
                       float(zs[0]+cc[:,2].mean()*dz)]
            # Submuestreo de la celda de planos en los 8 octantes de cada
            # voxel asignado; la malla de etiquetas NO se altera.
            # Se procesa por lotes para limitar memoria.
            weighted = 0.0
            for batch_start in range(0, len(coords), 120000):
                c = coords[batch_start:batch_start+120000]
                x0, y0, z0 = xs[c[:,0]], ys[c[:,1]], zs[c[:,2]]
                for sx in (-0.25, 0.25):
                    for sy in (-0.25, 0.25):
                        for sz in (-0.25, 0.25):
                            x, y, z = x0+sx*dx, y0+sy*dy, z0+sz*dz
                            keep = rock_mask(x, z, W, H)
                            for bit, (_, n, d) in enumerate(planes_list):
                                expected = bool(int(sig) & (1 << bit))
                                if expected:
                                    keep &= (n[0]*x+n[1]*y+n[2]*z >= d)
                                else:
                                    keep &= (n[0]*x+n[1]*y+n[2]*z < d)
                            weighted += int(np.count_nonzero(keep))/8.0
            estimated = weighted*cell_volume
            blocks.append(dict(id=ident, zone=zone, volume=estimated,
                               voxel_volume=count*cell_volume,
                               open=opened, center=center, contact=contact,
                               signature=int(sig), method='subvoxel_2x2x2',
                               sample_count=count))
    return xs, ys, zs, labels, blocks


def test_region_invariants():
    """Pruebas deterministas de exclusividad, conectividad y firmas."""
    import json as _json
    fam = [dict(dd=d, dip=60., xx=0., yy=y, spacing=2., persistence=5., phi=30.)
           for d, y in ((30., 6.), (150., 7.), (270., 8.))]
    xs, ys, zs, labels, blocks = compute_regions(5.8,5.8,15.,0.,36,2,_json.dumps(fam))
    assert len(blocks) > 0
    assert len({b['id'] for b in blocks}) == len(blocks)
    assert np.all(labels >= 0)
    structure = ndi.generate_binary_structure(3, 1)
    pl = planes(fam,2,5.8,0.)
    for b in blocks:
        mask = labels == b['id']
        assert mask.any()
        assert ndi.label(mask,structure)[1] == 1
        assert 0 <= b['volume'] <= b['voxel_volume'] + 1e-8
        assert b['signature'] == sum(bit << i for i, bit in enumerate(signature_at(np.array(b['center']),pl)))
    return {'blocks':len(blocks), 'connected':True, 'exclusive':True}



def _sample_grid_labels(points, axes, labels):
    """Nearest voxel-center lookup; outside-grid points remain unassigned."""
    indices = []
    valid = np.ones(len(points), dtype=bool)
    for k, axis in enumerate(axes):
        step = float(axis[1] - axis[0])
        v = (points[:, k] - axis[0]) / step
        valid &= (v >= -0.5) & (v <= len(axis) - 0.5)
        indices.append(np.clip(np.rint(v).astype(int), 0, len(axis) - 1))
    out = np.zeros(len(points), dtype=np.int32)
    out[valid] = labels[indices[0][valid], indices[1][valid], indices[2][valid]]
    return out


def _block_points(axes, labels, block_id, limit=25000):
    idx = np.argwhere(labels == int(block_id))
    if len(idx) > limit:
        idx = idx[np.linspace(0, len(idx)-1, limit, dtype=int)]
    return np.column_stack([axes[k][idx[:, k]] for k in range(3)]) if len(idx) else np.empty((0, 3))


def _match_block(source_block, source_axes, source_labels, target_axes,
                 target_labels, target_blocks, coverage_min=0.90,
                 purity_min=0.90, dominance_min=0.90):
    """Signature-aware, reciprocal and one-to-one spatial correspondence.

    A target is never accepted based solely on maximal overlap. The reverse
    association must identify the source block as its dominant source label.
    """
    source_id = int(source_block['id'])
    src = _block_points(source_axes, source_labels, source_id)
    result = dict(status='Sin correspondencia válida', block_id=None,
                  volume=None, coverage=0., purity=None, others=0,
                  reason='No se encontró una región compatible')
    if not len(src):
        return result
    mapped = _sample_grid_labels(src, target_axes, target_labels)
    candidates = {int(b['id']): b for b in target_blocks
                  if b.get('signature') == source_block.get('signature')}
    compatible = [(bid, int(np.count_nonzero(mapped == bid)))
                  for bid in candidates]
    compatible = sorted(((bid, n) for bid, n in compatible if n),
                        key=lambda item: -item[1])
    if not compatible:
        result['reason'] = 'No hay componentes con la misma firma de planos'
        return result
    bid, nmatch = compatible[0]
    coverage = nmatch / len(src)
    result['coverage'] = 100 * coverage
    result['others'] = sum(1 for other_id, count in compatible[1:]
                           if count >= max(2, int(.01 * len(src))))
    candidate_pts = _block_points(target_axes, target_labels, bid)
    back = _sample_grid_labels(candidate_pts, source_axes, source_labels)
    purity = float(np.mean(back == source_id)) if len(back) else 0.
    result['purity'] = 100 * purity
    counts = np.bincount(back, minlength=max(source_id + 1, 1))
    dominant = int(np.argmax(counts[1:]) + 1) if len(counts) > 1 and counts[1:].max() else 0
    dominant_share = (counts[dominant] / len(back)) if len(back) else 0.
    if coverage < coverage_min:
        result['reason'] = 'Cobertura espacial insuficiente'
    elif purity < purity_min:
        result['reason'] = 'La región candidata contiene mayoritariamente otros volúmenes'
    elif dominant != source_id or dominant_share < dominance_min:
        result['reason'] = 'Correspondencia inversa no exclusiva'
    elif result['others']:
        result['reason'] = 'La región original se divide en varias componentes'
    else:
        result.update(status='Correspondencia provisional', block_id=bid,
                      volume=float(candidates[bid]['volume']),
                      reason='Firma y correspondencia recíproca compatibles')
    return result


def compare_block_resolution(block, xs, ys, zs, labels, W, H, L, az,
                             nplanes, systems_json, resolutions=(48, 72)):
    """Auditoría conservadora: firma, correspondencia mutua y comparación directa.

    Los identificadores B no son persistentes entre mallas. Si un candidato
    no supera los controles, su volumen NO entra en la comparación.
    """
    base_axes = (xs, ys, zs)
    history, snapshots = [], []
    for resolution in resolutions:
        ax, ay, zz, newlabels, newblocks = compute_regions(
            W, H, L, az, int(resolution), nplanes, systems_json)
        axes = (ax, ay, zz)
        matched = _match_block(block, base_axes, labels, axes, newlabels, newblocks)
        history.append({
            'Resolución': int(resolution),
            'Correspondencia': matched['status'],
            'Bloque equivalente': ('B' + str(matched['block_id'])) if matched['block_id'] else 'Sin correspondencia válida',
            'Volumen comparable (m³)': matched['volume'],
            'Cobertura (%)': matched['coverage'],
            'Pureza inversa (%)': matched['purity'],
            'Otros bloques coincidentes': matched['others'],
            'Motivo': matched['reason']})
        snapshots.append((axes, newlabels, newblocks, matched))

    direct = None
    if len(snapshots) == 2 and all(s[3]['block_id'] is not None for s in snapshots):
        a, b = snapshots
        ba = next(item for item in a[2] if item['id'] == a[3]['block_id'])
        ab = _match_block(ba, a[0], a[1], b[0], b[1], b[2])
        bb = next(item for item in b[2] if item['id'] == b[3]['block_id'])
        reverse = _match_block(bb, b[0], b[1], a[0], a[1], a[2])
        direct = (ab['block_id'] == bb['id'] and
                  reverse['block_id'] == ba['id'] and
                  ab['status'] == 'Correspondencia provisional' and
                  reverse['status'] == 'Correspondencia provisional')
    comparable = len(history) == 2 and all(r['Volumen comparable (m³)'] is not None for r in history) and direct is True
    delta = (100 * abs(history[1]['Volumen comparable (m³)'] - history[0]['Volumen comparable (m³)']) /
             max(history[1]['Volumen comparable (m³)'], 1e-12)) if comparable else None
    if not comparable:
        # Do not display even provisional volumes when the direct comparison fails.
        for row in history:
            row['Volumen comparable (m³)'] = None
            if row['Correspondencia'] == 'Correspondencia provisional':
                row['Correspondencia'] = 'Sin correspondencia válida'
                row['Motivo'] = 'No se confirma la correspondencia directa 48 ↔ 72'
    return {'status': 'ok', 'history': history, 'volume_change_pct': delta,
            'ambiguous': not comparable, 'direct_match': direct,
            'validated': False}
