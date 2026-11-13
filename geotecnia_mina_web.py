"""Geotecnia Mina Web v5 — prototipo geométrico, NO herramienta de diseño."""
import io, itertools, math, json
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from scipy.spatial import ConvexHull

st.set_page_config(page_title='Geotecnia Mina v6', layout='wide')
st.title('Geotecnia Mina · v6')
st.caption('Discontinuidades · cuñas geométricas aproximadas · sostenimiento 2D')
st.warning('PROTOTIPO DE INVESTIGACIÓN. Las cuñas se calculan contra un PLANO TANGENTE HORIZONTAL en la clave, NO contra la bóveda curva real. No se calcula FS 3D ni se dimensiona sostenimiento. No utilizar para decisiones de seguridad.')

with st.sidebar:
    st.header('Galería')
    W=st.number_input('Ancho (m)',1.,25.,5.8,0.1)
    H=st.number_input('Altura (m)',1.,25.,5.8,0.1)
    L=st.number_input('Longitud (m)',2.,100.,15.,0.5)
    az=st.number_input('Azimut eje (°)',0.,359.9,0.)
    slope=st.number_input('Pendiente eje (°)',-45.,45.,0.)
    dxf=st.file_uploader('Sección DXF de AutoCAD (opcional)',type=['dxf'])
    st.caption('El DXF se visualiza en 2D, pero NO interviene en el cálculo de cuñas. El modelo de cuñas usa la clave de la sección estándar.')

# Sección de herradura: dos hastiales rectos y bóveda semicircular.
r=W/2; spring=H-r
ang=np.linspace(0,np.pi,90)
section=np.vstack([[-r,0],[-r,spring], *[[r*math.cos(a),spring+r*math.sin(a)] for a in ang[::-1]], [r,0]])
# Correct arc from left to right: theta pi -> 0
section=np.vstack([[-r,0],[-r,spring], np.column_stack((r*np.cos(np.linspace(np.pi,0,90)),spring+r*np.sin(np.linspace(np.pi,0,90)))), [r,0]])

systems=[]
methods=['Mohr-Coulomb (junta)','Barton-Bandis (junta)','Patton (junta)','Hoek-Brown (macizo; no junta)']
tab_data,tab_wedge,tab_support,tab_info=st.tabs(['Discontinuidades','Cuñas 3D + cortes 2D','Sostenimiento 2D','Datos / DXF'])
with tab_data:
    st.write('Abre solo el sistema que quieras editar. Cada plano se define por su orientación y por un punto de paso en coordenadas locales de la galería.')
    for i in range(3):
        with st.expander(f'Sistema {i+1}',expanded=(i==0)):
            a,b=st.columns(2)
            with a:
                dipdir=st.number_input('Dirección de buzamiento (°)',0.,359.9,[30.,150.,270.][i],key=f'dd{i}')
                dip=st.number_input('Buzamiento (°)',0.,90.,60.,key=f'dp{i}')
                offset=st.number_input('Posición longitudinal del plano (m)',0.,L,[L*.42,L*.5,L*.58][i],key=f'off{i}')
                xoff=st.number_input('Desplazamiento transversal del plano (m)',-W,W,0.,key=f'xo{i}')
                persist=st.number_input('Persistencia estimada (m)',0.1,100.,5.,key=f'pers{i}')
                spacing=st.number_input('Espaciado (m)',0.01,50.,1.5,key=f'spc{i}')
            with b:
                method=st.selectbox('Criterio resistente',methods,key=f'met{i}')
                props={}
                if method.startswith('Mohr'):
                    props['c_MPa']=st.number_input('Cohesión de junta (MPa)',0.,20.,0.05,key=f'c{i}')
                    props['phi_deg']=st.number_input('Fricción de junta (°)',0.,89.,30.,key=f'phi{i}')
                elif method.startswith('Barton'):
                    props['JRC']=st.number_input('JRC',0.,25.,10.,key=f'jrc{i}')
                    props['JCS_MPa']=st.number_input('JCS (MPa)',0.01,500.,30.,key=f'jcs{i}')
                    props['phi_r_deg']=st.number_input('Fricción residual (°)',0.,89.,25.,key=f'pr{i}')
                elif method.startswith('Patton'):
                    props['phi_b_deg']=st.number_input('Fricción básica (°)',0.,89.,25.,key=f'pb{i}')
                    props['i_deg']=st.number_input('Inclinación de asperidades (°)',0.,60.,10.,key=f'asp{i}')
                    st.caption('Rama de baja tensión únicamente; no se modela la rotura de asperidades.')
                else:
                    props['UCS_MPa']=st.number_input('UCS intacta (MPa)',0.1,500.,52.6,key=f'ucs{i}')
                    props['mi']=st.number_input('mi',0.1,100.,20.,key=f'mi{i}')
                    props['GSI']=st.number_input('GSI',1.,100.,50.,key=f'gsi{i}')
                    props['D']=st.number_input('Factor D',0.,1.,0.,key=f'd{i}')
                    st.info('Hoek-Brown describe el macizo rocoso. No se aplica como resistencia de una superficie de junta.')
                props['water_MPa']=st.number_input('Presión de agua de junta (MPa)',0.,10.,0.,key=f'water{i}')
        systems.append(dict(dipdir=dipdir,dip=dip,offset=offset,xoff=xoff,persist=persist,spacing=spacing,method=method,props=props))

# Local coordinate system: X transverse right, Y tunnel axis, Z up.
# Dip directions measured clockwise from north, transformed relative to tunnel azimuth.
def plane_normal(s):
    theta=np.deg2rad(s['dipdir']-az)
    d=np.deg2rad(s['dip'])
    # horizontal down-dip direction: (sin(theta), cos(theta))
    return np.array([np.sin(theta)*np.sin(d), np.cos(theta)*np.sin(d), np.cos(d)])

def find_tetra():
    # Exact intersection of 3 discontinuity planes with z=H, within a local tangent-plane approximation.
    ns=[plane_normal(s) for s in systems]
    ps=[np.array([s['xoff'],s['offset'],H]) for s in systems]
    A=np.stack(ns)
    if abs(np.linalg.det(A))<1e-8:return None,'Los tres planos no forman un vértice único (orientaciones degeneradas).'
    rhs=np.array([np.dot(n,p) for n,p in zip(ns,ps)])
    apex=np.linalg.solve(A,rhs)
    if apex[2]<=H+1e-5:return None,'El vértice común está por debajo de la clave; no se forma una cuña de techo en esta aproximación.'
    # Each pair of discontinuities intersects the roof tangent z=H.
    verts=[apex]
    for i,j in [(0,1),(0,2),(1,2)]:
        mat=np.array([ns[i],ns[j],[0,0,1]])
        if abs(np.linalg.det(mat))<1e-8:return None,'Una intersección es paralela al plano tangente de la clave.'
        q=np.linalg.solve(mat,np.array([rhs[i],rhs[j],H]))
        verts.append(q)
    verts=np.array(verts)
    if np.max(np.abs(verts[:,0]))>W/2 or np.min(verts[:,1])<0 or np.max(verts[:,1])>L:
        return None,'La cuña candidata excede el ancho o longitud del tramo. Ajusta posiciones de planos o dimensiones.'
    # The tetrahedron has 3 joint faces and one roof face. Test finite persistence conservatively.
    max_edge=max(np.linalg.norm(verts[i]-verts[j]) for i,j in itertools.combinations(range(4),2))
    if max_edge>min(s['persist'] for s in systems):
        return None,f'La cuña candidata requiere una extensión ~{max_edge:.2f} m, mayor que la persistencia introducida. Auméntala solo si está justificado.'
    hull=ConvexHull(verts)
    if hull.volume<1e-7:return None,'Volumen degenerado.'
    return (verts,hull),None

with tab_wedge:
    st.subheader('Cuña de techo — aproximación de cuatro planos')
    st.caption('Tres planos de discontinuidad + plano horizontal tangente en clave. La posición de cada plano se define en Discontinuidades. No se han modelado bloques en hastiales ni la intersección exacta con la bóveda curva.')
    result,error=find_tetra()
    fig=go.Figure()
    # Section at each end
    for yy in (0,L):
        fig.add_trace(go.Scatter3d(x=section[:,0],y=np.full(len(section),yy),z=section[:,1],mode='lines',line=dict(color='#64748b',width=7),name='Contorno galería',showlegend=(yy==0)))
    for x,z in section:
        pass
    for k in (0,len(section)//4,len(section)//2,3*len(section)//4,len(section)-1):
        fig.add_trace(go.Scatter3d(x=[section[k,0]]*2,y=[0,L],z=[section[k,1]]*2,mode='lines',line=dict(color='#cbd5e1',width=2),showlegend=False))
    if result:
        verts,hull=result
        # 4 triangular faces (3 joints, 1 roof tangent)
        fig.add_trace(go.Mesh3d(x=verts[:,0],y=verts[:,1],z=verts[:,2],i=hull.simplices[:,0],j=hull.simplices[:,1],k=hull.simplices[:,2],color='#e8792e',opacity=.78,flatshading=True,name='Cuña cerrada (aprox.)',showlegend=True))
        for i,j in itertools.combinations(range(4),2):
            fig.add_trace(go.Scatter3d(x=verts[[i,j],0],y=verts[[i,j],1],z=verts[[i,j],2],mode='lines',line=dict(color='#7c2d12',width=5),showlegend=False))
        st.success('Se ha construido un tetraedro cerrado mediante intersecciones geométricas. Es una cuña aproximada de techo, no una comprobación de estabilidad.')
        density=st.number_input('Peso específico de la roca (kN/m³)',10.,40.,26.4,0.1)
        c1,c2,c3=st.columns(3)
        c1.metric('Volumen geométrico (m³)',f'{hull.volume:.3f}')
        c2.metric('Peso propio (kN)',f'{hull.volume*density:.1f}')
        c3.metric('FS 3D','No calculado')
    else:
        st.info('No se ha identificado una cuña cerrada de techo con los datos actuales: '+error)
    # Show the geometry at a useful size, with a short local segment around the candidate.
    if result:
        center=np.mean(result[0],axis=0)
        ext=max(4.,float(np.max(np.ptp(result[0],axis=0)))*1.8)
    else:
        center=np.array([0.,L/2,H/2]); ext=max(W,H,5.)*1.5
    fig.update_layout(height=740,margin=dict(l=0,r=0,t=10,b=0),scene=dict(
        xaxis=dict(title='X transversal (m)',range=[center[0]-ext/2,center[0]+ext/2]),
        yaxis=dict(title='Y eje (m)',range=[center[1]-ext/2,center[1]+ext/2]),
        zaxis=dict(title='Z altura (m)',range=[center[2]-ext/2,center[2]+ext/2]),
        aspectmode='cube',camera=dict(eye=dict(x=1.5,y=-1.8,z=1.35))),legend=dict(orientation='h'))
    st.plotly_chart(fig,use_container_width=True)
    st.subheader('Cortes 2D de las discontinuidades')
    st.caption('Trazas matemáticas de planos infinitos. No equivalen a polígonos cerrados ni prueban que exista una cuña desprendible.')
    section_y=st.slider('Posición del corte transversal Y (m)',0.,float(L),float(L/2),step=0.1)
    fig_cross=go.Figure()
    fig_cross.add_trace(go.Scatter(x=section[:,0],y=section[:,1],mode='lines',line=dict(color='#334155',width=5),name='Galería'))
    xx=np.linspace(-W*1.2,W*1.2,160)
    for idx,sys in enumerate(systems):
        n=plane_normal(sys); p=np.array([sys['xoff'],sys['offset'],H]); rhs=float(n@p)
        if abs(n[2])>1e-7:
            zz=(rhs-n[0]*xx-n[1]*section_y)/n[2]
            fig_cross.add_trace(go.Scatter(x=xx,y=zz,mode='lines',name=f'Sistema {idx+1}',line=dict(width=3,dash='dash')))
        else:
            xconst=(rhs-n[1]*section_y)/n[0] if abs(n[0])>1e-7 else None
            if xconst is not None:fig_cross.add_vline(x=xconst,line_dash='dash')
    fig_cross.update_layout(height=440,xaxis=dict(title='X (m)',range=[-W*1.2,W*1.2],scaleanchor='y',scaleratio=1),yaxis=dict(title='Z (m)',range=[-H*.4,H*1.7]),margin=dict(l=0,r=0,t=10,b=0))
    st.plotly_chart(fig_cross,use_container_width=True)
    fig_long=go.Figure()
    yy=np.linspace(0,L,200)
    for idx,sys in enumerate(systems):
        n=plane_normal(sys);p=np.array([sys['xoff'],sys['offset'],H]);rhs=float(n@p)
        if abs(n[2])>1e-7:
            zz=(rhs-n[1]*yy)/n[2]
            fig_long.add_trace(go.Scatter(x=yy,y=zz,mode='lines',name=f'Sistema {idx+1}',line=dict(dash='dash')))
    fig_long.add_hline(y=H,line_color='#64748b',annotation_text='Plano tangente en clave')
    fig_long.update_layout(height=330,xaxis_title='Y longitudinal (m)',yaxis=dict(title='Z (m)',range=[-H*.4,H*2]),margin=dict(l=0,r=0,t=5,b=0))
    st.plotly_chart(fig_long,use_container_width=True)

supports={
'ST-I':(2.0,2.0,0.03,0,0),
'ST-II':(2.0,2.0,0.03,0,0),
'ST-III':(1.5,1.5,0.08,0,0),
'ST-IV superficial':(1.5,1.5,0.12,8,1.5),
'ST-IV profundo':(1.5,1.5,0.15,10,1.0),
'Personalizado':(1.5,1.5,0.08,0,0)}
with tab_support:
    st.subheader('Sección transversal 2D · disposición esquemática')
    preset=st.selectbox('Tipo de sostenimiento',list(supports))
    bolt_type=st.selectbox('Tipo de perno', ['Resina (barra corrugada)','Split Set (fricción)','Swellex (expansivo)','Sin pernos'])
    dx,dy,shot,cable,cs=supports[preset]
    with st.expander('Modificar elementos del sostenimiento',expanded=(preset=='Personalizado')):
        boltlen=st.number_input('Longitud pernos (m)',0.5,10.,2.4)
        boltcap=st.number_input('Capacidad nominal por perno (kN)',1.,1000.,324.)
        spacing=st.number_input('Espaciado transversal pernos (m)',0.3,5.,dx)
        shot=st.number_input('Espesor gunita (m)',0.,0.5,shot,0.01)
        cable=st.number_input('Longitud cables (m; 0 = ninguno)',0.,20.,float(cable))
        mesh=st.checkbox('Mostrar malla',value=preset in ('ST-III','ST-IV superficial','ST-IV profundo'))
    fig2=go.Figure()
    fig2.add_trace(go.Scatter(x=section[:,0],y=section[:,1],mode='lines',line=dict(color='#334155',width=5),name='Excavación'))
    # approximate shotcrete outward contour (not offset geometry exact)
    if shot>0:
        outward=section.copy(); outward[1:-1,0]+=np.sign(outward[1:-1,0])*shot; outward[1:-1,1]+=shot
        fig2.add_trace(go.Scatter(x=outward[:,0],y=outward[:,1],mode='lines',line=dict(color='#f97316',width=8),name=f'Gunita {shot*100:.0f} cm'))
    if mesh:
        fig2.add_trace(go.Scatter(x=section[1:-1,0],y=section[1:-1,1],mode='lines',line=dict(color='#16a34a',width=2,dash='dot'),name='Malla (esquemática)'))
    perimeter=np.vstack([[-r,spring],np.column_stack((r*np.cos(np.linspace(np.pi,0,120)),spring+r*np.sin(np.linspace(np.pi,0,120)))),[r,spring]])
    seg=np.linalg.norm(np.diff(perimeter,axis=0),axis=1); cum=np.r_[0,np.cumsum(seg)]
    positions=np.arange(spacing/2,cum[-1],spacing) if bolt_type!='Sin pernos' else []
    for k,s in enumerate(positions):
        idx=np.searchsorted(cum,s)-1; idx=max(0,min(idx,len(seg)-1)); t=(s-cum[idx])/max(seg[idx],1e-9)
        p=perimeter[idx]*(1-t)+perimeter[idx+1]*t
        normal=np.array([p[0],p[1]-spring]); normal=normal/max(np.linalg.norm(normal),1e-8)
        q=p+boltlen*normal
        fig2.add_trace(go.Scatter(x=[p[0],q[0]],y=[p[1],q[1]],mode='lines+markers',line=dict(color='#2563eb',width=4),marker=dict(size=4),name=f'{bolt_type} · {boltlen:g} m' if k==0 else None,showlegend=(k==0)))
    if cable>0:
        for k,angle in enumerate((math.pi*.30,math.pi*.5,math.pi*.70)):
            p=np.array([r*math.cos(angle),spring+r*math.sin(angle)])
            n=np.array([math.cos(angle),math.sin(angle)])
            q=p+cable*n
            fig2.add_trace(go.Scatter(x=[p[0],q[0]],y=[p[1],q[1]],mode='lines',line=dict(color='#a855f7',width=5),name='Cables (esquemáticos)' if k==0 else None,showlegend=k==0))
    fig2.update_layout(height=700,xaxis=dict(title='Ancho (m)',scaleanchor='y',scaleratio=1),yaxis=dict(title='Altura (m)'),margin=dict(l=10,r=10,t=10,b=10),legend=dict(orientation='h'))
    st.plotly_chart(fig2,use_container_width=True)
    st.caption('Representación esquemática de disposición transversal; no comprueba anclajes, capacidades, espaciamiento longitudinal ni FS. El piso no lleva sostenimiento.')
    st.info(f'Pernos: {bolt_type}. Capacidad nominal introducida: {boltcap:g} kN. Esta capacidad NO es la fuerza movilizable frente a una cuña.')

with tab_info:
    st.subheader('Resumen de parámetros y DXF')
    st.download_button('Descargar parámetros JSON',data=json.dumps({'galeria':dict(ancho=W,alto=H,longitud=L,azimut=az,pendiente=slope),'sistemas':systems},ensure_ascii=False,indent=2),file_name='geotecnia_parametros.json',mime='application/json')
    if dxf is not None:
        try:
            import ezdxf
            doc=ezdxf.read(io.StringIO(dxf.getvalue().decode('utf-8-sig',errors='replace')))
            msp=doc.modelspace(); f=go.Figure(); n=0
            for ent in msp:
                typ=ent.dxftype()
                pts=None
                if typ=='LINE':pts=[ent.dxf.start[:2],ent.dxf.end[:2]]
                elif typ=='LWPOLYLINE':pts=list(ent.get_points('xy'))
                elif typ=='CIRCLE':
                    t=np.linspace(0,2*np.pi,100); cen=ent.dxf.center; rad=ent.dxf.radius;pts=np.column_stack((cen.x+rad*np.cos(t),cen.y+rad*np.sin(t)))
                elif typ=='ARC':
                    t=np.deg2rad(np.linspace(ent.dxf.start_angle,ent.dxf.end_angle if ent.dxf.end_angle>=ent.dxf.start_angle else ent.dxf.end_angle+360,70));cen=ent.dxf.center;rad=ent.dxf.radius;pts=np.column_stack((cen.x+rad*np.cos(t),cen.y+rad*np.sin(t)))
                if pts is not None:
                    p=np.array(pts,dtype=float);f.add_trace(go.Scatter(x=p[:,0],y=p[:,1],mode='lines',showlegend=False));n+=1
            f.update_layout(height=500,xaxis=dict(scaleanchor='y',title='X (unidades DXF)'),yaxis=dict(title='Y (unidades DXF)'))
            st.plotly_chart(f,use_container_width=True)
            st.caption(f'{n} entidades dibujadas. Verifica unidades, cierre y escala antes de usar. Este DXF no alimenta el motor de cuñas.')
        except Exception as exc:st.error(f'No se pudo interpretar el DXF: {exc}')
    st.info('Próximo desarrollo: intersección exacta con contorno curvo, bloques de hastiales, análisis cinemático y FS 3D contrastado con casos de referencia. Hasta entonces los resultados son exclusivamente geométricos preliminares.')
