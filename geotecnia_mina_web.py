"""Geotecnia Mina Web v8 — visualizador geométrico educativo, NO software de diseño."""
import io, json, math, itertools
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from scipy.spatial import ConvexHull

st.set_page_config(page_title='Geotecnia Mina v8',layout='wide')
st.title('Geotecnia Mina · v8')
st.caption('Geometría dependiente de las discontinuidades · cortes 2D vinculados · sostenimiento superpuesto · DXF')
st.warning('PROTOTIPO GEOMÉTRICO: las cuñas candidatas se obtienen de tres planos y un techo horizontal tangente, NO de la bóveda curva. Si no hay cierre geométrico no se dibuja cuña. No calcula FS ni valida sostenimiento. NO usar para dimensionamiento.')

with st.sidebar:
    st.header('Galería')
    W=st.number_input('Ancho (m)',1.,25.,5.8,.1)
    H=st.number_input('Altura (m)',1.,25.,5.8,.1)
    L=st.number_input('Longitud (m)',2.,100.,15.,.5)
    az=st.number_input('Azimut del eje (°)',0.,359.9,0.)
    slope=st.number_input('Pendiente del eje (°)',-45.,45.,0.)
    dxf=st.file_uploader('Sección AutoCAD DXF (opcional)',type=['dxf'])
    st.caption('El DXF se muestra como referencia 2D; no se utiliza para construir cuñas.')

r=W/2; spring=H-r
arc=np.linspace(np.pi,0,100)
section=np.vstack(([-r,0],[-r,spring],np.column_stack((r*np.cos(arc),spring+r*np.sin(arc))),[r,0]))
methods=['Mohr-Coulomb (junta)','Barton-Bandis (junta)','Patton (junta)','Hoek-Brown (macizo, no junta)']
t_data,t_wedge,t_support,t_dxf=st.tabs(['Discontinuidades','Cuñas 3D y cortes 2D','Sostenimiento 2D','Datos y DXF'])
systems=[]
with t_data:
    st.caption('Abre cada familia solo cuando quieras editarla. Las orientaciones se introducen en grados geográficos.')
    for i in range(3):
        with st.expander(f'Sistema {i+1}',expanded=False):
            a,b=st.columns(2)
            with a:
                dd=st.number_input('Dirección de buzamiento (°)',0.,359.9,[30.,150.,270.][i],key=f'dd{i}')
                dp=st.number_input('Buzamiento (°)',0.,90.,60.,key=f'dp{i}')
                off=st.number_input('Paso longitudinal Y (m)',0.,L,[L*.42,L*.5,L*.58][i],key=f'off{i}')
                xo=st.number_input('Paso transversal X (m)',-W,W,0.,key=f'xo{i}')
                pers=st.number_input('Persistencia estimada (m)',.1,100.,5.,key=f'pers{i}')
                sp=st.number_input('Espaciamiento (m)',.01,50.,1.5,key=f'spc{i}')
            with b:
                method=st.selectbox('Criterio resistente',methods,key=f'm{i}')
                props={}
                if method.startswith('Mohr'):
                    props['cohesion_MPa']=st.number_input('Cohesión (MPa)',0.,20.,.05,key=f'c{i}')
                    props['friccion_deg']=st.number_input('Ángulo de fricción (°)',0.,89.,30.,key=f'phi{i}')
                elif method.startswith('Barton'):
                    props['JRC']=st.number_input('JRC',0.,25.,10.,key=f'jrc{i}')
                    props['JCS_MPa']=st.number_input('JCS (MPa)',.01,500.,30.,key=f'jcs{i}')
                    props['friccion_residual_deg']=st.number_input('Fricción residual (°)',0.,89.,25.,key=f'pr{i}')
                elif method.startswith('Patton'):
                    props['friccion_basica_deg']=st.number_input('Fricción básica (°)',0.,89.,25.,key=f'pb{i}')
                    props['asperidad_deg']=st.number_input('Inclinación de asperidades (°)',0.,60.,10.,key=f'asp{i}')
                    st.caption('Modelo de baja tensión; no se calcula rotura de asperidades.')
                else:
                    props['UCS_MPa']=st.number_input('UCS (MPa)',.1,500.,52.6,key=f'ucs{i}')
                    props['mi']=st.number_input('mi',.1,100.,20.,key=f'mi{i}')
                    props['GSI']=st.number_input('GSI',1.,100.,50.,key=f'gsi{i}')
                    props['D']=st.number_input('D',0.,1.,0.,key=f'd{i}')
                    st.info('Hoek-Brown caracteriza el macizo, no una junta de deslizamiento.')
                props['agua_MPa']=st.number_input('Presión de agua (MPa)',0.,10.,0.,key=f'water{i}')
        systems.append(dict(dipdir=dd,dip=dp,offset=off,xoff=xo,persist=pers,spacing=sp,method=method,props=props))

def normal(s):
    theta=np.deg2rad(s['dipdir']-az); dip=np.deg2rad(s['dip'])
    return np.array([np.sin(theta)*np.sin(dip),np.cos(theta)*np.sin(dip),np.cos(dip)])

def field_tetra():
    ns=[normal(s) for s in systems]
    points=[np.array([s['xoff'],s['offset'],H]) for s in systems]
    A=np.array(ns); rhs=np.array([n@p for n,p in zip(ns,points)])
    if abs(np.linalg.det(A))<1e-9:return None,'Planos degenerados: no tienen un vértice común único.'
    apex=np.linalg.solve(A,rhs)
    if apex[2]<=H+1e-7:return None,'El vértice está por debajo del plano de clave.'
    v=[apex]
    for i,j in [(0,1),(0,2),(1,2)]:
        mat=np.array([ns[i],ns[j],[0,0,1.]])
        if abs(np.linalg.det(mat))<1e-9:return None,'Una intersección es paralela al plano de clave.'
        v.append(np.linalg.solve(mat,np.array([rhs[i],rhs[j],H])))
    v=np.array(v)
    if not np.isfinite(v).all():return None,'Coordenadas no válidas.'
    if np.max(np.abs(v[:,0]))>W/2 or v[:,1].min()<0 or v[:,1].max()>L:
        return None,'El tetraedro queda fuera del ancho o longitud del tramo.'
    try:hull=ConvexHull(v)
    except Exception:return None,'Geometría degenerada.'
    if hull.volume<1e-7:return None,'Volumen degenerado.'
    edge=max(np.linalg.norm(v[i]-v[j]) for i,j in itertools.combinations(range(4),2))
    warning=None
    if edge>min(s['persist'] for s in systems):warning=f'La extensión máxima es {edge:.2f} m y supera la persistencia indicada. La geometría se muestra SOLO como candidata no verificada.'
    return (v,hull,warning),None

# Demostración cerrada explícita: tres caras de junta y plano horizontal de clave.
def demo_tetra():
    v=np.array([[0.,L*.5,H+min(2.1,H*.35)],[-min(r*.65,1.6),L*.5-1.4,H],[min(r*.65,1.6),L*.5-1.4,H],[0.,L*.5+1.5,H]])
    return v,ConvexHull(v),'Ejemplo didáctico: orientaciones y posiciones NO proceden de las tres familias introducidas.'

def slice_polygon(vertices,axis,value):
    # Intersección de un tetraedro con un plano de coordenada constante.
    pts=[]
    for i,j in itertools.combinations(range(4),2):
        a,b=vertices[i],vertices[j]; da=a[axis]-value; db=b[axis]-value
        if abs(da)<1e-9:pts.append(a)
        if abs(db)<1e-9:pts.append(b)
        if da*db<0:
            t=-da/(db-da);pts.append(a+t*(b-a))
    unique=[]
    for p in pts:
        if not any(np.linalg.norm(p-q)<1e-7 for q in unique):unique.append(p)
    if len(unique)<3:return None
    q=np.array(unique)
    dims=[k for k in range(3) if k!=axis]
    center=q[:,dims].mean(axis=0)
    order=np.argsort(np.arctan2(q[:,dims[1]]-center[1],q[:,dims[0]]-center[0]))
    return q[order]

with t_wedge:
    st.subheader('Cuñas geométricas · vista 3D y secciones')
    mode=st.radio('Origen de la geometría', ['Tres discontinuidades introducidas','Ejemplo cerrado para comprobar el visor'],horizontal=True, key='modo_cuna')
    if mode.startswith('Ejemplo'):
        v,hull,warning=demo_tetra(); error=None
        st.info('El ejemplo es un tetraedro matemáticamente cerrado, útil para comprobar el 3D y los cortes 2D. NO representa una cuña medida en la mina.')
    else:
        result,error=field_tetra()
        if result:v,hull,warning=result
        else:v=hull=warning=None
    fig=go.Figure()
    for yy in [0,L]:
        fig.add_trace(go.Scatter3d(x=section[:,0],y=np.full(len(section),yy),z=section[:,1],mode='lines',line=dict(color='#64748b',width=6),name='Galería',showlegend=yy==0))
    for k in [0,20,45,70,100]:
        fig.add_trace(go.Scatter3d(x=[section[k,0]]*2,y=[0,L],z=[section[k,1]]*2,mode='lines',line=dict(color='#cbd5e1',width=2),showlegend=False))
    if v is not None:
        fig.add_trace(go.Mesh3d(x=v[:,0],y=v[:,1],z=v[:,2],i=hull.simplices[:,0],j=hull.simplices[:,1],k=hull.simplices[:,2],color='#ea580c',opacity=.82,flatshading=True,name='Tetraedro geométrico'))
        for i,j in itertools.combinations(range(4),2):
            fig.add_trace(go.Scatter3d(x=v[[i,j],0],y=v[[i,j],1],z=v[[i,j],2],mode='lines',line=dict(color='#7c2d12',width=6),showlegend=False))
        if warning:st.warning(warning)
        gamma=st.number_input('Peso específico (kN/m³)',10.,40.,26.4,.1)
        c1,c2,c3=st.columns(3)
        c1.metric('Volumen tetraedro (m³)',f'{hull.volume:.3f}')
        c2.metric('Peso (kN)',f'{hull.volume*gamma:.1f}')
        c3.metric('FS','No calculado')
        center=v.mean(axis=0);extent=max(W,5.,np.max(np.ptp(v,axis=0))*2)
    else:
        st.info('No se ha formado una cuña tetraédrica candidata: '+str(error))
        center=np.array([0,L/2,H/2]);extent=max(W,H,6.)*2
    fig.update_layout(height=750,margin=dict(l=0,r=0,t=0,b=0),scene=dict(
        xaxis=dict(title='X (m)',range=[center[0]-extent/2,center[0]+extent/2]),
        yaxis=dict(title='Y (m)',range=[center[1]-extent/2,center[1]+extent/2]),
        zaxis=dict(title='Z (m)',range=[max(0,center[2]-extent/2),center[2]+extent/2]),
        aspectmode='cube',camera=dict(eye=dict(x=1.35,y=-1.35,z=.95)), bgcolor='rgba(0,0,0,0)'))
    st.plotly_chart(fig,use_container_width=True, config={'scrollZoom': True})
    st.subheader('Cortes 2D del mismo bloque 3D')
    a,b=st.columns(2)
    with a:
        ycut=st.slider('Corte transversal Y (m)',0.,float(L),float(L/2),.1)
        fc=go.Figure()
        fc.add_trace(go.Scatter(x=section[:,0],y=section[:,1],mode='lines',line=dict(color='#334155',width=5),name='Galería'))
        poly=slice_polygon(v,1,ycut) if v is not None else None
        if poly is not None:
            fc.add_trace(go.Scatter(x=np.r_[poly[:,0],poly[0,0]],y=np.r_[poly[:,2],poly[0,2]],fill='toself',mode='lines',line=dict(color='#c2410c',width=3),fillcolor='rgba(234,88,12,.5)',name='Sección de cuña'))
        else:fc.add_annotation(text='Sin intersección poligonal en este corte',xref='paper',yref='paper',x=.5,y=.9,showarrow=False)
        fc.update_layout(height=480,xaxis=dict(title='X (m)',scaleanchor='y',scaleratio=1),yaxis_title='Z (m)',margin=dict(l=0,r=0,t=5,b=0))
        st.plotly_chart(fc,use_container_width=True)
    with b:
        xcut=st.slider('Corte longitudinal X (m)',float(-r),float(r),0.,.1)
        fl=go.Figure()
        fl.add_hline(y=H,line_dash='dash',line_color='#64748b')
        poly=slice_polygon(v,0,xcut) if v is not None else None
        if poly is not None:
            fl.add_trace(go.Scatter(x=np.r_[poly[:,1],poly[0,1]],y=np.r_[poly[:,2],poly[0,2]],fill='toself',mode='lines',line=dict(color='#c2410c',width=3),fillcolor='rgba(234,88,12,.5)',name='Sección de cuña'))
        else:fl.add_annotation(text='Sin intersección poligonal en este corte',xref='paper',yref='paper',x=.5,y=.9,showarrow=False)
        fl.update_layout(height=480,xaxis_title='Y (m)',yaxis=dict(title='Z (m)',scaleanchor='x',scaleratio=1),margin=dict(l=0,r=0,t=5,b=0))
        st.plotly_chart(fl,use_container_width=True)
    st.caption('Los polígonos 2D son intersecciones geométricas del tetraedro mostrado, no trazas independientes. La base horizontal representa una aproximación a la clave.')

presets={
    'ST-I':(2.,2.,.03,0.,2.,False),
    'ST-II':(2.,2.,.03,0.,2.,False),
    'ST-III':(1.5,1.5,.08,0.,1.5,True),
    'ST-IV superficial':(1.5,1.5,.12,8.,1.5,True),
    'ST-IV profundo':(1.5,1.5,.15,10.,1.,True),
    'Personalizado':(1.5,1.5,.08,0.,1.5,True)}
with t_support:
    st.subheader('Sostenimiento transversal 2D con cuña superpuesta')
    st.caption('El corte de cuña utiliza el MISMO bloque calculado en la pestaña 3D. Cambia al modo de tres discontinuidades para que responda a sus orientaciones.')
    c1,c2=st.columns(2)
    with c1:
        preset=st.selectbox('Sostenimiento tipo',list(presets))
        bolt_type=st.selectbox('Pernos',['Resina','Split Set','Swellex','Sin pernos'])
    with c2:
        shot_on=st.checkbox('Gunita',value=True)
        mesh_on=st.checkbox('Malla metálica',value=presets[preset][5])
    dx,dy,shot_default,cable_default,cable_spacing,_=presets[preset]
    with st.expander('Pernos · propiedades y patrón',expanded=False):
        boltlen=st.number_input('Longitud pernos (m)',.5,12.,2.4)
        boltcap=st.number_input('Capacidad nominal (kN)',1.,1000.,324.)
        spacing=st.number_input('Separación transversal (m)',.3,5.,dx)
        longspacing=st.number_input('Separación longitudinal (m)',.3,5.,dy)
    if shot_on:
        with st.expander('Gunita · espesor y propiedades',expanded=True):
            shot_cm=st.number_input('Espesor total (cm)',0.,50.,shot_default*100,.5)
            shot_fc=st.number_input('Resistencia compresión (MPa)',1.,100.,35.)
            shot_ft=st.number_input('Resistencia tracción (MPa)',0.,20.,3.)
            shot_E=st.number_input('Módulo elástico (GPa)',1.,80.,30.)
            shot_phase=st.selectbox('Aplicación',['Una capa','Dos capas (inicial y final)'])
            if shot_phase.startswith('Dos'):
                shot_first=st.number_input('Primera capa (cm)',0.,float(shot_cm),min(5.,float(shot_cm)),.5)
                st.caption(f'Segunda capa: {shot_cm-shot_first:.1f} cm')
    else:shot_cm=0.
    cable_on=st.checkbox('Cables de anclaje',value=cable_default>0)
    if cable_on:
        with st.expander('Cables · propiedades y patrón',expanded=True):
            cablelen=st.number_input('Longitud cables (m)',1.,25.,cable_default if cable_default>0 else 8.)
            cablecap=st.number_input('Capacidad nominal cable (kN)',1.,1500.,350.)
            cable_spacing=st.number_input('Separación transversal cables (m)',.3,6.,cable_spacing)
    else:cablelen=0.
    if mesh_on:
        with st.expander('Malla · propiedades',expanded=False):
            mesh_type=st.text_input('Tipo de malla','Electrosoldada / metálica')
            mesh_overlap=st.number_input('Solape (%)',0.,50.,10.)
    # La geometría se comparte con la pestaña de cuñas; no se crea un bloque independiente.
    st.markdown('**Cuña y sostenimiento en la misma sección**')
    cut_support=st.slider('Posición del corte transversal (Y, m)',0.,float(L),float(L/2),.1,key='corte_sostenimiento')
    show_wedge=st.checkbox('Mostrar cuña en sección de sostenimiento',value=True)
    f=go.Figure()
    if show_wedge and v is not None:
        cutpoly=slice_polygon(v,1,cut_support)
        if cutpoly is not None:
            f.add_trace(go.Scatter(x=np.r_[cutpoly[:,0],cutpoly[0,0]],y=np.r_[cutpoly[:,2],cutpoly[0,2]],
                mode='lines',fill='toself',fillcolor='rgba(234,88,12,0.38)',
                line=dict(color='#c2410c',width=3),name='Cuña (corte del bloque 3D)'))
        else:
            st.info('El plano de corte seleccionado no atraviesa el bloque. Desplaza el control Y.')
    elif show_wedge:
        st.warning('No hay cuña candidata cerrada con los datos actuales. Consulta el diagnóstico de Cuñas 3D; no se dibuja una cuña ficticia.')
    f.add_trace(go.Scatter(x=section[:,0],y=section[:,1],mode='lines',line=dict(color='#334155',width=5),name='Excavación'))
    if shot_cm>0:
        # Trazo representativo: el espesor gráfico NO es una equidistancia exacta.
        ext=section.copy();ext[1:-1,0]+=np.sign(ext[1:-1,0])*shot_cm/100;ext[1:-1,1]+=shot_cm/100
        f.add_trace(go.Scatter(x=ext[:,0],y=ext[:,1],mode='lines',line=dict(color='#f97316',width=9),name=f'Gunita {shot_cm:g} cm'))
    if mesh_on:f.add_trace(go.Scatter(x=section[1:-1,0],y=section[1:-1,1],mode='lines',line=dict(color='#16a34a',width=3,dash='dot'),name='Malla'))
    # Hastiales y bóveda, excluyendo piso.
    per=np.vstack(([-r,0],[-r,spring],np.column_stack((r*np.cos(arc),spring+r*np.sin(arc))),[r,0]))
    seg=np.linalg.norm(np.diff(per,axis=0),axis=1);cum=np.r_[0,np.cumsum(seg)]
    if bolt_type!='Sin pernos':
        for k,s in enumerate(np.arange(spacing/2,cum[-1],spacing)):
            ix=np.clip(np.searchsorted(cum,s)-1,0,len(seg)-1)
            t=(s-cum[ix])/max(seg[ix],1e-9);p=per[ix]*(1-t)+per[ix+1]*t
            if abs(p[0])>=r-1e-5 and p[1]<=spring: n=np.array([np.sign(p[0]),0.])
            else:
                n=np.array([p[0],p[1]-spring]);n/=max(np.linalg.norm(n),1e-8)
            q=p+boltlen*n
            f.add_trace(go.Scatter(x=[p[0],q[0]],y=[p[1],q[1]],mode='lines+markers',line=dict(color='#2563eb',width=4),marker=dict(size=4),name=f'Pernos {bolt_type}' if k==0 else None,showlegend=k==0))
    if cablelen>0:
        for k,s in enumerate(np.arange(cable_spacing/2,math.pi*r,cable_spacing)):
            angle=math.pi-s/r;p=np.array([r*math.cos(angle),spring+r*math.sin(angle)])
            n=np.array([math.cos(angle),math.sin(angle)]);q=p+cablelen*n
            f.add_trace(go.Scatter(x=[p[0],q[0]],y=[p[1],q[1]],mode='lines',line=dict(color='#9333ea',width=5),name='Cables' if k==0 else None,showlegend=k==0))
    f.update_layout(height=650,xaxis=dict(title='X (m)',scaleanchor='y',scaleratio=1),yaxis_title='Z (m)',margin=dict(l=0,r=0,t=5,b=0),legend=dict(orientation='h'))
    st.plotly_chart(f,use_container_width=True)
    st.caption('Cuña: corte geométrico del mismo tetraedro del visor 3D (si existe). Pernos, cables, malla y gunita: disposición esquemática. No verifica interacción anclaje-cuñas, capacidad, FS ni bóveda curva. No se sostiene el piso.')

with t_dxf:
    payload={'galeria':dict(ancho=W,alto=H,longitud=L,azimut=az,pendiente=slope),'discontinuidades':systems}
    st.download_button('Descargar datos JSON',json.dumps(payload,ensure_ascii=False,indent=2),file_name='geotecnia_mina_datos.json',mime='application/json')
    if dxf is not None:
        try:
            import ezdxf
            doc=ezdxf.read(io.StringIO(dxf.getvalue().decode('utf-8-sig',errors='replace')))
            figd=go.Figure();n=0
            for ent in doc.modelspace():
                typ=ent.dxftype();pts=None
                if typ=='LINE':pts=[ent.dxf.start[:2],ent.dxf.end[:2]]
                elif typ=='LWPOLYLINE':pts=list(ent.get_points('xy'))
                elif typ=='POLYLINE':pts=[v.dxf.location[:2] for v in ent.vertices]
                elif typ in ('CIRCLE','ARC'):
                    cen=ent.dxf.center;rad=ent.dxf.radius
                    if typ=='CIRCLE':angles=np.linspace(0,2*np.pi,120)
                    else:
                        start=ent.dxf.start_angle;end=ent.dxf.end_angle
                        angles=np.deg2rad(np.linspace(start,end if end>=start else end+360,120))
                    pts=np.column_stack((cen.x+rad*np.cos(angles),cen.y+rad*np.sin(angles)))
                if pts is not None:
                    p=np.array(pts,dtype=float);figd.add_trace(go.Scatter(x=p[:,0],y=p[:,1],mode='lines',showlegend=False));n+=1
            figd.update_layout(height=550,xaxis=dict(title='X DXF',scaleanchor='y',scaleratio=1),yaxis_title='Y DXF')
            st.plotly_chart(figd,use_container_width=True)
            st.caption(f'{n} entidades representadas. Comprueba unidades y escala. El DXF no modifica el cálculo geométrico.')
        except Exception as exc:st.error(f'No se ha podido leer el DXF: {exc}')
    else:st.info('Sube un archivo DXF en el panel lateral para ver su sección.')
