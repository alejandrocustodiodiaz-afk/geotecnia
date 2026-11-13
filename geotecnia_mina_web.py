
import io
import json
import math
import numpy as np
import plotly.graph_objects as go
import streamlit as st
from scipy import ndimage as ndi
from skimage.measure import marching_cubes

st.set_page_config(page_title='Geotecnia Mina v13', layout='wide')
st.title('Geotecnia Mina · v13')
st.caption('Bloques candidatos de bóveda y hastiales | selección individual | cortes vinculados')
st.warning('HERRAMIENTA PRELIMINAR. Los planos se consideran infinitos y la persistencia no limita su extensión. Un bloque cerrado no equivale a una cuña liberable. Sin cálculo de FS ni dimensionamiento válido.')

with st.sidebar:
    st.header('Galería')
    W=st.number_input('Ancho (m)',1.0,25.0,5.8,0.1)
    H=st.number_input('Altura (m)',1.0,25.0,5.8,0.1)
    L=st.number_input('Longitud (m)',3.0,60.0,15.0,0.5)
    az=st.number_input('Azimut del eje (°)',0.0,359.9,0.0)
    gamma=st.number_input('Peso específico (kN/m³)',10.0,40.0,26.4,0.1)
    st.divider()
    st.header('Detección y visualización')
    nplanes=st.slider('Planos por familia',1,5,3)
    resolution=st.slider('Resolución de malla',30,70,50,5)
    minvol=st.number_input('Volumen mínimo visual (m³)',0.0,100.0,0.25,0.05)
    show_small=st.checkbox('Incluir candidatos pequeños',True)
    show_open=st.checkbox('Incluir regiones abiertas (solo diagnóstico)',False)
    display=st.radio('Vista 3D',['Bloque seleccionado','Hasta 2 por zona','Todos los candidatos filtrados'],index=0)
    st.caption('Los filtros son visuales; nunca demuestran seguridad.')

r=W/2
spring=H-r
arc=np.linspace(np.pi,0,121)
section=np.vstack(([-r,0],[-r,spring],np.column_stack((r*np.cos(arc),spring+r*np.sin(arc))),[r,0]))
t1,t2,t3,t4=st.tabs(['Bloques 3D','Discontinuidades','Cortes y sostenimiento','Exportación / DXF'])

systems=[]
with t2:
    st.info('La dirección de buzamiento se expresa en grados respecto al norte; el azimut de galería define el sistema local. La posición de cada familia sitúa el plano central.')
    for i in range(3):
        with st.expander(f'Familia {i+1}',expanded=(i==0)):
            a,b=st.columns(2)
            with a:
                dd=st.number_input('Dirección de buzamiento (°)',0.0,359.9,[30.0,150.0,270.0][i],key=f'dd{i}')
                dip=st.number_input('Buzamiento (°)',0.0,90.0,60.0,key=f'dip{i}')
                px=st.number_input('Posición transversal X (m)',-W,W,0.0,key=f'px{i}')
                py=st.number_input('Posición longitudinal Y (m)',0.0,L,L*[.42,.50,.58][i],key=f'py{i}')
                sp=st.number_input('Espaciamiento (m)',.1,20.0,1.5,key=f'sp{i}')
                pers=st.number_input('Persistencia estimada (m)',.1,100.0,5.0,key=f'pers{i}')
            with b:
                method=st.selectbox('Resistencia de juntas',['Mohr-Coulomb','Barton-Bandis','Patton'],key=f'method{i}')
                if method=='Mohr-Coulomb':
                    props={'c_MPa':st.number_input('Cohesión (MPa)',0.0,20.0,.05,key=f'c{i}'),'phi_deg':st.number_input('Fricción (°)',0.0,89.0,30.0,key=f'phi{i}')}
                elif method=='Barton-Bandis':
                    props={'JRC':st.number_input('JRC',0.0,25.0,10.0,key=f'jrc{i}'),'JCS_MPa':st.number_input('JCS (MPa)',.1,500.0,30.0,key=f'jcs{i}'),'phi_r_deg':st.number_input('Fricción residual (°)',0.0,89.0,25.0,key=f'pr{i}')}
                else:
                    props={'phi_b_deg':st.number_input('Fricción básica (°)',0.0,89.0,25.0,key=f'pb{i}'),'asperidad_deg':st.number_input('Asperidad (°)',0.0,60.0,10.0,key=f'asp{i}')}
                props['agua_MPa']=st.number_input('Presión de agua (MPa)',0.0,10.0,0.0,key=f'water{i}')
            systems.append(dict(dd=dd,dip=dip,x=px,y=py,spacing=sp,persistence=pers,method=method,props=props))

@st.cache_data(show_spinner=False)
def calculate(width,height,length,azimuth,geos,n,res):
    rr=width/2; ss=height-rr; margin=max(2.0,width*.6)
    nx=res; ny=min(105,max(35,round(res*length/(width+2*margin))))
    nz=min(85,max(35,round(res*(height+margin)/(width+2*margin))))
    xx=np.linspace(-rr-margin,rr+margin,nx)
    yy=np.linspace(0,length,ny)
    zz=np.linspace(0,height+margin,nz)
    dx,dy,dz=xx[1]-xx[0],yy[1]-yy[0],zz[1]-zz[0]
    X,Y,Z=np.meshgrid(xx,yy,zz,indexing='ij')
    tunnel=(np.abs(X)<=rr)&((Z<=ss)|((Z>ss)&(X*X+(Z-ss)**2<=rr*rr)))
    rock=~tunnel
    signature=np.zeros(X.shape,dtype=np.uint32)
    bit=0
    for dd,dip,px,py,spacing in geos:
        theta=np.deg2rad(dd-azimuth); dipr=np.deg2rad(dip)
        normal=np.array([np.sin(theta)*np.sin(dipr),np.cos(theta)*np.sin(dipr),np.cos(dipr)])
        base=normal@np.array([px,py,height])
        signed=normal[0]*X+normal[1]*Y+normal[2]*Z-base
        for k in range(n):
            shift=(k-(n-1)/2)*spacing
            signature|=((signed>=shift).astype(np.uint32)<<bit)
            bit+=1
    # Etiquetar cada celda por su firma y conectividad 3D.
    contact=rock&ndi.binary_dilation(tunnel,structure=ndi.generate_binary_structure(3,1))
    labels=np.zeros(rock.shape,dtype=np.int32)
    blocks=[]; next_id=0
    structure=ndi.generate_binary_structure(3,1)
    for sig in np.unique(signature[contact]):
        components,_=ndi.label(rock&(signature==sig),structure=structure)
        for component_id in np.unique(components[contact]):
            if component_id==0: continue
            region=components==component_id
            count=int(region.sum())
            if count<5: continue
            adjacent=region&ndi.binary_dilation(tunnel,structure=structure)
            # Contacto con superficie de excavación: bóveda o hastiales.
            roof=adjacent&(Z>ss)
            left=adjacent&(X<0)&(Z<=ss)
            right=adjacent&(X>=0)&(Z<=ss)
            contacts={'Bóveda':int(roof.sum()),'Hastial izquierdo':int(left.sum()),'Hastial derecho':int(right.sum())}
            zone=max(contacts,key=contacts.get)
            if contacts[zone]==0: continue
            open_bound=bool(region[0].any() or region[-1].any() or region[:,0].any() or region[:,-1].any() or region[:,:,0].any() or region[:,:,-1].any())
            ids=np.argwhere(region)
            ctr=[float(xx[ids[:,0]].mean()),float(yy[ids[:,1]].mean()),float(zz[ids[:,2]].mean())]
            touch=np.argwhere(adjacent)
            contact_ctr=[float(xx[touch[:,0]].mean()),float(yy[touch[:,1]].mean()),float(zz[touch[:,2]].mean())]
            next_id+=1; labels[region]=next_id
            blocks.append(dict(id=next_id,zone=zone,contacts=contacts,volume=count*dx*dy*dz,open=open_bound,center=ctr,contact_center=contact_ctr))
    return labels,blocks,(xx,yy,zz)

geo=tuple((s['dd'],s['dip'],s['x'],s['y'],s['spacing']) for s in systems)
with st.spinner('Calculando regiones y contactos...'):
    labels,all_blocks,axes=calculate(W,H,L,az,geo,nplanes,resolution)
xx,yy,zz=axes
step=(xx[1]-xx[0],yy[1]-yy[0],zz[1]-zz[0])
valid=[b for b in all_blocks if (show_open or not b['open']) and (show_small or b['volume']>=minvol)]
valid.sort(key=lambda b:b['volume'],reverse=True)
colors=['#e76f51','#2a9d8f','#457b9d','#9b5de5','#e9a23b','#d45087','#6b8e23','#b56576']
for j,b in enumerate(valid): b['color']=colors[j%len(colors)]

@st.cache_data(show_spinner=False)
def surface_for(block_id,labels,xx,yy,zz):
    mask=np.pad((labels==block_id).astype(np.uint8),1)
    if not mask.any(): return None
    step=(xx[1]-xx[0],yy[1]-yy[0],zz[1]-zz[0])
    verts,faces,_,_=marching_cubes(mask,level=.5,spacing=step)
    verts+=np.array([xx[0]-step[0],yy[0]-step[1],zz[0]-step[2]])
    return verts,faces

with t1:
    st.subheader('Bloques candidatos en contacto con la excavación')
    a,b,c=st.columns(3)
    a.metric('Regiones con contacto',len(all_blocks))
    b.metric('Cerradas dentro del dominio',sum(not q['open'] for q in all_blocks))
    c.metric('Abiertas (no verificadas)',sum(q['open'] for q in all_blocks))
    st.caption('«Cerrada» significa que no toca los límites numéricos; no acredita liberación ni estabilidad.')
    if not valid:
        st.warning('No hay regiones con estos filtros. Esto NO demuestra ausencia de cuñas. Revisa la geometría y los límites del dominio.')
        selected=None; visible=[]
    else:
        options={f'B{q["id"]} · {q["zone"]} · {q["volume"]:.2f} m³'+(' · ABIERTA' if q['open'] else ''):q for q in valid}
        choice=st.selectbox('Seleccionar bloque para estudiar',list(options))
        selected=options[choice]
        if display=='Bloque seleccionado': visible=[selected]
        elif display=='Hasta 2 por zona':
            visible=[]
            for zone in ['Bóveda','Hastial izquierdo','Hastial derecho']:
                visible.extend([q for q in valid if q['zone']==zone][:2])
        else: visible=valid
        if selected['open']:
            st.error('El bloque seleccionado llega al límite del dominio: su volumen está truncado y NO representa una cuña finita verificada.')
        else:
            st.info('Región cerrada numéricamente. Falta verificar persistencia, mecanismos de salida y equilibrio.')
    fig=go.Figure()
    for y in (0,L):
        fig.add_trace(go.Scatter3d(x=section[:,0],y=np.full(len(section),y),z=section[:,1],mode='lines',line=dict(color='#475569',width=6),showlegend=False))
    for k in (0,24,48,72,96,120):
        fig.add_trace(go.Scatter3d(x=[section[k,0]]*2,y=[0,L],z=[section[k,1]]*2,mode='lines',line=dict(color='#cbd5e1',width=2),showlegend=False))
    for q in visible:
        result=surface_for(q['id'],labels,xx,yy,zz)
        if result is None: continue
        verts,faces=result
        fig.add_trace(go.Mesh3d(x=verts[:,0],y=verts[:,1],z=verts[:,2],i=faces[:,0],j=faces[:,1],k=faces[:,2],color=q['color'],opacity=.72,name=f'B{q["id"]}',hovertemplate=f'B{q["id"]}<br>{q["zone"]}<br>{q["volume"]:.2f} m³<extra></extra>'))
    # Proporciones reales y zoom de cámara, sin escala artificial.
    fig.update_layout(height=780,margin=dict(l=0,r=0,t=0,b=0),showlegend=False,scene=dict(xaxis_title='X (m)',yaxis_title='Y (m)',zaxis_title='Z (m)',aspectmode='data',camera=dict(eye=dict(x=1.3,y=-1.45,z=.85),projection=dict(type='orthographic'))))
    st.plotly_chart(fig,use_container_width=True,config={'scrollZoom':True})
    st.dataframe([{'Bloque':f'B{q["id"]}','Zona principal':q['zone'],'Volumen discretizado (m³)':round(q['volume'],3),'Peso aproximado (kN)':round(q['volume']*gamma,1),'Estado geométrico':'Abierto' if q['open'] else 'Cerrado en dominio','FS':'No calculado'} for q in valid],hide_index=True,use_container_width=True)
    with st.expander('Todas las regiones detectadas (incluidas las ocultas)'):
        st.dataframe([{'Bloque':f'B{q["id"]}','Zona':q['zone'],'Volumen (m³)':round(q['volume'],3),'Abierto':q['open']} for q in all_blocks],hide_index=True,use_container_width=True)

presets={'ST-I':(2.0,.03,0.0,False),'ST-II':(2.0,.03,0.0,False),'ST-III':(1.5,.08,0.0,True),'ST-IV superficial':(1.5,.12,8.0,True),'ST-IV profundo':(1.5,.15,10.0,True),'Personalizado':(1.5,.08,0.0,True)}
with t3:
    st.subheader('Cortes 2D vinculados al bloque seleccionado')
    if selected is None: st.info('No hay un bloque seleccionado; se muestra la excavación.')
    else: st.caption(f'Centro del contacto de B{selected["id"]}: X={selected["contact_center"][0]:.2f}, Y={selected["contact_center"][1]:.2f}, Z={selected["contact_center"][2]:.2f} m')
    preset=st.selectbox('Sostenimiento',list(presets))
    sp,shot_default,cable_default,mesh_default=presets[preset]
    ca,cb=st.columns(2)
    with ca:
        bolt_type=st.selectbox('Pernos',['Resina','Split Set','Swellex','Sin pernos'])
        bolt_length=st.number_input('Longitud de pernos (m)',.5,12.0,2.4)
        bolt_spacing=st.number_input('Separación transversal pernos (m)',.3,5.0,sp)
        bolt_long=st.number_input('Separación longitudinal pernos (m)',.3,5.0,1.5)
        bolt_capacity=st.number_input('Capacidad nominal perno (kN)',1.0,1000.0,324.0)
    with cb:
        shot=st.checkbox('Gunita',True)
        shot_cm=st.number_input('Espesor de gunita (cm)',0.0,50.0,shot_default*100,.5)
        fc=st.number_input('fc gunita (MPa)',1.0,100.0,35.0)
        ft=st.number_input('ft gunita (MPa)',0.0,20.0,3.0)
        mesh=st.checkbox('Malla metálica',mesh_default)
        cables=st.checkbox('Cables',cable_default>0)
        cable_length=st.number_input('Longitud cables (m)',1.0,25.0,cable_default or 8.0)
        cable_spacing=st.number_input('Separación cables (m)',.3,6.0,1.0 if preset=='ST-IV profundo' else 1.5)
    ycenter=selected['contact_center'][1] if selected else L/2
    xcenter=selected['contact_center'][0] if selected else 0.0
    ycut=st.slider('Corte transversal Y (m)',0.0,float(L),float(np.clip(round(ycenter,1),0,L)),.1)
    xcut=st.slider('Corte longitudinal X (m)',float(xx[0]),float(xx[-1]),float(np.clip(round(xcenter,1),xx[0],xx[-1])),.1)
    iy=int(np.argmin(abs(yy-ycut))); ix=int(np.argmin(abs(xx-xcut)))
    f=go.Figure()
    for q in visible:
        mask=(labels[:,iy,:]==q['id']).T.astype(float)
        if mask.any():
            f.add_trace(go.Contour(x=xx,y=zz,z=mask,contours=dict(start=.5,end=.5,size=1,coloring='none'),line=dict(color=q['color'],width=3),showscale=False,name=f'B{q["id"]}'))
    f.add_trace(go.Scatter(x=section[:,0],y=section[:,1],mode='lines',line=dict(color='#334155',width=5),name='Excavación'))
    if shot and shot_cm>0:
        # Offset exterior radial en bóveda; lateral en hastiales.
        points=section[1:-1].copy(); norms=[]
        for p in points:
            if p[1]<=spring+1e-8: norms.append([np.sign(p[0]),0])
            else:
                v=np.array([p[0],p[1]-spring]); norms.append(v/max(np.linalg.norm(v),1e-9))
        ext=points+(shot_cm/100)*np.array(norms)
        f.add_trace(go.Scatter(x=ext[:,0],y=ext[:,1],mode='lines',line=dict(color='#f97316',width=7),name='Gunita'))
    if mesh:
        f.add_trace(go.Scatter(x=section[1:-1,0],y=section[1:-1,1],mode='lines',line=dict(color='#16a34a',dash='dot',width=3),name='Malla'))
    seg=np.linalg.norm(np.diff(section,axis=0),axis=1); cumulative=np.r_[0,np.cumsum(seg)]
    if bolt_type!='Sin pernos':
        for k,s in enumerate(np.arange(bolt_spacing/2,cumulative[-1],bolt_spacing)):
            idx=int(np.clip(np.searchsorted(cumulative,s)-1,0,len(seg)-1))
            t=(s-cumulative[idx])/max(seg[idx],1e-9); p=section[idx]*(1-t)+section[idx+1]*t
            if p[1]<=spring: normal2=np.array([np.sign(p[0]),0.0])
            else:
                normal2=np.array([p[0],p[1]-spring]); normal2/=max(np.linalg.norm(normal2),1e-9)
            q=p+bolt_length*normal2
            f.add_trace(go.Scatter(x=[p[0],q[0]],y=[p[1],q[1]],mode='lines',line=dict(color='#2563eb',width=4),name='Pernos',showlegend=(k==0)))
    if cables:
        for k,s in enumerate(np.arange(cable_spacing/2,math.pi*r,cable_spacing)):
            angle=math.pi-s/r; p=np.array([r*math.cos(angle),spring+r*math.sin(angle)]); q=p+cable_length*np.array([math.cos(angle),math.sin(angle)])
            f.add_trace(go.Scatter(x=[p[0],q[0]],y=[p[1],q[1]],mode='lines',line=dict(color='#9333ea',width=4),name='Cables',showlegend=(k==0)))
    f.update_layout(height=650,xaxis=dict(title='X (m)',scaleanchor='y',scaleratio=1),yaxis_title='Z (m)',legend=dict(orientation='h'))
    st.plotly_chart(f,use_container_width=True)
    fl=go.Figure()
    for q in visible:
        mask=(labels[ix,:,:]==q['id']).T.astype(float)
        if mask.any(): fl.add_trace(go.Contour(x=yy,y=zz,z=mask,contours=dict(start=.5,end=.5,size=1,coloring='none'),line=dict(color=q['color'],width=3),showscale=False,name=f'B{q["id"]}'))
    fl.update_layout(height=400,xaxis_title='Y (m)',yaxis_title='Z (m)')
    st.plotly_chart(fl,use_container_width=True)
    st.caption('Cortes y volúmenes aproximados según malla. Sostenimientos esquemáticos; no se calcula resistencia movilizada ni FS.')

with t4:
    data={'version':13,'galeria':{'ancho':W,'alto':H,'longitud':L,'azimut':az,'peso_especifico':gamma},'familias':systems}
    st.download_button('Exportar parámetros JSON',json.dumps(data,ensure_ascii=False,indent=2),file_name='geotecnia_v13.json',mime='application/json')
    dxf=st.file_uploader('Sección AutoCAD DXF (referencia visual)',type=['dxf'])
    if dxf:
        try:
            import ezdxf
            doc=ezdxf.read(io.StringIO(dxf.getvalue().decode('utf-8-sig',errors='replace')))
            fd=go.Figure()
            for ent in doc.modelspace():
                kind=ent.dxftype(); pts=None
                if kind=='LINE': pts=np.array([ent.dxf.start[:2],ent.dxf.end[:2]],float)
                elif kind=='LWPOLYLINE': pts=np.array(list(ent.get_points('xy')),float)
                elif kind=='CIRCLE':
                    ang=np.linspace(0,2*np.pi,120); p=ent.dxf.center; rad=ent.dxf.radius
                    pts=np.column_stack((p.x+rad*np.cos(ang),p.y+rad*np.sin(ang)))
                if pts is not None: fd.add_trace(go.Scatter(x=pts[:,0],y=pts[:,1],mode='lines',showlegend=False))
            fd.update_layout(height=480,xaxis=dict(scaleanchor='y',scaleratio=1))
            st.plotly_chart(fd,use_container_width=True)
            st.caption('DXF solo visual: no modifica la sección de cálculo.')
        except Exception as exc: st.error(f'Error de lectura DXF: {exc}')
