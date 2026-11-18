import io, json, math
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from scipy import ndimage as ndi
from skimage.measure import marching_cubes, find_contours
from motor_geometrico_v17_2 import reconstruct_curved_block, connected_curved_components

st.set_page_config(page_title='Geotecnia Mina v17.2 experimental', page_icon='⛏️', layout='wide', initial_sidebar_state='expanded')
st.markdown("""<style>
/* Tema oscuro consistente: texto de widgets, valores, métricas y pestañas */
html,body,.stApp,[data-testid="stAppViewContainer"],[data-testid="stSidebar"]{background:#0b1421!important;color:#f1f7ff!important}
[data-testid="stSidebar"]{background:#101d2f!important}
.stApp p,.stApp label,.stApp span,.stApp small,.stApp div[data-testid="stMarkdownContainer"],
[data-testid="stWidgetLabel"] *,[data-testid="stCaptionContainer"] *,
[data-testid="stSidebar"] p,[data-testid="stSidebar"] label,
[data-testid="stMetricLabel"] *,[data-testid="stMetricValue"] *,
[data-testid="stTabs"] button,[data-testid="stTabs"] button *{color:#eaf3ff!important}
[data-testid="stCaptionContainer"],.stApp small{color:#c5d6eb!important}
[data-testid="stMetric"]{background:#18283e;border:1px solid #47617f;border-radius:12px;padding:12px}
[data-testid="stMetricValue"]{font-size:1.55rem!important;font-weight:750!important}
[data-testid="stMetricLabel"]{font-weight:650!important}
[data-testid="stNumberInput"] input,[data-testid="stTextInput"] input,
[data-testid="stSelectbox"] [data-baseweb="select"] *,
[data-testid="stNumberInput"] button{color:#142235!important}
[data-testid="stSlider"] [data-testid="stTickBar"] *,[data-testid="stSlider"] [data-testid="stThumbValue"]{color:#ffffff!important}
[data-testid="stTabs"] button{font-weight:700!important;font-size:.95rem!important}
.stApp h1,.stApp h2,.stApp h3,.stApp h4{color:#fff!important}
.block-container{padding-top:1rem;max-width:1750px}
</style>""",unsafe_allow_html=True)
COL=['#36d6bb','#68a8ff','#eeb65d','#d58bf0','#ef7b88','#b5d66b']
st.title('⛏️ GEOTECNIA MINA  ·  v17.2 experimental')
st.caption('Galerías y cámaras · visualizador 3D · estereograma Schmidt/Wulff · Potvin · proyectos · sensibilidad')
st.warning('PROTOTIPO GEOMÉTRICO. Las regiones se detectan mediante discretización; las superficies 3D de regiones cerradas se reconstruyen a partir de ecuaciones analíticas de planos y herradura. La persistencia no recorta los planos. Bloque cerrado ≠ cuña liberable. No se calcula FS validado ni se dimensiona sostenimiento. No usar para decisiones de seguridad.')

# Carga de escenarios: antes de crear widgets para que sus valores se restauren.
with st.expander('📁 Proyectos · guardar o recuperar escenario', expanded=False):
    uploaded=st.file_uploader('Importar escenario JSON de Geotecnia Mina',type=['json'],key='scenario_upload')
    if uploaded is not None and st.button('Aplicar escenario importado'):
        try:
            project=json.load(uploaded)
            geo=project.get('geometria',{})
            for key,source in [('w_input','W'),('h_input','H'),('l_input','L'),('az_input','azimut'),('gamma_input','gamma')]:
                if source in geo: st.session_state[key]=float(geo[source])
            for i,fam in enumerate(project.get('sistemas',[])[:3]):
                for target,source in [('dd','dd'),('dip','dip'),('yy','yy'),('xx','xx'),('space','spacing'),('pers','persistence'),('phi','phi')]:
                    if source in fam: st.session_state[f'{target}{i}']=float(fam[source])
            st.session_state['loaded_project_support']=project.get('sostenimiento',{})
            st.success('Escenario cargado. Comprueba los parámetros antes de interpretar resultados.')
            st.rerun()
        except (ValueError,TypeError,KeyError) as exc:
            st.error(f'No se pudo importar el escenario: {exc}')

with st.sidebar:
 st.header('⚙️ Geometría de la galería')
 W=st.number_input('Ancho (m)',2.,20.,5.8,.1,key='w_input')
 H=st.number_input('Alto (m)',2.,20.,5.8,.1,key='h_input')
 L=st.number_input('Longitud analizada (m)',5.,40.,15.,1.,key='l_input')
 az=st.number_input('Azimut del eje (° desde N)',0.,359.9,0.,1.,key='az_input')
 gamma=st.number_input('Peso específico roca (kN/m³)',15.,40.,26.4,.1,key='gamma_input')
 res=st.slider('Resolución de detección de regiones',36,90,60,6)
 nplanes=st.slider('Planos paralelos por familia',1,4,2)
 minvol=st.number_input('Volumen mínimo mostrado (m³)',0.,20.,.05,.05)
 st.divider()
 st.header('🧱 Sostenimiento')
 typ=st.selectbox('Tipo', ['ST-I','ST-II','ST-III','ST-IV superficial','ST-IV profundo','Personalizado'],index=2)
 presets={'ST-I':(2.4,2.,.03,0.,False),'ST-II':(2.4,2.,.03,0.,False),'ST-III':(2.4,1.5,.08,0.,True),'ST-IV superficial':(2.4,1.5,.12,8.,True),'ST-IV profundo':(2.4,1.5,.15,10.,True),'Personalizado':(2.4,1.5,.08,0.,True)}
 bl,bs,shot,cl,mesh=presets[typ]
 boltlen=st.number_input('Longitud pernos (m)',.5,15.,bl,.1)
 boltspace=st.number_input('Separación pernos (m)',.5,5.,bs,.1)
 shot=st.number_input('Espesor gunita (m)',0.,.5,shot,.01)
 cablelen=st.number_input('Longitud cables (m)',0.,20.,cl,.5)
 mesh=st.checkbox('Malla metálica',value=mesh)
 show_bolts=True # La visibilidad real se controla en el visor 3D
 st.caption('La representación de sostenimiento es geométrica; no implica capacidad resistente verificada.')

r=W/2.; spring=max(0.,H-r)
theta=np.linspace(np.pi,0,100)
section=np.vstack(([-r,0],[-r,spring],np.column_stack((r*np.cos(theta),spring+r*np.sin(theta))),[r,0]))

def normal(dd,dip):
 a=np.deg2rad(dd);b=np.deg2rad(dip)
 return np.array([np.sin(a)*np.sin(b),np.cos(a)*np.sin(b),np.cos(b)])
def angular(a,b):
 return float(np.rad2deg(np.arccos(np.clip(abs(np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b))),0,1))))
def pole(dd,dip,projection):
 # Hemisferio inferior: polo opuesto a la normal ascendente.
 trend=np.deg2rad((dd+180)%360); plunge=90-dip
 rad=np.sqrt(2)*np.sin(np.deg2rad((90-plunge)/2)) if projection=='Schmidt (igual área)' else np.tan(np.deg2rad((90-plunge)/2))
 return rad*np.sin(trend),rad*np.cos(trend)
def great_circle(dd,dip,projection):
 n=normal(dd,dip); v=np.array([0.,0.,1.]);u=np.cross(n,v)
 if np.linalg.norm(u)<1e-7:u=np.array([1.,0.,0.])
 u=u/np.linalg.norm(u);v=np.cross(n,u);v=v/np.linalg.norm(v)
 pts=[]
 for t in np.linspace(0,2*np.pi,361):
  p=u*np.cos(t)+v*np.sin(t)
  if p[2]>0:p=-p
  plunge=np.rad2deg(np.arcsin(np.clip(-p[2],0,1)))
  trend=np.arctan2(p[0],p[1]);rad=np.sqrt(2)*np.sin(np.deg2rad((90-plunge)/2)) if projection=='Schmidt (igual área)' else np.tan(np.deg2rad((90-plunge)/2))
  pts.append((rad*np.sin(trend),rad*np.cos(trend)))
 return np.array(pts)

def tunnel_surface(section,y0,y1):
    # Superficie interior de la excavación extruida: no se cierra el piso.
    v0=np.column_stack([section[:,0],np.full(len(section),y0),section[:,1]])
    v1=np.column_stack([section[:,0],np.full(len(section),y1),section[:,1]])
    verts=np.vstack([v0,v1]); n=len(section)
    faces=[]
    for k in range(n-1):
        faces.append([k,k+1,k+n]);faces.append([k+1,k+n+1,k+n])
    return verts,np.asarray(faces)

def intersection_orientation(a,b):
    # Dos planos: dirección de la recta, no prueba de liberación.
    u=np.cross(normal(a['dd'],a['dip']),normal(b['dd'],b['dip']))
    if np.linalg.norm(u)<1e-8:return None
    u=u/np.linalg.norm(u)
    if u[2]>0:u=-u
    return (float(np.degrees(np.arctan2(u[0],u[1]))%360),float(np.degrees(np.arcsin(np.clip(-u[2],-1,1)))))

with st.expander('📐 Familias de discontinuidades · editar orientación y posición',expanded=True):
 cols=st.columns(3);systems=[]
 for i,col in enumerate(cols):
  with col:
   st.markdown(f'**Familia J{i+1}**')
   dd=st.number_input('Dirección de buzamiento (°)',0.,359.9,[30.,150.,270.][i],key=f'dd{i}')
   dip=st.number_input('Buzamiento (°)',0.,90.,60.,key=f'dip{i}')
   yy=st.number_input('Paso longitudinal (m)',0.,L,L*(.4+.1*i),key=f'yy{i}')
   xx=st.number_input('Paso transversal (m)',-W,W,0.,key=f'xx{i}')
   spacing=st.number_input('Espaciado (m)',.1,20.,2.,key=f'space{i}')
   persistence=st.number_input('Persistencia registrada (m)',.1,50.,5.,key=f'pers{i}')
   phi=st.number_input('Fricción de junta (°)',0.,70.,30.,key=f'phi{i}')
   systems.append(dict(dd=dd,dip=dip,yy=yy,xx=xx,spacing=spacing,persistence=persistence,phi=phi))

@st.cache_data(show_spinner=False,max_entries=3)
def compute(W,H,L,az,res,nplanes,systems_json):
 systems=json.loads(systems_json);r=W/2;ss=max(0,H-r);margin=max(2.,W*.5)
 nx=res;ny=max(30,min(105,round(res*L/(W+2*margin))));nz=max(35,min(85,round(res*(H+margin)/(W+2*margin))))
 xs=np.linspace(-r-margin,r+margin,nx);ys=np.linspace(0,L,ny);zs=np.linspace(0,H+margin,nz)
 X,Y,Z=np.meshgrid(xs,ys,zs,indexing='ij');tunnel=(abs(X)<=r)&((Z<=ss)|((Z>ss)&(X*X+(Z-ss)**2<=r*r)))
 rock=~tunnel;signature=np.zeros(rock.shape,dtype=np.uint32);bit=0
 for s in systems:
  n=normal((s['dd']-az)%360,s['dip']);base=np.dot(n,[s['xx'],s['yy'],H]);signed=n[0]*X+n[1]*Y+n[2]*Z-base
  for k in range(nplanes):
   offset=(k-(nplanes-1)/2)*s['spacing'];signature|=((signed>=offset).astype(np.uint32)<<bit);bit+=1
 structure=ndi.generate_binary_structure(3,1);adjacent=rock&ndi.binary_dilation(tunnel,structure=structure)
 labels=np.zeros(rock.shape,dtype=np.int16);blocks=[];ident=0
 voxel=np.prod([xs[1]-xs[0],ys[1]-ys[0],zs[1]-zs[0]])
 for sig in np.unique(signature[adjacent]):
  region_labels,_=ndi.label(rock&(signature==sig),structure)
  for comp in np.unique(region_labels[adjacent]):
   if comp==0:continue
   reg=region_labels==comp;count=int(reg.sum())
   if count<5:continue
   touching=reg&adjacent;roof=int((touching&(Z>ss)).sum());left=int((touching&(Z<=ss)&(X<0)).sum());right=int((touching&(Z<=ss)&(X>=0)).sum())
   zone=['Bóveda','Hastial izq.','Hastial der.'][np.argmax([roof,left,right])]
   opened=bool(reg[0].any() or reg[-1].any() or reg[:,0].any() or reg[:,-1].any() or reg[:,:,0].any() or reg[:,:,-1].any())
   ident+=1;labels[reg]=ident
   coords=np.argwhere(reg);cc=np.argwhere(touching)
   c=np.mean(coords,axis=0);tc=np.mean(cc,axis=0)
   center=[float(xs[0]+c[0]*(xs[1]-xs[0])),float(ys[0]+c[1]*(ys[1]-ys[0])),float(zs[0]+c[2]*(zs[1]-zs[0]))]
   contact=[float(xs[0]+tc[0]*(xs[1]-xs[0])),float(ys[0]+tc[1]*(ys[1]-ys[0])),float(zs[0]+tc[2]*(zs[1]-zs[0]))]
   blocks.append(dict(id=ident,zone=zone,volume=count*voxel,open=opened,center=center,contact=contact))
 return xs,ys,zs,labels,blocks

@st.cache_data(show_spinner=False,max_entries=50)
def surface(labels,ident,dx,dy,dz,x0,y0,z0):
 mask=np.pad((labels==ident).astype(np.uint8),1)
 if not mask.any():return None
 verts,faces,_,_=marching_cubes(mask,.5,spacing=(dx,dy,dz));verts+=np.array([x0-dx,y0-dy,z0-dz]);return verts,faces

@st.cache_data(show_spinner=False,max_entries=35)
def surface_analytic(labels,ident,xs,ys,zs,W,H,L,az,nplanes,systems_json,quality):
 # Reconstrucción subvoxel por campos implícitos de los planos y la herradura.
 # La detección y clasificación del bloque siguen siendo aproximadas.
 idx=np.argwhere(labels==ident)
 if len(idx)<5:return None
 systems=json.loads(systems_json)
 dx,dy,dz=xs[1]-xs[0],ys[1]-ys[0],zs[1]-zs[0]
 lo=idx.min(axis=0);hi=idx.max(axis=0)
 margin=np.array([max(dx*2,.30),max(dy*2,.30),max(dz*2,.30)])
 mins=np.maximum([xs[0],ys[0],zs[0]],[xs[lo[0]]-margin[0],ys[lo[1]]-margin[1],zs[lo[2]]-margin[2]])
 maxs=np.minimum([xs[-1],ys[-1],zs[-1]],[xs[hi[0]]+margin[0],ys[hi[1]]+margin[1],zs[hi[2]]+margin[2]])
 nside=int(quality)
 lengths=maxs-mins
 # Isotropic spacing, capped to keep interactive performance.
 step=max(float(max(lengths)/nside),.035)
 dims=np.maximum(5,np.ceil(lengths/step).astype(int)+1)
 dims=np.minimum(dims,140)
 axes=[np.linspace(mins[i],maxs[i],dims[i]) for i in range(3)]
 X,Y,Z=np.meshgrid(*axes,indexing='ij')
 center=np.array([xs[idx[:,0]].mean(),ys[idx[:,1]].mean(),zs[idx[:,2]].mean()])
 field=np.full(X.shape,-np.inf,dtype=np.float32)
 for fam in systems:
  n=normal((fam['dd']-az)%360,fam['dip'])
  base=np.dot(n,[fam['xx'],fam['yy'],H])
  for k in range(nplanes):
   off=(k-(nplanes-1)/2)*fam['spacing']
   signed=n[0]*X+n[1]*Y+n[2]*Z-base-off
   side=1. if (np.dot(n,center)-base-off)>=0 else -1.
   np.maximum(field,-side*signed,out=field)
 r=W/2.;spring=max(0.,H-r)
 # Distancia exterior aproximada a herradura: frontera exacta de hastiales y arco.
 outside=np.where(Z<=spring,np.abs(X)-r,np.sqrt(X*X+(Z-spring)**2)-r)
 np.maximum(field,-outside,out=field)
 # Límite de dominio; para regiones cerradas no debe cortar el sólido.
 for coord,lower,upper in [(X,xs[0],xs[-1]),(Y,ys[0],ys[-1]),(Z,zs[0],zs[-1])]:
  np.maximum(field,lower-coord,out=field)
  np.maximum(field,coord-upper,out=field)
 if not (np.nanmin(field)<0<np.nanmax(field)):return None
 verts,faces,_,_=marching_cubes(field,level=0.,spacing=tuple(float(a[1]-a[0]) for a in axes))
 verts+=mins
 # La caja de muestreo no debe generar tapas artificiales.
 eps=step*1.1
 on_box=np.any((verts<=mins+eps)|(verts>=maxs-eps),axis=1)
 return verts,faces,float(np.mean(on_box))

with st.spinner('Calculando regiones de roca adyacentes a la excavación...'):
 xs,ys,zs,labels,blocks=compute(W,H,L,az,res,nplanes,json.dumps(systems,sort_keys=True))
visible=[b for b in blocks if b['volume']>=minvol]
closed=[b for b in visible if not b['open']]
if not visible:st.error('No se han identificado regiones con los parámetros actuales.');st.stop()

m1,m2,m3,m4=st.columns(4)
m1.metric('Regiones en contacto',len(visible));m2.metric('Cerradas en dominio',len(closed));m3.metric('Abiertas al borde',len(visible)-len(closed));m4.metric('FS validado','No calculado')
tabs=st.tabs(['🧊 Cuñas 3D','🧭 Estereograma y mediciones','📏 Cortes 2D','🧱 Sostenimiento','🏗️ Cámaras · Potvin','📈 Sensibilidad','📥 Exportación','🧪 Motor geométrico v17.2'])

with tabs[0]:
 c1,c2,c3=st.columns([1.2,1,1])
 with c1:
  options=sorted(visible,key=lambda b:(b['open'],-b['volume']))
  selected=st.selectbox('Región seleccionada',options,format_func=lambda b:f"B{b['id']} · {b['zone']} · {b['volume']:.2f} m³ · {'abierta' if b['open'] else 'cerrada'}")
 with c2:mode=st.radio('Mostrar regiones', ['Todas las regiones','Todas las cerradas','Solo seleccionada'],horizontal=False,help='Abiertas/cerradas se refiere al borde del dominio, no a la estabilidad')
 with c3:
  scale=st.slider('Escala de análisis de cuña',.5,2.,1.,.05)
  focus=st.checkbox('Acercar cámara a la cuña',False)
  isolate=st.checkbox('Aislar cuña seleccionada (ocultar galería y otras regiones)',False)
  identify=st.checkbox('Identificar bloques con etiquetas B',True)
  quality=st.slider('Calidad de superficie 3D',55,115,85,10)
 st.markdown('#### 👁️ Herramientas de visualización 3D')
 with st.expander('Mostrar / ocultar elementos, etiquetas y cámara',expanded=True):
  vc1,vc2,vc3=st.columns(3)
  with vc1:
   view_tunnel=st.checkbox('Galería',True,key='view_tunnel')
   view_wedge=st.checkbox('Cuñas',True,key='view_wedge')
   st.caption('Trazas J1–J3: pendientes de implementación geométrica')
  with vc2:
   view_bolts=st.checkbox('Pernos',True,key='view_bolts')
   view_cables=st.checkbox('Cables',True,key='view_cables')
   view_axes=st.checkbox('Ejes y números',True,key='view_axes')
  with vc3:
   view_grid=st.checkbox('Cuadrícula',False,key='view_grid')
   view_labels=st.checkbox('Títulos de ejes',True,key='view_labels')
   label_size=st.slider('Tamaño de números',12,24,17,1,key='label_size')
  camera_view=st.selectbox('Vista de cámara',['Isométrica','Frontal','Lateral','Superior','Centrar cuña'],key='camera_view')
  st.caption('Para examinar B15/B16: selecciona el bloque, activa Aislar y escoge Centrar cuña. Puedes ocultar ejes y números.')
  if st.button('Restablecer visualización',key='reset_view'):
   for k,v in {'view_tunnel':True,'view_wedge':True,'view_bolts':True,'view_cables':True,'view_axes':True,'view_grid':False,'view_labels':True,'label_size':17,'camera_view':'Isométrica'}.items():st.session_state[k]=v
   st.rerun()
 st.caption(f'**Regiones disponibles:** {len(visible)} en contacto; {len(closed)} cerradas y {len(visible)-len(closed)} abiertas. El modo «Todas las regiones» permite comparar su ubicación, pero no indica que sean cuñas desprendibles.')
 st.caption('Escala: transforma una copia visual de la región seleccionada alrededor de su centro; el volumen de escenario se multiplica por escala³. No modifica los planos ni representa una cuña geológicamente demostrada.')
 a,b,c,d=st.columns(4)
 a.metric('Volumen geométrico',f"{selected['volume']:.3f} m³")
 b.metric('Peso geométrico',f"{selected['volume']*gamma:.1f} kN")
 c.metric('Volumen escalado',f"{selected['volume']*scale**3:.3f} m³")
 d.metric('Peso escalado',f"{selected['volume']*gamma*scale**3:.1f} kN")
 fig=go.Figure();ymin=max(0,selected['contact'][1]-2.5) if focus else 0;ymax=min(L,selected['contact'][1]+2.5) if focus else L
 tv,tf=tunnel_surface(section,ymin,ymax)
 if view_tunnel and not isolate:fig.add_trace(go.Mesh3d(x=tv[:,0],y=tv[:,1],z=tv[:,2],i=tf[:,0],j=tf[:,1],k=tf[:,2],color='#81b8e6',opacity=.16,flatshading=True,hoverinfo='skip',name='Galería'))
 if view_tunnel and not isolate:
  for y in [ymin,ymax]:fig.add_trace(go.Scatter3d(x=section[:,0],y=np.full(len(section),y),z=section[:,1],mode='lines',line=dict(color='#a2b6ca',width=5),showlegend=False,hoverinfo='skip'))
 if view_tunnel and not isolate:
  for x,z in section[::8]:fig.add_trace(go.Scatter3d(x=[x,x],y=[ymin,ymax],z=[z,z],mode='lines',line=dict(color='#9cc7ed',width=3),showlegend=False,hoverinfo='skip'))
 shown=([selected] if isolate else ([selected] if mode=='Solo seleccionada' else ([v for v in visible if not v['open']] if mode=='Todas las cerradas' else visible))) if view_wedge else []
 shown=sorted(shown,key=lambda b:(b['id']!=selected['id'],-b['volume']))[:35]
 st.caption(f'**Visualización 3D:** {len(shown)} de {len(visible) if mode=="Todas las regiones" else len(closed) if mode=="Todas las cerradas" else 1} regiones del filtro. Verde = seleccionada; otros colores = regiones adicionales. Máximo 35 para mantener fluidez.')
 for j,v in enumerate(shown):
  result=(surface_analytic(labels,v['id'],xs,ys,zs,W,H,L,az,nplanes,json.dumps(systems,sort_keys=True),quality) if not v['open'] else None)
  if result is None:result=surface(labels,v['id'],xs[1]-xs[0],ys[1]-ys[0],zs[1]-zs[0],xs[0],ys[0],zs[0]);
  if result is None:continue
  verts,faces=result[:2]
  if v['id']==selected['id'] and scale!=1:verts=(verts-np.array(v['center']))*scale+np.array(v['center'])
  color='#34e3a4' if v['id']==selected['id'] else {'Bóveda':'#f6bb54','Hastial izq.':'#63b3ed','Hastial der.':'#e99ed8'}.get(v['zone'],COL[j%len(COL)])
  fig.add_trace(go.Mesh3d(x=verts[:,0],y=verts[:,1],z=verts[:,2],i=faces[:,0],j=faces[:,1],k=faces[:,2],color=color,opacity=.86 if v['id']==selected['id'] else (.24 if v['open'] else .58),name=f"B{v['id']}",flatshading=True,lighting=dict(ambient=.55,diffuse=.85,specular=.35,roughness=.35,fresnel=.1),hovertemplate=f"B{v['id']} · {v['zone']}<br>{v['volume']:.2f} m³<br>{'Abierta al borde' if v['open'] else 'Cerrada en dominio'}<extra></extra>"))
  if identify:
   # Marcador situado en el centro de la región, con texto independiente del hover.
   center=np.asarray(v['center'])
   fig.add_trace(go.Scatter3d(x=[center[0]],y=[center[1]],z=[center[2]],mode='markers+text',
     marker=dict(size=5,color='#ffffff',line=dict(color='#101d2f',width=1)),
     text=[f"B{v['id']}"],textposition='top center',textfont=dict(size=15,color='#ffffff'),
     showlegend=False,hoverinfo='skip'))
 def bolts():
  # Anclajes radiales simplificados: puntos en bóveda y hastiales, con estaciones longitudinales.
  arc=np.linspace(np.pi-.12,.12,max(5,round(np.pi*r/boltspace)+1))
  stations=np.arange(boltspace/2,L,boltspace)
  for y in stations:
   for t in arc:
    p=np.array([r*np.cos(t),y,spring+r*np.sin(t)]);out=np.array([np.cos(t),0.,np.sin(t)])
    yield p,out,boltlen,'Perno'
   for side in [-1,1]:
    for z in np.arange(boltspace/2,spring,boltspace):yield np.array([side*r,y,z]),np.array([side,0.,0.]),boltlen,'Perno'
   if cablelen>0:
    for t in np.linspace(np.pi-.25,.25,5):yield np.array([r*np.cos(t),y,spring+r*np.sin(t)]),np.array([np.cos(t),0.,np.sin(t)]),cablelen,'Cable'
 def intersection(p,q,ident):
  delta=min(xs[1]-xs[0],ys[1]-ys[0],zs[1]-zs[0])/2
  n=max(3,int(np.linalg.norm(q-p)/delta)+1);t=np.linspace(0,1,n);pts=p[None,:]+t[:,None]*(q-p)[None,:]
  ix=np.rint((pts[:,0]-xs[0])/(xs[1]-xs[0])).astype(int);iy=np.rint((pts[:,1]-ys[0])/(ys[1]-ys[0])).astype(int);iz=np.rint((pts[:,2]-zs[0])/(zs[1]-zs[0])).astype(int)
  ok=(ix>=0)&(ix<len(xs))&(iy>=0)&(iy<len(ys))&(iz>=0)&(iz<len(zs));inside=np.zeros(n,bool);inside[ok]=labels[ix[ok],iy[ok],iz[ok]]==ident
  if not inside.any():return False,0.
  last=np.flatnonzero(inside)[-1];return True,float(np.linalg.norm(q-p)*(1-t[last]))
 hits=[]
 if (view_bolts or view_cables) and not isolate:
  for p,out,length,kind in bolts():
   if (kind=='Perno' and not view_bolts) or (kind=='Cable' and not view_cables):continue
   q=p+out*length
   if focus and not(ymin-.2<=p[1]<=ymax+.2):continue
   hit,tail=intersection(p,q,selected['id']);hits.append((kind,hit,tail))
   fig.add_trace(go.Scatter3d(x=[p[0],q[0]],y=[p[1],q[1]],z=[p[2],q[2]],mode='lines',line=dict(color=('#e8a34d' if hit else '#4c789e') if kind=='Perno' else '#c68bf0',width=5 if hit else 2),showlegend=False,hoverinfo='skip'))
 # Encuadre local de la región, evitando el gran vacío del visor anterior.
 cx,cy,cz=selected['center'] if (focus or isolate or camera_view=='Centrar cuña') else selected['contact'];span=max(W*.7,H*.7,3.0) if (focus or isolate or camera_view=='Centrar cuña') else max(W,H,L)
 xr=[cx-span*.65,cx+span*.65] if (focus or isolate or camera_view=='Centrar cuña') else [xs[0],xs[-1]]
 yr=[max(0,cy-span*.7),min(L,cy+span*.7)] if (focus or isolate or camera_view=='Centrar cuña') else [0,L]
 zr=[max(0,cz-span*.65),min(zs[-1],cz+span*.65)] if (focus or isolate or camera_view=='Centrar cuña') else [0,zs[-1]]
 # Plotly: texto grande y blanco, cuadrícula opcional y cámara configurable.
 cameras={'Isométrica':dict(x=1.4,y=-1.7,z=1.15),'Frontal':dict(x=0,y=-2.5,z=.55),
          'Lateral':dict(x=2.5,y=0,z=.35),'Superior':dict(x=.01,y=-.01,z=2.8),
          'Centrar cuña':dict(x=1.35,y=-1.6,z=1.05)}
 def axis_style(title,rng):
  return dict(title=dict(text=title if view_labels else '',font=dict(size=label_size+1,color='#ffffff')),
              range=rng,visible=view_axes,showticklabels=view_axes,
              tickfont=dict(size=label_size,color='#ffffff'),tickcolor='#ffffff',
              showgrid=view_grid,gridcolor='#50647e',gridwidth=1,
              showbackground=False,zeroline=False,ticks='outside',ticklen=5,
              nticks=4)
 fig.update_layout(height=780,margin=dict(l=0,r=0,t=0,b=0),paper_bgcolor='#0b1421',
     font=dict(color='#ffffff',size=label_size),
     scene=dict(domain=dict(x=[0,1],y=[0,1]),bgcolor='#0b1421',
                xaxis=axis_style('Transversal X (m)',xr),
                yaxis=axis_style('Eje Y (m)',yr),
                zaxis=axis_style('Altura Z (m)',zr),
                aspectmode='data',camera=dict(eye=cameras[camera_view],
                    projection=dict(type='orthographic'))),showlegend=False)

 st.plotly_chart(fig,use_container_width=True)
 # Diagnóstico explícito de bordes: basado en los mismos vóxeles que la clasificación.
 mask=(labels==selected['id'])
 borders=[]
 for touched,label in [(mask[0,:,:].any(),'X mínimo'),(mask[-1,:,:].any(),'X máximo'),
                       (mask[:,0,:].any(),'Y inicial'),(mask[:,-1,:].any(),'Y final'),
                       (mask[:,:,0].any(),'Z inferior'),(mask[:,:,-1].any(),'Z superior')]:
  if touched:borders.append(label)
 idx=np.argwhere(mask)
 if len(idx):
  mins=np.array([xs[idx[:,0].min()],ys[idx[:,1].min()],zs[idx[:,2].min()]])
  maxs=np.array([xs[idx[:,0].max()],ys[idx[:,1].max()],zs[idx[:,2].max()]])
  extents=maxs-mins
  st.markdown(f"**Diagnóstico B{selected['id']}:** {'Abierta al dominio' if borders else 'Cerrada en dominio'} · ",unsafe_allow_html=False)
  st.write('Bordes alcanzados:', ', '.join(borders) if borders else 'Ninguno de los seis límites del dominio discretizado')
  st.write(f'Dimensiones aproximadas de su caja envolvente (X × Y × Z): **{extents[0]:.2f} × {extents[1]:.2f} × {extents[2]:.2f} m**. Son extensiones voxelizadas, no dimensiones verificadas de una cuña.')
  if mode=='Todas las cerradas' and len(closed)>1:
   st.caption('Si solo distingues una cuña, utiliza «Aislar cuña seleccionada» y alterna entre B15 y B16: pueden solaparse en proyección o una puede quedar oculta.')
 st.caption('Galería azul semitransparente, región seleccionada verde; otras regiones en amarillo (bóveda), azul (hastial izquierdo) o rosa (derecho).')
 st.caption('Superficie 3D reconstruida mediante ecuaciones de planos y sección de herradura para regiones cerradas; identificación del bloque aún basada en vóxeles. Las caras y volúmenes necesitan verificación geométrica independiente.')
 st.info('La clasificación abierta/cerrada se refiere exclusivamente a los límites del dominio discretizado. No se ha comprobado la liberación cinemática ni la estabilidad.')
 if view_bolts or view_cables:
  h=[x for x in hits if x[1]]
  st.write(f"**Intersecciones geométricas aproximadas con B{selected['id']}:** {len(h)} de {len(hits)} elementos representados en la ventana 3D.")
  if h:st.write(f"Longitud exterior tras la última intersección: mínimo **{min(x[2] for x in h):.2f} m**. No equivale a longitud de anclaje efectiva verificada.")

with tabs[1]:
 st.subheader('Estereograma · selección de polos y medición angular')
 a,b,c=st.columns(3)
 with a:projection=st.selectbox('Proyección',['Schmidt (igual área)','Wulff (igual ángulo)'])
 with b:reference=st.selectbox('Plano de referencia',['Muro de cámara','Techo de cámara','Hastial de galería','Plano personalizado'])
 with c:show_circles=st.checkbox('Mostrar grandes círculos',True)
 refdd=st.number_input('Dirección de buzamiento del plano de referencia (°)',0.,359.9,90.,key='refdd')
 refdip=st.number_input('Buzamiento del plano de referencia (°)',0.,90.,90. if reference!='Techo de cámara' else 0.,key=f'refdip_{reference}')
 entries=[dict(name=f'J{i+1}',dd=s['dd'],dip=s['dip'],color=COL[i]) for i,s in enumerate(systems)]
 entries.append(dict(name='Referencia',dd=refdd,dip=refdip,color='#ffffff'))
 chart=go.Figure();t=np.linspace(0,2*np.pi,360)
 chart.add_trace(go.Scatter(x=np.sin(t),y=np.cos(t),mode='lines',line=dict(color='#7b90a7',width=2),showlegend=False,hoverinfo='skip'))
 for rad in [.25,.5,.75]:chart.add_trace(go.Scatter(x=rad*np.sin(t),y=rad*np.cos(t),mode='lines',line=dict(color='#30445c',width=1),showlegend=False,hoverinfo='skip'))
 for ang in range(0,180,30):
  q=np.deg2rad(ang);chart.add_trace(go.Scatter(x=[-np.sin(q),np.sin(q)],y=[-np.cos(q),np.cos(q)],mode='lines',line=dict(color='#30445c',width=1),showlegend=False,hoverinfo='skip'))
 for e in entries:
  if show_circles:
   gc=great_circle(e['dd'],e['dip'],projection)
   chart.add_trace(go.Scatter(x=gc[:,0],y=gc[:,1],mode='lines',line=dict(color=e['color'],width=1,dash='dot'),opacity=.55,showlegend=False,hoverinfo='skip'))
  x,y=pole(e['dd'],e['dip'],projection)
  chart.add_trace(go.Scatter(x=[x],y=[y],mode='markers+text',text=[e['name']],textposition='top center',marker=dict(size=14,color=e['color'],line=dict(color='#0b1421',width=2)),name=e['name'],customdata=[[e['dd'],e['dip']]],hovertemplate=f"{e['name']}<br>Dir. buzamiento %{{customdata[0]}}°<br>Buzamiento %{{customdata[1]}}°<extra></extra>"))
 chart.update_layout(height=590,margin=dict(l=10,r=10,t=10,b=10),paper_bgcolor='#0b1421',plot_bgcolor='#0b1421',font_color='#e1edf8',xaxis=dict(range=[-1.13,1.13],visible=False,scaleanchor='y',scaleratio=1),yaxis=dict(range=[-1.13,1.13],visible=False),legend=dict(orientation='h',y=-.02))
 st.caption('Haz clic en dos polos para medir su separación angular. También puedes elegirlos en los desplegables.')
 if 'picked' not in st.session_state:st.session_state.picked=[]
 event=st.plotly_chart(chart,use_container_width=True,on_select='rerun',selection_mode='points',key='stereonet')
 if event and event.selection and event.selection.points:
  # El evento puede contener puntos de círculos; solo se aceptan las trazas de polos.
  picked=[]
  for pt in event.selection.points:
   name=pt.get('legendgroup') or pt.get('name')
   curve=pt.get('curve_number',pt.get('curveNumber',-1))
   if isinstance(curve,int) and 0<=curve<len(chart.data):
    trace=chart.data[curve]
    if trace.name in [e['name'] for e in entries] and 'markers' in (trace.mode or ''):picked.append(trace.name)
  if picked:st.session_state.picked=list(dict.fromkeys(picked))[-2:]
 names=[e['name'] for e in entries]
 c1,c2=st.columns(2)
 with c1:p1=st.selectbox('Polo 1',names,index=names.index(st.session_state.picked[0]) if st.session_state.picked and st.session_state.picked[0] in names else len(names)-1)
 with c2:p2=st.selectbox('Polo 2',names,index=names.index(st.session_state.picked[-1]) if len(st.session_state.picked)>1 and st.session_state.picked[-1] in names else 0)
 e1=next(e for e in entries if e['name']==p1);e2=next(e for e in entries if e['name']==p2)
 angle=angular(normal(e1['dd'],e1['dip']),normal(e2['dd'],e2['dip']))
 st.metric(f'Separación angular entre {p1} y {p2}',f'{angle:.2f}°')
 st.markdown('**Líneas de intersección entre familias (orientación geográfica)**')
 intersections=[]
 for i in range(len(systems)):
  for j in range(i+1,len(systems)):
   ori=intersection_orientation(systems[i],systems[j])
   intersections.append({'Familias':f'J{i+1} × J{j+1}','Dirección (°)':round(ori[0],1) if ori else 'Indeterminada','Inmersión (°)':round(ori[1],1) if ori else 'Indeterminada'})
 st.dataframe(pd.DataFrame(intersections),hide_index=True,use_container_width=True)
 st.caption('Estas orientaciones por sí solas no determinan si existe una cuña cinemáticamente liberable.')
 st.caption('Ángulo agudo entre los planos (equivalente a la separación mínima entre ejes de polos). No es una distancia en metros.')
 rows=[{'Discontinuidad':e['name'],'Dir. buzamiento (°)':e['dd'],'Buzamiento (°)':e['dip'],'Ángulo con referencia (°)':round(angular(normal(refdd,refdip),normal(e['dd'],e['dip'])),2)} for e in entries[:-1]]
 st.dataframe(pd.DataFrame(rows),hide_index=True,use_container_width=True)
 st.download_button('Descargar mediciones CSV',pd.DataFrame(rows).to_csv(index=False).encode('utf-8-sig'),'angulos_polos.csv','text/csv')
 st.info('Para usar una curva de B de Potvin debe verificarse qué ángulo define exactamente esa curva. El ángulo entre polos y el ángulo polo–plano no son intercambiables.')

# Cortes de resolución independiente del mallado voxel, usando planos analíticos.
# El recorte de dominio y la clasificación siguen siendo preliminares.
def slice_region(plane, position, block, nx=360, nz=300):
    r=W/2.; spring=max(0.,H-r)
    if plane=='transversal':
        h=np.linspace(-r-2.0,r+2.0,nx)
    else:
        h=np.linspace(0,L,nx)
    z=np.linspace(-.35,H+3.,nz)
    U,Z=np.meshgrid(h,z)
    if plane=='transversal': X=U;Y=np.full_like(U,position)
    else: X=np.full_like(U,position);Y=U
    center=np.asarray(block['center'])
    # Campo negativo dentro de la celda definida por las mismas caras de discontinuidad.
    field=np.full_like(U,-1.e6,dtype=float)
    for fam in systems:
        n=normal((fam['dd']-az)%360,fam['dip'])
        base=np.dot(n,[fam['xx'],fam['yy'],H])
        for k in range(nplanes):
            offset=(k-(nplanes-1)/2)*fam['spacing']
            signed=n[0]*X+n[1]*Y+n[2]*Z-base-offset
            side=1. if (np.dot(n,center)-base-offset)>=0 else -1.
            field=np.maximum(field,-side*signed)
    # Roca exterior a la excavación; sección en herradura con piso excluido.
    outside=np.where(Z<=spring,np.abs(X)-r,np.sqrt(X*X+(Z-spring)**2)-r)
    field=np.maximum(field,-outside)
    # Dominio finito: no es una superficie geológica de cierre.
    for coord,lo,hi in ((X,xs[0],xs[-1]),(Y,ys[0],ys[-1]),(Z,zs[0],zs[-1])):
        field=np.maximum(field,lo-coord)
        field=np.maximum(field,coord-hi)
    return h,z,field

def make_slice(plane,position,block):
    h,z,field=slice_region(plane,position,block)
    fig=go.Figure()
    # Extraer isolíneas continuas, en lugar de dibujar cuadrados del mallado.
    contours=find_contours(field,0.0)
    for poly in contours:
        if len(poly)<5:continue
        xx=np.interp(poly[:,1],np.arange(len(h)),h)
        zz=np.interp(poly[:,0],np.arange(len(z)),z)
        fig.add_trace(go.Scatter(x=xx,y=zz,mode='lines',fill='toself',
            fillcolor='rgba(30,190,145,.55)',line=dict(color='#00c998',width=2.6),
            name='Región analítica',hovertemplate='%{x:.2f} m / %{y:.2f} m<extra></extra>'))
    r=W/2.;spring=max(0.,H-r)
    if plane=='transversal':
        fig.add_trace(go.Scatter(x=section[:,0],y=section[:,1],mode='lines',
            line=dict(color='#2563eb',width=5),name='Contorno de excavación'))
        fig.add_trace(go.Scatter(x=[-r,r],y=[0,0],mode='lines',
            line=dict(color='#64748b',width=2,dash='dash'),name='Solera (sin sostenimiento)'))
        xr=[-r-2,r+2];xlabel='Transversal X (m)'
    else:
        roof=spring+np.sqrt(max(0,r*r-position*position)) if abs(position)<=r else None
        if roof is not None:
            fig.add_trace(go.Scatter(x=[0,L,L,0,0],y=[0,0,roof,roof,0],
                mode='lines',fill='toself',fillcolor='rgba(59,130,246,.09)',
                line=dict(color='#2563eb',width=3),name='Excavación longitudinal'))
        xr=[0,L];xlabel='Longitudinal Y (m)'
    fig.update_layout(height=530,margin=dict(l=60,r=20,t=55,b=55),
        title=('Transversal Y' if plane=='transversal' else 'Longitudinal X')+f' = {position:.2f} m',
        paper_bgcolor='#ffffff',plot_bgcolor='#f8fafc',font=dict(color='#172b45',size=13),
        xaxis=dict(title=xlabel,range=xr,showgrid=True,gridcolor='#dce5ee',zeroline=False),
        yaxis=dict(title='Altura Z (m)',range=[-.5,H+3.2],showgrid=True,gridcolor='#dce5ee',
                   scaleanchor='x',scaleratio=1,zeroline=False),
        legend=dict(orientation='h',y=-.19),hovermode='closest')
    return fig

with tabs[2]:
 st.subheader('Cortes 2D · contornos continuos y excavación completa')
 st.caption('Secciones de los planos analíticos que delimitan la región seleccionada. No son una comprobación de liberación cinemática ni de persistencia de juntas.')
 cuty=st.slider('Posición del corte transversal Y (m)',0.,float(L),float(np.clip(selected['contact'][1],0,L)),.1)
 cutx=st.slider('Posición del corte longitudinal X (m)',float(-W/2),float(W/2),float(np.clip(selected['contact'][0],-W/2,W/2)),.1)
 ca,cb=st.columns(2)
 with ca:st.plotly_chart(make_slice('transversal',cuty,selected),use_container_width=True)
 with cb:st.plotly_chart(make_slice('longitudinal',cutx,selected),use_container_width=True)
 st.info('Los contornos son continuos e independientes de la resolución voxel. Las caras son prolongaciones de planos sin recorte por persistencia; una región cerrada en el dominio no demuestra que sea una cuña desprendible.')

with tabs[3]:
 st.subheader('Sostenimiento seleccionado')
 st.write(f'**{typ}** · Pernos {boltlen:.1f} m · patrón aproximado {boltspace:.1f} × {boltspace:.1f} m · gunita {shot*100:.0f} cm · cables {cablelen:.1f} m · malla {"sí" if mesh else "no"}')
 st.write('**Comparador preliminar de escenarios**')
 variants=pd.DataFrame([{'Alternativa':'Sin sostenimiento','Pernos':'No','Gunita (cm)':0,'Cables (m)':0},{'Alternativa':'Solo pernos','Pernos':f'{boltlen:.1f} m','Gunita (cm)':0,'Cables (m)':0},{'Alternativa':'Solo gunita','Pernos':'No','Gunita (cm)':round(shot*100,1),'Cables (m)':0},{'Alternativa':'Combinado','Pernos':f'{boltlen:.1f} m','Gunita (cm)':round(shot*100,1),'Cables (m)':cablelen}]);st.dataframe(variants,hide_index=True,use_container_width=True)
 st.warning('No se asigna resistencia ni FS a estas alternativas. Para hacerlo hacen falta mecanismos cinemáticos, áreas de contacto, propiedades resistentes, cargas y un modelo verificado de transferencia de esfuerzos.')
 st.markdown('**Interpretación de intersecciones:** un perno que atraviesa un bloque no necesariamente lo estabiliza; hay que verificar orientación, longitud anclada fuera del bloque, adherencia y capacidad.')

with tabs[4]:
 st.subheader('Cámaras mineras · análisis de orientación para Mathews–Potvin')
 st.write('La tabla del estereograma proporciona el ángulo entre el plano de cada discontinuidad y el muro o techo de referencia.')
 st.dataframe(pd.DataFrame(rows),hide_index=True,use_container_width=True)
 q=st.number_input('Q′ (índice modificado, introducido por el usuario)',.01,1000.,2.,.1)
 A=st.number_input('Factor A (introducido por el usuario)',0.,5.,1.,.05)
 B=st.number_input('Factor B (obtenido de curva Potvin, introducido por el usuario)',0.,5.,.5,.05)
 C=st.number_input('Factor C (introducido por el usuario)',0.,20.,2.,.1)
 st.metric('Número de estabilidad modificado N′',f'{q*A*B*C:.3f}')
 st.info('N′=Q′×A×B×C. Esta versión NO infiere automáticamente B de los ángulos: requiere seleccionar y digitalizar una curva bibliográfica identificada y verificar su definición angular. Tampoco evalúa el radio hidráulico ni traza límites de estabilidad.')

with tabs[5]:
 st.subheader('Sensibilidad geométrica · escenarios ilustrativos')
 st.caption('Variación del volumen y del peso por escala de visualización. NO es un análisis de factor de seguridad ni cambia las discontinuidades reales.')
 multipliers=np.arange(.5,1.51,.1)
 df_sens=pd.DataFrame({'Escala visual':np.round(multipliers,2),'Volumen visual (m³)':selected['volume']*multipliers**3,'Peso visual (kN)':selected['volume']*gamma*multipliers**3})
 st.line_chart(df_sens.set_index('Escala visual')[['Volumen visual (m³)']],height=300)
 st.dataframe(df_sens.round(3),hide_index=True,use_container_width=True)
 st.download_button('Descargar sensibilidad CSV',df_sens.to_csv(index=False).encode('utf-8-sig'),'sensibilidad_geometrica_v162.csv','text/csv')
 st.info('No se reproduce el FS simplificado de GeologgIA: la estabilidad requiere comprobar cinemática, fuerzas y sostenimiento efectivo.')

with tabs[6]:
 data={'version':'17.1 experimental','geometria':{'W':W,'H':H,'L':L,'azimut':az,'gamma':gamma},'sistemas':systems,'sostenimiento':{'tipo':typ,'pernos_m':boltlen,'separacion_m':boltspace,'gunita_m':shot,'cables_m':cablelen,'malla':mesh},'bloques':visible,'mediciones':rows,'advertencia':'Prototipo sin FS ni verificación cinemática'}
 st.download_button('Guardar proyecto / resultados JSON',json.dumps(data,ensure_ascii=False,indent=2).encode('utf-8'),'geotecnia_mina_v161.json','application/json')
 st.download_button('Descargar bloques CSV',pd.DataFrame(visible).to_csv(index=False).encode('utf-8-sig'),'bloques_v161.csv','text/csv')
 st.caption('Exportaciones de datos geométricos preliminares. No constituyen informe de diseño.')


with tabs[7]:
 st.subheader('Motor v17.2 · componentes conectadas y bóveda curva')
 st.warning('MOTOR EXPERIMENTAL: la excavación se evalúa con su perfil curvo, pero el volumen se ESTIMA numéricamente. Una celda de planos puede contener varias componentes. No hay persistencia finita, prueba cinemática ni factor de seguridad. NO usar para dimensionamiento.')
 st.write(f"Bloque de referencia: **B{selected['id']}**, {selected['zone']} · volumen voxelizado {selected['volume']:.4f} m³")

 st.markdown('### Separación de componentes conectadas (nuevo)')
 st.caption('Se discretiza la celda convexa recortada por la herradura y se identifican regiones de roca conectadas. La componente se asocia al centro del bloque voxelizado. Una misma celda puede contener varias componentes.')
 resolutions=st.selectbox('Precisión del análisis de componentes', ['Rápida (36/54/72)', 'Alta (48/72/96)'],index=0,key='v172_resolution')
 if st.button('Identificar componentes de la región seleccionada',key='v172_components'):
  with st.spinner('Separando componentes de roca y comprobando resolución...'):
   rr=(36,54,72) if resolutions.startswith('Rápida') else (48,72,96)
   comp=connected_curved_components(selected,systems,nplanes,W,H,L,az,resolutions=rr)
  if comp['status']=='ok':
   ca,cb,cc=st.columns(3)
   ca.metric('Volumen componente asignada',f"{comp['selected_component_volume']:.4f} m³")
   cb.metric('Volumen total de celda',f"{comp['total_volume']:.4f} m³")
   cc.metric('Componentes separadas',str(len(comp['components'])))
   st.write(f"Asociación con B{selected['id']}: **{comp['association']}** · distancia al centro: {comp['association_distance']:.3f} m")
   if comp['association']!='centroid_inside_component':
    st.warning('El centro del bloque no cae dentro de una componente del nuevo mallado. Se ha asignado la más próxima: correspondencia NO demostrada.')
   st.dataframe(pd.DataFrame(comp['progress']).round(5),hide_index=True,use_container_width=True)
   st.markdown('**Componentes de la celda en la resolución final**')
   st.dataframe(pd.DataFrame(comp['components']).round(5),hide_index=True,use_container_width=True)
   if comp['components']:
    try:
     m=comp['component_mask']
     padded=np.pad(m.astype(np.uint8),1)
     verts,tri,_,_=marching_cubes(padded,level=.5,spacing=(comp['grid_step'],)*3)
     verts += comp['grid_origin'] - .5*comp['grid_step']
     fig=go.Figure(go.Mesh3d(x=verts[:,0],y=verts[:,1],z=verts[:,2],
                i=tri[:,0],j=tri[:,1],k=tri[:,2],color='#38d9a9',opacity=.95,flatshading=True,
                name=f"Componente B{selected['id']}"))
     fig.update_layout(height=560,paper_bgcolor='#0b1421',font_color='white',
           scene=dict(bgcolor='#0b1421',aspectmode='data',
           xaxis=dict(title='X (m)',color='white',nticks=4),
           yaxis=dict(title='Y (m)',color='white',nticks=4),
           zaxis=dict(title='Z (m)',color='white',nticks=4)))
     st.plotly_chart(fig,use_container_width=True)
    except (ValueError,RuntimeError) as exc:
     st.warning(f'No se pudo dibujar la componente: {exc}')
   st.warning('El volumen sigue dependiendo de la resolución. No es un volumen geométrico exacto ni valida la liberación de la cuña.')
  else:
   st.error(f"No se pudieron identificar componentes: {comp['status']}")
 st.divider()
 st.markdown('### Comparación de la celda completa con muestreo Sobol (v17.1)')

 c1,c2=st.columns(2)
 with c1:
  power=st.select_slider('Precisión de muestreo (potencia de 2)',options=[12,13,14,15,16,17],value=16,key='v171_power')
 with c2:
  repeats=st.select_slider('Repeticiones independientes',options=[2,4,6,8],value=4,key='v171_repeats')
 st.caption('El volumen estimado corresponde a toda la porción de la celda convexa que queda fuera de la excavación, no necesariamente a una única región conectada B.')
 if st.button('Reconstruir y comparar con bóveda curva',key='rebuild_v171'):
  with st.spinner('Reconstruyendo celda y estimando volumen con muestreo Sobol...'):
   result=reconstruct_curved_block(selected,systems,nplanes,W,H,L,az,powers=(max(10,power-4),max(11,power-2),power),repeats=repeats)
  if result['status']=='ok':
   vertices=result['vertices'];faces=result['faces']
   a,b,c,d=st.columns(4)
   a.metric('Celda convexa completa',f"{result['volume_convex_cell']:.4f} m³")
   b.metric('Exterior a bóveda curva (estimado)',f"{result['volume_curved_estimate']:.4f} m³")
   c.metric('Volumen voxelizado',f"{selected['volume']:.4f} m³")
   d.metric('Diferencia estimada',f"{100*result['relative_difference']:+.1f}%")
   st.caption(f"Dispersión entre {repeats} réplicas: {result['replicate_sd']:.5f} m³. Es una medida de variabilidad del muestreo, no de error geométrico total.")
   if abs(result['relative_difference'])>.2:
    st.error('DISCREPANCIA IMPORTANTE: la celda y la región voxelizada podrían no representar la misma componente, o el modelo voxelizado podría ser impreciso. No aceptar este volumen como validado.')
   else:
    st.info('Concordancia numérica preliminar. No demuestra identidad geométrica, continuidad ni liberación de la cuña.')
   st.markdown('**Convergencia del volumen numérico**')
   st.dataframe(pd.DataFrame(result['progress']).round(5),hide_index=True,use_container_width=True)
   st.markdown('**Discontinuidades y límites de la celda**')
   st.dataframe(pd.DataFrame(result['faces_diagnostic']),hide_index=True,use_container_width=True)
   st.caption('Las caras mostradas a continuación corresponden a la celda convexa completa ANTES de recortarla con la bóveda curva. No son la malla final de la cuña.')
   g=go.Figure()
   g.add_trace(go.Mesh3d(x=vertices[:,0],y=vertices[:,1],z=vertices[:,2],i=faces[:,0],j=faces[:,1],k=faces[:,2],color='#38d9a9',opacity=.45,flatshading=True,name='Celda completa'))
   g.add_trace(go.Scatter3d(x=vertices[:,0],y=vertices[:,1],z=vertices[:,2],mode='markers',marker=dict(color='white',size=4),name='Vértices'))
   g.update_layout(height=620,paper_bgcolor='#0b1421',font_color='white',scene=dict(bgcolor='#0b1421',aspectmode='data',xaxis=dict(title='X (m)',color='white',nticks=4),yaxis=dict(title='Y (m)',color='white',nticks=4),zaxis=dict(title='Z (m)',color='white',nticks=4)))
   st.plotly_chart(g,use_container_width=True)
   df_out=pd.DataFrame(result['progress'])
   st.download_button('Exportar convergencia CSV',df_out.to_csv(index=False).encode('utf-8-sig'),f"B{selected['id']}_convergencia_v17_1.csv",'text/csv')
   st.info('Pendiente: identificar componentes conectadas, persistencia finita y construir la malla exacta del recorte curvo. FS no calculado.')
  else:
   st.error(f"No se obtuvo una celda válida: {result['status']}")
   st.caption(result.get('detail',''))
