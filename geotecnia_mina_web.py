import io, json, math
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from scipy import ndimage as ndi
from skimage.measure import marching_cubes

st.set_page_config(page_title='Geotecnia Mina v15', page_icon='⛏️', layout='wide', initial_sidebar_state='expanded')
st.markdown('''<style>
.stApp,[data-testid="stSidebar"]{background:#0b1421;color:#e6edf7}
[data-testid="stMetric"]{background:#142438;border:1px solid #2b435d;border-radius:12px;padding:12px}
[data-testid="stTabs"] button{font-weight:650}
.block-container{padding-top:1.1rem;max-width:1600px}
h1,h2,h3{color:#e9f4ff!important}
</style>''', unsafe_allow_html=True)
COL=['#36d6bb','#68a8ff','#eeb65d','#d58bf0','#ef7b88','#b5d66b']
st.title('⛏️ GEOTECNIA MINA  ·  v15')
st.caption('Galerías y cámaras · cuñas geométricas · estereograma interactivo · medición angular · sostenimiento')
st.warning('PROTOTIPO GEOMÉTRICO. Los bloques se generan mediante planos infinitos discretizados; la persistencia no recorta todavía los planos. Bloque cerrado ≠ cuña liberable. No se calcula FS validado ni se dimensiona sostenimiento. No usar para decisiones de seguridad.')

with st.sidebar:
 st.header('⚙️ Geometría de la galería')
 W=st.number_input('Ancho (m)',2.,20.,5.8,.1)
 H=st.number_input('Alto (m)',2.,20.,5.8,.1)
 L=st.number_input('Longitud analizada (m)',5.,40.,15.,1.)
 az=st.number_input('Azimut del eje (° desde N)',0.,359.9,0.,1.)
 gamma=st.number_input('Peso específico roca (kN/m³)',15.,40.,26.4,.1)
 res=st.slider('Resolución de discretización',36,90,54,6)
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
 show_bolts=st.checkbox('Mostrar pernos/cables 3D',True)
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

with st.spinner('Calculando regiones de roca adyacentes a la excavación...'):
 xs,ys,zs,labels,blocks=compute(W,H,L,az,res,nplanes,json.dumps(systems,sort_keys=True))
visible=[b for b in blocks if b['volume']>=minvol]
closed=[b for b in visible if not b['open']]
if not visible:st.error('No se han identificado regiones con los parámetros actuales.');st.stop()

m1,m2,m3,m4=st.columns(4)
m1.metric('Regiones en contacto',len(visible));m2.metric('Cerradas en dominio',len(closed));m3.metric('Abiertas al borde',len(visible)-len(closed));m4.metric('FS validado','No calculado')
tabs=st.tabs(['🧊 Cuñas 3D','🧭 Estereograma y mediciones','📏 Cortes 2D','🧱 Sostenimiento','🏗️ Cámaras · Potvin','📥 Exportación'])

with tabs[0]:
 c1,c2,c3=st.columns([1.2,1,1])
 with c1:
  options=sorted(visible,key=lambda b:(b['open'],-b['volume']))
  selected=st.selectbox('Región seleccionada',options,format_func=lambda b:f"B{b['id']} · {b['zone']} · {b['volume']:.2f} m³ · {'abierta' if b['open'] else 'cerrada'}")
 with c2:mode=st.radio('Mostrar', ['Solo seleccionada','Todas las cerradas','Todas las regiones'],horizontal=False)
 with c3:
  scale=st.slider('Escala de análisis de cuña',.5,2.,1.,.05)
  focus=st.checkbox('Acercar cámara a la cuña',True)
 st.caption('Escala: transforma una copia visual de la región seleccionada alrededor de su centro; el volumen de escenario se multiplica por escala³. No modifica los planos ni representa una cuña geológicamente demostrada.')
 a,b,c,d=st.columns(4)
 a.metric('Volumen geométrico',f"{selected['volume']:.3f} m³")
 b.metric('Peso geométrico',f"{selected['volume']*gamma:.1f} kN")
 c.metric('Volumen escalado',f"{selected['volume']*scale**3:.3f} m³")
 d.metric('Peso escalado',f"{selected['volume']*gamma*scale**3:.1f} kN")
 fig=go.Figure();ymin=max(0,selected['contact'][1]-4) if focus else 0;ymax=min(L,selected['contact'][1]+4) if focus else L
 for y in [ymin,ymax]:fig.add_trace(go.Scatter3d(x=section[:,0],y=np.full(len(section),y),z=section[:,1],mode='lines',line=dict(color='#a2b6ca',width=5),showlegend=False,hoverinfo='skip'))
 for x,z in section[::12]:fig.add_trace(go.Scatter3d(x=[x,x],y=[ymin,ymax],z=[z,z],mode='lines',line=dict(color='#5c768f',width=2),showlegend=False,hoverinfo='skip'))
 shown=[selected] if mode=='Solo seleccionada' else ([v for v in visible if not v['open']] if mode=='Todas las cerradas' else visible)
 shown=sorted(shown,key=lambda b:b['volume'],reverse=True)[:35]
 for j,v in enumerate(shown):
  result=surface(labels,v['id'],xs[1]-xs[0],ys[1]-ys[0],zs[1]-zs[0],xs[0],ys[0],zs[0]);
  if result is None:continue
  verts,faces=result
  if v['id']==selected['id'] and scale!=1:verts=(verts-np.array(v['center']))*scale+np.array(v['center'])
  color='#36d6bb' if v['id']==selected['id'] else COL[j%len(COL)]
  fig.add_trace(go.Mesh3d(x=verts[:,0],y=verts[:,1],z=verts[:,2],i=faces[:,0],j=faces[:,1],k=faces[:,2],color=color,opacity=.88 if v['id']==selected['id'] else .35,name=f"B{v['id']}",flatshading=True,hovertemplate=f"B{v['id']}<br>{v['volume']:.2f} m³<extra></extra>"))
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
 if show_bolts:
  for p,out,length,kind in bolts():
   q=p+out*length
   if focus and not(ymin-.2<=p[1]<=ymax+.2):continue
   hit,tail=intersection(p,q,selected['id']);hits.append((kind,hit,tail))
   fig.add_trace(go.Scatter3d(x=[p[0],q[0]],y=[p[1],q[1]],z=[p[2],q[2]],mode='lines',line=dict(color=('#e8a34d' if hit else '#4c789e') if kind=='Perno' else '#c68bf0',width=5 if hit else 2),showlegend=False,hoverinfo='skip'))
 fig.update_layout(height=700,margin=dict(l=0,r=0,t=5,b=0),paper_bgcolor='#0b1421',font_color='#d9e8f6',scene=dict(bgcolor='#0b1421',xaxis=dict(title='Transversal X (m)',backgroundcolor='#0b1421',color='#9ab0c6'),yaxis=dict(title='Eje Y (m)',backgroundcolor='#0b1421',color='#9ab0c6',range=[ymin,ymax]),zaxis=dict(title='Altura Z (m)',backgroundcolor='#0b1421',color='#9ab0c6'),aspectmode='data',camera=dict(eye=dict(x=1.15,y=-1.3,z=.8),projection=dict(type='orthographic'))),showlegend=False)
 st.plotly_chart(fig,use_container_width=True)
 st.info('La clasificación abierta/cerrada se refiere exclusivamente a los límites del dominio discretizado. No se ha comprobado la liberación cinemática ni la estabilidad.')
 if show_bolts:
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
 st.caption('Ángulo agudo entre los planos (equivalente a la separación mínima entre ejes de polos). No es una distancia en metros.')
 rows=[{'Discontinuidad':e['name'],'Dir. buzamiento (°)':e['dd'],'Buzamiento (°)':e['dip'],'Ángulo con referencia (°)':round(angular(normal(refdd,refdip),normal(e['dd'],e['dip'])),2)} for e in entries[:-1]]
 st.dataframe(pd.DataFrame(rows),hide_index=True,use_container_width=True)
 st.download_button('Descargar mediciones CSV',pd.DataFrame(rows).to_csv(index=False).encode('utf-8-sig'),'angulos_polos.csv','text/csv')
 st.info('Para usar una curva de B de Potvin debe verificarse qué ángulo define exactamente esa curva. El ángulo entre polos y el ángulo polo–plano no son intercambiables.')

with tabs[2]:
 st.subheader('Cortes vinculados a la región seleccionada')
 cuty=st.slider('Posición del corte transversal Y (m)',0.,float(L),float(np.clip(selected['contact'][1],0,L)),.1)
 iy=int(np.argmin(abs(ys-cuty)));ix=int(np.argmin(abs(xs-selected['contact'][0])))
 ca,cb=st.columns(2)
 def heat(z,x,y,title,xlabel,ylabel):
  f=go.Figure();f.add_trace(go.Heatmap(z=z,x=x,y=y,colorscale=[[0,'#142438'],[1,'#36d6bb']],showscale=False,hovertemplate='X=%{x:.2f}, Y=%{y:.2f}<extra></extra>'))
  f.update_layout(title=title,height=430,paper_bgcolor='#0b1421',plot_bgcolor='#0b1421',font_color='#d9e8f6',xaxis_title=xlabel,yaxis_title=ylabel,margin=dict(l=20,r=10,t=50,b=35));return f
 with ca:
  f=heat(np.where(labels[:,iy,:]==selected['id'],1.,np.nan).T,xs,zs,f'Corte transversal Y={ys[iy]:.2f} m','Transversal (m)','Altura (m)')
  f.add_trace(go.Scatter(x=section[:,0],y=section[:,1],mode='lines',line=dict(color='#eeb65d',width=4),name='Excavación'))
  st.plotly_chart(f,use_container_width=True)
 with cb:
  f=heat(np.where(labels[ix,:,:]==selected['id'],1.,np.nan).T,ys,zs,f'Corte longitudinal X={xs[ix]:.2f} m','Longitud (m)','Altura (m)')
  st.plotly_chart(f,use_container_width=True)
 st.caption('El corte utiliza las mismas etiquetas voxelizadas que el modelo 3D. Los bordes escalonados dependen de la resolución.')

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
 data={'version':'15','geometria':{'W':W,'H':H,'L':L,'azimut':az,'gamma':gamma},'sistemas':systems,'sostenimiento':{'tipo':typ,'pernos_m':boltlen,'separacion_m':boltspace,'gunita_m':shot,'cables_m':cablelen,'malla':mesh},'bloques':visible,'mediciones':rows,'advertencia':'Prototipo sin FS ni verificación cinemática'}
 st.download_button('Descargar resultados JSON',json.dumps(data,ensure_ascii=False,indent=2).encode('utf-8'),'geotecnia_mina_v15.json','application/json')
 st.download_button('Descargar bloques CSV',pd.DataFrame(visible).to_csv(index=False).encode('utf-8-sig'),'bloques_v15.csv','text/csv')
 st.caption('Exportaciones de datos geométricos preliminares. No constituyen informe de diseño.')
