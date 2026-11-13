import io, json, math
import numpy as np
import streamlit as st
import plotly.graph_objects as go

st.set_page_config(page_title='Geotecnia Mina Web',layout='wide')
st.title('Geotecnia Mina · secciones 2D y visor 3D · v4')
st.warning('VERSIÓN 4 · La vista 2D representa el sostenimiento y la 3D muestra planos e intersecciones ilustrativas. NO se calculan cuñas cerradas ni FS 3D. El FS inferior corresponde solo a un bloque hipotético definido manualmente. NO usar para dimensionamiento.')

with st.sidebar:
    st.header('Galería')
    width=st.number_input('Ancho (m)',0.5,30.0,5.8)
    height=st.number_input('Alto (m)',0.5,30.0,5.8)
    length=st.number_input('Longitud (m)',1.0,300.0,15.0)
    az=st.number_input('Azimut del eje (°)',0.0,359.9,0.0)
    slope=st.number_input('Pendiente del eje (°)',-45.0,45.0,0.0)
    uploaded=st.file_uploader('Importar sección DXF (opcional)',type=['dxf'])
    st.caption('DXF: LINE, ARC, CIRCLE y LWPOLYLINE/POLYLINE. Se interpreta en metros; se muestra como sección local.')

st.subheader('Tres sistemas de discontinuidades')
systems=[]
cols=st.columns(3)
for i,col in enumerate(cols):
    with col:
        st.markdown(f'**Sistema {i+1}**')
        dipdir=st.number_input('Dirección buzamiento (°)',0.0,359.9,float((i*60+30)%360),key=f'dd{i}')
        dip=st.number_input('Buzamiento (°)',0.0,90.0,60.0,key=f'dip{i}')
        spacing=st.number_input('Espaciamiento (m)',0.01,100.0,1.5,key=f'sp{i}')
        persistence=st.number_input('Persistencia (m)',0.1,1000.0,5.0,key=f'pe{i}')
        cohesion=st.number_input('Cohesión de junta (MPa)',0.0,100.0,0.05,key=f'c{i}')
        phi=st.number_input('Fricción (°)',0.0,89.0,30.0,key=f'phi{i}')
        jrc=st.number_input('JRC',0.0,30.0,10.0,key=f'jrc{i}')
        jcs=st.number_input('JCS (MPa)',0.01,500.0,30.0,key=f'jcs{i}')
        residual=st.number_input('Fricción residual (°)',0.0,89.0,25.0,key=f'res{i}')
        pressure=st.number_input('Presión agua (MPa)',0.0,100.0,0.0,key=f'u{i}')
        method=st.selectbox('Criterio resistente', ['Mohr-Coulomb','Barton-Bandis','Patton (rama de asperezas)','Hoek-Brown (macizo, no junta)'],key=f'method{i}')
        rough=st.number_input('Ángulo de asperezas Patton (°)',0.0,45.0,10.0,key=f'rough{i}')
        ucs=st.number_input('UCS macizo (MPa)',0.01,1000.0,52.6,key=f'ucs{i}')
        mi=st.number_input('mi Hoek-Brown',0.1,100.0,20.0,key=f'mi{i}')
        gsi=st.number_input('GSI',0.0,100.0,50.0,key=f'gsi{i}')
        d=st.number_input('D',0.0,1.0,0.0,key=f'd{i}')
        systems.append(dict(dip_direction=dipdir,dip=dip,spacing=spacing,persistence=persistence,cohesion=cohesion,phi=phi,jrc=jrc,jcs=jcs,residual_phi=residual,water_pressure=pressure,method=method,rough=rough,ucs=ucs,mi=mi,gsi=gsi,d=d))

# Coordinates: x horizontal across section, y tunnel axis, z vertical.
def to_world(x,y,z):
    a=math.radians(az); p=math.radians(slope)
    # horizontal local x perpendicular to axis, y along axis
    return (x*math.cos(a)+y*math.sin(a)*math.cos(p), -x*math.sin(a)+y*math.cos(a)*math.cos(p), z+y*math.sin(p))

fig=go.Figure()
r=width/2; wall=max(0,height-r)
xs=np.concatenate(([-r,-r],r*np.cos(np.linspace(math.pi,0,45)),[r,r]))
zs=np.concatenate(([0,wall],wall+r*np.sin(np.linspace(math.pi,0,45)),[wall,0]))
for yy in [0,length]:
    xyz=np.array([to_world(x,yy,z) for x,z in zip(xs,zs)])
    fig.add_trace(go.Scatter3d(x=xyz[:,0],y=xyz[:,1],z=xyz[:,2],mode='lines',line=dict(color='slategray',width=7),name='Sección'))
for j in range(0,len(xs),max(1,len(xs)//12)):
    a=to_world(xs[j],0,zs[j]); b=to_world(xs[j],length,zs[j])
    fig.add_trace(go.Scatter3d(x=[a[0],b[0]],y=[a[1],b[1]],z=[a[2],b[2]],mode='lines',line=dict(color='lightgray',width=2),showlegend=False))

# Three finite planes, oriented with dip/dip direction; position is illustrative.
plane_centers=[]; plane_normals=[]
for i,s in enumerate(systems):
    dd=math.radians(s['dip_direction']); dip=math.radians(s['dip'])
    strike=np.array([math.cos(dd),-math.sin(dd),0.0])
    down=np.array([math.sin(dd)*math.cos(dip),math.cos(dd)*math.cos(dip),-math.sin(dip)])
    center=np.array(to_world(0,length*(i+1)/4,height*.75))
    plane_centers.append(center)
    plane_normals.append(np.cross(strike,down)/np.linalg.norm(np.cross(strike,down)))
    scale=min(s['persistence'],length,12)/2
    corners=[center+u*scale*strike+v*scale*down for u,v in [(-1,-1),(1,-1),(1,1),(-1,1)]]
    pts=np.array(corners)
    fig.add_trace(go.Mesh3d(x=pts[:,0],y=pts[:,1],z=pts[:,2],i=[0,0],j=[1,2],k=[2,3],opacity=.35,name=f'Sistema {i+1}',hovertext=f'DD {s["dip_direction"]}°, dip {s["dip"]}°'))

# Intersección matemática de planos infinitos (NO equivale a una cuña cerrada).
st.subheader('Intersecciones de los tres planos')
st.caption('Los planos se sitúan de forma ilustrativa en el espacio. Para obtener cuñas reales es imprescindible definir la posición de cada discontinuidad y su extensión; una orientación media por familia no basta.')
for i,j in [(0,1),(0,2),(1,2)]:
    ni,nj=plane_normals[i],plane_normals[j]
    direction=np.cross(ni,nj)
    norm=np.linalg.norm(direction)
    if norm<1e-7:
        st.info(f'Sistemas {i+1} y {j+1}: planos paralelos o coincidentes.')
        continue
    direction=direction/norm
    A=np.stack([ni,nj,direction])
    b=np.array([np.dot(ni,plane_centers[i]),np.dot(nj,plane_centers[j]),0.0])
    origin=np.linalg.solve(A,b)
    extent=min(length,12.0)
    pts=np.stack([origin-extent*direction,origin+extent*direction])
    fig.add_trace(go.Scatter3d(x=pts[:,0],y=pts[:,1],z=pts[:,2],mode='lines',line=dict(width=7,dash='dash'),name=f'Intersección {i+1}-{j+1}'))

if uploaded is not None:
    try:
        import ezdxf
        doc=ezdxf.read(io.StringIO(uploaded.getvalue().decode('utf-8-sig',errors='replace')))
        msp=doc.modelspace(); n=0
        for ent in msp:
            kind=ent.dxftype(); pts=[]
            if kind=='LINE':
                pts=[ent.dxf.start,ent.dxf.end]
            elif kind in ('LWPOLYLINE','POLYLINE'):
                pts=list(ent.get_points('xy') if kind=='LWPOLYLINE' else [(v.dxf.location.x,v.dxf.location.y) for v in ent.vertices])
                pts=[(p[0],p[1]) for p in pts]
                if ent.is_closed and pts: pts.append(pts[0])
            elif kind in ('ARC','CIRCLE'):
                center=ent.dxf.center; radius=ent.dxf.radius
                angles=np.linspace(ent.dxf.start_angle,ent.dxf.end_angle if ent.dxf.end_angle>ent.dxf.start_angle else ent.dxf.end_angle+360,60) if kind=='ARC' else np.linspace(0,360,100)
                pts=[(center.x+radius*math.cos(math.radians(t)),center.y+radius*math.sin(math.radians(t))) for t in angles]
            if len(pts)>1:
                xyz=np.array([to_world(float(pt[0]),0,float(pt[1])) for pt in pts])
                fig.add_trace(go.Scatter3d(x=xyz[:,0],y=xyz[:,1],z=xyz[:,2],mode='lines',line=dict(color='orange',width=5),name='DXF' if n==0 else None,showlegend=n==0)); n+=1
        st.success(f'DXF importado: {n} entidades geométricas representadas. Comprueba escala, origen y unidades.')
    except Exception as exc:
        st.error(f'No se pudo leer el DXF: {exc}. Exporta como DXF ASCII desde AutoCAD.')
# Escala y cámara: aproximar el objeto en vez de encuadrar los extremos
# de líneas de intersección de planos infinitos.
fig.update_layout(
    height=680, margin=dict(l=0,r=0,b=0,t=0),
    scene=dict(aspectmode='manual',aspectratio=dict(x=1,y=max(1.0,length/width*.7),z=1),
        xaxis=dict(title='X (m)',range=[-width*1.1,width*1.1]),
        yaxis=dict(title='Y (m)',range=[-length*.2,length*1.2]),
        zaxis=dict(title='Z (m)',range=[-height*.5,height*1.7]),
        camera=dict(eye=dict(x=1.5,y=1.65,z=.95),projection=dict(type='perspective'))))
st.plotly_chart(fig,use_container_width=True,config={'scrollZoom':True,'displaylogo':False})
st.caption('Amplía la figura con la rueda del ratón. Las líneas discontinuas son intersecciones de planos infinitos; no indican necesariamente cuñas inestables.')

st.subheader('Resistencia por sistema de discontinuidades')
st.caption('Selecciona el criterio de cada sistema arriba. Las tensiones son valores introducidos por el usuario; no se deducen automáticamente de una cuña.')
normal=st.number_input('Tensión normal total de evaluación (MPa)',0.0,500.0,1.0)
for i,s in enumerate(systems):
    method=s['method']; effective=max(0.0,normal-s['water_pressure'])
    with st.expander(f'Sistema {i+1} · {method}',expanded=True):
        if method=='Mohr-Coulomb':
            tau=s['cohesion']+effective*math.tan(math.radians(s['phi']))
            st.metric('Resistencia al corte (MPa)',f'{tau:.4f}')
        elif method=='Barton-Bandis':
            if effective<=0:
                st.info('Se necesita tensión normal efectiva positiva.')
            else:
                angle=s['residual_phi']+s['jrc']*math.log10(s['jcs']/effective)
                if not (0<angle<89):
                    st.error('Parámetros fuera del rango admisible para esta formulación simplificada.')
                else:
                    tau=effective*math.tan(math.radians(angle))
                    st.metric('Resistencia al corte estimada (MPa)',f'{tau:.4f}')
                    st.caption('Sin corrección de escala; verificar el dominio de aplicación.')
        elif method.startswith('Patton'):
            angle=s['phi']+s['rough']
            if angle>=89:
                st.error('Ángulo total no admisible.')
            else:
                st.metric('Resistencia al corte idealizada (MPa)',f'{effective*math.tan(math.radians(angle)):.4f}')
                st.caption('Solo rama de deslizamiento sobre asperezas; no representa su rotura.')
        else:
            ucs,mi,gsi,d=s['ucs'],s['mi'],s['gsi'],s['d']
            mb=mi*math.exp((gsi-100)/(28-14*d))
            shb=math.exp((gsi-100)/(9-3*d))
            a=.5+(math.exp(-gsi/15)-math.exp(-20/3))/6
            sigma1=normal+ucs*(mb*normal/ucs+shb)**a
            st.metric('σ1 a rotura del macizo (MPa)',f'{sigma1:.4f}')
            st.warning('Hoek-Brown no es un criterio de resistencia al corte de una discontinuidad.')

payload=dict(section=dict(width=width,height=height,length=length,azimuth=az,slope=slope),systems=systems)
st.download_button('Descargar parámetros JSON',data=json.dumps(payload,indent=2,ensure_ascii=False),file_name='proyecto_geotecnia.json',mime='application/json')

st.divider()
st.header('Sostenimiento y FS preliminar de bloque definido por el usuario')
st.error('Este módulo NO calcula cuñas geométricas a partir de los tres planos. El FS mostrado es una estimación de equilibrio límite de un bloque con geometría y modo de deslizamiento INTRODUCIDOS por el usuario. No es válido para dimensionar sostenimiento ni sustituye a UnWedge.')

support_presets={
 'Sin sostenimiento': (0.0,0.0,0.0,0.0,0.0),
 'ST-I · pernos 2 × 2 m, gunita 3 cm':(2.0,2.0,2.4,324.0,3.0),
 'ST-II · configuración editable':(2.0,2.0,2.4,324.0,3.0),
 'ST-III · pernos 1,5 × 1,5 m, gunita 8 cm':(1.5,1.5,2.4,324.0,8.0),
 'ST-IV superficial · cables 8 m, gunita 12 cm':(1.5,1.5,2.4,324.0,12.0),
 'ST-IV profundo · cables 10 m, gunita 15 cm':(1.5,1.5,2.4,324.0,15.0),
 'Personalizado':(1.5,1.5,2.4,324.0,8.0)
}
preset=st.selectbox('Tipo de sostenimiento',list(support_presets))
px,py,bl,cap,shot=support_presets[preset]
a,b,c=st.columns(3)
with a:
    sx=st.number_input('Separación transversal pernos (m)',min_value=0.0,value=px,step=0.1)
    sy=st.number_input('Separación longitudinal pernos (m)',min_value=0.0,value=py,step=0.1)
    bolt_length=st.number_input('Longitud pernos (m)',min_value=0.0,value=bl,step=0.1)
    bolt_capacity=st.number_input('Capacidad nominal por perno (kN)',min_value=0.0,value=cap,step=10.0)
with b:
    shotcrete=st.number_input('Espesor de gunita (cm)',min_value=0.0,value=shot,step=1.0)
    cable_length=st.number_input('Longitud cables (m)',min_value=0.0,value=8.0 if 'superficial' in preset else 10.0 if 'profundo' in preset else 0.0,step=1.0)
    cable_capacity=st.number_input('Capacidad nominal cable (kN)',min_value=0.0,value=350.0,step=10.0)
    st.checkbox('Malla metálica',value=('ST-III' in preset or 'ST-IV' in preset))
with c:
    st.info('Los valores nominales y espesores se registran, pero NO se convierten automáticamente en resistencia movilizada. La aportación efectiva requiere conocer intersección, anclaje, orientación y modo de fallo.')

st.subheader('Sección transversal 2D · disposición del sostenimiento')
st.caption('Dibujo de distribución geométrica, NO comprobación estructural. Se muestra un anillo en la sección; la separación longitudinal se define arriba. El piso no se sostiene.')
show_bolts=st.checkbox('Mostrar pernos',value=True,key='view_bolts')
show_cables=st.checkbox('Mostrar cables',value=True,key='view_cables')
show_mesh=st.checkbox('Mostrar malla',value=True,key='view_mesh')
show_shot=st.checkbox('Mostrar gunita',value=True,key='view_shot')
show_labels=st.checkbox('Etiquetar longitudes',value=False,key='view_labels')
# Sección de herradura aproximada: hastiales verticales y bóveda semicircular.
r=width/2.0
wall=max(0.0,height-r)
# Trayectoria desde el arranque izquierdo hasta el derecho, sin el piso.
wall_n=24; arch_n=120
xl=np.full(wall_n,-r); zl=np.linspace(0,wall,wall_n)
theta=np.linspace(math.pi,0,arch_n)
xa=r*np.cos(theta); za=wall+r*np.sin(theta)
xr=np.full(wall_n,r); zr=np.linspace(wall,0,wall_n)
xc=np.concatenate([xl,xa,xr]); zc=np.concatenate([zl,za,zr])
# Normales exteriores al contorno, mediante derivada respecto a longitud de arco.
seg=np.sqrt(np.diff(xc)**2+np.diff(zc)**2)
cum=np.concatenate([[0],np.cumsum(seg)])
# Se recorre del hastial izquierdo hacia arriba, por la clave, y baja por el derecho.
# Normal exterior = (-dz/ds, dx/ds) para este sentido horario.
dx=np.gradient(xc,cum); dz=np.gradient(zc,cum)
nx=-dz; nz=dx
norm=np.hypot(nx,nz); nx/=norm; nz/=norm
fig2=go.Figure()
fig2.add_trace(go.Scatter(x=xc,y=zc,mode='lines',name='Excavación',line=dict(color='#334155',width=5)))
fig2.add_trace(go.Scatter(x=[-r,r],y=[0,0],mode='lines',name='Solera (sin soporte)',line=dict(color='#94a3b8',dash='dot',width=2)))
if show_shot and shotcrete>0:
    thickness=shotcrete/100.0
    fig2.add_trace(go.Scatter(x=xc+nx*thickness,y=zc+nz*thickness,mode='lines',name=f'Gunita {shotcrete:g} cm',line=dict(color='#e87924',width=9)))
if show_mesh and ('ST-III' in preset or 'ST-IV' in preset or preset=='Personalizado'):
    fig2.add_trace(go.Scatter(x=xc+nx*.07,y=zc+nz*.07,mode='lines',name='Malla (esquemática)',line=dict(color='#22a06b',dash='dash',width=3)))
# Reparto sobre longitud de contorno, incluyendo extremos, sin inventar recuentos reales.
perimeter=float(cum[-1]); bolt_count=0
if sx>0 and bolt_length>0 and show_bolts and preset!='Sin sostenimiento':
    positions=np.arange(sx/2,perimeter,sx)
    for pos in positions:
        xx=np.interp(pos,cum,xc); zz=np.interp(pos,cum,zc)
        ux=np.interp(pos,cum,nx); uz=np.interp(pos,cum,nz)
        fig2.add_trace(go.Scatter(x=[xx,xx+bolt_length*ux],y=[zz,zz+bolt_length*uz],mode='lines+markers',
            line=dict(color='#2563eb',width=3),marker=dict(size=[6,3]),name='Pernos 2,4 m' if bolt_count==0 else None,
            showlegend=bolt_count==0,hovertemplate=f'Perno L={bolt_length:.2f} m<extra></extra>'))
        if show_labels: fig2.add_annotation(x=xx+bolt_length*ux,y=zz+bolt_length*uz,text=f'{bolt_length:g} m',showarrow=False,font=dict(size=9))
        bolt_count+=1
# Los cables se dibujan en una sección de referencia, no como intersecciones reales con una cuña.
if show_cables and cable_length>0 and preset!='Sin sostenimiento':
    for j,pos in enumerate([perimeter*.25,perimeter*.5,perimeter*.75]):
        xx=np.interp(pos,cum,xc); zz=np.interp(pos,cum,zc)
        ux=np.interp(pos,cum,nx); uz=np.interp(pos,cum,nz)
        fig2.add_trace(go.Scatter(x=[xx,xx+cable_length*ux],y=[zz,zz+cable_length*uz],mode='lines+markers',
            line=dict(color='#b45309',width=4,dash='dash'),marker=dict(size=[7,3]),name='Cables (ejemplo)' if j==0 else None,
            showlegend=j==0,hovertemplate=f'Cable L={cable_length:.1f} m<extra></extra>'))
max_extent=max(width,height,bolt_length,cable_length)+2
fig2.update_layout(height=670,margin=dict(l=5,r=5,t=10,b=5),
    xaxis=dict(title='Transversal X (m)',scaleanchor='y',scaleratio=1,range=[-max_extent,max_extent]),
    yaxis=dict(title='Altura Z (m)',range=[-1,max_extent+height*.5]),
    legend=dict(orientation='h',y=-.16),dragmode='pan')
st.plotly_chart(fig2,use_container_width=True,config={'scrollZoom':True,'displaylogo':False})
st.caption(f'Perímetro aproximado sin piso: {perimeter:.2f} m. Pernos dibujados por anillo: {bolt_count}. La distribución es esquemática y no considera interferencias, juntas ni anclaje efectivo.')

st.subheader('Equilibrio límite simplificado · bloque hipotético')
st.caption('Se considera deslizamiento en UN plano con fuerza normal aproximada W cos(β) y fuerza tangencial W sen(β). No se resuelven deslizamientos por dos planos, caída libre, fuerzas 3D ni contacto real con pernos.')
ca,cb,cc=st.columns(3)
with ca:
    volume=st.number_input('Volumen del bloque (m³) — introducido',min_value=0.01,value=5.0)
    unitweight=st.number_input('Peso específico (kN/m³)',min_value=1.0,value=26.4)
    beta=st.number_input('Inclinación plano de deslizamiento (°)',min_value=0.0,max_value=89.0,value=55.0)
with cb:
    area=st.number_input('Área del plano de deslizamiento (m²)',min_value=0.01,value=6.0)
    selected=st.selectbox('Sistema de discontinuidades del plano de deslizamiento',range(3),format_func=lambda i:f'Sistema {i+1}')
    effective_bolts=st.number_input('Número de pernos efectivamente anclados a la cuña',min_value=0,value=0,step=1)
with cc:
    effective_cables=st.number_input('Número de cables efectivamente anclados a la cuña',min_value=0,value=0,step=1)
    bolt_fraction=st.slider('Fracción movilizada de capacidad de anclajes (hipótesis)',0.0,1.0,0.0,0.05)
    st.caption('Valor por defecto 0: no atribuye resistencia sin verificación del anclaje.')

s=systems[selected]
W=volume*unitweight
normal_force=W*math.cos(math.radians(beta))
driving=W*math.sin(math.radians(beta))
water_force=s['water_pressure']*1000.0*area
normal_effective=max(0.0,normal_force-water_force)
sigma_eff=normal_effective/area/1000.0
method=s['method']
tau=None
if method=='Mohr-Coulomb':
    tau=s['cohesion']+sigma_eff*math.tan(math.radians(s['phi']))
elif method=='Barton-Bandis' and sigma_eff>0:
    angle=s['residual_phi']+s['jrc']*math.log10(s['jcs']/sigma_eff)
    if 0<angle<89: tau=sigma_eff*math.tan(math.radians(angle))
elif method.startswith('Patton') and s['phi']+s['rough']<89:
    tau=sigma_eff*math.tan(math.radians(s['phi']+s['rough']))

if method.startswith('Hoek'):
    st.warning('Hoek-Brown no puede utilizarse directamente como resistencia de junta para este FS. Selecciona Mohr-Coulomb, Barton-Bandis o Patton.')
elif tau is None:
    st.warning('No puede calcularse el FS con estos parámetros. Revisa la tensión efectiva y el dominio del criterio seleccionado.')
elif driving<=1e-9:
    st.warning('No existe fuerza motriz por deslizamiento en este esquema (β = 0°); FS no definido.')
else:
    shear_resistance=tau*1000.0*area
    anchorage=bolt_fraction*(effective_bolts*bolt_capacity+effective_cables*cable_capacity)
    fs0=shear_resistance/driving
    fs1=(shear_resistance+anchorage)/driving
    m1,m2,m3=st.columns(3)
    m1.metric('FS sin sostenimiento (hipotético)',f'{fs0:.2f}')
    m2.metric('FS con anclajes supuestos (hipotético)',f'{fs1:.2f}')
    m3.metric('Peso del bloque',f'{W:.1f} kN')
    st.caption(f'Resistencia al corte: {shear_resistance:.1f} kN · Fuerza motriz: {driving:.1f} kN · Aportación supuesta de anclajes: {anchorage:.1f} kN. Gunita y malla NO incluidas en el FS.')
    st.warning('FS NO VALIDADO: geometría, persistencia, presión hidráulica y resistencia movilizada deben contrastarse con un modelo de cuña tridimensional y una metodología de diseño validada.')
