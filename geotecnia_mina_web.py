import io, json, math
import numpy as np
import streamlit as st
import plotly.graph_objects as go

st.set_page_config(page_title='Geotecnia Mina Web',layout='wide')
st.title('Geotecnia Mina · visor geotécnico 3D')
st.warning('Versión exploratoria: calcula líneas de intersección entre planos, pero NO identifica cuñas cerradas ni calcula su factor de seguridad o sostenimiento. No usar para diseño.')

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
fig.update_layout(height=620,scene=dict(aspectmode='data',xaxis_title='X (m)',yaxis_title='Y (m)',zaxis_title='Z (m)'),margin=dict(l=0,r=0,b=0,t=10))
fig.update_layout(scene_camera=dict(eye=dict(x=1.65,y=1.8,z=1.15)),scene=dict(aspectmode='data',xaxis_title='X (m)',yaxis_title='Y (m)',zaxis_title='Z (m)'),height=750)
st.plotly_chart(fig,use_container_width=True)
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
