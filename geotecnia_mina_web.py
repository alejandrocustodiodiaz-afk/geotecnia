
"""Geotecnia Mina Web v10 - búsqueda preliminar de múltiples cuñas."""
import io
import json
import itertools
import math

import numpy as np
import plotly.graph_objects as go
import streamlit as st
from scipy.spatial import ConvexHull
from skimage.measure import marching_cubes

st.set_page_config(
    page_title="Geotecnia Mina v10",
    layout="wide"
)

st.title("Geotecnia Mina · v10")
st.caption(
    "Cuñas múltiples en bóveda y hastiales · "
    "vistas 3D y 2D · sostenimiento"
)
st.warning(
    "PROTOTIPO GEOMÉTRICO. Se muestran candidatos obtenidos "
    "de combinaciones de planos. No se garantiza que se hayan "
    "identificado todas las cuñas reales, ni que sean liberables. "
    "La persistencia se comprueba de forma simplificada. "
    "No se calcula FS. No utilizar para dimensionamiento."
)

with st.sidebar:
    st.header("Galería")
    W = st.number_input("Ancho (m)", 1.0, 25.0, 5.8, 0.1)
    H = st.number_input("Altura (m)", 1.0, 25.0, 5.8, 0.1)
    L = st.number_input("Longitud (m)", 2.0, 100.0, 15.0, 0.5)
    az = st.number_input("Azimut eje (°)", 0.0, 359.9, 0.0)
    slope = st.number_input("Pendiente eje (°)", -45.0, 45.0, 0.0)
    gamma = st.number_input(
        "Peso específico (kN/m³)", 10.0, 40.0, 26.4, 0.1
    )
    number = st.slider(
        "Planos por familia", 1, 5, 3,
        help="Aumentarlo incrementa las combinaciones y el cálculo."
    )
    limit = st.slider("Máximo de candidatos a representar", 3, 30, 15)
    dxf = st.file_uploader("Sección DXF", type=["dxf"])

r = W / 2
spring = H - r
angles = np.linspace(np.pi, 0, 120)

section = np.vstack((
    [-r, 0],
    [-r, spring],
    np.column_stack((
        r * np.cos(angles),
        spring + r * np.sin(angles)
    )),
    [r, 0]
))

methods = [
    "Mohr-Coulomb",
    "Barton-Bandis",
    "Patton",
    "Hoek-Brown (macizo)"
]

tab1, tab2, tab3, tab4 = st.tabs([
    "Discontinuidades",
    "Todas las cuñas 3D",
    "Cortes y sostenimiento",
    "Datos y DXF"
])

systems = []

with tab1:
    for i in range(3):
        with st.expander(f"Familia {i + 1}", expanded=i == 0):
            a, b = st.columns(2)
            with a:
                dd = st.number_input(
                    "Dirección buzamiento (°)", 0.0, 359.9,
                    [30.0, 150.0, 270.0][i], key=f"dd{i}"
                )
                dp = st.number_input(
                    "Buzamiento (°)", 0.0, 90.0,
                    60.0, key=f"dp{i}"
                )
                off = st.number_input(
                    "Posición longitudinal (m)", 0.0, L,
                    L * [0.42, 0.50, 0.58][i],
                    key=f"off{i}"
                )
                xo = st.number_input(
                    "Posición transversal (m)", -W, W,
                    0.0, key=f"xo{i}"
                )
                spacing = st.number_input(
                    "Espaciamiento (m)", 0.1, 20.0,
                    1.5, key=f"sp{i}"
                )
                persistence = st.number_input(
                    "Persistencia (m)", 0.1, 100.0,
                    5.0, key=f"pe{i}"
                )
            with b:
                method = st.selectbox(
                    "Criterio resistente", methods, key=f"m{i}"
                )
                props = {}
                if method == "Mohr-Coulomb":
                    props["c_MPa"] = st.number_input(
                        "Cohesión (MPa)", 0.0, 20.0,
                        0.05, key=f"c{i}"
                    )
                    props["phi"] = st.number_input(
                        "Fricción (°)", 0.0, 89.0,
                        30.0, key=f"phi{i}"
                    )
                elif method == "Barton-Bandis":
                    props["JRC"] = st.number_input(
                        "JRC", 0.0, 25.0, 10.0, key=f"jrc{i}"
                    )
                    props["JCS"] = st.number_input(
                        "JCS (MPa)", 0.1, 500.0,
                        30.0, key=f"jcs{i}"
                    )
                    props["phi_r"] = st.number_input(
                        "Fricción residual (°)", 0.0, 89.0,
                        25.0, key=f"pr{i}"
                    )
                elif method == "Patton":
                    props["phi_b"] = st.number_input(
                        "Fricción básica (°)", 0.0, 89.0,
                        25.0, key=f"pb{i}"
                    )
                    props["i"] = st.number_input(
                        "Ángulo asperidades (°)", 0.0, 60.0,
                        10.0, key=f"as{i}"
                    )
                else:
                    props["UCS"] = st.number_input(
                        "UCS (MPa)", 0.1, 500.0,
                        52.6, key=f"ucs{i}"
                    )
                    props["GSI"] = st.number_input(
                        "GSI", 1.0, 100.0, 50.0, key=f"gsi{i}"
                    )
                    st.info(
                        "Hoek-Brown describe el macizo, "
                        "no la resistencia de una junta."
                    )
                props["agua_MPa"] = st.number_input(
                    "Presión de agua (MPa)", 0.0, 10.0,
                    0.0, key=f"water{i}"
                )

            systems.append(dict(
                dd=dd, dp=dp, off=off, xo=xo,
                spacing=spacing, persistence=persistence,
                method=method, props=props
            ))


def roof(x):
    x = np.asarray(x)
    return spring + np.sqrt(np.maximum(0, r*r - x*x))


def normal(s):
    theta = np.deg2rad(s["dd"] - az)
    dip = np.deg2rad(s["dp"])
    return np.array([
        np.sin(theta) * np.sin(dip),
        np.cos(theta) * np.sin(dip),
        np.cos(dip)
    ])


def generate_planes():
    families = []
    for s in systems:
        n = normal(s)
        p = np.array([s["xo"], s["off"], H])
        base = n @ p
        planes = []
        for k in range(number):
            shift = (k - (number - 1) / 2) * s["spacing"]
            planes.append((n, base + shift, s["persistence"]))
        families.append(planes)
    return families


def tetra_candidate(planes, surface):
    ns = np.array([p[0] for p in planes])
    rhs = np.array([p[1] for p in planes])

    if abs(np.linalg.det(ns)) < 1e-7:
        return None

    apex = np.linalg.solve(ns, rhs)

    if surface == "Bóveda":
        axis, value = 2, spring
        if apex[2] <= H:
            return None
        if abs(apex[0]) > r * 1.6:
            return None
    elif surface == "Hastial derecho":
        axis, value = 0, 0.0
        if apex[0] <= r:
            return None
        if apex[2] < -1 or apex[2] > H + 1:
            return None
    else:
        axis, value = 0, 0.0
        if apex[0] >= -r:
            return None
        if apex[2] < -1 or apex[2] > H + 1:
            return None

    vertices = [apex]
    for i, j in [(0, 1), (0, 2), (1, 2)]:
        mat = np.array([
            ns[i], ns[j], np.eye(3)[axis]
        ])
        if abs(np.linalg.det(mat)) < 1e-8:
            return None
        point = np.linalg.solve(
            mat, [rhs[i], rhs[j], value]
        )
        vertices.append(point)

    vertices = np.array(vertices)

    if not np.isfinite(vertices).all():
        return None

    if vertices[:, 1].max() < 0 or vertices[:, 1].min() > L:
        return None

    try:
        hull = ConvexHull(vertices)
    except Exception:
        return None

    if hull.volume < 1e-6:
        return None

    edge = max(
        np.linalg.norm(vertices[i] - vertices[j])
        for i, j in itertools.combinations(range(4), 2)
    )

    valid_persistence = all(
        edge <= p[2] for p in planes
    )
    return vertices, valid_persistence, edge


@st.cache_data(show_spinner=False)
def clip_mesh(vertices_data, surface, width, height, length):
    """Intersección numérica aproximada del tetraedro con roca exterior."""
    v = np.array(vertices_data)
    rr = width / 2
    ss = height - rr

    hull = ConvexHull(v)
    lo = np.maximum(
        v.min(axis=0) - .1,
        [-width * 2, 0, -0.1]
    )
    hi = np.minimum(
        v.max(axis=0) + .1,
        [width * 2, length, height + width * 2]
    )

    if np.any(hi - lo < .05):
        return None

    sizes = hi - lo
    nxyz = np.maximum(
        20,
        np.minimum(
            62,
            np.ceil(42 * sizes / max(sizes)).astype(int)
        )
    )

    axes = [
        np.linspace(lo[i], hi[i], int(nxyz[i]))
        for i in range(3)
    ]
    X, Y, Z = np.meshgrid(*axes, indexing="ij")

    inside = np.maximum.reduce([
        e[0]*X + e[1]*Y + e[2]*Z + e[3]
        for e in hull.equations
    ])

    if surface == "Bóveda":
        rr2 = np.sqrt(np.maximum(0, rr*rr-X*X))
        boundary = ss + rr2 - Z
        boundary = np.maximum(boundary, np.abs(X)-rr)
    elif surface == "Hastial derecho":
        boundary = rr-X
        boundary = np.maximum(boundary, -Z)
        boundary = np.maximum(boundary, Z-ss)
    else:
        boundary = X+rr
        boundary = np.maximum(boundary, -Z)
        boundary = np.maximum(boundary, Z-ss)

    field = np.maximum(inside, boundary)
    field = np.maximum(field, -Y)
    field = np.maximum(field, Y-length)

    if field.min() >= 0 or field.max() <= 0:
        return None

    step = np.array([
        a[1]-a[0] for a in axes
    ])

    try:
        verts, faces, _, _ = marching_cubes(
            field.astype(np.float32),
            level=0,
            spacing=tuple(step)
        )
    except Exception:
        return None

    verts += lo
    volume = np.count_nonzero(field < 0) * np.prod(step)

    if volume < .005:
        return None

    return verts, faces, float(volume)


def mesh_cut(vertices, faces, axis, value):
    """Segmentos del corte obtenidos de la misma malla 3D."""
    segments = []
    for tri in faces:
        points = vertices[tri]
        distances = points[:, axis] - value
        intersections = []

        for i, j in [(0, 1), (1, 2), (2, 0)]:
            a, b = distances[i], distances[j]
            if a*b < 0:
                t = -a / (b-a)
                intersections.append(
                    points[i] + t*(points[j]-points[i])
                )

        if len(intersections) == 2:
            segments.append(intersections)

    return segments


def compute_candidates():
    families = generate_planes()
    results = []
    seen = set()

    for indices in itertools.product(range(number), repeat=3):
        planes = [
            families[i][indices[i]] for i in range(3)
        ]
        for surface in [
            "Bóveda", "Hastial izquierdo", "Hastial derecho"
        ]:
            result = tetra_candidate(planes, surface)
            if result is None:
                continue

            v, persistence_ok, edge = result
            key = (
                surface,
                tuple(np.round(v[0], 2))
            )
            if key in seen:
                continue
            seen.add(key)

            mesh = clip_mesh(
                tuple(map(tuple, v)), surface, W, H, L
            )
            if mesh is None:
                continue

            mv, mf, volume = mesh
            results.append(dict(
                name=f"C{len(results)+1}",
                surface=surface,
                verts=mv,
                faces=mf,
                volume=volume,
                weight=volume*gamma,
                persistence_ok=persistence_ok,
                edge=edge,
                center=mv.mean(axis=0),
                planes=tuple(i+1 for i in indices)
            ))

    results.sort(key=lambda c: c["volume"], reverse=True)
    return results


with st.spinner("Buscando cuñas geométricas candidatas..."):
    all_candidates = compute_candidates()

candidates = all_candidates[:limit]

colors = [
    "#ea580c", "#2563eb", "#16a34a",
    "#9333ea", "#dc2626", "#0891b2",
    "#ca8a04", "#db2777", "#4f46e5"
]

with tab2:
    st.subheader("Cuñas candidatas en bóveda y hastiales")

    c1, c2, c3 = st.columns(3)
    c1.metric("Candidatos encontrados", len(all_candidates))
    c2.metric("Representados", len(candidates))
    c3.metric(
        "Con persistencia compatible",
        sum(c["persistence_ok"] for c in all_candidates)
    )

    if len(all_candidates) > limit:
        st.info(
            "Se muestran los candidatos de mayor volumen. "
            "Aumenta el límite para visualizar más."
        )

    fig = go.Figure()

    for y in [0, L]:
        fig.add_trace(go.Scatter3d(
            x=section[:, 0],
            y=np.full(len(section), y),
            z=section[:, 1],
            mode="lines",
            line=dict(color="#64748b", width=5),
            name="Galería",
            showlegend=y == 0
        ))

    for i, c in enumerate(candidates):
        v, faces = c["verts"], c["faces"]
        color = colors[i % len(colors)]
        fig.add_trace(go.Mesh3d(
            x=v[:, 0], y=v[:, 1], z=v[:, 2],
            i=faces[:, 0],
            j=faces[:, 1],
            k=faces[:, 2],
            color=color,
            opacity=.75,
            flatshading=True,
            name=f'{c["name"]} · {c["surface"]}',
            showlegend=True
        ))

    fig.update_layout(
        height=780,
        margin=dict(l=0, r=0, t=0, b=0),
        scene=dict(
            xaxis_title="X (m)",
            yaxis_title="Y (m)",
            zaxis_title="Z (m)",
            aspectmode="data",
            camera=dict(
                eye=dict(x=1.5, y=-1.5, z=.9)
            )
        ),
        legend=dict(orientation="h")
    )
    st.plotly_chart(
        fig,
        use_container_width=True,
        config={"scrollZoom": True}
    )

    if candidates:
        rows = []
        for c in candidates:
            rows.append({
                "Cuña": c["name"],
                "Ubicación": c["surface"],
                "Volumen aprox. (m³)": round(c["volume"], 3),
                "Peso aprox. (kN)": round(c["weight"], 1),
                "Persistencia compatible": (
                    "Sí, preliminar"
                    if c["persistence_ok"] else "No"
                ),
                "FS": "No calculado"
            })
        st.dataframe(rows, use_container_width=True)
    else:
        st.info(
            "No se han encontrado candidatos con estas "
            "combinaciones y límites de búsqueda. "
            "Esto no demuestra ausencia de cuñas reales."
        )


presets = {
    "ST-I": (2.0, .03, 0.0, False),
    "ST-II": (2.0, .03, 0.0, False),
    "ST-III": (1.5, .08, 0.0, True),
    "ST-IV superficial": (1.5, .12, 8.0, True),
    "ST-IV profundo": (1.5, .15, 10.0, True),
    "Personalizado": (1.5, .08, 0.0, True)
}

with tab3:
    st.subheader("Cuñas y sostenimiento en la misma sección")

    preset = st.selectbox("Sostenimiento tipo", list(presets))
    bolt_spacing, shot_default, cable_default, mesh_default = (
        presets[preset]
    )

    a, b = st.columns(2)
    with a:
        bolt_type = st.selectbox(
            "Tipo de perno",
            ["Resina", "Split Set", "Swellex", "Sin pernos"]
        )
        bolt_length = st.number_input(
            "Longitud pernos (m)", .5, 12.0, 2.4
        )
        bolt_spacing = st.number_input(
            "Separación transversal pernos (m)",
            .3, 5.0, bolt_spacing
        )
        bolt_longspacing = st.number_input(
            "Separación longitudinal pernos (m)",
            .3, 5.0, 1.5
        )
        bolt_capacity = st.number_input(
            "Capacidad nominal perno (kN)",
            1.0, 1000.0, 324.0
        )
    with b:
        shot_on = st.checkbox("Gunita", True)
        shot_cm = st.number_input(
            "Espesor gunita (cm)", 0.0, 50.0,
            shot_default*100, .5,
            disabled=not shot_on
        )
        shot_fc = st.number_input(
            "fc gunita (MPa)", 1.0, 100.0, 35.0
        )
        shot_ft = st.number_input(
            "ft gunita (MPa)", 0.0, 20.0, 3.0
        )
        shot_E = st.number_input(
            "E gunita (GPa)", 1.0, 80.0, 30.0
        )
        mesh_on = st.checkbox(
            "Malla metálica", mesh_default
        )
        cable_on = st.checkbox(
            "Cables", cable_default > 0
        )
        cable_length = st.number_input(
            "Longitud cables (m)", 1.0, 25.0,
            cable_default if cable_default else 8.0,
            disabled=not cable_on
        )
        cable_spacing = st.number_input(
            "Separación transversal cables (m)",
            .3, 6.0,
            1.0 if preset == "ST-IV profundo" else 1.5
        )
        cable_capacity = st.number_input(
            "Capacidad cable (kN)", 1.0, 1500.0, 350.0
        )

    if candidates:
        ycenters = [c["center"][1] for c in candidates]
        default_y = float(np.clip(ycenters[0], 0, L))
    else:
        default_y = L/2

    ycut = st.slider(
        "Corte transversal Y (m)",
        0.0, float(L),
        round(default_y, 1), .1
    )

    xcut = st.slider(
        "Corte longitudinal X (m)",
        -float(W), float(W), 0.0, .1
    )

    f = go.Figure()
    f.add_trace(go.Scatter(
        x=section[:, 0],
        y=section[:, 1],
        mode="lines",
        line=dict(color="#334155", width=5),
        name="Excavación"
    ))

    for i, c in enumerate(candidates):
        segments = mesh_cut(
            c["verts"], c["faces"], 1, ycut
        )
        color = colors[i % len(colors)]
        for j, segment in enumerate(segments):
            p, q = segment
            f.add_trace(go.Scatter(
                x=[p[0], q[0]],
                y=[p[2], q[2]],
                mode="lines",
                line=dict(color=color, width=3),
                name=c["name"],
                showlegend=j == 0
            ))

    if shot_on and shot_cm > 0:
        ext = section.copy()
        ext[1:-1, 0] += (
            np.sign(ext[1:-1, 0]) * shot_cm/100
        )
        ext[1:-1, 1] += shot_cm/100
        f.add_trace(go.Scatter(
            x=ext[:, 0],
            y=ext[:, 1],
            mode="lines",
            line=dict(color="#f97316", width=7),
            name="Gunita (esquema)"
        ))

    if mesh_on:
        f.add_trace(go.Scatter(
            x=section[1:-1, 0],
            y=section[1:-1, 1],
            mode="lines",
            line=dict(
                color="#16a34a",
                width=3,
                dash="dot"
            ),
            name="Malla"
        ))

    seg = np.linalg.norm(
        np.diff(section, axis=0), axis=1
    )
    cum = np.r_[0, np.cumsum(seg)]

    if bolt_type != "Sin pernos":
        for k, s in enumerate(
            np.arange(
                bolt_spacing/2, cum[-1], bolt_spacing
            )
        ):
            ix = np.clip(
                np.searchsorted(cum, s)-1,
                0, len(seg)-1
            )
            t = (s-cum[ix])/max(seg[ix], 1e-9)
            p = (
                section[ix]*(1-t)
                + section[ix+1]*t
            )

            if p[1] <= spring and abs(p[0]) > r-.01:
                n = np.array([np.sign(p[0]), 0.0])
            else:
                n = np.array([p[0], p[1]-spring])
                n /= max(np.linalg.norm(n), 1e-9)

            q = p + bolt_length*n
            f.add_trace(go.Scatter(
                x=[p[0], q[0]],
                y=[p[1], q[1]],
                mode="lines",
                line=dict(color="#2563eb", width=4),
                name="Pernos" if k == 0 else None,
                showlegend=k == 0
            ))

    if cable_on:
        for k, s in enumerate(
            np.arange(
                cable_spacing/2,
                math.pi*r,
                cable_spacing
            )
        ):
            angle = math.pi-s/r
            p = np.array([
                r*math.cos(angle),
                spring+r*math.sin(angle)
            ])
            n = np.array([
                math.cos(angle),
                math.sin(angle)
            ])
            q = p+cable_length*n
            f.add_trace(go.Scatter(
                x=[p[0], q[0]],
                y=[p[1], q[1]],
                mode="lines",
                line=dict(color="#9333ea", width=5),
                name="Cables" if k == 0 else None,
                showlegend=k == 0
            ))

    f.update_layout(
        height=700,
        xaxis=dict(
            title="X (m)",
            scaleanchor="y",
            scaleratio=1
        ),
        yaxis_title="Z (m)",
        legend=dict(orientation="h")
    )
    st.plotly_chart(f, use_container_width=True)

    st.subheader("Corte longitudinal de las mismas mallas 3D")
    fl = go.Figure()

    for i, c in enumerate(candidates):
        segments = mesh_cut(
            c["verts"], c["faces"], 0, xcut
        )
        color = colors[i % len(colors)]
        for j, segment in enumerate(segments):
            p, q = segment
            fl.add_trace(go.Scatter(
                x=[p[1], q[1]],
                y=[p[2], q[2]],
                mode="lines",
                line=dict(color=color, width=3),
                name=c["name"],
                showlegend=j == 0
            ))

    fl.update_layout(
        height=480,
        xaxis_title="Y (m)",
        yaxis_title="Z (m)"
    )
    st.plotly_chart(fl, use_container_width=True)

    st.caption(
        "Ambos cortes proceden de las mallas 3D representadas. "
        "La gunita y los anclajes son esquemáticos; "
        "no se comprueba su interacción resistente."
    )


with tab4:
    data = {
        "version": "10",
        "galeria": {
            "ancho": W,
            "alto": H,
            "longitud": L,
            "azimut": az,
            "pendiente": slope
        },
        "discontinuidades": systems
    }

    st.download_button(
        "Descargar datos JSON",
        json.dumps(data, indent=2, ensure_ascii=False),
        file_name="geotecnia_mina_v10.json",
        mime="application/json"
    )

    if dxf is not None:
        try:
            import ezdxf
            doc = ezdxf.read(
                io.StringIO(
                    dxf.getvalue().decode(
                        "utf-8-sig",
                        errors="replace"
                    )
                )
            )
            fd = go.Figure()
            count = 0

            for ent in doc.modelspace():
                typ = ent.dxftype()
                pts = None

                if typ == "LINE":
                    pts = [
                        ent.dxf.start[:2],
                        ent.dxf.end[:2]
                    ]
                elif typ == "LWPOLYLINE":
                    pts = list(ent.get_points("xy"))
                elif typ == "POLYLINE":
                    pts = [
                        p.dxf.location[:2]
                        for p in ent.vertices
                    ]
                elif typ in ("CIRCLE", "ARC"):
                    center = ent.dxf.center
                    radius = ent.dxf.radius
                    if typ == "CIRCLE":
                        ang = np.linspace(
                            0, 2*np.pi, 120
                        )
                    else:
                        a0 = ent.dxf.start_angle
                        a1 = ent.dxf.end_angle
                        if a1 < a0:
                            a1 += 360
                        ang = np.deg2rad(
                            np.linspace(a0, a1, 120)
                        )
                    pts = np.column_stack((
                        center.x+radius*np.cos(ang),
                        center.y+radius*np.sin(ang)
                    ))

                if pts is not None:
                    p = np.array(pts, dtype=float)
                    fd.add_trace(go.Scatter(
                        x=p[:, 0],
                        y=p[:, 1],
                        mode="lines",
                        showlegend=False
                    ))
                    count += 1

            fd.update_layout(
                height=550,
                xaxis=dict(
                    title="X DXF",
                    scaleanchor="y",
                    scaleratio=1
                ),
                yaxis_title="Y DXF"
            )
            st.plotly_chart(fd, use_container_width=True)
            st.caption(
                f"{count} entidades DXF representadas. "
                "El DXF no modifica la geometría calculada."
            )
        except Exception as exc:
            st.error(f"Error leyendo DXF: {exc}")
    else:
        st.info(
            "Puedes subir un DXF desde el panel lateral."
        )
