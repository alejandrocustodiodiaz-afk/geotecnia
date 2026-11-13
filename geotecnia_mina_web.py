
"""Geotecnia Mina Web v11 - visualizador preliminar de cuñas."""
import io
import json
import math
import itertools

import numpy as np
import plotly.graph_objects as go
import streamlit as st
from scipy.spatial import ConvexHull
from skimage.measure import marching_cubes

st.set_page_config(
    page_title="Geotecnia Mina v11",
    layout="wide"
)

st.title("Geotecnia Mina · v11")
st.caption(
    "Cuñas de bóveda y hastiales · filtros de revisión · "
    "cortes 2D y 3D · sostenimiento"
)
st.warning(
    "MODELO GEOMÉTRICO PRELIMINAR. Los bloques se generan "
    "mediante combinaciones simplificadas de tres planos. "
    "No se verifica liberación cinemática, equilibrio límite, "
    "persistencia real ni FS. No utilizar para dimensionamiento."
)

with st.sidebar:
    st.header("Geometría de galería")
    W = st.number_input("Ancho (m)", 1.0, 25.0, 5.8, 0.1)
    H = st.number_input("Altura (m)", 1.0, 25.0, 5.8, 0.1)
    L = st.number_input("Longitud (m)", 2.0, 100.0, 15.0, 0.5)
    az = st.number_input("Azimut eje (°)", 0.0, 359.9, 0.0)
    slope = st.number_input("Pendiente eje (°)", -45.0, 45.0, 0.0)
    gamma = st.number_input(
        "Peso específico (kN/m³)", 10.0, 40.0, 26.4, 0.1
    )

    st.divider()
    st.header("Visualización de cuñas")

    mode = st.radio(
        "Modo de revisión",
        ["Revisión de seguridad", "Vista simplificada"],
        index=0
    )

    min_volume = st.number_input(
        "Volumen mínimo orientativo (m³)",
        0.0, 100.0, 0.25, 0.05
    )

    nplanes = st.slider(
        "Planos por familia", 1, 5, 3
    )

    max_zone = st.slider(
        "Máximo por zona en vista simplificada",
        1, 10, 3
    )

    show_small = st.checkbox(
        "Mostrar también candidatos menores al umbral",
        value=False
    )

    only_persistent = st.checkbox(
        "Filtrar por persistencia simplificada",
        value=False,
        help=(
            "Esta comprobación no sustituye una evaluación "
            "de la geometría real de las discontinuidades."
        )
    )

    dxf = st.file_uploader(
        "Sección AutoCAD DXF",
        type=["dxf"]
    )

r = W / 2
spring = H - r
arc = np.linspace(np.pi, 0, 120)

section = np.vstack((
    [-r, 0],
    [-r, spring],
    np.column_stack((
        r * np.cos(arc),
        spring + r * np.sin(arc)
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
    "Cuñas 3D",
    "Cortes y sostenimiento",
    "Datos y DXF"
])

systems = []

with tab1:
    st.subheader("Familias de discontinuidades")
    for i in range(3):
        with st.expander(
            f"Familia {i + 1}",
            expanded=i == 0
        ):
            a, b = st.columns(2)

            with a:
                dd = st.number_input(
                    "Dirección de buzamiento (°)",
                    0.0, 359.9,
                    [30.0, 150.0, 270.0][i],
                    key=f"dd{i}"
                )
                dp = st.number_input(
                    "Buzamiento (°)",
                    0.0, 90.0, 60.0,
                    key=f"dp{i}"
                )
                off = st.number_input(
                    "Posición longitudinal (m)",
                    0.0, L,
                    L * [0.42, 0.50, 0.58][i],
                    key=f"off{i}"
                )
                xo = st.number_input(
                    "Posición transversal (m)",
                    -W, W, 0.0,
                    key=f"xo{i}"
                )
                spacing = st.number_input(
                    "Espaciamiento (m)",
                    0.1, 20.0, 1.5,
                    key=f"sp{i}"
                )
                persistence = st.number_input(
                    "Persistencia estimada (m)",
                    0.1, 100.0, 5.0,
                    key=f"pe{i}"
                )

            with b:
                method = st.selectbox(
                    "Criterio resistente",
                    methods,
                    key=f"method{i}"
                )
                props = {}

                if method == "Mohr-Coulomb":
                    props["cohesion_MPa"] = st.number_input(
                        "Cohesión (MPa)",
                        0.0, 20.0, 0.05,
                        key=f"c{i}"
                    )
                    props["friccion_deg"] = st.number_input(
                        "Fricción (°)",
                        0.0, 89.0, 30.0,
                        key=f"phi{i}"
                    )
                elif method == "Barton-Bandis":
                    props["JRC"] = st.number_input(
                        "JRC", 0.0, 25.0, 10.0,
                        key=f"jrc{i}"
                    )
                    props["JCS_MPa"] = st.number_input(
                        "JCS (MPa)",
                        0.1, 500.0, 30.0,
                        key=f"jcs{i}"
                    )
                    props["phi_residual"] = st.number_input(
                        "Fricción residual (°)",
                        0.0, 89.0, 25.0,
                        key=f"pr{i}"
                    )
                elif method == "Patton":
                    props["phi_basica"] = st.number_input(
                        "Fricción básica (°)",
                        0.0, 89.0, 25.0,
                        key=f"pb{i}"
                    )
                    props["asperidad"] = st.number_input(
                        "Ángulo asperidades (°)",
                        0.0, 60.0, 10.0,
                        key=f"as{i}"
                    )
                else:
                    props["UCS_MPa"] = st.number_input(
                        "UCS (MPa)",
                        0.1, 500.0, 52.6,
                        key=f"ucs{i}"
                    )
                    props["GSI"] = st.number_input(
                        "GSI", 1.0, 100.0, 50.0,
                        key=f"gsi{i}"
                    )
                    st.info(
                        "Hoek-Brown es un criterio de macizo, "
                        "no de resistencia de juntas."
                    )

                props["agua_MPa"] = st.number_input(
                    "Presión de agua (MPa)",
                    0.0, 10.0, 0.0,
                    key=f"water{i}"
                )

            systems.append(dict(
                dd=dd,
                dp=dp,
                off=off,
                xo=xo,
                spacing=spacing,
                persistence=persistence,
                method=method,
                props=props
            ))


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
        family = []
        for k in range(nplanes):
            shift = (
                k - (nplanes - 1) / 2
            ) * s["spacing"]
            family.append((
                n, base + shift, s["persistence"]
            ))
        families.append(family)
    return families


def make_tetra(planes, zone):
    ns = np.array([p[0] for p in planes])
    rhs = np.array([p[1] for p in planes])

    if abs(np.linalg.det(ns)) < 1e-8:
        return None

    apex = np.linalg.solve(ns, rhs)

    if zone == "Bóveda":
        axis = 2
        value = spring
        if apex[2] <= H:
            return None
        if abs(apex[0]) > r * 1.6:
            return None
    elif zone == "Hastial izquierdo":
        axis = 0
        value = 0.0
        if apex[0] >= -r:
            return None
        if apex[2] < -1 or apex[2] > H + 1:
            return None
    else:
        axis = 0
        value = 0.0
        if apex[0] <= r:
            return None
        if apex[2] < -1 or apex[2] > H + 1:
            return None

    vertices = [apex]

    for i, j in [(0, 1), (0, 2), (1, 2)]:
        mat = np.array([
            ns[i],
            ns[j],
            np.eye(3)[axis]
        ])
        if abs(np.linalg.det(mat)) < 1e-8:
            return None

        vertices.append(
            np.linalg.solve(
                mat, [rhs[i], rhs[j], value]
            )
        )

    vertices = np.array(vertices)

    if not np.isfinite(vertices).all():
        return None

    if (
        vertices[:, 1].max() < 0
        or vertices[:, 1].min() > L
    ):
        return None

    try:
        hull = ConvexHull(vertices)
    except Exception:
        return None

    if hull.volume < 1e-7:
        return None

    edge = max(
        np.linalg.norm(vertices[i] - vertices[j])
        for i, j in itertools.combinations(range(4), 2)
    )

    persistence_ok = all(
        edge <= p[2] for p in planes
    )

    return vertices, edge, persistence_ok


@st.cache_data(show_spinner=False)
def clip_mesh(vtuple, zone, width, height, length):
    """Recorte numérico aproximado, no operación CAD exacta."""
    v = np.array(vtuple, dtype=float)
    rr = width / 2
    ss = height - rr

    try:
        hull = ConvexHull(v)
    except Exception:
        return None

    lo = np.maximum(
        v.min(axis=0) - .1,
        [-width*2, 0.0, -.1]
    )
    hi = np.minimum(
        v.max(axis=0) + .1,
        [width*2, length, height+width*2]
    )

    if np.any(hi-lo < .05):
        return None

    size = hi-lo
    dims = np.maximum(
        20,
        np.minimum(
            60,
            np.ceil(42*size/max(size)).astype(int)
        )
    )

    axes = [
        np.linspace(lo[i], hi[i], int(dims[i]))
        for i in range(3)
    ]

    X, Y, Z = np.meshgrid(*axes, indexing="ij")

    signed = np.maximum.reduce([
        e[0]*X + e[1]*Y + e[2]*Z + e[3]
        for e in hull.equations
    ])

    if zone == "Bóveda":
        roof = ss + np.sqrt(
            np.maximum(0, rr*rr-X*X)
        )
        boundary = np.maximum(
            roof-Z, np.abs(X)-rr
        )
    elif zone == "Hastial derecho":
        boundary = np.maximum.reduce([
            rr-X, -Z, Z-ss
        ])
    else:
        boundary = np.maximum.reduce([
            X+rr, -Z, Z-ss
        ])

    field = np.maximum(signed, boundary)
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

    volume = float(
        np.count_nonzero(field < 0)*np.prod(step)
    )

    if volume < .001:
        return None

    return verts, faces, volume


def mesh_cut(vertices, faces, axis, value):
    """Extrae segmentos del corte de una malla 3D."""
    segments = []

    for tri in faces:
        pts = vertices[tri]
        d = pts[:, axis]-value
        intersections = []

        for i, j in [(0, 1), (1, 2), (2, 0)]:
            a, b = d[i], d[j]

            if a*b < 0:
                t = -a/(b-a)
                intersections.append(
                    pts[i]+t*(pts[j]-pts[i])
                )

        if len(intersections) == 2:
            segments.append(intersections)

    return segments


def find_candidates():
    families = generate_planes()
    found = []

    for indices in itertools.product(
        range(nplanes), repeat=3
    ):
        planes = [
            families[i][indices[i]]
            for i in range(3)
        ]

        for zone in [
            "Bóveda",
            "Hastial izquierdo",
            "Hastial derecho"
        ]:
            result = make_tetra(planes, zone)

            if result is None:
                continue

            v, edge, persistence_ok = result
            mesh = clip_mesh(
                tuple(map(tuple, v)),
                zone, W, H, L
            )

            if mesh is None:
                continue

            mv, mf, volume = mesh

            found.append(dict(
                zone=zone,
                vertices=mv,
                faces=mf,
                volume=volume,
                weight=volume*gamma,
                center=mv.mean(axis=0),
                edge=edge,
                persistence_ok=persistence_ok,
                indices=indices
            ))

    return found


def near_duplicate(a, b):
    """Solo detecta candidatos casi coincidentes; no fusiona sólidos."""
    if a["zone"] != b["zone"]:
        return False

    va = a["volume"]
    vb = b["volume"]

    if max(va, vb) <= 0:
        return False

    volume_difference = abs(va-vb)/max(va, vb)
    center_distance = np.linalg.norm(
        a["center"]-b["center"]
    )

    scale = max(min(va, vb)**(1/3), .1)

    return (
        volume_difference < .02
        and center_distance < .05*scale
    )


def filter_candidates(found):
    ordered = sorted(
        found,
        key=lambda c: c["volume"],
        reverse=True
    )

    # El filtro de coincidencia es heurístico.
    # No acredita que los sólidos sean idénticos.
    unique = []

    for c in ordered:
        if not any(
            near_duplicate(c, other)
            for other in unique
        ):
            unique.append(c)

    selected = []

    for zone in [
        "Bóveda",
        "Hastial izquierdo",
        "Hastial derecho"
    ]:
        zone_items = [
            c for c in unique
            if c["zone"] == zone
        ]

        if only_persistent:
            zone_items = [
                c for c in zone_items
                if c["persistence_ok"]
            ]

        if not show_small:
            zone_items = [
                c for c in zone_items
                if c["volume"] >= min_volume
            ]

        if mode == "Vista simplificada":
            zone_items = zone_items[:max_zone]

        selected.extend(zone_items)

    return unique, selected


with st.spinner("Calculando candidatos geométricos..."):
    raw_candidates = find_candidates()

unique_candidates, candidates = filter_candidates(
    raw_candidates
)

colors = [
    "#ea580c", "#2563eb", "#16a34a",
    "#9333ea", "#dc2626", "#0891b2",
    "#ca8a04", "#db2777", "#4f46e5",
    "#65a30d", "#0f766e", "#b45309"
]

for i, c in enumerate(candidates):
    c["name"] = f"C{i+1}"
    c["color"] = colors[i % len(colors)]


with tab2:
    st.subheader("Cuñas geométricas de bóveda y hastiales")

    a, b, c = st.columns(3)
    a.metric(
        "Candidatos generados",
        len(raw_candidates)
    )
    b.metric(
        "Tras filtro de coincidencia",
        len(unique_candidates)
    )
    c.metric(
        "Cuñas visibles",
        len(candidates)
    )

    st.caption(
        "El filtro de coincidencia es aproximado. "
        "Puede agrupar candidatos distintos muy similares; "
        "los resultados originales se conservan en el recuento."
    )

    if not show_small:
        small_count = sum(
            x["volume"] < min_volume
            for x in unique_candidates
        )
        if small_count:
            st.warning(
                f"{small_count} candidatos inferiores a "
                f"{min_volume:.2f} m³ están ocultos por "
                "el filtro visual. Esto NO significa "
                "que sean seguros."
            )

    fig = go.Figure()

    for yy in [0, L]:
        fig.add_trace(go.Scatter3d(
            x=section[:, 0],
            y=np.full(len(section), yy),
            z=section[:, 1],
            mode="lines",
            line=dict(
                color="#64748b",
                width=5
            ),
            name="Galería",
            showlegend=yy == 0
        ))

    for k in [0, 25, 50, 75, 100]:
        fig.add_trace(go.Scatter3d(
            x=[section[k, 0]]*2,
            y=[0, L],
            z=[section[k, 1]]*2,
            mode="lines",
            line=dict(
                color="#cbd5e1",
                width=2
            ),
            showlegend=False
        ))

    for item in candidates:
        v = item["vertices"]
        faces = item["faces"]

        fig.add_trace(go.Mesh3d(
            x=v[:, 0],
            y=v[:, 1],
            z=v[:, 2],
            i=faces[:, 0],
            j=faces[:, 1],
            k=faces[:, 2],
            color=item["color"],
            opacity=.72,
            flatshading=True,
            name=(
                f'{item["name"]} · '
                f'{item["zone"]}'
            ),
            hovertemplate=(
                f'{item["name"]}<br>'
                f'{item["zone"]}<br>'
                f'Volumen: {item["volume"]:.3f} m³'
                '<extra></extra>'
            )
        ))

    fig.update_layout(
        height=750,
        margin=dict(l=0, r=0, t=0, b=0),
        scene=dict(
            xaxis_title="X (m)",
            yaxis_title="Y (m)",
            zaxis_title="Z (m)",
            aspectmode="data",
            camera=dict(
                eye=dict(x=1.45, y=-1.4, z=.9)
            )
        ),
        legend=dict(
            orientation="h",
            y=-.1
        )
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
        config={"scrollZoom": True}
    )

    if candidates:
        rows = []

        for item in candidates:
            rows.append({
                "Cuña": item["name"],
                "Ubicación": item["zone"],
                "Volumen (m³)": round(
                    item["volume"], 3
                ),
                "Peso (kN)": round(
                    item["weight"], 1
                ),
                "Persistencia simplificada": (
                    "Compatible"
                    if item["persistence_ok"]
                    else "Incompatible"
                ),
                "FS": "No calculado"
            })

        st.dataframe(
            rows,
            use_container_width=True,
            hide_index=True
        )
    else:
        st.info(
            "No hay candidatos visibles con los filtros "
            "seleccionados. Esto no demuestra que "
            "la galería esté libre de cuñas."
        )

    with st.expander(
        "Consultar todos los candidatos, incluidos los ocultos"
    ):
        st.dataframe([
            {
                "Zona": item["zone"],
                "Volumen (m³)": round(
                    item["volume"], 3
                ),
                "Peso (kN)": round(
                    item["weight"], 1
                ),
                "Persistencia compatible": (
                    "Sí" if item["persistence_ok"] else "No"
                )
            }
            for item in sorted(
                raw_candidates,
                key=lambda x: x["volume"],
                reverse=True
            )
        ], use_container_width=True, hide_index=True)


presets = {
    "ST-I": (2.0, 2.0, .03, 0.0, False),
    "ST-II": (2.0, 2.0, .03, 0.0, False),
    "ST-III": (1.5, 1.5, .08, 0.0, True),
    "ST-IV superficial": (1.5, 1.5, .12, 8.0, True),
    "ST-IV profundo": (1.5, 1.5, .15, 10.0, True),
    "Personalizado": (1.5, 1.5, .08, 0.0, True)
}

with tab3:
    st.subheader("Sección 2D y sostenimiento")

    preset = st.selectbox(
        "Sostenimiento tipo",
        list(presets)
    )

    bolt_spacing, bolt_long, shot_default, cable_default, mesh_default = (
        presets[preset]
    )

    a, b = st.columns(2)

    with a:
        bolt_type = st.selectbox(
            "Tipo de perno",
            ["Resina", "Split Set", "Swellex", "Sin pernos"]
        )
        bolt_length = st.number_input(
            "Longitud pernos (m)",
            .5, 12.0, 2.4
        )
        bolt_spacing = st.number_input(
            "Separación transversal (m)",
            .3, 5.0, bolt_spacing
        )
        bolt_long = st.number_input(
            "Separación longitudinal (m)",
            .3, 5.0, bolt_long
        )
        bolt_capacity = st.number_input(
            "Capacidad nominal perno (kN)",
            1.0, 1000.0, 324.0
        )

    with b:
        shot_on = st.checkbox("Gunita", True)
        shot_cm = st.number_input(
            "Espesor gunita (cm)",
            0.0, 50.0, shot_default*100, .5,
            disabled=not shot_on
        )
        shot_fc = st.number_input(
            "fc gunita (MPa)",
            1.0, 100.0, 35.0
        )
        shot_ft = st.number_input(
            "ft gunita (MPa)",
            0.0, 20.0, 3.0
        )
        shot_E = st.number_input(
            "E gunita (GPa)",
            1.0, 80.0, 30.0
        )
        mesh_on = st.checkbox(
            "Malla metálica", mesh_default
        )
        cable_on = st.checkbox(
            "Cables de anclaje",
            cable_default > 0
        )
        cable_length = st.number_input(
            "Longitud cables (m)",
            1.0, 25.0,
            cable_default if cable_default else 8.0,
            disabled=not cable_on
        )
        cable_spacing = st.number_input(
            "Separación cables (m)",
            .3, 6.0,
            1.0 if preset == "ST-IV profundo" else 1.5
        )
        cable_capacity = st.number_input(
            "Capacidad cable (kN)",
            1.0, 1500.0, 350.0
        )

    if candidates:
        y_default = float(np.clip(
            candidates[0]["center"][1], 0, L
        ))
    else:
        y_default = L/2

    ycut = st.slider(
        "Corte transversal Y (m)",
        0.0, float(L),
        round(y_default, 1), .1
    )

    xcut = st.slider(
        "Corte longitudinal X (m)",
        -float(W), float(W), 0.0, .1
    )

    f = go.Figure()

    for item in candidates:
        segments = mesh_cut(
            item["vertices"],
            item["faces"],
            1, ycut
        )

        for j, segm in enumerate(segments):
            p, q = segm
            f.add_trace(go.Scatter(
                x=[p[0], q[0]],
                y=[p[2], q[2]],
                mode="lines",
                line=dict(
                    color=item["color"],
                    width=3
                ),
                name=item["name"],
                showlegend=j == 0
            ))

    f.add_trace(go.Scatter(
        x=section[:, 0],
        y=section[:, 1],
        mode="lines",
        line=dict(
            color="#334155",
            width=5
        ),
        name="Excavación"
    ))

    if shot_on and shot_cm > 0:
        ext = section.copy()
        ext[1:-1, 0] += (
            np.sign(ext[1:-1, 0])*shot_cm/100
        )
        ext[1:-1, 1] += shot_cm/100

        f.add_trace(go.Scatter(
            x=ext[:, 0],
            y=ext[:, 1],
            mode="lines",
            line=dict(
                color="#f97316",
                width=8
            ),
            name="Gunita (esquemática)"
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

    lengths = np.linalg.norm(
        np.diff(section, axis=0),
        axis=1
    )
    cumulative = np.r_[
        0, np.cumsum(lengths)
    ]

    if bolt_type != "Sin pernos":
        for k, s in enumerate(
            np.arange(
                bolt_spacing/2,
                cumulative[-1],
                bolt_spacing
            )
        ):
            ix = int(np.clip(
                np.searchsorted(cumulative, s)-1,
                0, len(lengths)-1
            ))

            t = (
                (s-cumulative[ix])
                / max(lengths[ix], 1e-9)
            )
            p = (
                section[ix]*(1-t)
                + section[ix+1]*t
            )

            if p[1] <= spring and abs(p[0]) > r-.01:
                n = np.array([
                    np.sign(p[0]), 0.0
                ])
            else:
                n = np.array([
                    p[0], p[1]-spring
                ])
                n /= max(
                    np.linalg.norm(n), 1e-9
                )

            q = p+bolt_length*n

            f.add_trace(go.Scatter(
                x=[p[0], q[0]],
                y=[p[1], q[1]],
                mode="lines",
                line=dict(
                    color="#2563eb",
                    width=4
                ),
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
                line=dict(
                    color="#9333ea",
                    width=5
                ),
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

    st.plotly_chart(
        f, use_container_width=True
    )

    st.subheader("Corte longitudinal")

    fl = go.Figure()

    for item in candidates:
        segments = mesh_cut(
            item["vertices"],
            item["faces"],
            0, xcut
        )

        for j, segm in enumerate(segments):
            p, q = segm

            fl.add_trace(go.Scatter(
                x=[p[1], q[1]],
                y=[p[2], q[2]],
                mode="lines",
                line=dict(
                    color=item["color"],
                    width=3
                ),
                name=item["name"],
                showlegend=j == 0
            ))

    fl.update_layout(
        height=450,
        xaxis_title="Y (m)",
        yaxis_title="Z (m)"
    )

    st.plotly_chart(
        fl, use_container_width=True
    )

    st.caption(
        "Los cortes proceden de las mallas 3D aproximadas. "
        "Pernos, cables, malla y gunita son representaciones "
        "esquemáticas. No se calcula su contribución al FS."
    )


with tab4:
    data = {
        "version": "11",
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
        "Descargar parámetros JSON",
        json.dumps(
            data,
            indent=2,
            ensure_ascii=False
        ),
        file_name="geotecnia_mina_v11.json",
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
                    pts = list(
                        ent.get_points("xy")
                    )
                elif typ == "POLYLINE":
                    pts = [
                        p.dxf.location[:2]
                        for p in ent.vertices
                    ]
                elif typ in ("CIRCLE", "ARC"):
                    center = ent.dxf.center
                    radius = ent.dxf.radius

                    if typ == "CIRCLE":
                        angles_dxf = np.linspace(
                            0, 2*np.pi, 120
                        )
                    else:
                        start = ent.dxf.start_angle
                        end = ent.dxf.end_angle

                        if end < start:
                            end += 360

                        angles_dxf = np.deg2rad(
                            np.linspace(
                                start, end, 120
                            )
                        )

                    pts = np.column_stack((
                        center.x
                        + radius*np.cos(angles_dxf),
                        center.y
                        + radius*np.sin(angles_dxf)
                    ))

                if pts is not None:
                    p = np.array(
                        pts, dtype=float
                    )
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

            st.plotly_chart(
                fd, use_container_width=True
            )
            st.caption(
                f"{count} entidades representadas. "
                "El DXF no modifica la geometría."
            )

        except Exception as exc:
            st.error(
                f"Error al leer el DXF: {exc}"
            )
    else:
        st.info(
            "Sube un archivo DXF desde el panel lateral."
        )
