
import io
import json
import math

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from scipy import ndimage as ndi
from skimage.measure import marching_cubes

# ==========================================================
# GEOTECNIA MINA WEB - VERSION 14
# Visualizador preliminar de bloques en galerias
# ==========================================================

st.set_page_config(
    page_title="Geotecnia Mina v14",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
.block-container {
    max-width: 1600px;
    padding-top: 1rem;
}
[data-testid="stMetricValue"] {
    font-size: 1.6rem;
}
</style>
""", unsafe_allow_html=True)

st.title("Geotecnia Mina · Versión 14")
st.caption(
    "Bloques de bóveda y hastiales | "
    "Visualización 3D | Cortes 2D | Sostenimiento"
)

st.warning(
    "MODELO PRELIMINAR. La geometría utiliza planos "
    "continuos y una malla discretizada. La persistencia "
    "real y la liberación cinemática no están verificadas. "
    "No se calcula el factor de seguridad. "
    "No utilizar para dimensionar sostenimientos."
)

# ==========================================================
# 1. CONFIGURACION
# ==========================================================

with st.sidebar:

    st.header("Geometría de la galería")

    W = st.number_input(
        "Ancho (m)", 1.0, 25.0, 5.8, 0.1
    )

    H = st.number_input(
        "Altura (m)", 1.0, 25.0, 5.8, 0.1
    )

    L = st.number_input(
        "Longitud (m)", 3.0, 60.0, 15.0, 0.5
    )

    az = st.number_input(
        "Azimut del eje (°)", 0.0, 359.9, 0.0
    )

    gamma = st.number_input(
        "Peso específico (kN/m³)",
        10.0, 40.0, 26.4, 0.1
    )

    st.divider()

    st.header("Detección de bloques")

    nplanes = st.slider(
        "Planos por familia", 1, 5, 3
    )

    resolution = st.slider(
        "Resolución de malla", 30, 70, 50, 5
    )

    minvol = st.number_input(
        "Volumen mínimo visual (m³)",
        0.0, 100.0, 0.25, 0.05
    )

    show_small = st.checkbox(
        "Incluir candidatos pequeños",
        value=False
    )

    show_open = st.checkbox(
        "Mostrar regiones abiertas (diagnóstico)",
        value=False
    )

    display = st.radio(
        "Visualización 3D",
        [
            "Bloque seleccionado",
            "Hasta 2 por zona",
            "Todos los candidatos filtrados"
        ],
        index=0
    )

    st.caption(
        "Los filtros son exclusivamente visuales. "
        "Una región oculta puede requerir revisión."
    )

# ==========================================================
# 2. SECCION HERRADURA
# ==========================================================

r = W / 2.0
spring = H - r

angles = np.linspace(np.pi, 0, 121)

section = np.vstack([
    [-r, 0],
    [-r, spring],
    np.column_stack([
        r * np.cos(angles),
        spring + r * np.sin(angles)
    ]),
    [r, 0]
])

tabs = st.tabs([
    "Cuñas 3D",
    "Discontinuidades",
    "Cortes y sostenimiento",
    "Exportación y DXF"
])

t3d, tdisc, t2d, tdata = tabs

# ==========================================================
# 3. FAMILIAS DE DISCONTINUIDADES
# ==========================================================

systems = []

with tdisc:

    st.subheader("Familias de discontinuidades")

    st.info(
        "Se utilizan tres familias. El espaciamiento "
        "genera planos paralelos. La persistencia se "
        "registra, pero todavía no recorta dichos planos."
    )

    for i in range(3):

        with st.expander(
            f"Familia {i + 1}",
            expanded=(i == 0)
        ):

            a, b = st.columns(2)

            with a:

                dd = st.number_input(
                    "Dirección de buzamiento (°)",
                    0.0, 359.9,
                    [30.0, 150.0, 270.0][i],
                    key=f"dd{i}"
                )

                dip = st.number_input(
                    "Buzamiento (°)",
                    0.0, 90.0, 60.0,
                    key=f"dip{i}"
                )

                px = st.number_input(
                    "Posición transversal X (m)",
                    -W, W, 0.0,
                    key=f"px{i}"
                )

                py = st.number_input(
                    "Posición longitudinal Y (m)",
                    0.0, L,
                    L * [0.42, 0.50, 0.58][i],
                    key=f"py{i}"
                )

                spacing = st.number_input(
                    "Espaciamiento (m)",
                    0.1, 20.0, 1.5,
                    key=f"sp{i}"
                )

                persistence = st.number_input(
                    "Persistencia (m)",
                    0.1, 100.0, 5.0,
                    key=f"pers{i}"
                )

            with b:

                method = st.selectbox(
                    "Resistencia de juntas",
                    [
                        "Mohr-Coulomb",
                        "Barton-Bandis",
                        "Patton"
                    ],
                    key=f"method{i}"
                )

                if method == "Mohr-Coulomb":

                    props = {
                        "c_MPa": st.number_input(
                            "Cohesión (MPa)",
                            0.0, 20.0, 0.05,
                            key=f"c{i}"
                        ),
                        "phi_deg": st.number_input(
                            "Fricción (°)",
                            0.0, 89.0, 30.0,
                            key=f"phi{i}"
                        )
                    }

                elif method == "Barton-Bandis":

                    props = {
                        "JRC": st.number_input(
                            "JRC",
                            0.0, 25.0, 10.0,
                            key=f"jrc{i}"
                        ),
                        "JCS_MPa": st.number_input(
                            "JCS (MPa)",
                            0.1, 500.0, 30.0,
                            key=f"jcs{i}"
                        ),
                        "phi_r_deg": st.number_input(
                            "Fricción residual (°)",
                            0.0, 89.0, 25.0,
                            key=f"pr{i}"
                        )
                    }

                else:

                    props = {
                        "phi_b_deg": st.number_input(
                            "Fricción básica (°)",
                            0.0, 89.0, 25.0,
                            key=f"pb{i}"
                        ),
                        "asperidad_deg": st.number_input(
                            "Asperidad (°)",
                            0.0, 60.0, 10.0,
                            key=f"asp{i}"
                        )
                    }

                props["agua_MPa"] = st.number_input(
                    "Presión de agua (MPa)",
                    0.0, 10.0, 0.0,
                    key=f"water{i}"
                )

            systems.append({
                "dd": dd,
                "dip": dip,
                "x": px,
                "y": py,
                "spacing": spacing,
                "persistence": persistence,
                "method": method,
                "props": props
            })

# ==========================================================
# 4. CALCULO GEOMETRICO
# ==========================================================

@st.cache_data(show_spinner=False)
def calculate(
    width, height, length,
    azimuth, geos, n, res
):

    rr = width / 2.0
    ss = height - rr

    margin = max(2.0, width * 0.6)

    nx = res

    ny = min(
        105,
        max(
            35,
            round(
                res * length /
                (width + 2 * margin)
            )
        )
    )

    nz = min(
        85,
        max(
            35,
            round(
                res * (height + margin) /
                (width + 2 * margin)
            )
        )
    )

    xx = np.linspace(
        -rr - margin, rr + margin, nx
    )

    yy = np.linspace(0, length, ny)

    zz = np.linspace(
        0, height + margin, nz
    )

    dx = xx[1] - xx[0]
    dy = yy[1] - yy[0]
    dz = zz[1] - zz[0]

    X, Y, Z = np.meshgrid(
        xx, yy, zz, indexing="ij"
    )

    # Hueco de excavación de sección herradura
    tunnel = (
        (np.abs(X) <= rr)
        &
        (
            (Z <= ss)
            |
            (
                (Z > ss)
                &
                (
                    X**2 + (Z - ss)**2
                    <= rr**2
                )
            )
        )
    )

    rock = ~tunnel

    signature = np.zeros(
        X.shape,
        dtype=np.uint32
    )

    bit = 0

    for dd, dip, px, py, spacing in geos:

        theta = np.deg2rad(
            dd - azimuth
        )

        dipr = np.deg2rad(dip)

        normal = np.array([
            np.sin(theta) * np.sin(dipr),
            np.cos(theta) * np.sin(dipr),
            np.cos(dipr)
        ])

        origin = np.array([
            px, py, height
        ])

        base = normal @ origin

        signed = (
            normal[0] * X
            + normal[1] * Y
            + normal[2] * Z
            - base
        )

        for k in range(n):

            shift = (
                k - (n - 1) / 2
            ) * spacing

            signature |= (
                (signed >= shift).astype(np.uint32)
                << bit
            )

            bit += 1

    structure = ndi.generate_binary_structure(
        3, 1
    )

    adjacent_rock = rock & ndi.binary_dilation(
        tunnel,
        structure=structure
    )

    labels = np.zeros(
        rock.shape,
        dtype=np.int32
    )

    blocks = []
    next_id = 0

    for sig in np.unique(
        signature[adjacent_rock]
    ):

        components, _ = ndi.label(
            rock & (signature == sig),
            structure=structure
        )

        touching_components = np.unique(
            components[adjacent_rock]
        )

        for component_id in touching_components:

            if component_id == 0:
                continue

            region = (
                components == component_id
            )

            count = int(region.sum())

            if count < 5:
                continue

            adjacent = (
                region & adjacent_rock
            )

            roof = (
                adjacent & (Z > ss)
            )

            left = (
                adjacent
                & (X < 0)
                & (Z <= ss)
            )

            right = (
                adjacent
                & (X >= 0)
                & (Z <= ss)
            )

            contacts = {
                "Bóveda": int(roof.sum()),
                "Hastial izquierdo": int(left.sum()),
                "Hastial derecho": int(right.sum())
            }

            zone = max(
                contacts,
                key=contacts.get
            )

            if contacts[zone] == 0:
                continue

            # Si alcanza cualquier límite del dominio,
            # no es un bloque finito verificado.
            open_bound = bool(
                region[0].any()
                or region[-1].any()
                or region[:, 0].any()
                or region[:, -1].any()
                or region[:, :, 0].any()
                or region[:, :, -1].any()
            )

            ids = np.argwhere(region)

            center = [
                float(xx[ids[:, 0]].mean()),
                float(yy[ids[:, 1]].mean()),
                float(zz[ids[:, 2]].mean())
            ]

            contact_ids = np.argwhere(adjacent)

            contact_center = [
                float(xx[contact_ids[:, 0]].mean()),
                float(yy[contact_ids[:, 1]].mean()),
                float(zz[contact_ids[:, 2]].mean())
            ]

            # Indicador descriptivo, NO prueba de
            # liberación cinemática.
            if open_bound:
                status = "Región abierta"
            else:
                status = "Cerrada numéricamente"

            next_id += 1

            labels[region] = next_id

            blocks.append({
                "id": next_id,
                "zone": zone,
                "contacts": contacts,
                "volume": float(
                    count * dx * dy * dz
                ),
                "open": open_bound,
                "center": center,
                "contact_center": contact_center,
                "status": status
            })

    return labels, blocks, (
        xx, yy, zz
    )


geo = tuple(
    (
        s["dd"],
        s["dip"],
        s["x"],
        s["y"],
        s["spacing"]
    )
    for s in systems
)

with st.spinner(
    "Calculando bloques y contactos..."
):

    labels, all_blocks, axes = calculate(
        W, H, L, az,
        geo, nplanes, resolution
    )

xx, yy, zz = axes

step = (
    xx[1] - xx[0],
    yy[1] - yy[0],
    zz[1] - zz[0]
)

valid = [
    b for b in all_blocks
    if (
        show_open or not b["open"]
    )
    and (
        show_small
        or b["volume"] >= minvol
    )
]

valid.sort(
    key=lambda b: b["volume"],
    reverse=True
)

colors = [
    "#e76f51",
    "#2a9d8f",
    "#457b9d",
    "#9b5de5",
    "#e9a23b",
    "#d45087",
    "#6b8e23",
    "#b56576"
]

for j, block in enumerate(valid):
    block["color"] = colors[
        j % len(colors)
    ]

# ==========================================================
# 5. SUPERFICIE 3D
# ==========================================================

@st.cache_data(show_spinner=False)
def surface_for(
    block_id, labels, xx, yy, zz
):

    mask = np.pad(
        (labels == block_id).astype(np.uint8),
        1
    )

    if not mask.any():
        return None

    spacing = (
        xx[1] - xx[0],
        yy[1] - yy[0],
        zz[1] - zz[0]
    )

    vertices, faces, _, _ = marching_cubes(
        mask,
        level=0.5,
        spacing=spacing
    )

    vertices += np.array([
        xx[0] - spacing[0],
        yy[0] - spacing[1],
        zz[0] - spacing[2]
    ])

    return vertices, faces


# ==========================================================
# 6. VISTA 3D
# ==========================================================

selected = None
visible = []

with t3d:

    st.subheader(
        "Bloques candidatos en bóveda y hastiales"
    )

    a, b, c = st.columns(3)

    a.metric(
        "Regiones en contacto",
        len(all_blocks)
    )

    b.metric(
        "Cerradas en el dominio",
        sum(
            not q["open"]
            for q in all_blocks
        )
    )

    c.metric(
        "Regiones abiertas",
        sum(
            q["open"]
            for q in all_blocks
        )
    )

    st.caption(
        "La clasificación 'cerrada' no verifica "
        "persistencia, salida cinemática ni estabilidad."
    )

    if valid:

        options = {
            (
                f'B{q["id"]} · {q["zone"]} · '
                f'{q["volume"]:.2f} m³'
            ): q
            for q in valid
        }

        choice = st.selectbox(
            "Seleccionar bloque",
            list(options)
        )

        selected = options[choice]

        if display == "Bloque seleccionado":

            visible = [selected]

        elif display == "Hasta 2 por zona":

            for zone in [
                "Bóveda",
                "Hastial izquierdo",
                "Hastial derecho"
            ]:

                visible.extend([
                    q for q in valid
                    if q["zone"] == zone
                ][:2])

        else:

            visible = valid

        if selected["open"]:

            st.error(
                "Región abierta: el volumen está "
                "limitado artificialmente por el "
                "dominio de cálculo."
            )

        else:

            st.info(
                "Bloque cerrado dentro del dominio. "
                "Todavía no se ha verificado que "
                "pueda desprenderse."
            )

        m1, m2, m3 = st.columns(3)

        m1.metric(
            "Volumen seleccionado",
            f'{selected["volume"]:.3f} m³'
        )

        m2.metric(
            "Peso aproximado",
            f'{selected["volume"] * gamma:.1f} kN'
        )

        m3.metric(
            "Factor de seguridad",
            "No calculado"
        )

    else:

        st.warning(
            "No aparecen bloques con los filtros "
            "seleccionados. Esto no demuestra "
            "ausencia de inestabilidades."
        )

    fig = go.Figure()

    # Contorno de la galería en sus extremos
    for y in [0, L]:

        fig.add_trace(go.Scatter3d(
            x=section[:, 0],
            y=np.full(len(section), y),
            z=section[:, 1],
            mode="lines",
            line=dict(
                color="#475569",
                width=7
            ),
            showlegend=False
        ))

    # Generatrices de la excavación
    for k in [0, 24, 48, 72, 96, 120]:

        fig.add_trace(go.Scatter3d(
            x=[section[k, 0]] * 2,
            y=[0, L],
            z=[section[k, 1]] * 2,
            mode="lines",
            line=dict(
                color="#cbd5e1",
                width=2
            ),
            showlegend=False
        ))

    for block in visible:

        result = surface_for(
            block["id"],
            labels, xx, yy, zz
        )

        if result is None:
            continue

        vertices, faces = result

        fig.add_trace(go.Mesh3d(
            x=vertices[:, 0],
            y=vertices[:, 1],
            z=vertices[:, 2],
            i=faces[:, 0],
            j=faces[:, 1],
            k=faces[:, 2],
            color=block["color"],
            opacity=0.78,
            flatshading=True,
            name=f'B{block["id"]}',
            hovertemplate=(
                f'Bloque B{block["id"]}<br>'
                f'{block["zone"]}<br>'
                f'{block["volume"]:.3f} m³'
                '<extra></extra>'
            )
        ))

    fig.update_layout(
        height=720,
        margin=dict(
            l=0, r=0, t=0, b=0
        ),
        showlegend=False,
        scene=dict(
            xaxis=dict(
                title="X (m)",
                showbackground=False
            ),
            yaxis=dict(
                title="Y (m)",
                showbackground=False
            ),
            zaxis=dict(
                title="Z (m)",
                showbackground=False
            ),
            aspectmode="data",
            camera=dict(
                eye=dict(
                    x=1.3,
                    y=-1.5,
                    z=0.85
                ),
                projection=dict(
                    type="orthographic"
                )
            )
        )
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
        config={
            "scrollZoom": True,
            "displaylogo": False
        }
    )

    st.subheader("Resultados geométricos")

    st.dataframe(
        [
            {
                "Bloque": f'B{q["id"]}',
                "Zona": q["zone"],
                "Volumen (m³)": round(
                    q["volume"], 3
                ),
                "Peso (kN)": round(
                    q["volume"] * gamma, 1
                ),
                "Estado": q["status"],
                "FS": "No calculado"
            }
            for q in valid
        ],
        hide_index=True,
        use_container_width=True
    )

    with st.expander(
        "Consultar todas las regiones detectadas"
    ):

        st.dataframe(
            [
                {
                    "Bloque": f'B{q["id"]}',
                    "Zona": q["zone"],
                    "Volumen (m³)": round(
                        q["volume"], 3
                    ),
                    "Abierto": q["open"]
                }
                for q in all_blocks
            ],
            hide_index=True,
            use_container_width=True
        )

# ==========================================================
# 7. SOSTENIMIENTOS
# ==========================================================

presets = {
    "ST-I": (2.0, 0.03, 0.0, False),
    "ST-II": (2.0, 0.03, 0.0, False),
    "ST-III": (1.5, 0.08, 0.0, True),
    "ST-IV superficial": (
        1.5, 0.12, 8.0, True
    ),
    "ST-IV profundo": (
        1.5, 0.15, 10.0, True
    ),
    "Personalizado": (
        1.5, 0.08, 0.0, True
    )
}

# ==========================================================
# 8. CORTES 2D
# ==========================================================

with t2d:

    st.subheader(
        "Cortes vinculados al modelo 3D"
    )

    if selected:

        st.caption(
            f'Bloque seleccionado: B{selected["id"]} '
            f'· {selected["zone"]}'
        )

    preset = st.selectbox(
        "Tipo de sostenimiento",
        list(presets)
    )

    boltsp, shotdefault, cabledefault, meshdefault = (
        presets[preset]
    )

    ca, cb = st.columns(2)

    with ca:

        bolt_type = st.selectbox(
            "Tipo de perno",
            [
                "Resina",
                "Split Set",
                "Swellex",
                "Sin pernos"
            ]
        )

        boltlen = st.number_input(
            "Longitud pernos (m)",
            0.5, 12.0, 2.4
        )

        boltsp = st.number_input(
            "Separación transversal pernos (m)",
            0.3, 5.0, boltsp
        )

        boltlong = st.number_input(
            "Separación longitudinal pernos (m)",
            0.3, 5.0, 1.5
        )

        boltcap = st.number_input(
            "Capacidad nominal perno (kN)",
            1.0, 1000.0, 324.0
        )

    with cb:

        shot = st.checkbox(
            "Gunita", True
        )

        shotcm = st.number_input(
            "Espesor gunita (cm)",
            0.0, 50.0,
            shotdefault * 100,
            0.5
        )

        fc = st.number_input(
            "fc gunita (MPa)",
            1.0, 100.0, 35.0
        )

        ft = st.number_input(
            "ft gunita (MPa)",
            0.0, 20.0, 3.0
        )

        E = st.number_input(
            "E gunita (GPa)",
            1.0, 80.0, 30.0
        )

        mesh = st.checkbox(
            "Malla metálica",
            meshdefault
        )

        cables = st.checkbox(
            "Cables de anclaje",
            cabledefault > 0
        )

        cablelen = st.number_input(
            "Longitud cables (m)",
            1.0, 25.0,
            cabledefault or 8.0
        )

        cablesp = st.number_input(
            "Separación cables (m)",
            0.3, 6.0,
            1.0 if preset == "ST-IV profundo"
            else 1.5
        )

    if selected:

        ycenter = selected[
            "contact_center"
        ][1]

        xcenter = selected[
            "contact_center"
        ][0]

    else:

        ycenter = L / 2
        xcenter = 0.0

    ycut = st.slider(
        "Corte transversal Y (m)",
        0.0,
        float(L),
        float(np.clip(
            round(ycenter, 1), 0, L
        )),
        0.1
    )

    xcut = st.slider(
        "Corte longitudinal X (m)",
        float(xx[0]),
        float(xx[-1]),
        float(np.clip(
            round(xcenter, 1),
            xx[0], xx[-1]
        )),
        0.1
    )

    iy = int(np.argmin(
        abs(yy - ycut)
    ))

    ix = int(np.argmin(
        abs(xx - xcut)
    ))

    f = go.Figure()

    # Bloques rellenos en sección transversal
    for block in visible:

        mask = (
            labels[:, iy, :]
            == block["id"]
        ).T.astype(float)

        if not mask.any():
            continue

        f.add_trace(go.Heatmap(
            x=xx,
            y=zz,
            z=np.where(
                mask > 0, 1.0, np.nan
            ),
            colorscale=[
                [0, block["color"]],
                [1, block["color"]]
            ],
            opacity=0.35,
            showscale=False,
            hoverinfo="skip",
            name=f'B{block["id"]}'
        ))

        f.add_trace(go.Contour(
            x=xx,
            y=zz,
            z=mask,
            contours=dict(
                start=0.5,
                end=0.5,
                size=1,
                coloring="none"
            ),
            line=dict(
                color=block["color"],
                width=3
            ),
            showscale=False,
            showlegend=False
        ))

    # Excavación
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

    # Gunita con desplazamiento normal exterior
    if shot and shotcm > 0:

        points = section[1:-1].copy()

        normals = []

        for p in points:

            if p[1] <= spring + 1e-8:

                normals.append([
                    np.sign(p[0]), 0.0
                ])

            else:

                v = np.array([
                    p[0],
                    p[1] - spring
                ])

                normals.append(
                    v / max(
                        np.linalg.norm(v),
                        1e-9
                    )
                )

        ext = (
            points
            + shotcm / 100
            * np.array(normals)
        )

        f.add_trace(go.Scatter(
            x=ext[:, 0],
            y=ext[:, 1],
            mode="lines",
            line=dict(
                color="#f97316",
                width=7
            ),
            name="Gunita"
        ))

    if mesh:

        f.add_trace(go.Scatter(
            x=section[1:-1, 0],
            y=section[1:-1, 1],
            mode="lines",
            line=dict(
                color="#16a34a",
                dash="dot",
                width=3
            ),
            name="Malla"
        ))

    # Pernos: representación transversal esquemática
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
                boltsp / 2,
                cumulative[-1],
                boltsp
            )
        ):

            idx = int(np.clip(
                np.searchsorted(
                    cumulative, s
                ) - 1,
                0, len(lengths) - 1
            ))

            t = (
                s - cumulative[idx]
            ) / max(
                lengths[idx], 1e-9
            )

            p = (
                section[idx] * (1 - t)
                + section[idx + 1] * t
            )

            if p[1] <= spring:

                normal2 = np.array([
                    np.sign(p[0]), 0.0
                ])

            else:

                normal2 = np.array([
                    p[0],
                    p[1] - spring
                ])

                normal2 /= max(
                    np.linalg.norm(normal2),
                    1e-9
                )

            q = p + boltlen * normal2

            f.add_trace(go.Scatter(
                x=[p[0], q[0]],
                y=[p[1], q[1]],
                mode="lines",
                line=dict(
                    color="#2563eb",
                    width=4
                ),
                name="Pernos",
                showlegend=(k == 0)
            ))

    # Cables: representación esquemática
    if cables:

        for k, s in enumerate(
            np.arange(
                cablesp / 2,
                math.pi * r,
                cablesp
            )
        ):

            angle = (
                math.pi - s / r
            )

            p = np.array([
                r * math.cos(angle),
                spring + r * math.sin(angle)
            ])

            q = (
                p
                + cablelen
                * np.array([
                    math.cos(angle),
                    math.sin(angle)
                ])
            )

            f.add_trace(go.Scatter(
                x=[p[0], q[0]],
                y=[p[1], q[1]],
                mode="lines",
                line=dict(
                    color="#9333ea",
                    width=4
                ),
                name="Cables",
                showlegend=(k == 0)
            ))

    f.update_layout(
        height=680,
        xaxis=dict(
            title="X (m)",
            scaleanchor="y",
            scaleratio=1
        ),
        yaxis_title="Z (m)",
        legend=dict(
            orientation="h"
        ),
        margin=dict(
            l=30, r=20,
            t=20, b=30
        )
    )

    st.plotly_chart(
        f,
        use_container_width=True
    )

    # ======================================================
    # CORTE LONGITUDINAL
    # ======================================================

    st.subheader(
        "Corte longitudinal"
    )

    fl = go.Figure()

    for block in visible:

        mask = (
            labels[ix, :, :]
            == block["id"]
        ).T.astype(float)

        if not mask.any():
            continue

        fl.add_trace(go.Heatmap(
            x=yy,
            y=zz,
            z=np.where(
                mask > 0, 1.0, np.nan
            ),
            colorscale=[
                [0, block["color"]],
                [1, block["color"]]
            ],
            opacity=0.4,
            showscale=False,
            hoverinfo="skip"
        ))

        fl.add_trace(go.Contour(
            x=yy,
            y=zz,
            z=mask,
            contours=dict(
                start=0.5,
                end=0.5,
                size=1,
                coloring="none"
            ),
            line=dict(
                color=block["color"],
                width=3
            ),
            showscale=False
        ))

    fl.update_layout(
        height=440,
        xaxis_title="Y (m)",
        yaxis_title="Z (m)",
        margin=dict(
            l=30, r=20,
            t=20, b=30
        )
    )

    st.plotly_chart(
        fl,
        use_container_width=True
    )

    st.caption(
        "Los bloques 2D proceden de las mismas celdas "
        "que el modelo 3D. Pernos y cables son esquemas "
        "de una sección, no intersecciones 3D verificadas. "
        "No se calcula su aportación al FS."
    )

# ==========================================================
# 9. EXPORTACION Y DXF
# ==========================================================

with tdata:

    st.subheader(
        "Exportación de parámetros"
    )

    data = {
        "version": 14,
        "galeria": {
            "ancho": W,
            "alto": H,
            "longitud": L,
            "azimut": az,
            "peso_especifico": gamma
        },
        "familias": systems
    }

    st.download_button(
        "Exportar parámetros JSON",
        json.dumps(
            data,
            ensure_ascii=False,
            indent=2
        ),
        file_name="geotecnia_v14.json",
        mime="application/json"
    )

    dxf = st.file_uploader(
        "Sección AutoCAD DXF",
        type=["dxf"]
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

            for ent in doc.modelspace():

                kind = ent.dxftype()
                pts = None

                if kind == "LINE":

                    pts = np.array([
                        ent.dxf.start[:2],
                        ent.dxf.end[:2]
                    ], float)

                elif kind == "LWPOLYLINE":

                    pts = np.array(
                        list(ent.get_points("xy")),
                        float
                    )

                elif kind == "CIRCLE":

                    a = np.linspace(
                        0, 2 * np.pi, 120
                    )

                    p = ent.dxf.center
                    rad = ent.dxf.radius

                    pts = np.column_stack([
                        p.x + rad * np.cos(a),
                        p.y + rad * np.sin(a)
                    ])

                if pts is not None:

                    fd.add_trace(
                        go.Scatter(
                            x=pts[:, 0],
                            y=pts[:, 1],
                            mode="lines",
                            showlegend=False
                        )
                    )

            fd.update_layout(
                height=500,
                xaxis=dict(
                    scaleanchor="y",
                    scaleratio=1
                )
            )

            st.plotly_chart(
                fd,
                use_container_width=True
            )

            st.caption(
                "El DXF es una referencia visual. "
                "No modifica la geometría de cálculo."
            )

        except Exception as exc:

            st.error(
                f"Error al leer DXF: {exc}"
            )
