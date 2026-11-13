
import io
import json
import math

import numpy as np
import plotly.graph_objects as go
import streamlit as st
from scipy import ndimage as ndi
from skimage.measure import marching_cubes

st.set_page_config(page_title="Geotecnia Mina v12", layout="wide")
st.title("Geotecnia Mina · v12")
st.caption("Bloques de bóveda y hastiales · Modelo 3D y cortes 2D")
st.warning(
    "PROTOTIPO GEOMÉTRICO. Los bloques se obtienen mediante "
    "discretización espacial y planos continuos. No se verifica "
    "persistencia real, liberación cinemática ni factor de seguridad. "
    "No utilizar para dimensionamiento."
)

with st.sidebar:
    st.header("Geometría de la galería")
    W = st.number_input("Ancho (m)", 1.0, 25.0, 5.8, 0.1)
    H = st.number_input("Altura (m)", 1.0, 25.0, 5.8, 0.1)
    L = st.number_input("Longitud (m)", 3.0, 60.0, 15.0, 0.5)
    az = st.number_input("Azimut del eje (°)", 0.0, 359.9, 0.0)
    gamma = st.number_input(
        "Peso específico (kN/m³)", 10.0, 40.0, 26.4, 0.1
    )

    st.divider()
    st.header("Control de cuñas")
    view = st.radio(
        "Modo de visualización",
        ["Revisión de seguridad", "Vista simplificada"]
    )
    nplanes = st.slider("Planos por familia", 1, 5, 3)
    resolution = st.slider("Resolución del cálculo", 35, 75, 55, 5)
    minvol = st.number_input(
        "Volumen mínimo visual (m³)", 0.0, 100.0, 0.25, 0.05
    )
    show_all = st.checkbox(
        "Mostrar también bloques pequeños", value=True
    )
    show_open = st.checkbox(
        "Mostrar bloques abiertos en el límite del modelo",
        value=True
    )
    maxzone = st.slider(
        "Máximo por zona en vista simplificada", 1, 8, 3
    )
    dxf = st.file_uploader(
        "Sección AutoCAD DXF", type=["dxf"]
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

t1, t2, t3, t4 = st.tabs([
    "Discontinuidades",
    "Bloques 3D",
    "Cortes y sostenimiento",
    "Datos y DXF"
])

systems = []

with t1:
    st.info(
        "Las familias generan planos sucesivos según su "
        "espaciamiento. La persistencia se registra, pero "
        "todavía no limita geométricamente los planos."
    )

    for i in range(3):
        with st.expander(
            f"Familia {i + 1}", expanded=i == 0
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
                posy = st.number_input(
                    "Posición longitudinal Y (m)",
                    0.0, L,
                    L * [0.42, 0.50, 0.58][i],
                    key=f"y{i}"
                )
                posx = st.number_input(
                    "Posición transversal X (m)",
                    -W, W, 0.0,
                    key=f"x{i}"
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
                    [
                        "Mohr-Coulomb",
                        "Barton-Bandis",
                        "Patton"
                    ],
                    key=f"m{i}"
                )

                if method == "Mohr-Coulomb":
                    props = {
                        "c_MPa": st.number_input(
                            "Cohesión (MPa)",
                            0.0, 20.0, 0.05,
                            key=f"c{i}"
                        ),
                        "phi": st.number_input(
                            "Fricción (°)",
                            0.0, 89.0, 30.0,
                            key=f"ph{i}"
                        )
                    }
                elif method == "Barton-Bandis":
                    props = {
                        "JRC": st.number_input(
                            "JRC", 0.0, 25.0, 10.0,
                            key=f"jrc{i}"
                        ),
                        "JCS_MPa": st.number_input(
                            "JCS (MPa)",
                            0.1, 500.0, 30.0,
                            key=f"jcs{i}"
                        ),
                        "phi_r": st.number_input(
                            "Fricción residual (°)",
                            0.0, 89.0, 25.0,
                            key=f"pr{i}"
                        )
                    }
                else:
                    props = {
                        "phi_b": st.number_input(
                            "Fricción básica (°)",
                            0.0, 89.0, 25.0,
                            key=f"pb{i}"
                        ),
                        "asperidad": st.number_input(
                            "Asperidad (°)",
                            0.0, 60.0, 10.0,
                            key=f"as{i}"
                        )
                    }

                props["agua_MPa"] = st.number_input(
                    "Presión de agua (MPa)",
                    0.0, 10.0, 0.0,
                    key=f"w{i}"
                )

            systems.append(dict(
                dd=dd,
                dip=dip,
                y=posy,
                x=posx,
                spacing=spacing,
                persistence=persistence,
                method=method,
                props=props
            ))


@st.cache_data(show_spinner=False)
def calculate(width, height, length, azimuth, specs, n, res):
    rr = width / 2
    ss = height - rr
    margin = max(2.5, width * 0.65)

    nx = res
    ny = max(
        35, int(res * length / (width + 2 * margin))
    )
    nz = max(
        35, int(res * (height + margin) / (width + 2 * margin))
    )

    nx = min(nx, 75)
    ny = min(ny, 110)
    nz = min(nz, 85)

    xx = np.linspace(-rr - margin, rr + margin, nx)
    yy = np.linspace(0, length, ny)
    zz = np.linspace(0, height + margin, nz)

    dx = xx[1] - xx[0]
    dy = yy[1] - yy[0]
    dz = zz[1] - zz[0]

    X, Y, Z = np.meshgrid(
        xx, yy, zz, indexing="ij"
    )

    inside = (
        (np.abs(X) < rr)
        & (
            (Z < ss)
            | (
                (X * X + (Z - ss) ** 2 < rr * rr)
                & (Z >= ss)
            )
        )
    )

    rock = ~inside
    signature = np.zeros(X.shape, dtype=np.uint32)
    bit = 0

    for s in specs:
        theta = np.deg2rad(s["dd"] - azimuth)
        dip = np.deg2rad(s["dip"])

        normal = np.array([
            np.sin(theta) * np.sin(dip),
            np.cos(theta) * np.sin(dip),
            np.cos(dip)
        ])

        origin = np.array([
            s["x"], s["y"], height
        ])

        base = float(normal @ origin)

        for k in range(n):
            shift = (
                k - (n - 1) / 2
            ) * s["spacing"]

            signed = (
                normal[0] * X
                + normal[1] * Y
                + normal[2] * Z
                - (base + shift)
            )

            signature |= (
                (signed >= 0).astype(np.uint32) << bit
            )
            bit += 1

    contact = rock & ndi.binary_dilation(
        inside, iterations=1
    )

    labels = np.zeros(rock.shape, dtype=np.int32)
    blocks = []
    nextid = 0

    structure = ndi.generate_binary_structure(3, 1)

    for sig in np.unique(signature[contact]):
        mask = rock & (signature == sig)

        components, count = ndi.label(
            mask, structure=structure
        )

        touched = np.unique(components[contact])

        for comp in touched:
            if comp == 0:
                continue

            region = components == comp
            count_vox = int(np.count_nonzero(region))

            if count_vox < 5:
                continue

            adj = region & ndi.binary_dilation(
                inside, iterations=1
            )

            roof_zone = adj & (Z >= ss)
            left_zone = (
                adj & (X < 0) & (Z < ss)
            )
            right_zone = (
                adj & (X >= 0) & (Z < ss)
            )

            contacts = {
                "Bóveda": int(roof_zone.sum()),
                "Hastial izquierdo": int(left_zone.sum()),
                "Hastial derecho": int(right_zone.sum())
            }

            zone = max(contacts, key=contacts.get)

            open_boundary = bool(
                np.any(region[0])
                or np.any(region[-1])
                or np.any(region[:, 0])
                or np.any(region[:, -1])
                or np.any(region[:, :, 0])
                or np.any(region[:, :, -1])
            )

            nextid += 1
            labels[region] = nextid

            indices = np.argwhere(region)

            center = np.array([
                xx[indices[:, 0]].mean(),
                yy[indices[:, 1]].mean(),
                zz[indices[:, 2]].mean()
            ])

            blocks.append({
                "id": nextid,
                "zone": zone,
                "contacts": contacts,
                "volume": count_vox * dx * dy * dz,
                "center": center.tolist(),
                "open": open_boundary,
                "count": count_vox
            })

    return labels, blocks, (xx, yy, zz)


specs = tuple(
    (
        s["dd"],
        s["dip"],
        s["x"],
        s["y"],
        s["spacing"]
    )
    for s in systems
)

geometry_specs = tuple(
    dict(
        dd=a,
        dip=b,
        x=c,
        y=d,
        spacing=e
    )
    for a, b, c, d, e in specs
)

with st.spinner(
    "Calculando regiones y contactos con la excavación..."
):
    labels, all_blocks, axes = calculate(
        W, H, L, az,
        geometry_specs,
        nplanes,
        resolution
    )

xx, yy, zz = axes

step = (
    xx[1] - xx[0],
    yy[1] - yy[0],
    zz[1] - zz[0]
)

blocks = [
    b for b in all_blocks
    if (
        show_all or b["volume"] >= minvol
    )
    and (
        show_open or not b["open"]
    )
]

blocks.sort(
    key=lambda b: b["volume"],
    reverse=True
)

if view == "Vista simplificada":
    selected = []

    for zone in [
        "Bóveda",
        "Hastial izquierdo",
        "Hastial derecho"
    ]:
        selected += [
            b for b in blocks
            if b["zone"] == zone
        ][:maxzone]

    blocks = selected

colors = [
    "#e76f51",
    "#2a9d8f",
    "#457b9d",
    "#9b5de5",
    "#f4a261",
    "#ef476f",
    "#06d6a0",
    "#118ab2",
    "#b56576"
]

for i, b in enumerate(blocks):
    b["color"] = colors[i % len(colors)]


def surface(block_id):
    mask = (labels == block_id).astype(np.uint8)
    padded = np.pad(mask, 1)

    if padded.max() == 0:
        return None

    vertices, faces, _, _ = marching_cubes(
        padded,
        level=0.5,
        spacing=step
    )

    vertices += np.array([
        xx[0] - step[0],
        yy[0] - step[1],
        zz[0] - step[2]
    ])

    return vertices, faces


with t2:
    st.subheader(
        "Bloques que contactan con bóveda y hastiales"
    )

    a, b, c = st.columns(3)

    a.metric(
        "Regiones identificadas",
        len(all_blocks)
    )
    b.metric(
        "Regiones visibles",
        len(blocks)
    )
    c.metric(
        "Regiones abiertas",
        sum(x["open"] for x in all_blocks)
    )

    if any(b["open"] for b in blocks):
        st.warning(
            "Algunas regiones alcanzan el límite exterior "
            "del dominio. Su volumen está truncado "
            "artificialmente y no representa una cuña cerrada."
        )

    fig = go.Figure()

    for y in [0, L]:
        fig.add_trace(go.Scatter3d(
            x=section[:, 0],
            y=np.full(len(section), y),
            z=section[:, 1],
            mode="lines",
            line=dict(
                color="#475569",
                width=6
            ),
            name="Galería",
            showlegend=y == 0
        ))

    for k in [0, 24, 48, 72, 96, 119]:
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

    for block in blocks:
        surf = surface(block["id"])

        if surf is None:
            continue

        v, faces = surf

        fig.add_trace(go.Mesh3d(
            x=v[:, 0],
            y=v[:, 1],
            z=v[:, 2],
            i=faces[:, 0],
            j=faces[:, 1],
            k=faces[:, 2],
            color=block["color"],
            opacity=0.65,
            flatshading=True,
            name=f'B{block["id"]} · {block["zone"]}',
            hovertemplate=(
                f'B{block["id"]}<br>'
                f'{block["zone"]}<br>'
                f'Volumen discretizado: '
                f'{block["volume"]:.2f} m³<br>'
                f'Abierto: '
                f'{"Sí" if block["open"] else "No"}'
                '<extra></extra>'
            )
        ))

    fig.update_layout(
        height=780,
        margin=dict(
            l=0, r=0, t=0, b=0
        ),
        showlegend=False,
        scene=dict(
            xaxis_title="X (m)",
            yaxis_title="Y (m)",
            zaxis_title="Z (m)",
            aspectmode="manual",
            aspectratio=dict(
                x=1,
                y=max(1, L / W),
                z=1.25
            ),
            camera=dict(
                eye=dict(
                    x=1.7,
                    y=-1.5,
                    z=1.0
                )
            )
        )
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
        config={"scrollZoom": True}
    )

    rows = []

    for block in blocks:
        rows.append({
            "Bloque": f'B{block["id"]}',
            "Zona principal": block["zone"],
            "Volumen discretizado (m³)": round(
                block["volume"], 3
            ),
            "Peso (kN)": round(
                block["volume"] * gamma, 1
            ),
            "Límite exterior": (
                "Sí (no cerrado)"
                if block["open"] else "No"
            ),
            "FS": "No calculado"
        })

    st.dataframe(
        rows,
        hide_index=True,
        use_container_width=True
    )

    with st.expander(
        "Consultar todas las regiones, incluidas las ocultas"
    ):
        st.dataframe([
            {
                "Bloque": f'B{x["id"]}',
                "Zona": x["zone"],
                "Volumen (m³)": round(x["volume"], 3),
                "Límite exterior": x["open"]
            }
            for x in sorted(
                all_blocks,
                key=lambda q: q["volume"],
                reverse=True
            )
        ], hide_index=True, use_container_width=True)


presets = {
    "ST-I": (2.0, 0.03, 0.0, False),
    "ST-II": (2.0, 0.03, 0.0, False),
    "ST-III": (1.5, 0.08, 0.0, True),
    "ST-IV superficial": (1.5, 0.12, 8.0, True),
    "ST-IV profundo": (1.5, 0.15, 10.0, True),
    "Personalizado": (1.5, 0.08, 0.0, True)
}

with t3:
    st.subheader(
        "Corte transversal y sostenimiento"
    )

    preset = st.selectbox(
        "Tipo de sostenimiento",
        list(presets)
    )

    boltsp, shotdefault, cabledefault, meshdefault = (
        presets[preset]
    )

    a, b = st.columns(2)

    with a:
        bolt_type = st.selectbox(
            "Pernos",
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
        bolt_long = st.number_input(
            "Separación longitudinal pernos (m)",
            0.3, 5.0, 1.5
        )
        bolt_cap = st.number_input(
            "Capacidad nominal perno (kN)",
            1.0, 1000.0, 324.0
        )

    with b:
        shot_on = st.checkbox("Gunita", True)
        shot_cm = st.number_input(
            "Espesor gunita (cm)",
            0.0, 50.0,
            shotdefault * 100,
            0.5,
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
        mesh_on = st.checkbox(
            "Malla", meshdefault
        )
        cable_on = st.checkbox(
            "Cables",
            cabledefault > 0
        )
        cablelen = st.number_input(
            "Longitud cables (m)",
            1.0, 25.0,
            cabledefault or 8.0,
            disabled=not cable_on
        )
        cablesp = st.number_input(
            "Separación transversal cables (m)",
            0.3, 6.0,
            1.0 if preset == "ST-IV profundo" else 1.5
        )

    if blocks:
        options = {
            f'B{x["id"]} · {x["zone"]}': x
            for x in blocks
        }

        choice = st.selectbox(
            "Centrar cortes en un bloque",
            list(options)
        )

        chosen = options[choice]

        y_default = float(
            yy[np.argmin(
                abs(yy - chosen["center"][1])
            )]
        )
        x_default = float(
            xx[np.argmin(
                abs(xx - chosen["center"][0])
            )]
        )
    else:
        y_default = L / 2
        x_default = 0.0

    ycut = st.number_input(
        "Corte transversal Y (m)",
        0.0, float(L),
        float(y_default), 0.1
    )

    xcut = st.number_input(
        "Corte longitudinal X (m)",
        float(xx[0]),
        float(xx[-1]),
        float(x_default), 0.1
    )

    iy = int(np.argmin(abs(yy - ycut)))
    ix = int(np.argmin(abs(xx - xcut)))

    f = go.Figure()

    for block in blocks:
        mask = (
            labels[:, iy, :] == block["id"]
        ).T.astype(int)

        if np.any(mask):
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
                name=f'B{block["id"]}',
                showlegend=True
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
            np.sign(ext[1:-1, 0])
            * shot_cm / 100
        )
        ext[1:-1, 1] += shot_cm / 100

        f.add_trace(go.Scatter(
            x=ext[:, 0],
            y=ext[:, 1],
            mode="lines",
            line=dict(
                color="#f97316",
                width=7
            ),
            name="Gunita (esquema)"
        ))

    if mesh_on:
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

    seg = np.linalg.norm(
        np.diff(section, axis=0),
        axis=1
    )
    cum = np.r_[0, np.cumsum(seg)]

    if bolt_type != "Sin pernos":
        for k, s in enumerate(
            np.arange(
                boltsp / 2,
                cum[-1],
                boltsp
            )
        ):
            i = int(np.clip(
                np.searchsorted(cum, s) - 1,
                0, len(seg) - 1
            ))

            t = (
                (s - cum[i])
                / max(seg[i], 1e-9)
            )

            p = (
                section[i] * (1 - t)
                + section[i + 1] * t
            )

            if (
                p[1] <= spring
                and abs(p[0]) > r - 0.01
            ):
                normal2 = np.array([
                    np.sign(p[0]), 0.0
                ])
            else:
                normal2 = np.array([
                    p[0], p[1] - spring
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
                showlegend=k == 0
            ))

    if cable_on:
        for k, s in enumerate(
            np.arange(
                cablesp / 2,
                math.pi * r,
                cablesp
            )
        ):
            angle = math.pi - s / r

            p = np.array([
                r * math.cos(angle),
                spring + r * math.sin(angle)
            ])

            q = p + cablelen * np.array([
                math.cos(angle),
                math.sin(angle)
            ])

            f.add_trace(go.Scatter(
                x=[p[0], q[0]],
                y=[p[1], q[1]],
                mode="lines",
                line=dict(
                    color="#9333ea",
                    width=4
                ),
                name="Cables",
                showlegend=k == 0
            ))

    f.update_layout(
        height=650,
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

    st.subheader(
        "Corte longitudinal de las mismas regiones"
    )

    fl = go.Figure()

    for block in blocks:
        mask = (
            labels[ix, :, :] == block["id"]
        ).T.astype(int)

        if np.any(mask):
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
                showscale=False,
                name=f'B{block["id"]}'
            ))

    fl.update_layout(
        height=440,
        xaxis_title="Y (m)",
        yaxis_title="Z (m)"
    )

    st.plotly_chart(
        fl, use_container_width=True
    )

    st.caption(
        "Los cortes proceden de las mismas regiones "
        "discretizadas del modelo 3D. Los anclajes son "
        "esquemáticos; no se calcula su eficacia resistente."
    )


with t4:
    data = {
        "version": 12,
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
        file_name="geotecnia_v12.json",
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

            for e in doc.modelspace():
                typ = e.dxftype()

                if typ == "LINE":
                    pts = np.array([
                        e.dxf.start[:2],
                        e.dxf.end[:2]
                    ])

                elif typ == "LWPOLYLINE":
                    pts = np.array(
                        list(e.get_points("xy"))
                    )

                elif typ == "CIRCLE":
                    a = np.linspace(
                        0, 2 * np.pi, 100
                    )
                    p = e.dxf.center
                    rr = e.dxf.radius

                    pts = np.column_stack((
                        p.x + rr * np.cos(a),
                        p.y + rr * np.sin(a)
                    ))

                else:
                    continue

                fd.add_trace(go.Scatter(
                    x=pts[:, 0],
                    y=pts[:, 1],
                    mode="lines",
                    showlegend=False
                ))

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
                "DXF únicamente como referencia. "
                "No modifica la geometría de cálculo."
            )

        except Exception as exc:
            st.error(
                f"No se pudo leer el DXF: {exc}"
            )
    else:
        st.info(
            "Sube un DXF desde el panel lateral "
            "para visualizarlo."
        )
