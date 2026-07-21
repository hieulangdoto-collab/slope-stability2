"""
Streamlit app for the Slope Stability LE Solver.

Run locally:   streamlit run app.py
Deploy online: push this repo to GitHub, then deploy on share.streamlit.io
               (entrypoint file: app.py)
"""
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from le_solver import (Material, Layer, Surcharge, PointLoad, Reinforcement,
                        factor_of_safety, generate_slices)
from visualize import plot_cross_section
from grid_search import grid_search, refine_with_rigorous_methods
from pso_search import pso_search

st.set_page_config(page_title="Slope Stability LE Solver", layout="wide")

# ---------------------------------------------------------------------------
# Fixed ground profile (2:1 slope, height 10 m) -- edit here if you want a
# different geometry; a future version could expose this in the sidebar too.
# ---------------------------------------------------------------------------
H = 10.0
SLOPE = 2.0
GROUND = [(-100.0, H), (0.0, H), (H * SLOPE, 0.0), (120.0, 0.0)]

st.title("Slope Stability — Limit Equilibrium Solver")
st.caption(
    "Fellenius / Bishop / Spencer / Morgenstern-Price · nhiều lớp vật liệu · "
    "tải trọng ngoài · gia cố · động đất giả tĩnh — xem `le_solver.py` để biết chi tiết lý thuyết."
)

# ---------------------------------------------------------------------------
# Sidebar: trial circle, materials, water table, loads, seismic
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("Vòng tròn thử")
    xc = st.slider("xc (m)", -15.0, 35.0, 16.0, 0.5)
    yc = st.slider("yc (m)", 8.0, 40.0, 20.0, 0.5)
    R = st.slider("R (m)", 5.0, 45.0, 20.5, 0.5)

    st.header("Lớp vật liệu")
    n_layers = st.number_input("Số lớp", 1, 4, 1)
    layers = []
    for i in range(n_layers):
        with st.expander(f"Lớp {i+1}", expanded=(i == 0)):
            c = st.number_input(f"c' (kPa) — lớp {i+1}", 0.0, 200.0, 5.0 if i == 0 else 20.0, key=f"c{i}")
            phi = st.number_input(f"φ' (°) — lớp {i+1}", 0.0, 45.0, 20.0 if i == 0 else 30.0, key=f"phi{i}")
            gamma = st.number_input(f"γ (kN/m³) — lớp {i+1}", 10.0, 25.0, 19.0, key=f"gamma{i}")
            if i < n_layers - 1:
                thickness = st.number_input(f"Độ dày (m) — lớp {i+1}", 0.5, 30.0, 4.0, key=f"th{i}")
            else:
                thickness = None
        mat = Material(f"Lop {i+1}", c=c, phi_deg=phi, unit_weight=gamma)
        layers.append((mat, thickness))

    # convert thickness (offset from ground) into absolute lower-boundary polylines
    layer_objs = []
    cum_offset = 0.0
    for i, (mat, thickness) in enumerate(layers):
        if thickness is None:
            layer_objs.append(Layer(mat, lower_boundary=None))
        else:
            cum_offset += thickness
            boundary = [(x, y - cum_offset) for x, y in GROUND]
            layer_objs.append(Layer(mat, lower_boundary=boundary))

    st.header("Mực nước ngầm")
    piezo_ratio = st.slider("Tỉ lệ mực nước (0=khô, 1=đầy)", 0.0, 1.0, 0.0, 0.05)
    piezo_pts = None
    if piezo_ratio > 0:
        PZ_MAX_DEPTH = 12.0
        piezo_pts = [(x, y - (1 - piezo_ratio) * PZ_MAX_DEPTH) for x, y in GROUND]

    st.header("Động đất giả tĩnh")
    kh = st.slider("kh", 0.0, 0.3, 0.0, 0.01)
    kv = st.slider("kv", -0.1, 0.1, 0.0, 0.01)

    st.header("Tải trọng ngoài")
    add_surcharge = st.checkbox("Thêm surcharge")
    surcharges = None
    if add_surcharge:
        sx0 = st.number_input("x bắt đầu", value=-1.5)
        sx1 = st.number_input("x kết thúc", value=4.0)
        sq = st.number_input("q (kPa)", value=30.0)
        surcharges = [Surcharge(x_start=sx0, x_end=sx1, q=sq)]

    st.header("Gia cố")
    add_reinf = st.checkbox("Thêm neo/gia cố")
    reinforcements = None
    if add_reinf:
        n_reinf = st.number_input("Số neo", 1, 5, 2)
        reinforcements = []
        for i in range(n_reinf):
            rx = st.number_input(f"x neo {i+1}", value=8.0 + i * 4, key=f"rx{i}")
            rt = st.number_input(f"T neo {i+1} (kN/m)", value=80.0, key=f"rt{i}")
            reinforcements.append(Reinforcement(x=rx, T=rt))

# ---------------------------------------------------------------------------
# Compute FS for all 4 methods
# ---------------------------------------------------------------------------
methods = ["fellenius", "bishop", "spencer", "mp"]
method_names = {"fellenius": "Fellenius", "bishop": "Bishop", "spencer": "Spencer", "mp": "Morgenstern-Price"}

fs_results = {}
for m in methods:
    fs_results[m] = factor_of_safety(
        GROUND, xc, yc, R, layer_objs, method=m, n_slices=40,
        piezo_pts=piezo_pts, surcharges=surcharges, reinforcements=reinforcements,
        kh=kh, kv=kv,
    )

# ---------------------------------------------------------------------------
# Layout: plot + results
# ---------------------------------------------------------------------------
col1, col2 = st.columns([2.2, 1])

with col1:
    fig = plot_cross_section(
        GROUND, layer_objs, xc, yc, R, fs_results=fs_results,
        piezo_pts=piezo_pts, surcharges=surcharges, reinforcements=reinforcements,
        n_slices=40, title="Mặt cắt và mặt trượt thử",
    )
    st.pyplot(fig, width='stretch')

with col2:
    st.subheader("Kết quả FS")
    for m in methods:
        fs = fs_results[m]
        if fs is None:
            st.metric(method_names[m], "không hội tụ")
        else:
            delta_color = "normal"
            st.metric(method_names[m], f"{fs:.3f}")
    ref_fs = fs_results["bishop"] or fs_results["fellenius"]
    if ref_fs is not None:
        if ref_fs < 1.0:
            st.error("⚠ Mái dốc KHÔNG ổn định (FS < 1.0)")
        elif ref_fs < 1.5:
            st.warning("Ổn định nhưng cận biên (1.0 – 1.5)")
        else:
            st.success("Ổn định tốt (FS ≥ 1.5)")

    st.divider()
    st.subheader("Tìm mặt trượt nguy hiểm nhất")
    search_method = st.radio("Thuật toán", ["Grid search (vét cạn)", "PSO"], horizontal=True)
    if st.button("Chạy tìm kiếm", width='stretch'):
        with st.spinner("Đang tìm..."):
            if search_method.startswith("Grid"):
                top, n_tested, n_valid = grid_search(
                    GROUND, xc_range=(-10, 30, 15), yc_range=(10, 35, 12),
                    r_range=(8, 40, 15), layers=layer_objs, n_slices=20, top_k=1,
                )
                if top:
                    best = refine_with_rigorous_methods(GROUND, top, layer_objs, n_slices=30)[0]
                    st.session_state["found_xc"] = best["xc"]
                    st.session_state["found_yc"] = best["yc"]
                    st.session_state["found_R"] = best["R"]
                    st.success(f"Đã dò {n_tested} vòng tròn, {n_valid} hợp lệ.")
            else:
                bounds = {"xc": (-10, 30), "yc": (10, 35), "R": (8, 40)}
                best_pos, best_fs, history = pso_search(
                    GROUND, bounds, layer_objs, n_particles=25, n_generations=40,
                    n_slices=20, seed=42,
                )
                st.session_state["found_xc"] = best_pos[0]
                st.session_state["found_yc"] = best_pos[1]
                st.session_state["found_R"] = best_pos[2]
                st.success(f"PSO hội tụ sau 40 generations.")

    if "found_xc" in st.session_state:
        fx, fy, fr = st.session_state["found_xc"], st.session_state["found_yc"], st.session_state["found_R"]
        st.write(f"**Kết quả:** xc={fx:.2f}, yc={fy:.2f}, R={fr:.2f}")
        st.caption("Chỉnh 3 slider ở sidebar về các giá trị trên để xem trực quan mặt trượt này.")

st.divider()
with st.expander("Về ứng dụng này"):
    st.markdown(
        "Ứng dụng minh họa phương pháp Cân bằng giới hạn (Limit Equilibrium) cho bài toán "
        "ổn định mái dốc. Toàn bộ lý thuyết và mã nguồn: xem `le_solver.py`, `visualize.py`, "
        "`grid_search.py`, `pso_search.py` trong repo này, và `README.md` để biết các giới hạn "
        "hiện tại của bộ solver."
    )
