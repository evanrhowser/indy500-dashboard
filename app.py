import math
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

st.title("Indy 500 Historical Dashboard")

df = pd.read_csv("data/indy500_results.csv")

wins = df["winner"].value_counts().head(10).sort_values()

st.subheader("All-Time Wins Leaders (Top 10)")
st.bar_chart(wins)

st.subheader("Compare Races")

options = [f"{row.year} - {row.winner}" for row in df.itertuples()]

col1, col2 = st.columns(2)
with col1:
    driver1 = st.selectbox("Driver 1", options, key="driver1")
with col2:
    driver2 = st.selectbox("Driver 2", options, key="driver2")

st.subheader("Indianapolis Motor Speedway")

# ── Track geometry ─────────────────────────────────────────────────────────────
# cx  = half-length of front/back straights
# cy  = half-length of short chutes
# r   = corner radius   tw = track width
cx, cy, r, tw = 1.6, 0.35, 0.9, 0.22

# Flat-space y extents (used by perspective functions below)
Y_NEAR = -(cy + r)   # front straight  (-1.25)
Y_FAR  =  (cy + r)   # back straight   (+1.25)

def ims_path(radius, n=80):
    """Clockwise rounded-rectangle path for one boundary, in flat top-down space."""
    px, py = [], []
    px += [-cx, cx];                    py += [-(cy + radius)] * 2
    t = np.linspace(-np.pi / 2, 0, n)
    px += (cx  + radius * np.cos(t)).tolist(); py += (-cy + radius * np.sin(t)).tolist()
    px += [cx + radius] * 2;            py += [-cy, cy]
    t = np.linspace(0, np.pi / 2, n)
    px += (cx  + radius * np.cos(t)).tolist(); py += ( cy + radius * np.sin(t)).tolist()
    px += [cx, -cx];                    py += [cy + radius] * 2
    t = np.linspace(np.pi / 2, np.pi, n)
    px += (-cx + radius * np.cos(t)).tolist(); py += ( cy + radius * np.sin(t)).tolist()
    px += [-(cx + radius)] * 2;         py += [cy, -cy]
    t = np.linspace(np.pi, 3 * np.pi / 2, n)
    px += (-cx + radius * np.cos(t)).tolist(); py += (-cy + radius * np.sin(t)).tolist()
    px.append(px[0]); py.append(py[0])
    return px, py

# ── Perspective helpers ────────────────────────────────────────────────────────
# Simulates an elevated viewing angle looking from in front of the near
# (front-straight) side.  Near side: full size.  Far side: narrower, closer.
#
#   _K  — perspective depth constant (larger = more subtle foreshortening)
#   _Vy — extra vertical-compression factor
#
_K  = 4.0
_Vy = 0.55

def _pscale(y):
    """Perspective scale at flat-space y coordinate."""
    return 1.0 / (1.0 + (np.asarray(y, dtype=float) - Y_NEAR) / _K)

def persp_arr(x_arr, y_arr):
    """Apply perspective to coordinate arrays; returns plain Python lists."""
    xa, ya = np.asarray(x_arr, dtype=float), np.asarray(y_arr, dtype=float)
    s  = _pscale(ya)
    xp = xa * s
    yp = Y_NEAR + (ya - Y_NEAR) * s * _Vy
    return xp.tolist(), yp.tolist()

def persp_pt(x, y):
    """Apply perspective to a single (x, y) point."""
    s  = float(_pscale(y))
    xp = x * s
    yp = Y_NEAR + (y - Y_NEAR) * s * _Vy
    return xp, yp

def persp_size(y, base=14, small=7):
    """Marker size: large when near (front), small when far (back)."""
    s = float(_pscale(y))
    return max(small, round(base * s))

# ── Build perspective-transformed track boundaries ─────────────────────────────
x_out, y_out = persp_arr(*ims_path(r))
x_in,  y_in  = persp_arr(*ims_path(r - tw))

# Pre-compute perspective S/F endpoints so they can be reused in both figures.
sf_x = 1.2
sf_outer = persp_pt(sf_x, Y_NEAR)
sf_inner = persp_pt(sf_x, Y_NEAR + tw)

# ── Axis bounds that fit the perspective-compressed track ─────────────────────
X_RANGE = [-2.7, 2.7]
Y_RANGE = [-1.6,  0.1]

# ── Static track figure ────────────────────────────────────────────────────────
fig = go.Figure()

fig.add_trace(go.Scatter(
    x=x_out, y=y_out,
    fill="toself", fillcolor="#8a8a8a",
    line=dict(color="white", width=1.5),
    mode="lines", showlegend=False,
))

fig.add_trace(go.Scatter(
    x=x_in, y=y_in,
    fill="toself", fillcolor="#2d6a2d",
    line=dict(color="white", width=1),
    mode="lines", showlegend=False,
))

fig.add_shape(
    type="line",
    x0=sf_outer[0], y0=sf_outer[1],
    x1=sf_inner[0], y1=sf_inner[1],
    line=dict(color="white", width=5),
)

label_r = r + 0.32
for ax, ay, angle, label in [
    ( cx, -cy, -np.pi / 4,     "Turn 1"),
    ( cx,  cy,  np.pi / 4,     "Turn 2"),
    (-cx,  cy,  3 * np.pi / 4, "Turn 3"),
    (-cx, -cy, -3 * np.pi / 4, "Turn 4"),
]:
    lx, ly = persp_pt(ax + label_r * np.cos(angle),
                      ay + label_r * np.sin(angle))
    fig.add_annotation(
        x=lx, y=ly, text=label, showarrow=False,
        font=dict(color="white", size=11),
        bgcolor="rgba(0,0,0,0.5)", borderpad=4,
    )

sf_lbl = persp_pt(sf_x, Y_NEAR - 0.2)
fig.add_annotation(
    x=sf_lbl[0], y=sf_lbl[1],
    text="Start / Finish", showarrow=False,
    font=dict(color="#FFD700", size=10),
)

fig.update_layout(
    plot_bgcolor="#1a4a1a",
    paper_bgcolor="#1a4a1a",
    xaxis=dict(visible=False, range=X_RANGE),
    yaxis=dict(visible=False, range=Y_RANGE),
    margin=dict(l=20, r=20, t=10, b=20),
    height=360,
)

st.plotly_chart(fig, use_container_width=True)

# ── Race simulation ────────────────────────────────────────────────────────────
st.subheader("Race Simulation")

if "race_running" not in st.session_state:
    st.session_state.race_running = False

if st.button("Race!"):
    st.session_state.race_running = True

if st.session_state.race_running:
    year1_str, name1 = driver1.split(" - ", 1)
    year2_str, name2 = driver2.split(" - ", 1)
    year1, year2 = int(year1_str), int(year2_str)

    row1 = df[(df["year"] == year1) & (df["winner"] == name1)].iloc[0]
    row2 = df[(df["year"] == year2) & (df["winner"] == name2)].iloc[0]
    speed1   = float(row1["avg_speed_mph"])
    speed2   = float(row2["avg_speed_mph"])
    car_num1 = str(int(row1["car_number"]))
    car_num2 = str(int(row2["car_number"]))

    # Center-line path in flat space (arc-length parametrization done here,
    # before perspective, so world-space spacing is uniform).
    r_cl = r - tw / 2

    def center_line(radius):
        """Clockwise center-line path from S/F all the way around back to S/F."""
        px, py = [], []
        px += [sf_x, cx];              py += [-(cy + radius)] * 2
        t = np.linspace(-np.pi / 2, 0, 80)
        px += (cx  + radius * np.cos(t)).tolist(); py += (-cy + radius * np.sin(t)).tolist()
        px += [cx + radius] * 2;       py += [-cy, cy]
        t = np.linspace(0, np.pi / 2, 80)
        px += (cx  + radius * np.cos(t)).tolist(); py += ( cy + radius * np.sin(t)).tolist()
        px += [cx, -cx];               py += [cy + radius] * 2
        t = np.linspace(np.pi / 2, np.pi, 80)
        px += (-cx + radius * np.cos(t)).tolist(); py += ( cy + radius * np.sin(t)).tolist()
        px += [-(cx + radius)] * 2;    py += [cy, -cy]
        t = np.linspace(np.pi, 3 * np.pi / 2, 80)
        px += (-cx + radius * np.cos(t)).tolist(); py += (-cy + radius * np.sin(t)).tolist()
        px += [-cx, sf_x];             py += [-(cy + radius)] * 2
        return np.array(px), np.array(py)

    tx, ty = center_line(r_cl)

    dxy     = np.diff(np.column_stack([tx, ty]), axis=0)
    cum_len = np.concatenate([[0], np.cumsum(np.hypot(dxy[:, 0], dxy[:, 1]))])
    frac    = cum_len / cum_len[-1]

    def pos_at(frame_idx, speed, n_frames, min_spd):
        """Returns (x_screen, y_screen, marker_size) perspective-projected."""
        f      = min(1.0, (frame_idx / n_frames) * (speed / min_spd))
        x_flat = float(np.interp(f, frac, tx))
        y_flat = float(np.interp(f, frac, ty))
        xp, yp = persp_pt(x_flat, y_flat)
        return xp, yp, persp_size(y_flat)

    min_spd    = min(speed1, speed2)
    lap_time_s = (2.5 / min_spd) * 3600
    N          = 120
    frame_ms   = int(lap_time_s * 1000 / N)

    pos1 = [pos_at(i, speed1, N, min_spd) for i in range(N + 1)]
    pos2 = [pos_at(i, speed2, N, min_spd) for i in range(N + 1)]

    # ── Arrow angles (clockwise degrees from "up" = Plotly convention) ─────────
    # Computed from screen-space positions so the arrow matches visible motion.
    def travel_angle(i, positions):
        for j in [i + 1, i - 1]:
            if 0 <= j < len(positions):
                dx = positions[j][0] - positions[i][0]
                dy = positions[j][1] - positions[i][1]
                if abs(dx) + abs(dy) > 1e-9:
                    return 90.0 - math.degrees(math.atan2(dy, dx))
        return 90.0   # default: pointing east (front-straight direction)

    angles1 = [travel_angle(i, pos1) for i in range(N + 1)]
    angles2 = [travel_angle(i, pos2) for i in range(N + 1)]

    # ── Motion trail (fading comet tail behind each car) ───────────────────────
    N_TRAIL = 12
    C1_RGB  = (255, 107, 107)   # matches #FF6B6B
    C2_RGB  = (79,  195, 247)   # matches #4FC3F7

    def trail_rgba(rgb, j):
        alpha = max(0.0, 1.0 - j / N_TRAIL * 1.3)
        return f"rgba({rgb[0]},{rgb[1]},{rgb[2]},{alpha:.2f})"

    trail1_colors = [trail_rgba(C1_RGB, j) for j in range(N_TRAIL)]
    trail2_colors = [trail_rgba(C2_RGB, j) for j in range(N_TRAIL)]
    trail_sizes   = [max(2.0, 8.0 * (1.0 - j / N_TRAIL)) for j in range(N_TRAIL)]

    def trail_xy(positions, i):
        xs = [positions[max(0, i - j)][0] for j in range(N_TRAIL)]
        ys = [positions[max(0, i - j)][1] for j in range(N_TRAIL)]
        return xs, ys

    # ── Build frames — traces: 2=car1, 3=car2, 4=trail1, 5=trail2 ─────────────
    frames = []
    for i in range(N + 1):
        t1x, t1y = trail_xy(pos1, i)
        t2x, t2y = trail_xy(pos2, i)
        frames.append(go.Frame(
            data=[
                go.Scatter(
                    x=[pos1[i][0]], y=[pos1[i][1]],
                    text=[f"#{car_num1}"],
                    marker=dict(size=pos1[i][2], angle=angles1[i]),
                ),
                go.Scatter(
                    x=[pos2[i][0]], y=[pos2[i][1]],
                    text=[f"#{car_num2}"],
                    marker=dict(size=pos2[i][2], angle=angles2[i]),
                ),
                go.Scatter(x=t1x, y=t1y,
                           marker=dict(size=trail_sizes, color=trail1_colors)),
                go.Scatter(x=t2x, y=t2y,
                           marker=dict(size=trail_sizes, color=trail2_colors)),
            ],
            traces=[2, 3, 4, 5],
            name=str(i),
        ))

    # ── Base figure ────────────────────────────────────────────────────────────
    t1x0, t1y0 = trail_xy(pos1, 0)
    t2x0, t2y0 = trail_xy(pos2, 0)

    fig_anim = go.Figure()

    fig_anim.add_trace(go.Scatter(          # trace 0 — asphalt
        x=x_out, y=y_out, fill="toself", fillcolor="#8a8a8a",
        line=dict(color="white", width=1.5), mode="lines", showlegend=False,
    ))
    fig_anim.add_trace(go.Scatter(          # trace 1 — infield
        x=x_in, y=y_in, fill="toself", fillcolor="#2d6a2d",
        line=dict(color="white", width=1), mode="lines", showlegend=False,
    ))
    fig_anim.add_trace(go.Scatter(          # trace 2 — car 1 (triangle + number)
        x=[pos1[0][0]], y=[pos1[0][1]],
        mode="markers+text",
        text=[f"#{car_num1}"],
        textposition="top center",
        textfont=dict(color="#FF6B6B", size=9),
        marker=dict(symbol="triangle-up", color="#FF6B6B",
                    size=pos1[0][2], angle=angles1[0],
                    line=dict(color="white", width=1.5)),
        name=f"{year1} · {name1}  ({speed1:.1f} mph)",
    ))
    fig_anim.add_trace(go.Scatter(          # trace 3 — car 2 (triangle + number)
        x=[pos2[0][0]], y=[pos2[0][1]],
        mode="markers+text",
        text=[f"#{car_num2}"],
        textposition="top center",
        textfont=dict(color="#4FC3F7", size=9),
        marker=dict(symbol="triangle-up", color="#4FC3F7",
                    size=pos2[0][2], angle=angles2[0],
                    line=dict(color="white", width=1.5)),
        name=f"{year2} · {name2}  ({speed2:.1f} mph)",
    ))
    fig_anim.add_trace(go.Scatter(          # trace 4 — car 1 trail
        x=t1x0, y=t1y0, mode="markers",
        marker=dict(size=trail_sizes, color=trail1_colors),
        showlegend=False,
    ))
    fig_anim.add_trace(go.Scatter(          # trace 5 — car 2 trail
        x=t2x0, y=t2y0, mode="markers",
        marker=dict(size=trail_sizes, color=trail2_colors),
        showlegend=False,
    ))

    fig_anim.add_shape(
        type="line",
        x0=sf_outer[0], y0=sf_outer[1],
        x1=sf_inner[0], y1=sf_inner[1],
        line=dict(color="white", width=5),
    )

    fig_anim.frames = frames

    fig_anim.update_layout(
        plot_bgcolor="#1a4a1a",
        paper_bgcolor="#1a4a1a",
        xaxis=dict(visible=False, range=X_RANGE),
        yaxis=dict(visible=False, range=Y_RANGE),
        legend=dict(
            x=0.01, y=0.01,
            bgcolor="rgba(0,0,0,0.55)",
            font=dict(color="white", size=11),
            bordercolor="rgba(255,255,255,0.25)",
            borderwidth=1,
        ),
        margin=dict(l=20, r=20, t=10, b=60),
        height=400,
        updatemenus=[{
            "type": "buttons",
            "showactive": False,
            "x": 0.5, "y": -0.02,
            "xanchor": "center", "yanchor": "top",
            "bgcolor": "#2a2a2a",
            "font": {"color": "white"},
            "buttons": [
                {
                    "label": "▶  Play",
                    "method": "animate",
                    "args": [None, {
                        "frame": {"duration": frame_ms, "redraw": True},
                        "fromcurrent": True,
                        "transition": {"duration": frame_ms, "easing": "linear"},
                        "mode": "immediate",
                    }],
                },
                {
                    "label": "⏸  Pause",
                    "method": "animate",
                    "args": [[None], {
                        "frame": {"duration": 0, "redraw": False},
                        "mode": "immediate",
                        "transition": {"duration": 0},
                    }],
                },
            ],
        }],
    )

    st.plotly_chart(fig_anim, use_container_width=True)
