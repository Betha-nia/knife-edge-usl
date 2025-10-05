import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.optimize import curve_fit
from scipy.special import erf
import math

    # Constantes
h = 6.62607015e-34       # J·s
c = 2.99792458e8         # m/s
PI = math.pi
GM_to_cm4_s_per_photon = 1e-50

MM_PER_INCH = 25.4

st.set_page_config(page_title="Ultrafast Spectroscopy Laser Calculator", layout="wide")
st.title("Data analysis panel")

# Abas
tab1, tab2, tab3, tab4 = st.tabs(["About us", "Knife Edge", "Photon flux", "Exciton per dot"])

# Tab 1: About us
with tab1:
    st.header("About us")
    st.write("""
        We are the Ultrafast Spectroscopy Group led by Professor Dr. Lázaro Padilha.

        Our research focuses on the study of nanomaterials at nanometric scales, analyzing their emission, absorption, size, and dynamics. To do this, we routinely work with key experimental parameters such as beam size, photon flux, and the number of excitons generated per quantum dot.

        This site was designed to serve as a fast and intuitive calculator for the laboratory, helping researchers define these parameters directly and optimize experimental setups.
    """)


# ---------- Shared state ----------
if "beam_data" not in st.session_state:
    # each item: dict {x_mm: float, orig_x: str, orig_unit: "mm"|"inch", orig_y: float}
    st.session_state.beam_data = []

if "new_point_buffer" not in st.session_state:
    st.session_state.new_point_buffer = ""
if "new_point_unit" not in st.session_state:
    st.session_state.new_point_unit = "mm"
if "message" not in st.session_state:
    st.session_state.message = ""
if "clear_buffer" not in st.session_state:
    st.session_state.clear_buffer = False

# editing
if "edit_index" not in st.session_state:
    st.session_state.edit_index = None
if "edit_x_buffer" not in st.session_state:
    st.session_state.edit_x_buffer = ""
if "edit_x_unit" not in st.session_state:
    st.session_state.edit_x_unit = "mm"
if "edit_y_buffer" not in st.session_state:
    st.session_state.edit_y_buffer = ""
if "clear_edit_buffer" not in st.session_state:
    st.session_state.clear_edit_buffer = False

# fit result
if "fit_result" not in st.session_state:
    st.session_state.fit_result = None

# store the highest power value entered so far
if "max_input_power" not in st.session_state:
    st.session_state.max_input_power = None  # float or None

def to_mm_value(x_value, unit):
    x = float(x_value)
    return x if unit == "mm" else x * MM_PER_INCH

def update_max_input_power():
    """Recalculate and update st.session_state.max_input_power from st.session_state.beam_data"""
    if not st.session_state.beam_data:
        st.session_state.max_input_power = None
    else:
        max_p = max(item["orig_y"] for item in st.session_state.beam_data)
        st.session_state.max_input_power = float(max_p)

def get_normalized_y(orig_y):
    """Return orig_y / max_input_power (or 0 if max is None or zero)"""
    max_p = st.session_state.max_input_power
    if (max_p is None) or (max_p == 0):
        return 0.0
    return float(orig_y) / float(max_p)

def knife_edge_model(x, P0, x0, w):
    return 0.5 * P0 * (1 - erf(np.sqrt(2) * (x - x0) / w))

def compute_delta_x_array(xs, orig_ys):
    """
    Determine x_initial as the position (x_mm) of the point with highest orig_y.
    Compute delta = x - x_initial; if any delta is negative, multiply all by -1.
    Return array of deltas (float) and x_initial.
    """
    idx_max = int(np.argmax(orig_ys))
    x_initial = xs[idx_max]
    deltas = xs - x_initial
    if np.any(deltas < 0):
        deltas = -deltas
    return deltas, x_initial

def format_scientific(value, sig=3):
    """Return string in format '1.83 × 10^16' with `sig` significant digits."""
    if value == 0:
        return "0"
    sign = "-" if value < 0 else ""
    v = abs(float(value))
    exp = int(math.floor(math.log10(v)))
    mant = v / (10 ** exp)
    mant_str = f"{mant:.{sig-1}f}"
    if "." in mant_str:
        mant_str = mant_str.rstrip("0").rstrip(".")
    return f"{sign}{mant_str} × 10^{exp}"

def add_point_callback():
    text = st.session_state.new_point_buffer.strip()
    unit = st.session_state.new_point_unit
    st.session_state.message = ""
    if not text:
        st.session_state.message = "warning:Please enter a point before submitting."
        return
    if text.count(",") != 1:
        st.session_state.message = "warning:Invalid format. Use: number, number"
        return
    p0, p1 = [p.strip() for p in text.split(",")]
    try:
        x_mm = to_mm_value(p0, unit)
        orig_y = float(p1)
        st.session_state.beam_data.append({
            "x_mm": float(x_mm),
            "orig_x": p0,
            "orig_unit": unit,
            "orig_y": float(orig_y)
        })
        st.session_state.clear_buffer = True
        update_max_input_power()
        st.session_state.message = "success:Point added"
        st.session_state.fit_result = None
    except ValueError:
        st.session_state.message = "warning:Invalid format. Use: number, number"

# ---------- Edição / remoção ----------
def start_edit(idx):
    st.session_state.edit_index = idx
    item = st.session_state.beam_data[idx]
    st.session_state.edit_x_buffer = item["orig_x"]
    st.session_state.edit_x_unit = item["orig_unit"]
    st.session_state.edit_y_buffer = str(item["orig_y"])
    st.session_state.clear_edit_buffer = False

def apply_edit():
    idx = st.session_state.edit_index
    if idx is None:
        st.session_state.message = "warning:No point selected for editing."
        return
    try:
        x_mm = to_mm_value(st.session_state.edit_x_buffer.strip(), st.session_state.edit_x_unit)
        orig_y_text = st.session_state.edit_y_buffer.strip()
        orig_y = float(orig_y_text)
        st.session_state.beam_data[idx] = {
            "x_mm": float(x_mm),
            "orig_x": st.session_state.edit_x_buffer.strip(),
            "orig_unit": st.session_state.edit_x_unit,
            "orig_y": float(orig_y)
        }
        update_max_input_power()
        st.session_state.message = "success:Point updated"
        st.session_state.edit_index = None
        st.session_state.clear_edit_buffer = True
        st.session_state.clear_buffer = True
        st.session_state.fit_result = None
    except ValueError:
        st.session_state.message = "warning:Invalid format. Use: number, number"

def remove_point(idx):
    if idx is None:
        st.session_state.message = "warning:No point selected for removal."
        return
    try:
        st.session_state.beam_data.pop(idx)
        update_max_input_power()
        st.session_state.message = "success:Point removed"
        st.session_state.clear_buffer = True
        st.session_state.fit_result = None
    except Exception:
        st.session_state.message = "warning:Error while removing point."

# ---------- Ajuste de curva ----------
def fit_curve():
    if len(st.session_state.beam_data) < 4:
        st.session_state.message = "warning:At least 4 points are required for a reliable fit."
        return
    sorted_data = sorted(st.session_state.beam_data, key=lambda d: d["x_mm"])
    xs_abs = np.array([d["x_mm"] for d in sorted_data])
    orig_ys = np.array([d["orig_y"] for d in sorted_data])

    xs_delta, x_initial = compute_delta_x_array(xs_abs, orig_ys)
    order = np.argsort(xs_delta)
    xs_fit = xs_delta[order]
    ys_fit = np.array([get_normalized_y(sorted_data[i]["orig_y"]) for i in order])

    P0_guess = max(ys_fit) * 2 if len(ys_fit) else 1.0
    x0_guess = np.median(xs_fit)
    w_guess = max(1e-3, (max(xs_fit) - min(xs_fit)) / 4.0)
    p0 = [P0_guess, x0_guess, w_guess]
    try:
        popt, pcov = curve_fit(knife_edge_model, xs_fit, ys_fit, p0=p0, maxfev=100000)
        perr = np.sqrt(np.diag(pcov))
        x_fit = np.linspace(min(xs_fit) - 0.1 * abs(min(xs_fit)), max(xs_fit) + 0.1 * abs(max(xs_fit)), 300)
        y_fit = knife_edge_model(x_fit, *popt)
        st.session_state.fit_result = {
            "popt": popt.tolist(),
            "perr": perr.tolist(),
            "x_fit": x_fit.tolist(),
            "y_fit": y_fit.tolist(),
            "x_initial": float(x_initial)
        }
        w_value = popt[2]
        w_err = perr[2] if len(perr) > 2 else 0.0
        st.session_state.message = f"success:Fit completed: w = {w_value:.1f} mm ± {w_err:.1f} mm"
    except Exception as e:
        st.session_state.message = f"warning:Fit error: {str(e)}"
        st.session_state.fit_result = None

# ---------- Tab 2: Knife Edge (UI) ----------
with tab2:
    st.header("Knife-edge")
    st.write("Enter position and measured power. The first power value entered is taken as the maximum; normalization uses the highest value entered so far. The graph shows ΔX = |X - X_initial| (positive).")

    col1, col2 = st.columns([1, 2])

    # Left column: input, table, editing
    with col1:
        # Clear pending input before rendering
        if st.session_state.get("clear_buffer", False):
            st.session_state.new_point_buffer = ""
            st.session_state.clear_buffer = False

        input_row = st.columns([2, 1])
        with input_row[0]:
            st.text_input(
                "New point (e.g.: 2.1, 75)",
                key="new_point_buffer",
                value=st.session_state.new_point_buffer,
                on_change=add_point_callback,
                help="Enter 'position, power' (power can be a percentage like 75) and press Enter."
            )
        with input_row[1]:
            st.selectbox("Position unit", options=["mm", "inch"], index=0, key="new_point_unit")

        if st.button("Add point"):
            add_point_callback()

        if st.session_state.message:
            level, msg = st.session_state.message.split(":", 1)
            if level == "warning":
                st.warning(msg)
            else:
                st.success(msg)

        if st.button("🧹 Clear all points", key="clear_beam"):
            st.session_state.beam_data = []
            st.session_state.message = ""
            st.session_state.clear_buffer = True
            st.session_state.fit_result = None
            st.session_state.max_input_power = None
            st.success("All points have been removed!")

        # Table
        if st.session_state.beam_data:
            table = []
            for i, item in enumerate(st.session_state.beam_data):
                norm_y = get_normalized_y(item["orig_y"])
                table.append({
                    "Index": i,
                    "Position (original)": f"{item['orig_x']} {item['orig_unit']}",
                    "Position (mm)": f"{item['x_mm']:.6f} mm",
                    "Entered power": f"{item['orig_y']}",
                    "Normalized intensity": f"{norm_y:.3f}"
                })
            df_display = pd.DataFrame(table)
            with st.expander("📋 Show point table"):
                st.dataframe(df_display)

            options = [f"{i}: {item['orig_x']} {item['orig_unit']} | {item['orig_y']}" for i, item in enumerate(st.session_state.beam_data)]
            st.markdown("**Select point to edit or remove**")
            sel_index_default = 0 if options else None
            selected = st.selectbox("Points", options=options, index=sel_index_default, key="select_for_edit") if options else None
            selected_idx = int(selected.split(":")[0]) if selected else None

            btn_cols = st.columns([1, 1, 1])
            with btn_cols[0]:
                if st.button("✏️ Edit point"):
                    if selected_idx is not None:
                        start_edit(selected_idx)
            with btn_cols[1]:
                if st.button("🗑️ Remove point"):
                    if selected_idx is not None:
                        remove_point(selected_idx)
            with btn_cols[2]:
                txt_content = "\n".join([
                    f"{i},{item['orig_x']},{item['orig_unit']},{item['orig_y']},{item['x_mm']:.6f}"
                    for i, item in enumerate(st.session_state.beam_data)
                ])
                st.download_button(
                    label="📁 Download data as .txt",
                    data=txt_content,
                    file_name="knife_edge_data.txt",
                    mime="text/plain"
                )

        # Edit point
        if st.session_state.edit_index is not None:
            st.markdown(f"**Editing point index {st.session_state.edit_index}**")
            if st.session_state.clear_edit_buffer:
                st.session_state.edit_x_buffer = ""
                st.session_state.edit_y_buffer = ""
                st.session_state.edit_x_unit = "mm"
                st.session_state.clear_edit_buffer = False

            edit_cols = st.columns([2, 1, 2])
            with edit_cols[0]:
                st.text_input("Position (value)", key="edit_x", value=st.session_state.edit_x_buffer)
            with edit_cols[1]:
                st.selectbox("Unit", options=["mm", "inch"], index=0, key="edit_x_unit")
            with edit_cols[2]:
                st.text_input("Power (value)", key="edit_y", value=st.session_state.edit_y_buffer)

            if st.session_state.get("edit_x", None) is not None:
                st.session_state.edit_x_buffer = st.session_state.edit_x
            if st.session_state.get("edit_y", None) is not None:
                st.session_state.edit_y_buffer = st.session_state.edit_y

            btns = st.columns([1, 1])
            with btns[0]:
                if st.button("Save edit"):
                    apply_edit()
            with btns[1]:
                if st.button("Cancel edit"):
                    st.session_state.edit_index = None
                    st.session_state.edit_x_buffer = ""
                    st.session_state.edit_y_buffer = ""
                    st.session_state.message = ""

    # Right column: graph + curve fitting button
    with col2:
        col_plot_top = st.columns([3, 1])
        with col_plot_top[1]:
            if st.button("Fit curve"):
                fit_curve()

        if st.session_state.beam_data:
            sorted_data = sorted(st.session_state.beam_data, key=lambda d: d["x_mm"])
            xs_abs = np.array([d["x_mm"] for d in sorted_data])
            orig_ys = np.array([d["orig_y"] for d in sorted_data])

            xs_delta, x_initial = compute_delta_x_array(xs_abs, orig_ys)
            order_plot = np.argsort(xs_delta)
            xs_plot = xs_delta[order_plot]
            ys_plot = np.array([get_normalized_y(sorted_data[i]["orig_y"]) for i in order_plot])
            ys_plot = np.clip(ys_plot, 0.0, 1.0)

            df_beam = pd.DataFrame({
                "Delta X (mm)": xs_plot,
                "Intensity": ys_plot
            })

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=df_beam["Delta X (mm)"],
                y=df_beam["Intensity"],
                mode='markers',
                marker=dict(size=8, color='red'),
                name='Points'
            ))

            if st.session_state.fit_result is not None:
                x_fit = np.array(st.session_state.fit_result["x_fit"])
                y_fit = np.array(st.session_state.fit_result["y_fit"])
                fig.add_trace(go.Scatter(
                    x=x_fit,
                    y=y_fit,
                    mode='lines',
                    line=dict(color='blue'),
                    name='Knife-edge fit'
                ))
                popt = np.array(st.session_state.fit_result["popt"])
                perr = np.array(st.session_state.fit_result["perr"])
                w_val = popt[2]
                w_err = perr[2] if perr.size > 2 else 0.0
                annotation_text = f"w = {w_val:.1f} mm ± {w_err:.1f} mm"
                fig.add_annotation(
                    x=0.98,
                    y=0.98,
                    xref="paper",
                    yref="paper",
                    xanchor="right",
                    yanchor="top",
                    text=annotation_text,
                    showarrow=False,
                    font=dict(family="Times New Roman", size=16, color="black"),
                    align="right",
                    bordercolor="black",
                    borderwidth=1,
                    borderpad=6,
                    bgcolor="rgba(255,255,255,0.9)"
                )

            # Auto x-axis limits with 10% padding
            x_min_plot = float(xs_plot.min()) if xs_plot.size else 0.0
            x_max_plot = float(xs_plot.max()) if xs_plot.size else 1.0
            x_range_span = x_max_plot - x_min_plot
            pad = 0.1 * x_range_span if x_range_span != 0 else max(0.5, 0.1 * abs(x_min_plot) if x_min_plot != 0 else 0.5)
            x_axis_range = [x_min_plot - pad, x_max_plot + pad]

            fig.update_layout(
                title="Knife-Edge Graph (ΔX in mm)",
                xaxis_title="ΔX (mm)",
                yaxis_title="Intensity",
                autosize=False,
                width=700,
                height=600,
                margin=dict(l=40, r=40, t=40, b=40),
                font=dict(family="Times New Roman", size=12),
                xaxis=dict(range=x_axis_range, scaleanchor="y", scaleratio=1, fixedrange=False, showline=True, linewidth=2,
                           linecolor='gray', mirror=True, ticks='outside',
                           tickfont=dict(family="Times New Roman", size = 12)),
                yaxis=dict(range=[0,1.1], fixedrange = False, showline = True,linewidth = 2.0,linecolor="gray",
                           mirror=True,ticks = "outside",tickfont = dict(family = "Times New Roman",size=12)),
                           dragmode="pan"
            )
            st.plotly_chart(fig,use_container_width=False)
        else:
            st.info("Not enought points to show the graph.")
                
            

# ---------- Tab 3: Photon flux ----------
with tab3:
    st.header("Photon flux")
    st.write("Calculates photon flux (photons / cm² per pulse) using the formula P·λ / (h·c·A·f). The beam area is considered elliptical: A = π·rx·ry.")

    # Constants
    h = 6.62607015e-34       # J·s
    c = 2.99792458e8         # m/s
    PI = np.pi

    col_a, col_b = st.columns([1, 1])

    with col_a:
        rx_um = st.number_input("Beam radius x — value", min_value=0.0, value=5.0, step=0.5, format="%.3f")
        st.caption("Radius x in μm")
        ry_um = st.number_input("Beam radius y — value", min_value=0.0, value=5.0, step=0.5, format="%.3f")
        st.caption("Radius y in μm")

        rep_rate = st.number_input("Repetition rate — value", min_value=0.0, value=80.0, step=1.0, format="%.3f")
        rep_scale = st.selectbox("Repetition rate unit", options=["kHz", "MHz"], index=1)

    with col_b:
        wavelength_nm = st.number_input("Wavelength (λ) — nm", min_value=0.0, value=800.0, step=1.0, format="%.3f")

        power_val = st.number_input("Average power — value", min_value=0.0, value=1.0, step=0.1, format="%.6f")
        power_scale = st.selectbox("Power unit", options=["mW", "μW"], index=0)

        if st.button("Calculate photon flux"):
            # Conversions
            rx_m = rx_um * 1e-6
            ry_m = ry_um * 1e-6
            area_m2 = PI * rx_m * ry_m
            if area_m2 <= 0:
                st.error("Invalid beam area. Please check the radii.")
            else:
                P_W = power_val * 1e-3 if power_scale == "mW" else power_val * 1e-6
                f_Hz = rep_rate * 1e3 if rep_scale == "kHz" else rep_rate * 1e6
                if f_Hz <= 0:
                    st.error("Repetition rate must be greater than zero.")
                else:
                    lam_m = wavelength_nm * 1e-9
                    E_photon = h * c / lam_m  # J
                    photons_per_pulse_per_m2 = (P_W / f_Hz) / (E_photon * area_m2)
                    photons_per_pulse_per_cm2 = photons_per_pulse_per_m2 / 1e4

                    # Scientific formatting
                    formatted = format_scientific(photons_per_pulse_per_cm2, sig=3)

                    st.subheader("Results")
                    st.write(f"- Radius x: {rx_um:.3f} μm ({rx_m:.3e} m); Radius y: {ry_um:.3f} μm ({ry_m:.3e} m)")
                    st.write(f"- Beam area (ellipse): {area_m2:.3e} m²")
                    st.write(f"- Average power: {P_W:.3e} W")
                    st.write(f"- Repetition rate: {f_Hz:.3e} Hz")
                    st.write(f"- Wavelength: {lam_m:.3e} m")
                    st.write(f"- Photon energy: {E_photon:.3e} J")

                    st.markdown("---")
                    st.markdown(f"**Photon flux per pulse:** **{formatted} photons / cm² / pulse**")
                    st.caption("Area used: A = π·rx·ry (rx and ry in meters).")

# ---------- Aba 4 ----------
# Tab 4: Exciton per dot
with tab4:
    st.header("Exciton per dot")
    st.write("Calculate the average number of excitons per quantum dot using the general multiphoton formula.")

    import math

    # Constants
    h = 6.62607015e-34       # J·s
    c = 2.99792458e8         # m/s
    PI = math.pi
    GM_to_cm4_s_per_photon = 1e-50

    # Layout: inputs on the left, results on the right
    col_inputs, col_results = st.columns([1, 1])


    with col_inputs:
        

        # Beam radius
        rx_um = st.number_input("Beam radius (x) — value", min_value=0.0, value=5.0, step=0.5, format="%.2f", key="rx_input")
        st.caption("Radius x in μm")
        ry_um = st.number_input("Beam radius (y) — value", min_value=0.0, value=5.0, step=0.5, format="%.2f", key="ry_input")
        st.caption("Radius y in μm")

        # Repetition rate
        rep_col1, rep_col2 = st.columns([1, 1])
        with rep_col1:
            rep_rate = st.number_input("Repetition rate — value", min_value=0.0, value=80.0, step=1.0, format="%.2f", key="rep_rate_input")
        with rep_col2:
            rep_unit = st.selectbox("Unit", options=["kHz", "MHz"], index=1, key="rep_unit_select")

        # Wavelength
        wavelength_nm = st.number_input("Wavelength (λ) — nm", min_value=0.0, value=800.0, step=1.0, format="%.0f", key="wavelength_input")

        # Pulse width (always in fs)
        pulse_width_fs = st.number_input("Pulse width (fs)", min_value=0.0, value=100.0, step=1.0, format="%.0f", key="pulse_width_input")


    with col_results:

        # Power
        power_col1, power_col2 = st.columns([1, 1])
        with power_col1:
            power_val = st.number_input("Average power — value", min_value=0.0, value=1.0, step=0.1, format="%.2f", key="power_val_input")
        with power_col2:
            power_unit = st.selectbox("Unit", options=["mW", "μW"], index=0, key="power_unit_select")

        # Cross section
        sigma_col1, sigma_col2 = st.columns([1, 1])
        with sigma_col1:
            sigma_val = st.number_input("Absorption cross section — value", min_value=0.0, value=1.0, step=0.1, format="%.2e", key="sigma_input")
        with sigma_col2:
            sigma_unit = st.selectbox("Unit", options=["cm²", "GM"], index=1, key="sigma_unit_select")

        # Calculate button
        calculate = st.button("Calculate excitons per dot", key="calculate_exciton_btn")

        

        if calculate:
            # Determine p from wavelength
            p = max(1, round(wavelength_nm / 400))

            # Unit conversions
            P_W = power_val * 1e-3 if power_unit == "mW" else power_val * 1e-6
            f_Hz = rep_rate * 1e3 if rep_unit == "kHz" else rep_rate * 1e6
            lam_m = wavelength_nm * 1e-9
            rx_m = rx_um * 1e-6
            ry_m = ry_um * 1e-6
            T_s = pulse_width_fs * 1e-15

            # Beam area (elliptical)
            A_m2 = PI * rx_m * ry_m

            # Photon energy
            E_photon = h * c / lam_m

            # Photon flux per pulse (photons / cm² / pulse)
            phi_per_pulse_m2 = (P_W / f_Hz) / (E_photon * A_m2)
            phi_per_pulse_cm2 = phi_per_pulse_m2 / 1e4

            # Cross section interpretation
            if p == 1 or sigma_unit == "cm²":
                sigma_cm2 = sigma_val
            elif p == 2 and sigma_unit == "GM":
                sigma_cm2 = sigma_val * GM_to_cm4_s_per_photon
            else:
                sigma_cm2 = sigma_val  # fallback

            # Apply formula
            denom_prefactor = p * math.sqrt(p)
            phi_term = phi_per_pulse_cm2 ** p
            pulse_term = (T_s * math.sqrt(PI)) ** (p - 1)
            N_mean = (sigma_cm2 / denom_prefactor) * (phi_term / pulse_term)

            # Poisson probabilities
            P0 = math.exp(-N_mean)
            P_ge1 = 1 - P0

            # Display results
            st.subheader("Result:")
            st.write(f"**⟨N⟩ (average excitons per dot):** {N_mean:.2f}")