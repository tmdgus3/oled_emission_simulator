import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import os
import json
import copy
import io
from scipy.interpolate import griddata
import urllib.request
import time

# --- Access Control ---
def check_password():
    def password_entered():
        if st.session_state["password"] == "oeml2025":  # 원하는 비밀번호로 설정
            st.session_state["authenticated"] = True
        else:
            st.error("비밀번호가 틀렸습니다.")

    if "authenticated" not in st.session_state:
        st.session_state["authenticated"] = False

    if not st.session_state["authenticated"]:
        st.text_input("🔒 비밀번호를 입력하세요:", type="password", on_change=password_entered, key="password")
        st.stop()

check_password()

# --- Page Configuration (MUST BE THE FIRST STREAMLIT COMMAND) ---
st.set_page_config(
    page_title="OLED Optical Simulator Pro",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- Robust Library Import ---
try:
    from tmm import coh_tmm, absorp_in_each_layer
    TMM_AVAILABLE = True
except ImportError:
    st.error("The 'tmm' library is not installed. Please run `pip install tmm` in your terminal and restart the app.")
    TMM_AVAILABLE = False
    coh_tmm = None

try:
    from scipy.interpolate import griddata
    SCIPY_AVAILABLE = True
except ImportError:
    st.error("The 'scipy' library is not installed. Please run `pip install scipy` in your terminal and restart the app.")
    SCIPY_AVAILABLE = False
    griddata = None

# ==============================================================================
# 1. DATA LOADING & MANAGEMENT
# ==============================================================================

@st.cache_data
def load_material_data():
    materials = {'Air': pd.DataFrame({'wavelength_nm': [400, 800], 'n': [1.0, 1.0], 'k': [0.0, 0.0]})}
    if not os.path.isdir('materials'):
        st.sidebar.error("The 'materials' directory was not found.")
        return materials
    for f in sorted([f for f in os.listdir('materials') if f.endswith('.csv')]):
        name = os.path.splitext(f)[0]
        try:
            df = pd.read_csv(os.path.join('materials', f))
            if all(c in df.columns for c in ['wavelength_nm', 'n', 'k']):
                materials[name] = df.sort_values(by='wavelength_nm').reset_index(drop=True)
            else:
                st.warning(f"⚠️ '{f}' is missing required columns ('wavelength_nm', 'n', 'k'). Skipping.")
        except Exception as e:
            st.error(f"Failed to load '{f}'. Reason: {e}")
    return materials

@st.cache_data
def load_emitter_data():
    emitters = {}
    if not os.path.isdir('emitters'):
        st.error("The 'emitters' directory was not found.")
        return emitters
    for f in sorted([f for f in os.listdir('emitters') if f.endswith('.csv')]):
        name = os.path.splitext(f)[0]
        try:
            df = pd.read_csv(os.path.join('emitters', f))
            if all(c in df.columns for c in ['wavelength_nm', 'intensity']):
                emitters[name] = df.sort_values(by='wavelength_nm').reset_index(drop=True)
            else:
                st.warning(f"⚠️ '{f}' is missing required columns ('wavelength_nm', 'intensity'). Skipping.")
        except Exception as e:
            st.error(f"Failed to load '{f}'. Reason: {e}")
    return emitters

@st.cache_data
def get_cie_data():
    """외부 CSV 파일에서 CIE 1931 색 일치 함수 데이터를 불러옵니다."""
    path = "data/raw/cie_data.csv"  # ← 실행 경로 기준 상대경로
    if not os.path.exists(path):
        st.error(f"❌ CIE 데이터 파일을 찾을 수 없습니다: {path}")
        st.stop()
    cie_df = pd.read_csv(path)
    return cie_df

# ==============================================================================
# 2. CORE PHYSICS & CALCULATION ENGINE
# ==============================================================================

def get_n_complex(material_name, wl, materials_db):
    """Interpolates n and k for a given material and wavelength."""
    if material_name not in materials_db:
        st.error(f"Material '{material_name}' not found in database. Calculation cannot proceed.")
        st.stop()
    mat_data = materials_db[material_name]
    min_wl, max_wl = mat_data['wavelength_nm'].min(), mat_data['wavelength_nm'].max()
    if not (min_wl <= wl <= max_wl):
        st.toast(f"Warning: Wavelength {wl:.1f}nm is outside data range for {material_name}. Extrapolating.", icon="⚠️")
    n = np.interp(wl, mat_data['wavelength_nm'], mat_data['n'])
    k = np.interp(wl, mat_data['wavelength_nm'], mat_data['k'])
    return n + 1j * k

def run_reflectance_simulation(stack, wavelengths, angle_deg, materials, polarization):
    if not TMM_AVAILABLE: return None, None
    view_direction, incident_medium, exit_medium, materials_list, thicknesses_list = prepare_stack_for_viewing(stack, materials)
    
    results_R, results_T = [], []
    d_list = [np.inf] + thicknesses_list + [np.inf]
    th_0 = np.deg2rad(angle_deg)
    
    for wl in wavelengths:
        n_list = [get_n_complex(incident_medium, wl, materials)] + [get_n_complex(m, wl, materials) for m in materials_list] + [get_n_complex(exit_medium, wl, materials)]
        if polarization == 's-polarization':
            tmm_results = coh_tmm('s', n_list, d_list, th_0, wl)
            results_R.append(tmm_results['R'])
            results_T.append(tmm_results['T'])
        elif polarization == 'p-polarization':
            tmm_results = coh_tmm('p', n_list, d_list, th_0, wl)
            results_R.append(tmm_results['R'])
            results_T.append(tmm_results['T'])
        else: # Average
            tmm_s = coh_tmm('s', n_list, d_list, th_0, wl)
            tmm_p = coh_tmm('p', n_list, d_list, th_0, wl)
            results_R.append((tmm_s['R'] + tmm_p['R']) / 2)
            results_T.append((tmm_s['T'] + tmm_p['T']) / 2)
    return np.array(results_R), np.array(results_T)

def run_emission_simulation(stack, emitter_pl, emissive_layer_index, emission_position, view_angle_deg, materials_db, polarization):
    if not TMM_AVAILABLE: return None, None, None, None, None
    
    view_direction, incident_medium, exit_medium, materials_list, thicknesses_list = prepare_stack_for_viewing(stack, materials_db)
    if view_direction == 'Bottom':
        emissive_layer_index = (len(materials_list) - 1) - emissive_layer_index
    
    wavelengths, intrinsic_pl = emitter_pl['wavelength_nm'].values, emitter_pl['intensity'].values
    final_emission_spectrum = []
    d_list_sim = [np.inf] + thicknesses_list + [np.inf]
    th_0 = np.deg2rad(view_angle_deg)
    
    for wl, pl in zip(wavelengths, intrinsic_pl):
        n_list_complex = np.array([get_n_complex(incident_medium, wl, materials_db)] + [get_n_complex(mat, wl, materials_db) for mat in materials_list] + [get_n_complex(exit_medium, wl, materials_db)])
        
        # Correctly use the tmm library: first run coh_tmm, then get absorption
        tmm_data_s = coh_tmm('s', n_list_complex, d_list_sim, th_0, wl)
        tmm_data_p = coh_tmm('p', n_list_complex, d_list_sim, th_0, wl)
        
        absorption_s = absorp_in_each_layer(tmm_data_s)
        absorption_p = absorp_in_each_layer(tmm_data_p)
        
        eml_index_tmm = emissive_layer_index + 1
        
        eml_absorption_s = absorption_s[eml_index_tmm]
        eml_absorption_p = absorption_p[eml_index_tmm]
        
        if polarization == 's-polarization':
            purcell_factor = eml_absorption_s
        elif polarization == 'p-polarization':
            purcell_factor = eml_absorption_p
        else: # Average
            purcell_factor = (eml_absorption_s + eml_absorption_p) / 2.0
        
        final_emission_spectrum.append(pl * purcell_factor)
        
    final_emission_spectrum = np.array(final_emission_spectrum)
    radiance, luminance = calculate_radiance_luminance(wavelengths, final_emission_spectrum)
    cie_coords = calculate_cie_coords(wavelengths, final_emission_spectrum)
    
    normalized_spectrum = final_emission_spectrum.copy()
    if normalized_spectrum.max() > 0:
        normalized_spectrum /= normalized_spectrum.max()
    return wavelengths, final_emission_spectrum, normalized_spectrum, radiance, luminance, cie_coords

def run_angle_sweep_simulation(stack, wavelengths, angles_deg, sim_type, materials_db, polarization):
    if not TMM_AVAILABLE: return None
    if materials_db is None:
        materials_db = materials
    view_direction, incident_medium, exit_medium, materials_list, thicknesses_list = prepare_stack_for_viewing(stack, materials_db)
    
    results_map = np.zeros((len(angles_deg), len(wavelengths)))
    d_list = [np.inf] + thicknesses_list + [np.inf]
    progress_bar = st.progress(0, text=f"Calculating {sim_type} map...")

    for i, angle in enumerate(angles_deg):
        th_0 = np.deg2rad(angle)
        for j, wl in enumerate(wavelengths):
            n_list = [get_n_complex(incident_medium, wl, materials_db)] + [get_n_complex(m, wl, materials_db) for m in materials_list] + [get_n_complex(exit_medium, wl, materials_db)]
            if polarization == 's-polarization':
                tmm_results = coh_tmm('s', n_list, d_list, th_0, wl)
                results_map[i, j] = tmm_results[sim_type]
            elif polarization == 'p-polarization':
                tmm_results = coh_tmm('p', n_list, d_list, th_0, wl)
                results_map[i, j] = tmm_results[sim_type]
            else: # Average
                tmm_s = coh_tmm('s', n_list, d_list, th_0, wl)
                tmm_p = coh_tmm('p', n_list, d_list, th_0, wl)
                results_map[i, j] = (tmm_s[sim_type] + tmm_p[sim_type]) / 2
        progress_bar.progress((i + 1) / len(angles_deg), text=f"Calculating... Angle {int(angle)}°")
    progress_bar.empty()
    return results_map

def run_emission_angle_sweep_simulation(stack, emitter_pl, emissive_layer_index, emission_position, angles_deg, materials_db, polarization):
    if not TMM_AVAILABLE: return None
    if materials_db is None:
        materials_db = materials
    view_direction, incident_medium, exit_medium, materials_list, thicknesses_list = prepare_stack_for_viewing(stack, materials_db)
    if view_direction == 'Bottom':
        emissive_layer_index = (len(materials_list) - 1) - emissive_layer_index
    
    wavelengths = emitter_pl['wavelength_nm'].values
    emission_map = np.zeros((len(angles_deg), len(wavelengths)))
    progress_bar = st.progress(0, text="Calculating emission map...")

    for i, angle in enumerate(angles_deg):
        _, spectrum, _, _, _, _ = run_emission_simulation(stack, emitter_pl, emissive_layer_index, emission_position, angle, materials_db, polarization)
        if spectrum is not None:
            emission_map[i, :] = spectrum
        progress_bar.progress((i + 1) / len(angles_deg), text=f"Calculating... Angle {int(angle)}°")
    
    progress_bar.empty()
    return emission_map

def run_advanced_sweep(base_stack, emitter_pl, emissive_layer_index, emission_position, sweep_params, materials_db):
    enabled_params = [p for p in sweep_params if p['enabled']]
    if not 1 <= len(enabled_params) <= 2:
        st.error("Please enable 1 (for a 1D scan) or 2 (for a 2D scan) parameters.")
        return None

    p1 = enabled_params[0]
    range1 = np.arange(p1['from'], p1['to'] + p1['step'], p1['step'])
    
    if len(enabled_params) == 1:
        results = []
        progress_bar = st.progress(0, f"Running 1D sweep over {p1['name']}...")
        for i, val1 in enumerate(range1):
            temp_stack, temp_materials, temp_angle = apply_sweep_value(base_stack, p1, val1, materials_db)
            _, _, _, radiance, luminance, _ = run_emission_simulation(temp_stack, emitter_pl, emissive_layer_index, emission_position, temp_angle, materials_db=temp_materials, polarization='Average')
            results.append({'val': val1, 'radiance': radiance, 'luminance': luminance})
            progress_bar.progress((i + 1) / len(range1))
        progress_bar.empty()
        return {'type': '1D', 'df': pd.DataFrame(results), 'param_name': p1['name']}
    
    else:
        p2 = enabled_params[1]
        range2 = np.arange(p2['from'], p2['to'] + p2['step'], p2['step'])
        results_map = np.zeros((len(range2), len(range1)))
        
        progress_bar = st.progress(0, f"Running 2D sweep over {p1['name']} and {p2['name']}...")
        for i, val2 in enumerate(range2):
            for j, val1 in enumerate(range1):
                temp_stack, temp_materials, temp_angle = apply_sweep_value(base_stack, p1, val1, materials_db)
                temp_stack, temp_materials, temp_angle = apply_sweep_value(temp_stack, p2, val2, temp_materials, temp_angle)
                _, _, _, radiance, luminance, _ = run_emission_simulation(temp_stack, emitter_pl, emissive_layer_index, emission_position, temp_angle, materials_db=temp_materials, polarization='Average')
                results_map[i, j] = radiance
            progress_bar.progress((i + 1) / len(range2))
        progress_bar.empty()
        return {'type': '2D', 'map': results_map, 'range1': range1, 'range2': range2, 'name1': p1['name'], 'name2': p2['name']}

def apply_sweep_value(stack, param, value, materials_db, angle=0):
    """Helper to apply a single parameter value to the stack or settings."""
    temp_stack = copy.deepcopy(stack)
    temp_materials = copy.deepcopy(materials_db) if materials_db else copy.deepcopy(materials)
    temp_angle = angle

    ptype, layer_idx, prop = param['parsed_name']

    if ptype == 'thickness':
        temp_stack['layer_stack'][layer_idx]['thickness'] = value
    elif ptype == 'angle':
        temp_angle = value
    elif ptype == 'ri':
        material_to_vary = temp_stack['layer_stack'][layer_idx]['material']
        original_df = temp_materials[material_to_vary].copy()
        if prop == 'n':
            original_df['n'] *= value
        elif prop == 'k':
            original_df['k'] *= value
        temp_materials[material_to_vary] = original_df
        
    return temp_stack, temp_materials, temp_angle

# ==============================================================================
# 4. UI COMPONENTS & HELPER FUNCTIONS
# ==============================================================================

def prepare_stack_for_viewing(stack, materials_db):
    k_top = get_n_complex(stack['layer_stack'][0]['material'], 550, materials_db).imag
    k_bottom = get_n_complex(stack['layer_stack'][-1]['material'], 550, materials_db).imag
    materials_list = [l['material'] for l in stack['layer_stack']]
    thicknesses_list = [l['thickness'] for l in stack['layer_stack']]
    if k_top > k_bottom + 0.5:
        view_direction = 'Bottom'
        materials_list.reverse()
        thicknesses_list.reverse()
        incident_medium, exit_medium = stack['exit_medium'], stack['incident_medium']
    else:
        view_direction = 'Top'
        incident_medium, exit_medium = stack['incident_medium'], stack['exit_medium']
    return view_direction, incident_medium, exit_medium, materials_list, thicknesses_list

def draw_device_structure(stack, selected_index, materials):
    fig, ax = plt.subplots(figsize=(2.5, 5))
    view_direction, _, _, _, _ = prepare_stack_for_viewing(stack, materials)
    layer_stack = stack['layer_stack']
    incident_medium = stack['incident_medium']
    exit_medium = stack['exit_medium']
    if view_direction == 'Bottom':
        layer_stack = list(reversed(layer_stack))
        selected_index = (len(layer_stack) - 1) - selected_index if selected_index is not None else None
        incident_medium, exit_medium = exit_medium, incident_medium
    total_thickness = sum(layer['thickness'] for layer in layer_stack)
    hatch_height = max(50, total_thickness * 0.1) if total_thickness > 0 else 50
    current_pos = 0
    ax.bar(0, hatch_height, bottom=current_pos, color='#e9ecef', edgecolor='black', hatch='\\\\', width=1)
    ax.text(0, current_pos + hatch_height / 2, f"{exit_medium}\n(Exit Medium)", ha='center', va='center', fontsize=8)
    current_pos += hatch_height
    for i, layer in enumerate(layer_stack):
        is_selected = (i == selected_index)
        ax.bar(0, layer['thickness'], bottom=current_pos, color=layer.get('color', '#f0f2f6'), 
               edgecolor='red' if is_selected else 'black', linewidth=2.5 if is_selected else 1, width=1)
        ax.text(0, current_pos + layer['thickness'] / 2, f"{layer['name']}\n{layer['thickness']:.1f} nm", 
                ha='center', va='center', fontsize=8, weight='bold' if is_selected else 'normal',
                bbox=dict(boxstyle="round,pad=0.3", fc='white', ec='none', alpha=0.7) if is_selected else None)
        current_pos += layer['thickness']
    ax.bar(0, hatch_height, bottom=current_pos, color='#e9ecef', edgecolor='black', hatch='//', width=1)
    ax.text(0, current_pos + hatch_height / 2, f"{incident_medium}\n(Incident Medium)", ha='center', va='center', fontsize=8)
    ax.set_xlim(-0.5, 0.5); ax.set_ylim(0, current_pos + hatch_height)
    ax.set_xticks([]); ax.set_yticks([])
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False); ax.spines['left'].set_visible(False)
    ax.set_title("Device Structure", fontsize=11)
    plt.tight_layout()
    return fig

def plot_nk_data(material_name, materials):
    if material_name not in materials:
        st.warning("Material not found.")
        return
    df = materials[material_name]
    fig, ax1 = plt.subplots(figsize=(6, 4))
    color = 'tab:blue'
    ax1.set_xlabel('Wavelength (nm)'); ax1.set_ylabel('Refractive Index (n)', color=color)
    ax1.plot(df['wavelength_nm'], df['n'], color=color, marker='o', markersize=4, linestyle='-')
    ax1.tick_params(axis='y', labelcolor=color); ax1.grid(True, linestyle='--', alpha=0.6)
    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('Extinction Coeff. (k)', color=color)
    ax2.plot(df['wavelength_nm'], df['k'], color=color, marker='x', markersize=4, linestyle='--')
    ax2.tick_params(axis='y', labelcolor=color)
    fig.tight_layout()
    st.pyplot(fig)

def calculate_cie_coords(wavelengths, spectrum):
    """Calculates CIE 1931 (x,y) coordinates from a spectrum."""
    cie_df = get_cie_data()
    if cie_df is None:
        return None

    # Interpolate CIE functions to match the spectrum's wavelengths
    x_bar = np.interp(wavelengths, cie_df['wavelength'], cie_df['x'])
    y_bar = np.interp(wavelengths, cie_df['wavelength'], cie_df['y'])
    z_bar = np.interp(wavelengths, cie_df['wavelength'], cie_df['z'])
    
    # Calculate tristimulus values
    X = np.trapz(spectrum * x_bar, wavelengths)
    Y = np.trapz(spectrum * y_bar, wavelengths)
    Z = np.trapz(spectrum * z_bar, wavelengths)
    
    # Calculate chromaticity coordinates
    total = X + Y + Z
    if total == 0:
        return 0.333, 0.333 # Return white point if no light
    
    x = X / total
    y = Y / total
    return x, y

def calculate_radiance_luminance(wavelengths, spectrum):
    """Calculates radiance and luminance from a spectrum."""
    cie_df = get_cie_data()
    if cie_df is None:
        return 0, 0
    y_interp = np.interp(wavelengths, cie_df['wavelength'], cie_df['y'])
    radiance = np.trapz(spectrum, wavelengths)
    luminance = np.trapz(spectrum * y_interp, wavelengths)
    return radiance, luminance

def calculate_dissipated_power(u_vals, wavelength_nm):
    """사용자 정의: u_inplane vs dissipated power 그래프용 더미 계산"""
    center = 0.9
    width = 0.02
    power = np.exp(-((u_vals - center)**2) / (2 * width**2)) * 15
    return power

def plot_cie_diagram(cie_coords=None):
    """Generates a plot of the CIE 1931 color space with a color-filled background."""
    if not SCIPY_AVAILABLE:
        st.warning("`scipy` is not installed. Cannot generate color-filled CIE plot. Please run `pip install scipy`.")
        return None
        
    cie_data = get_cie_data()
    if cie_data is None:
        st.warning("Cannot plot CIE diagram because CIE data could not be loaded.")
        return None

    # Calculate chromaticity coordinates for the spectral locus
    xyz_sum = cie_data['x'] + cie_data['y'] + cie_data['z']
    locus_x = cie_data['x'] / xyz_sum
    locus_y = cie_data['y'] / xyz_sum

    # Generate a grid of points for the color background
    nx, ny = 200, 200
    grid_x, grid_y = np.meshgrid(np.linspace(0, 0.8, nx), np.linspace(0, 0.9, ny))

    # Interpolate the CIE data onto the grid
    grid_z = griddata(np.vstack((locus_x, locus_y)).T, cie_data['z'], (grid_x, grid_y), method='cubic')
    grid_z = np.nan_to_num(grid_z)

    # Convert CIE xy to RGB for display
    X = grid_x
    Y = grid_y
    Z = 1 - X - Y
    
    # CIE XYZ to sRGB matrix
    M = np.array([[ 3.2404542, -1.5371385, -0.4985314],
                  [-0.9692660,  1.8760108,  0.0415560],
                  [ 0.0556434, -0.2040259,  1.0572252]])

    XYZ = np.dstack((X, Y, Z))
    RGB = np.dot(XYZ, M.T)
    RGB = np.clip(RGB, 0, 1)

    # Gamma correction
    RGB = np.where(RGB <= 0.0031308, 12.92 * RGB, 1.055 * (RGB**(1/2.4)) - 0.055)
    RGB = np.clip(RGB, 0, 1)

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(RGB, extent=(0, 0.8, 0, 0.9), origin='lower', aspect='auto')
    
    # Plot the spectral locus
    ax.plot(locus_x, locus_y, 'k-', linewidth=1.5, label='Spectral Locus')
    ax.plot([locus_x.iloc[-1], locus_x.iloc[0]], [locus_y.iloc[-1], locus_y.iloc[0]], 'k--', linewidth=1.5)

    # Plot the calculated CIE coordinate if provided
    if cie_coords:
        x, y = cie_coords
        ax.plot(x, y, 'ko', markersize=10, mfc='none', mew=2, label=f'Device Emission')
        ax.plot(x, y, 'w+', markersize=8, mew=2)
    
    ax.set_xlabel('CIE x')
    ax.set_ylabel('CIE y')
    ax.set_title('CIE 1931 Color Space')
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlim(-0.05, 0.8)
    ax.set_ylim(-0.05, 0.9)
    ax.legend()
    plt.tight_layout()
    return fig

# ==============================================================================
# 5. SESSION STATE & UI INITIALIZATION
# ==============================================================================

def initialize_session_state(materials, emitters):
    if 'init' not in st.session_state:
        st.session_state.presets = {
            "Default Blue OLED": [
                {'name': 'Anode', 'material': 'ITO', 'thickness': 40.0, 'color': '#d4edda'},
                {'name': 'HTL', 'material': 'NPB', 'thickness': 75.0, 'color': '#fff3cd'},
                {'name': 'EML', 'material': 'Alq3', 'thickness': 60.0, 'color': '#d1ecf1'},
                {'name': 'Cathode', 'material': 'Al', 'thickness': 100.0, 'color': '#ced4da'}
            ]
        }
        st.session_state.current_stack_name = "Default Blue OLED"
        st.session_state.layer_stack = st.session_state.presets["Default Blue OLED"]
        st.session_state.selected_layer_index = 2
        st.session_state.incident_medium = 'Air'
        st.session_state.exit_medium = 'Glass' if 'Glass' in materials else 'Air'
        
        # Initialize separate storage for each result type
        st.session_state.results = {}
        st.session_state.active_plot_key = None # Which result to show
        
        # Initialize sweep parameters
        st.session_state.sweep_params = [
            {'enabled': True, 'name': 'L3 EML Thickness', 'parsed_name': ('thickness', 2, 'd'), 'from': 10, 'to': 100, 'step': 10},
            {'enabled': False, 'name': 'L2 HTL Thickness', 'parsed_name': ('thickness', 1, 'd'), 'from': 10, 'to': 100, 'step': 10},
        ]
        st.session_state.init = True

def render_dissipated_power_tab():
    st.markdown("## 🔥 Dissipated Power Analysis")
    st.markdown("In-plane wavevector (`u_inplane`) vs. dissipated power")

    wavelength = st.slider("Wavelength (nm)", 400, 700, 530)
    u_min, u_max = st.slider("u_inplane Range", 0.0, 2.0, (0.0, 2.0), step=0.01)
    num_points = st.number_input("Number of u_inplane points", 50, 1000, 200, step=10)

    col1, col2, col3 = st.columns(3)
    with col1:
        log_x = st.checkbox("Log X-Axis", False)
        abs_x = st.checkbox("Abs X-Axis", False)
        inv_x = st.checkbox("Invert X-Axis", False)
        grid_x = st.checkbox("Grid X", True)
    with col2:
        log_y = st.checkbox("Log Y-Axis", False)
        abs_y = st.checkbox("Abs Y-Axis", False)
        inv_y = st.checkbox("Invert Y-Axis", False)
        grid_y = st.checkbox("Grid Y", True)

    if st.button("Run Dissipated Power Simulation"):
        u_vals = np.linspace(u_min, u_max, num_points)
        power_vals = calculate_dissipated_power(u_vals, wavelength)

        if abs_y:
            power_vals = np.abs(power_vals)
        if abs_x:
            u_vals = np.abs(u_vals)

        fig, ax = plt.subplots()
        ax.plot(u_vals, power_vals, color="purple", linewidth=2)
        ax.set_xlabel("u_inplane (-)")
        ax.set_ylabel("Dissipated Power (arb. unit)")
        ax.set_title(f"Dissipated Power @ {wavelength} nm")

        if log_x: ax.set_xscale("log")
        if log_y: ax.set_yscale("log")
        if inv_x: ax.invert_xaxis()
        if inv_y: ax.invert_yaxis()
        if grid_x or grid_y: ax.grid(True, alpha=0.5)

        st.pyplot(fig)


# ==============================================================================
# 6. MAIN APP UI
# ==============================================================================
def main():
    st.title("💡 Advanced OLED Optical Simulator")

    if not TMM_AVAILABLE or not SCIPY_AVAILABLE:
        st.stop()

    materials = load_material_data()
    emitters = load_emitter_data()
    initialize_session_state(materials, emitters)

    main_tabs = st.tabs(["**⚙️ Setup & Analysis**", "**📈 Results**", "**🔥 Dissipated Power**"])

    # --- SETUP & ANALYSIS TAB ---
    with main_tabs[0]:
        setup_cols = st.columns([2, 2, 1])

        with setup_cols[0]:
            st.subheader("Layer Structure")
            with st.container(border=True):
                st.markdown("**Presets**")
                preset_name = st.selectbox("Load Preset", options=list(st.session_state.presets.keys()))
                if st.button("Load Selected Preset", use_container_width=True):
                    st.session_state.layer_stack = st.session_state.presets[preset_name]
                    st.session_state.current_stack_name = preset_name
                    st.rerun()

                with st.expander("Save or Delete Presets"):
                    new_preset_name = st.text_input("Enter preset name:", st.session_state.current_stack_name)
                    c1, c2 = st.columns(2)
                    if c1.button("Save", use_container_width=True):
                        if new_preset_name:
                            st.session_state.presets[new_preset_name] = st.session_state.layer_stack
                            st.session_state.current_stack_name = new_preset_name
                            st.toast(f"Preset '{new_preset_name}' saved!")
                            st.rerun()
                    if c2.button("Delete", use_container_width=True, disabled=(len(st.session_state.presets) <= 1)):
                        if new_preset_name in st.session_state.presets:
                            del st.session_state.presets[new_preset_name]
                            st.session_state.current_stack_name = list(st.session_state.presets.keys())[0]
                            st.session_state.layer_stack = st.session_state.presets[st.session_state.current_stack_name]
                            st.toast(f"Preset '{new_preset_name}' deleted.")
                            st.rerun()

                st.markdown("**Surrounding Media**")
                material_options = list(materials.keys())
                st.session_state.incident_medium = st.selectbox("Incident (Top)", material_options, index=material_options.index(st.session_state.incident_medium))
                st.session_state.exit_medium = st.selectbox("Exit (Bottom)", material_options, index=material_options.index(st.session_state.exit_medium))
                
                st.markdown("**Layer Operations**")
                c1, c2, c3, c4 = st.columns(4)
                if c1.button("➕ Add", use_container_width=True, help="Add new layer below selected"):
                    idx = st.session_state.selected_layer_index
                    new_layer = {'name': 'New Layer', 'material': 'Alq3', 'thickness': 20.0, 'color': '#f5c6cb'}
                    st.session_state.layer_stack.insert(idx + 1, new_layer)
                    st.session_state.selected_layer_index = idx + 1
                    st.rerun()
                if c2.button("➖ Del", use_container_width=True, help="Remove selected layer", disabled=(len(st.session_state.layer_stack) <= 1)):
                    idx = st.session_state.selected_layer_index
                    st.session_state.layer_stack.pop(idx)
                    st.session_state.selected_layer_index = min(idx, len(st.session_state.layer_stack) - 1)
                    st.rerun()
                if c3.button("🔼 Up", use_container_width=True, help="Move selected layer up", disabled=(st.session_state.selected_layer_index == 0)):
                    idx = st.session_state.selected_layer_index
                    st.session_state.layer_stack.insert(idx - 1, st.session_state.layer_stack.pop(idx))
                    st.session_state.selected_layer_index = idx - 1
                    st.rerun()
                if c4.button("🔽 Down", use_container_width=True, help="Move selected layer down", disabled=(st.session_state.selected_layer_index >= len(st.session_state.layer_stack) - 1)):
                    idx = st.session_state.selected_layer_index
                    st.session_state.layer_stack.insert(idx + 1, st.session_state.layer_stack.pop(idx))
                    st.session_state.selected_layer_index = idx + 1
                    st.rerun()
                
                st.markdown("---")
                
                table_cols = st.columns([1, 4, 4, 3, 1])
                table_cols[0].write("**Select**")
                table_cols[1].write("**Layer Name**")
                table_cols[2].write("**Material**")
                table_cols[3].write("**Thickness (nm)**")
                table_cols[4].write("**Color**")

                with table_cols[0]:
                    if st.session_state.selected_layer_index >= len(st.session_state.layer_stack):
                        st.session_state.selected_layer_index = len(st.session_state.layer_stack) - 1
                    
                    selected_index = st.radio("Select Layer", range(len(st.session_state.layer_stack)),
                        format_func=lambda i: f"L{i+1}", index=st.session_state.selected_layer_index,
                        key="layer_selector_radio_inline", label_visibility="collapsed")
                    st.session_state.selected_layer_index = selected_index

                for i, layer in enumerate(st.session_state.layer_stack):
                    with table_cols[1]:
                        st.session_state.layer_stack[i]['name'] = st.text_input("Name", value=layer['name'], key=f"name_{i}", label_visibility="collapsed")
                    with table_cols[2]:
                        st.session_state.layer_stack[i]['material'] = st.selectbox("Material", options=material_options, index=material_options.index(layer['material']), key=f"material_{i}", label_visibility="collapsed")
                    with table_cols[3]:
                        st.session_state.layer_stack[i]['thickness'] = st.number_input("Thickness", value=layer['thickness'], min_value=0.0, step=1.0, format="%.1f", key=f"thickness_{i}", label_visibility="collapsed")
                    with table_cols[4]:
                        st.session_state.layer_stack[i]['color'] = st.color_picker("Color", value=layer['color'], key=f"color_{i}", label_visibility="collapsed")

        with setup_cols[1]:
            st.subheader("Material Properties (n,k)")
            with st.container(border=True, height=600):
                selected_material = st.session_state.layer_stack[st.session_state.selected_layer_index]['material']
                plot_nk_data(selected_material, materials)

        with setup_cols[2]:
            st.subheader("Device Visualization")
            current_stack = {'incident_medium': st.session_state.incident_medium, 'exit_medium': st.session_state.exit_medium, 'layer_stack': st.session_state.layer_stack}
            device_fig = draw_device_structure(current_stack, st.session_state.selected_layer_index, materials)
            st.pyplot(device_fig)
            
            st.markdown("---")
            st.subheader("Database Lists")
            db_tabs = st.tabs(["Materials", "Emitters"])
            with db_tabs[0]:
                st.dataframe(pd.DataFrame(list(materials.keys()), columns=['Material Name']), use_container_width=True, height=200)
            with db_tabs[1]:
                if emitters:
                    st.dataframe(pd.DataFrame(list(emitters.keys()), columns=['Emitter Name']), use_container_width=True, height=200)

    # --- RESULTS TAB ---
    with main_tabs[1]:
        result_cols = st.columns([2, 3])
        
        with result_cols[0]:
            st.header("Simulation Parameters")
            with st.container(border=True):
                sim_tabs = st.tabs(["**1D Spectrum**", "**2D Angle Dependence**", "**Variable Mode**", "**Results Manager**"])
                
                with sim_tabs[0]: # 1D Spectrum
                    st.subheader("Emission Spectrum")
                    if not emitters:
                        st.warning("No emitter files found.")
                    else:
                        layer_names = [layer['name'] for layer in st.session_state.layer_stack]
                        eml_idx = st.selectbox("Emissive Layer", range(len(layer_names)), format_func=lambda x: f"{x+1}. {layer_names[x]}", index=st.session_state.selected_layer_index, key="eml_select_1d")
                        pl_name = st.selectbox("Intrinsic PL", list(emitters.keys()), key="pl_select_1d")
                        c1, c2 = st.columns(2)
                        angle_em = c1.slider("Viewing Angle (°)", 0, 89, 0, key='e_angle_1d')
                        pos_em = c2.slider("Emitter Position", 0.0, 1.0, 0.5, 0.05, help="Position within the EML (0=start, 1=end)", key="pos_em_1d")
                        polarization_1d = st.selectbox("Polarization", ["Average", "s-polarization", "p-polarization"], key="pol_1d")
                        if st.button("Calculate Emission", type="primary", use_container_width=True, key="calc_emission_1d"):
                            emitter_pl_data = emitters[pl_name]
                            wl_em, spec_em, norm_spec_em, radiance, luminance, cie_coords = run_emission_simulation(current_stack, emitter_pl_data, eml_idx, pos_em, angle_em, materials, polarization_1d)
                            if wl_em is not None:
                                result_key = f"1D Emission @ {angle_em}° ({polarization_1d})"
                                st.session_state.results[result_key] = {'type': 'Emission_1D', 'data': {'wl': wl_em, 'spec': norm_spec_em, 'pl_data': emitter_pl_data, 'pl_name': pl_name, 'angle': angle_em, 'radiance': radiance, 'luminance': luminance, 'cie': cie_coords}}
                                st.session_state.active_plot_key = result_key
                                st.rerun()
                    st.markdown("---")
                    st.subheader("Reflectance / Transmittance")
                    angle_rt = st.slider("Viewing Angle (°)", -89, 89, 0, key='r_angle_1d')
                    wl_range_rt = st.slider("Wavelength Range (nm)", 380, 850, (400, 800), key='r_wl_1d')
                    polarization_rt = st.selectbox("Polarization", ["Average", "s-polarization", "p-polarization"], key="pol_rt_1d")
                    if st.button("Calculate R/T Spectrum", use_container_width=True, key="calc_rt_1d"):
                        wl_pts = np.linspace(wl_range_rt[0], wl_range_rt[1], 200)
                        R, T = run_reflectance_simulation(current_stack, wl_pts, angle_rt, materials, polarization_rt)
                        if R is not None:
                            result_key = f"1D R/T @ {angle_rt}° ({polarization_rt})"
                            st.session_state.results[result_key] = {'type': 'RT_1D', 'data': {'wl': wl_pts, 'R': R, 'T': T, 'angle': angle_rt}}
                            st.session_state.active_plot_key = result_key
                            st.rerun()

                with sim_tabs[1]: # 2D Angle Dependence
                    st.subheader("Angle-Dependent Emission")
                    if not emitters:
                        st.warning("No emitter files found.")
                    else:
                        layer_names_2d = [layer['name'] for layer in st.session_state.layer_stack]
                        eml_idx_2d = st.selectbox("Emissive Layer", range(len(layer_names_2d)), format_func=lambda x: f"{x+1}. {layer_names_2d[x]}", index=st.session_state.selected_layer_index, key="eml_select_2d")
                        pl_name_2d = st.selectbox("Intrinsic PL", list(emitters.keys()), key="pl_select_2d")
                        angle_range_em_2d = st.slider("Angle Range (°)", -89, 89, (-89, 89), key='e_angle_2d')
                        pos_em_2d = st.slider("Emitter Position", 0.0, 1.0, 0.5, 0.05, help="Position within the EML", key="pos_em_2d")
                        polarization_em_2d = st.selectbox("Polarization", ["Average", "s-polarization", "p-polarization"], key="pol_em_2d")
                        if st.button("Calculate Emission Map", type="primary", use_container_width=True, key="calc_emission_2d"):
                            emitter_pl_data = emitters[pl_name_2d]
                            angles_deg = np.linspace(angle_range_em_2d[0], angle_range_em_2d[1], 90)
                            emission_map = run_emission_angle_sweep_simulation(current_stack, emitter_pl_data, eml_idx_2d, pos_em_2d, angles_deg, materials, polarization_em_2d)
                            if emission_map is not None:
                                result_key = f"2D Emission vs Angle ({polarization_em_2d})"
                                st.session_state.results[result_key] = {'type': 'Emission_2D', 'data': {'map': emission_map, 'wavelengths': emitter_pl_data['wavelength_nm'].values, 'angles': angles_deg}}
                                st.session_state.active_plot_key = result_key
                                st.rerun()
                    st.markdown("---")
                    st.subheader("Angle-Dependent R/T")
                    angle_range_rt_2d = st.slider("Angle Range (°)", -89, 89, (-89, 89), key='rt_angle_2d')
                    wl_range_rt_2d = st.slider("Wavelength Range (nm)", 380, 850, (400, 800), key='rt_wl_2d')
                    polarization_rt_2d = st.selectbox("Polarization", ["Average", "s-polarization", "p-polarization"], key="pol_rt_2d")
                    if st.button("Calculate R/T Map", use_container_width=True, key="calc_rt_2d"):
                        angles_deg = np.linspace(angle_range_rt_2d[0], angle_range_rt_2d[1], 90)
                        wavelengths = np.linspace(wl_range_rt_2d[0], wl_range_rt_2d[1], 100)
                        R_map = run_angle_sweep_simulation(current_stack, wavelengths, angles_deg, 'R', materials, polarization_rt_2d)
                        T_map = run_angle_sweep_simulation(current_stack, wavelengths, angles_deg, 'T', materials, polarization_rt_2d)
                        if R_map is not None:
                            result_key = f"2D R/T vs Angle ({polarization_rt_2d})"
                            st.session_state.results[result_key] = {'type': 'RT_2D', 'data': {'R_map': R_map, 'T_map': T_map, 'wavelengths': wavelengths, 'angles': angles_deg}}
                            st.session_state.active_plot_key = result_key
                            st.rerun()

                with sim_tabs[2]: # Variable Mode
                    st.subheader("Variable Mode Settings")
                    if not emitters:
                        st.warning("No emitter files found.")
                    else:
                        st.write("**Base Emission Settings for Sweep**")
                        pl_name_sweep = st.selectbox("Intrinsic PL", list(emitters.keys()), key="pl_select_sweep")
                        eml_idx_sweep = st.selectbox("Emissive Layer", range(len(st.session_state.layer_stack)), format_func=lambda x: f"{x+1}. {st.session_state.layer_stack[x]['name']}", index=st.session_state.selected_layer_index, key="eml_select_sweep")
                        pos_em_sweep = st.slider("Emitter Position", 0.0, 1.0, 0.5, 0.05, help="Position within the EML", key="pos_em_sweep")
                        
                        st.markdown("---")
                        st.write("**Define Sweep Parameters**")
                        
                        for i, p in enumerate(st.session_state.sweep_params):
                            cols = st.columns([1, 4, 2, 2, 2, 1])
                            p['enabled'] = cols[0].checkbox("", value=p['enabled'], key=f"en_{i}")
                            
                            param_options = {'Viewing Angle': ('angle', -1, 'angle')}
                            for j, layer in enumerate(st.session_state.layer_stack):
                                param_options[f"L{j+1} {layer['name']} Thickness"] = ('thickness', j, 'd')
                                param_options[f"L{j+1} {layer['name']} RI (n) Multiplier"] = ('ri', j, 'n')
                                param_options[f"L{j+1} {layer['name']} RI (k) Multiplier"] = ('ri', j, 'k')
                            
                            param_name = cols[1].selectbox("Parameter", list(param_options.keys()), index=list(param_options.keys()).index(p['name']) if p['name'] in param_options else 0, key=f"p_{i}", label_visibility="collapsed")
                            p['name'] = param_name
                            p['parsed_name'] = param_options[param_name]

                            p['from'] = cols[2].number_input("From", value=p['from'], key=f"from_{i}", label_visibility="collapsed")
                            p['to'] = cols[3].number_input("To", value=p['to'], key=f"to_{i}", label_visibility="collapsed")
                            p['step'] = cols[4].number_input("Step", value=p['step'], key=f"step_{i}", label_visibility="collapsed")

                            if cols[5].button("➖", key=f"del_{i}", help="Remove parameter"):
                                st.session_state.sweep_params.pop(i)
                                st.rerun()

                        if st.button("➕ Add Parameter", use_container_width=True):
                            st.session_state.sweep_params.append({'enabled': False, 'name': 'Viewing Angle', 'parsed_name': ('angle', -1, 'angle'), 'from': 0, 'to': 80, 'step': 10})
                            st.rerun()

                        if st.button("Run Sweep", type="primary", use_container_width=True):
                            sweep_results = run_advanced_sweep(current_stack, emitters[pl_name_sweep], eml_idx_sweep, pos_em_sweep, st.session_state.sweep_params, materials)
                            if sweep_results:
                                result_key = f"Sweep of {len(sweep_results.get('df', []))} points" if sweep_results['type'] == '1D' else f"2D Sweep: {sweep_results['name1']} vs {sweep_results['name2']}"
                                st.session_state.results[result_key] = {'type': 'Sweep', 'data': sweep_results}
                                st.session_state.active_plot_key = result_key
                                st.rerun()
                
                with sim_tabs[3]: # Results Manager
                    st.header("Results Overview")
                    if not st.session_state.results:
                        st.info("No results yet. Run a simulation from one of the other tabs.")
                    else:
                        for key, result in st.session_state.results.items():
                            with st.expander(f"Result: {key}"):
                                st.write(result)
                        if st.button("Clear All Results", use_container_width=True):
                            st.session_state.results = {}
                            st.rerun()

                with main_tabs[2]:  # 🔥 Dissipated Power
                    render_dissipated_power_tab()

        
        with result_cols[1]:
            st.header("Output Plots")
            
            available_results = list(st.session_state.results.keys())
            
            if not available_results:
                st.info("Select parameters on the left and click a 'Calculate' button to see results here.")
            else:
                active_plot_key = st.selectbox(
                    "Select a result to view",
                    options=available_results,
                    index=available_results.index(st.session_state.active_plot_key) if st.session_state.active_plot_key in available_results else 0
                )

                if 'view_direction_info' in st.session_state:
                    st.info(st.session_state.view_direction_info)

                result_to_show = st.session_state.results[active_plot_key]
                plot_type = result_to_show['type']
                data = result_to_show['data']
                
                if plot_type == 'Emission_1D':
                    st.subheader("Emission Analysis")
                    plot_cols = st.columns(2)
                    with plot_cols[0]:
                        fig, ax = plt.subplots()
                        pl_norm = data['pl_data']['intensity'] / data['pl_data']['intensity'].max()
                        ax.plot(data['pl_data']['wavelength_nm'], pl_norm, 'k--', label=f"Intrinsic PL ({data['pl_name']})", alpha=0.6)
                        ax.plot(data['wl'], data['spec'], label=f"Final Emission @ {data['angle']}°", color='crimson', linewidth=2)
                        ax.set_xlabel("Wavelength (nm)"); ax.set_ylabel("Normalized Intensity")
                        ax.grid(True, alpha=0.5); ax.legend(); ax.set_ylim(bottom=0)
                        st.pyplot(fig)
                    with plot_cols[1]:
                        cie_fig = plot_cie_diagram(data['cie'])
                        if cie_fig:
                            st.pyplot(cie_fig)
                    metric_cols = st.columns(3)
                    metric_cols[0].metric("Radiance (arb. units)", f"{data['radiance']:.3f}")
                    metric_cols[1].metric("Luminance (arb. units)", f"{data['luminance']:.3f}")
                    if data['cie']:
                        metric_cols[2].metric("CIE (x, y)", f"({data['cie'][0]:.3f}, {data['cie'][1]:.3f})")
                
                elif plot_type == 'RT_1D':
                    st.subheader("R/T Spectrum Result")
                    fig, ax = plt.subplots()
                    ax.plot(data['wl'], data['R'] * 100, label='Reflectance')
                    ax.plot(data['wl'], data['T'] * 100, label='Transmittance', linestyle='--')
                    ax.set_title(f"Spectrum at {data['angle']}°"); ax.set_xlabel("Wavelength (nm)"); ax.set_ylabel("Percent (%)")
                    ax.set_ylim(0, 100); ax.grid(True, alpha=0.5); ax.legend()
                    st.pyplot(fig)

                elif plot_type in ['Emission_2D', 'RT_2D']:
                    st.subheader("2D Angle-Dependent Result")
                    if plot_type == 'Emission_2D':
                        plot_data = data
                        data_to_plot = plot_data['map']
                        plot_title = "Angle-Resolved Emission"
                        cbar_label = "Intensity (arb. units)"
                        is_percentage = False
                    else: # RT_2D
                        plot_data = data
                        plot_type_select = st.radio("Plot Type", ["Reflectance", "Transmittance"], horizontal=True, key="2d_rt_type")
                        data_to_plot = plot_data['R_map'] if plot_type_select == "Reflectance" else plot_data['T_map']
                        plot_title = f"Angle-Resolved {plot_type_select}"
                        cbar_label = f"{plot_type_select} (%)"
                        data_to_plot *= 100
                        is_percentage = True

                    with st.container(border=True):
                        st.write("**Plot Controls**")
                        c1, c2 = st.columns(2)
                        colormap = c1.selectbox("Colormap", ('viridis', 'plasma', 'inferno', 'magma', 'jet', 'rainbow'), key="2d_cmap")
                        log_scale = c2.checkbox("Log Color Scale", key="2d_log")
                        use_manual_range = st.checkbox("Set Manual Color Bar Range", key="2d_manual_range")
                        if use_manual_range:
                            c1, c2 = st.columns(2)
                            default_min = 0.0
                            default_max = 100.0 if is_percentage else 1.0
                            cbar_min = c1.number_input("Min", value=default_min, format="%.2f", key="2d_min")
                            cbar_max = c2.number_input("Max", value=default_max, format="%.2f", key="2d_max")
                        else: cbar_min, cbar_max = None, None

                    fig, ax = plt.subplots(figsize=(8, 6))
                    norm = None
                    if log_scale:
                        vmin = cbar_min if use_manual_range else np.min(data_to_plot[data_to_plot > 0])
                        norm = mcolors.LogNorm(vmin=max(vmin, 1e-9), vmax=cbar_max)
                    elif use_manual_range:
                        norm = mcolors.Normalize(vmin=cbar_min, vmax=cbar_max)
                    im = ax.pcolormesh(plot_data['wavelengths'], plot_data['angles'], data_to_plot, shading='gouraud', cmap=colormap, norm=norm)
                    cbar = fig.colorbar(im, ax=ax); cbar.set_label(cbar_label)
                    ax.set_xlabel("Wavelength (nm)"); ax.set_ylabel("Angle (degrees)"); ax.set_title(plot_title)
                    st.pyplot(fig)
                    
                elif plot_type == 'Sweep':
                    sweep_data = data
                    if sweep_data['type'] == '1D':
                        st.subheader("1D Sweep Result")
                        fig, ax = plt.subplots()
                        ax.plot(sweep_data['df']['val'], sweep_data['df']['radiance'], label='Radiance', marker='o')
                        ax.plot(sweep_data['df']['val'], sweep_data['df']['luminance'], label='Luminance', marker='x')
                        ax.set_xlabel(f"{sweep_data['param_name']}"); ax.set_ylabel("Value (arb. units)")
                        ax.set_title(f"Performance vs. {sweep_data['param_name']}"); ax.grid(True, alpha=0.5); ax.legend()
                        st.pyplot(fig)
                    elif sweep_data['type'] == '2D':
                        st.subheader("2D Sweep Result")
                        with st.container(border=True):
                            st.write("**Plot Controls**")
                            c1, c2 = st.columns(2)
                            colormap = c1.selectbox("Colormap", ('viridis', 'plasma', 'inferno', 'magma', 'jet', 'rainbow'), key="sweep_2d_cmap")
                            use_manual_range = c2.checkbox("Set Manual Color Bar Range", key="sweep_2d_manual_range")
                            if use_manual_range:
                                c1, c2 = st.columns(2)
                                cbar_min = c1.number_input("Min", value=float(sweep_data['map'].min()), format="%.3f", key="sweep_2d_min")
                                cbar_max = c2.number_input("Max", value=float(sweep_data['map'].max()), format="%.3f", key="sweep_2d_max")
                            else:
                                cbar_min, cbar_max = None, None
                                
                        fig, ax = plt.subplots(figsize=(8, 6))
                        im = ax.pcolormesh(sweep_data['range1'], sweep_data['range2'], sweep_data['map'], shading='gouraud', cmap=colormap, vmin=cbar_min, vmax=cbar_max)
                        cbar = fig.colorbar(im, ax=ax); cbar.set_label("Radiance (arb. units)")
                        ax.set_xlabel(f"{sweep_data['name1']}"); ax.set_ylabel(f"{sweep_data['name2']}")
                        ax.set_title(f"Radiance vs. {sweep_data['name1']} and {sweep_data['name2']}")
                        st.pyplot(fig)

if __name__ == '__main__':
    main()
