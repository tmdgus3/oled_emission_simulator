import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import os
import json
import copy
import io
import urllib.request

# --- Page Configuration (MUST BE THE FIRST STREAMLIT COMMAND) ---
st.set_page_config(
    page_title="OLED Optical Simulator",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- Robust Library Import ---
try:
    from tmm import coh_tmm
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
# 1. CORE PHYSICS & CALCULATION ENGINE
# ==============================================================================

def make_2x2_array(a, b, c, d, dtype=complex):
    """A utility function to quickly create a 2x2 numpy array."""
    my_array = np.empty((2, 2), dtype=dtype)
    my_array[0, 0] = a; my_array[0, 1] = b
    my_array[1, 0] = c; my_array[1, 1] = d
    return my_array

def snell_law(n1, n2, th1):
    """
    Calculates the angle in a new medium using Snell's Law.
    Handles complex refractive indices and total internal reflection.
    """
    arg = n1 / n2 * np.sin(th1)
    if isinstance(arg, complex):
        return np.arcsin(arg)
    if abs(arg) > 1:
        return np.arcsin(arg + 0j)
    return np.arcsin(arg)

def calculate_intensity_profile(n_list, d_list, th_0, lam_vac):
    """
    Calculates the electric field intensity profile |E|^2 throughout the stack.
    This is a core function for emission calculations. It averages s and p polarizations.
    """
    num_layers = len(n_list)
    th_list = np.array([snell_law(n_list[0], n, th_0) for n in n_list])
    kz_list = 2 * np.pi * n_list * np.cos(th_list) / lam_vac

    intensities = []
    for pol in ['s', 'p']:
        M_list = []
        for i in range(num_layers - 1):
            if pol == 's':
                Y_i, Y_j = n_list[i] * np.cos(th_list[i]), n_list[i+1] * np.cos(th_list[i+1])
            else: # p-polarization
                Y_i, Y_j = np.cos(th_list[i]) / n_list[i], np.cos(th_list[i+1]) / n_list[i+1]
            if abs(Y_i) < 1e-9: Y_i = 1e-9
            M_list.append(0.5 * make_2x2_array(1 + Y_j/Y_i, 1 - Y_j/Y_i, 1 - Y_j/Y_i, 1 + Y_j/Y_i))

        P_list = [make_2x2_array(np.exp(1j*kz_list[i]*d_list[i]), 0, 0, np.exp(-1j*kz_list[i]*d_list[i])) for i in range(1, num_layers - 1)]

        T_total = M_list[0]
        for i in range(len(P_list)):
            T_total = T_total @ P_list[i] @ M_list[i+1]
        
        coeffs = [np.array([[1/T_total[0,0]],[0]])]
        for i in range(len(P_list) - 1, -1, -1):
            coeffs.insert(0, P_list[i] @ M_list[i+1] @ coeffs[0])
        coeffs.insert(0, M_list[0] @ coeffs[0])

        z_pts_pol, I_pts_pol = [], []
        current_z = 0
        for i in range(1, num_layers - 1):
            d_layer = d_list[i]
            z_in_layer = np.linspace(0, d_layer, 30)
            A, B = coeffs[i][0,0], coeffs[i][1,0]
            kz = kz_list[i]
            for z in z_in_layer:
                intensity = np.abs(A * np.exp(1j * kz * z) + B * np.exp(-1j * kz * z))**2
                I_pts_pol.append(intensity)
                z_pts_pol.append(current_z + z)
            current_z += d_layer
        intensities.append(np.array(I_pts_pol))
    
    avg_intensity = (intensities[0] + intensities[1]) / 2.0
    return np.array(z_pts_pol), avg_intensity

# ==============================================================================
# 2. DATA LOADING & MANAGEMENT
# ==============================================================================

@st.cache_data
def load_material_data():
    materials = {'Air': pd.DataFrame({'wavelength_nm': [400, 800], 'n': [1.0, 1.0], 'k': [0.0, 0.0]})}
    if not os.path.isdir('materials'):
        st.error("The 'materials' directory was not found.")
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

def get_n_complex(material_name, wl, materials_db=None):
    if materials_db is None:
        materials_db = materials
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

# ==============================================================================
# 3. SIMULATION RUNNERS
# ==============================================================================

def run_reflectance_simulation(stack, wavelengths, angle_deg):
    if not TMM_AVAILABLE: return None, None
    view_direction, incident_medium, exit_medium, materials_list, thicknesses_list = prepare_stack_for_viewing(stack)
    st.session_state.view_direction_info = f"Auto-detected **{view_direction} Emission**. Simulating light from **{incident_medium}** side."
    results_R, results_T = [], []
    d_list = [np.inf] + thicknesses_list + [np.inf]
    th_0 = np.deg2rad(angle_deg)
    for wl in wavelengths:
        n_list = [get_n_complex(incident_medium, wl)] + [get_n_complex(m, wl) for m in materials_list] + [get_n_complex(exit_medium, wl)]
        tmm_s = coh_tmm('s', n_list, d_list, th_0, wl)
        tmm_p = coh_tmm('p', n_list, d_list, th_0, wl)
        results_R.append((tmm_s['R'] + tmm_p['R']) / 2)
        results_T.append((tmm_s['T'] + tmm_p['T']) / 2)
    return np.array(results_R), np.array(results_T)

def run_emission_simulation(stack, emitter_pl, emissive_layer_index, emission_position, view_angle_deg, materials_db=None):
    if not TMM_AVAILABLE: return None, None, None, None, None
    view_direction, incident_medium, exit_medium, materials_list, thicknesses_list = prepare_stack_for_viewing(stack, materials_db=materials_db)
    if view_direction == 'Bottom':
        emissive_layer_index = (len(materials_list) - 1) - emissive_layer_index
    st.session_state.view_direction_info = f"Auto-detected **{view_direction} Emission**. Simulating light exiting from **{exit_medium}** side."
    wavelengths, intrinsic_pl = emitter_pl['wavelength_nm'].values, emitter_pl['intensity'].values
    final_emission_spectrum = []
    d_list_sim = [np.inf] + thicknesses_list + [np.inf]
    depth_to_eml_start = sum(thicknesses_list[:emissive_layer_index])
    eml_thickness = thicknesses_list[emissive_layer_index]
    emitter_depth = depth_to_eml_start + (eml_thickness * emission_position)
    th_0 = np.deg2rad(view_angle_deg)
    
    for wl, pl in zip(wavelengths, intrinsic_pl):
        n_list_complex = np.array([get_n_complex(incident_medium, wl, materials_db)] + [get_n_complex(mat, wl, materials_db) for mat in materials_list] + [get_n_complex(exit_medium, wl, materials_db)])
        z_pts, intensity_profile = calculate_intensity_profile(n_list_complex, d_list_sim, th_0, wl)
        intensity_at_emitter = np.interp(emitter_depth, z_pts, intensity_profile, left=0, right=0)
        final_emission_spectrum.append(pl * intensity_at_emitter)
        
    final_emission_spectrum = np.array(final_emission_spectrum)
    radiance, luminance = calculate_radiance_luminance(wavelengths, final_emission_spectrum)
    cie_coords = calculate_cie_coords(wavelengths, final_emission_spectrum)
    
    normalized_spectrum = final_emission_spectrum.copy()
    if normalized_spectrum.max() > 0:
        normalized_spectrum /= normalized_spectrum.max()
    return wavelengths, normalized_spectrum, radiance, luminance, cie_coords

def run_angle_sweep_simulation(stack, wavelengths, angles_deg, sim_type='R'):
    if not TMM_AVAILABLE: return None
    view_direction, incident_medium, exit_medium, materials_list, thicknesses_list = prepare_stack_for_viewing(stack)
    st.session_state.view_direction_info = f"Auto-detected **{view_direction} Emission**. Simulating light from **{incident_medium}** side."
    
    results_map = np.zeros((len(angles_deg), len(wavelengths)))
    d_list = [np.inf] + thicknesses_list + [np.inf]
    progress_bar = st.progress(0, text=f"Calculating {sim_type} map...")

    for i, angle in enumerate(angles_deg):
        th_0 = np.deg2rad(angle)
        for j, wl in enumerate(wavelengths):
            n_list = [get_n_complex(incident_medium, wl)] + [get_n_complex(m, wl) for m in materials_list] + [get_n_complex(exit_medium, wl)]
            tmm_s = coh_tmm('s', n_list, d_list, th_0, wl)
            tmm_p = coh_tmm('p', n_list, d_list, th_0, wl)
            results_map[i, j] = (tmm_s[sim_type] + tmm_p[sim_type]) / 2
        progress_bar.progress((i + 1) / len(angles_deg), text=f"Calculating... Angle {int(angle)}°")
    progress_bar.empty()
    return results_map

def run_emission_angle_sweep_simulation(stack, emitter_pl, emissive_layer_index, emission_position, angles_deg):
    if not TMM_AVAILABLE: return None
    view_direction, incident_medium, exit_medium, materials_list, thicknesses_list = prepare_stack_for_viewing(stack)
    if view_direction == 'Bottom':
        emissive_layer_index = (len(materials_list) - 1) - emissive_layer_index
    st.session_state.view_direction_info = f"Auto-detected **{view_direction} Emission**. Simulating light exiting from **{exit_medium}** side."
    
    wavelengths = emitter_pl['wavelength_nm'].values
    emission_map = np.zeros((len(angles_deg), len(wavelengths)))
    progress_bar = st.progress(0, text="Calculating emission map...")

    for i, angle in enumerate(angles_deg):
        _, spectrum, _, _, _ = run_emission_simulation(stack, emitter_pl, emissive_layer_index, emission_position, angle)
        if spectrum is not None:
            emission_map[i, :] = spectrum
        progress_bar.progress((i + 1) / len(angles_deg), text=f"Calculating... Angle {int(angle)}°")
    
    progress_bar.empty()
    return emission_map

def run_advanced_sweep(base_stack, emitter_pl, emissive_layer_index, emission_position, sweep_params):
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
            temp_stack, temp_materials, temp_angle = apply_sweep_value(base_stack, p1, val1)
            _, _, radiance, luminance, _ = run_emission_simulation(temp_stack, emitter_pl, emissive_layer_index, emission_position, temp_angle, materials_db=temp_materials)
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
                temp_stack, temp_materials, temp_angle = apply_sweep_value(base_stack, p1, val1)
                temp_stack, temp_materials, temp_angle = apply_sweep_value(temp_stack, p2, val2, temp_materials, temp_angle)
                _, _, radiance, _, _ = run_emission_simulation(temp_stack, emitter_pl, emissive_layer_index, emission_position, temp_angle, materials_db=temp_materials)
                results_map[i, j] = radiance
            progress_bar.progress((i + 1) / len(range2))
        progress_bar.empty()
        return {'type': '2D', 'map': results_map, 'range1': range1, 'range2': range2, 'name1': p1['name'], 'name2': p2['name']}

def apply_sweep_value(stack, param, value, materials_db=None, angle=0):
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

def prepare_stack_for_viewing(stack, materials_db=None):
    if materials_db is None:
        materials_db = materials
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

def draw_device_structure(stack, selected_index=None):
    fig, ax = plt.subplots(figsize=(2.5, 5))
    view_direction, _, _, _, _ = prepare_stack_for_viewing(stack)
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

def plot_nk_data(material_name):
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

@st.cache_data
def get_cie_data():
    """Loads CIE 1931 color matching functions from an internal string."""
    cie_data_string = """wavelength,x,y,z
380,0.001368,0.000039,0.00645
385,0.002236,0.000064,0.01055
390,0.004243,0.00012,0.02005
395,0.00765,0.000217,0.03621
400,0.01431,0.000396,0.06785
405,0.02319,0.00064,0.1102
410,0.04351,0.00121,0.2074
415,0.07763,0.00218,0.3713
420,0.13438,0.004,0.6456
425,0.21477,0.0073,1.03905
430,0.2839,0.0116,1.3856
435,0.3285,0.01684,1.62296
440,0.34828,0.023,1.74706
445,0.34806,0.0298,1.7826
450,0.3362,0.038,1.77211
455,0.3187,0.048,1.7441
460,0.2908,0.06,1.6692
465,0.2511,0.0739,1.5281
470,0.19536,0.09098,1.28764
475,0.1421,0.1126,1.0419
480,0.09564,0.13902,0.81295
485,0.05795,0.1693,0.6162
490,0.03201,0.20802,0.46518
495,0.0147,0.2586,0.3533
500,0.0049,0.323,0.272
505,0.0024,0.4073,0.2123
510,0.0093,0.503,0.1582
515,0.0291,0.6082,0.1117
520,0.06327,0.71,0.07825
525,0.1096,0.7932,0.05725
530,0.1655,0.862,0.04216
535,0.22575,0.91485,0.02984
540,0.2904,0.954,0.0203
545,0.3597,0.9803,0.0134
550,0.43345,0.99495,0.00875
555,0.51205,1.0,0.00575
560,0.5945,0.995,0.0039
565,0.6784,0.9786,0.00275
570,0.7621,0.952,0.0021
575,0.8425,0.9154,0.0018
580,0.9163,0.87,0.00165
585,0.9786,0.8163,0.0014
590,1.0263,0.757,0.0011
595,1.0567,0.6949,0.0008
600,1.0622,0.631,0.00065
605,1.0456,0.5668,0.00051
610,1.0026,0.503,0.00034
615,0.9384,0.4412,0.00024
620,0.85445,0.381,0.00019
625,0.7514,0.321,0.00012
630,0.6424,0.265,0.00008
635,0.5419,0.217,0.00005
640,0.4479,0.175,0.00003
645,0.3608,0.1382,0.00002
650,0.2835,0.107,0.0
655,0.2187,0.0816,0.0
660,0.1649,0.061,0.0
665,0.1212,0.04458,0.0
670,0.0874,0.032,0.0
675,0.0636,0.0232,0.0
680,0.04677,0.017,0.0
685,0.0329,0.01192,0.0
690,0.0227,0.00821,0.0
695,0.01584,0.005723,0.0
700,0.011359,0.004102,0.0
705,0.008111,0.002929,0.0
710,0.00579,0.002091,0.0
715,0.004109,0.001484,0.0
720,0.002899,0.001047,0.0
725,0.002049,0.00074,0.0
730,0.001439,0.00052,0.0
735,0.000999,0.000361,0.0
740,0.00069,0.000249,0.0
745,0.000476,0.000172,0.0
750,0.000332,0.00012,0.0
755,0.000235,0.000085,0.0
760,0.000166,0.00006,0.0
765,0.000117,0.000042,0.0
770,0.000083,0.00003,0.0
775,0.000059,0.000021,0.0
780,0.000042,0.000015,0.0
"""
    csv_file = io.StringIO(cie_data_string)
    cie_df = pd.read_csv(csv_file)
    return cie_df

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

materials = load_material_data()
emitters = load_emitter_data()

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
    st.session_state.results = {
        'Emission_1D': None,
        'RT_1D': None,
        'Emission_2D': None,
        'RT_2D': None,
        'Sweep': None
    }
    st.session_state.active_plot_key = None # Which result to show
    
    # Initialize sweep parameters
    st.session_state.sweep_params = [
        {'enabled': True, 'name': 'L3 EML Thickness', 'parsed_name': ('thickness', 2, 'd'), 'from': 10, 'to': 100, 'step': 10},
        {'enabled': False, 'name': 'L2 HTL Thickness', 'parsed_name': ('thickness', 1, 'd'), 'from': 10, 'to': 100, 'step': 10},
    ]
    st.session_state.init = True

# ==============================================================================
# 6. MAIN APP UI
# ==============================================================================

st.title("💡 Advanced OLED Optical Simulator")

if not TMM_AVAILABLE or not SCIPY_AVAILABLE:
    st.stop()

main_tabs = st.tabs(["**⚙️ Setup & Analysis**", "**📈 Results**"])

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
            plot_nk_data(selected_material)

    with setup_cols[2]:
        st.subheader("Device Visualization")
        current_stack = {'incident_medium': st.session_state.incident_medium, 'exit_medium': st.session_state.exit_medium, 'layer_stack': st.session_state.layer_stack}
        device_fig = draw_device_structure(current_stack, st.session_state.selected_layer_index)
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
            sim_tabs = st.tabs(["**1D Spectrum**", "**2D Angle Dependence**", "**Sweep Mode**"])
            
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
                    if st.button("Calculate Emission", type="primary", use_container_width=True, key="calc_emission_1d"):
                        emitter_pl_data = emitters[pl_name]
                        wl_em, spec_em, radiance, luminance, cie_coords = run_emission_simulation(current_stack, emitter_pl_data, eml_idx, pos_em, angle_em)
                        if wl_em is not None:
                            st.session_state.results['Emission_1D'] = {'wl': wl_em, 'spec': spec_em, 'pl_data': emitter_pl_data, 'pl_name': pl_name, 'angle': angle_em, 'radiance': radiance, 'luminance': luminance, 'cie': cie_coords}
                            st.session_state.active_plot_key = 'Emission_1D'
                            st.rerun()
                st.markdown("---")
                st.subheader("Reflectance / Transmittance")
                angle_rt = st.slider("Viewing Angle (°)", -89, 89, 0, key='r_angle_1d')
                wl_range_rt = st.slider("Wavelength Range (nm)", 380, 850, (400, 800), key='r_wl_1d')
                if st.button("Calculate R/T Spectrum", use_container_width=True, key="calc_rt_1d"):
                    wl_pts = np.linspace(wl_range_rt[0], wl_range_rt[1], 200)
                    R, T = run_reflectance_simulation(current_stack, wl_pts, angle_rt)
                    if R is not None:
                        st.session_state.results['RT_1D'] = {'wl': wl_pts, 'R': R, 'T': T, 'angle': angle_rt}
                        st.session_state.active_plot_key = 'RT_1D'
                        st.rerun()

            with sim_tabs[1]: # 2D Angle Dependence
                st.subheader("Angle-Dependent Emission")
                if not emitters:
                    st.warning("No emitter files found.")
                else:
                    layer_names_2d = [layer['name'] for layer in st.session_state.layer_stack]
                    eml_idx_2d = st.selectbox("Emissive Layer", range(len(layer_names_2d)), format_func=lambda x: f"{x+1}. {layer_names_2d[x]}", index=st.session_state.selected_layer_index, key="eml_select_2d")
                    pl_name_2d = st.selectbox("Intrinsic PL", list(emitters.keys()), key="pl_select_2d")
                    angle_range_em_2d = st.slider("Angle Range (°)", 0, 89, (0, 89), key='e_angle_2d')
                    pos_em_2d = st.slider("Emitter Position", 0.0, 1.0, 0.5, 0.05, help="Position within the EML", key="pos_em_2d")
                    if st.button("Calculate Emission Map", type="primary", use_container_width=True, key="calc_emission_2d"):
                        emitter_pl_data = emitters[pl_name_2d]
                        angles_deg = np.linspace(angle_range_em_2d[0], angle_range_em_2d[1], 45)
                        emission_map = run_emission_angle_sweep_simulation(current_stack, emitter_pl_data, eml_idx_2d, pos_em_2d, angles_deg)
                        if emission_map is not None:
                            st.session_state.results['Emission_2D'] = {'map': emission_map, 'wavelengths': emitter_pl_data['wavelength_nm'].values, 'angles': angles_deg}
                            st.session_state.active_plot_key = 'Emission_2D'
                            st.rerun()
                st.markdown("---")
                st.subheader("Angle-Dependent R/T")
                angle_range_rt_2d = st.slider("Angle Range (°)", -89, 89, (-80, 80), key='rt_angle_2d')
                wl_range_rt_2d = st.slider("Wavelength Range (nm)", 380, 850, (400, 800), key='rt_wl_2d')
                if st.button("Calculate R/T Map", use_container_width=True, key="calc_rt_2d"):
                    angles_deg = np.linspace(angle_range_rt_2d[0], angle_range_rt_2d[1], 90)
                    wavelengths = np.linspace(wl_range_rt_2d[0], wl_range_rt_2d[1], 100)
                    R_map = run_angle_sweep_simulation(current_stack, wavelengths, angles_deg, 'R')
                    T_map = run_angle_sweep_simulation(current_stack, wavelengths, angles_deg, 'T')
                    if R_map is not None:
                        st.session_state.results['RT_2D'] = {'R_map': R_map, 'T_map': T_map, 'wavelengths': wavelengths, 'angles': angles_deg}
                        st.session_state.active_plot_key = 'RT_2D'
                        st.rerun()

            with sim_tabs[2]: # Sweep Mode
                st.subheader("Sweep Settings")
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
                        sweep_results = run_advanced_sweep(current_stack, emitters[pl_name_sweep], eml_idx_sweep, pos_em_sweep, st.session_state.sweep_params)
                        if sweep_results:
                            st.session_state.results['Sweep'] = sweep_results
                            st.session_state.active_plot_key = 'Sweep'
                            st.rerun()

    with result_cols[1]:
        st.header("Output Plots")
        
        available_results = {k: v for k, v in st.session_state.results.items() if v is not None}
        
        if not available_results:
            st.info("Select parameters on the left and click a 'Calculate' button to see results here.")
        else:
            # Let user select which plot to view
            active_plot_key = st.selectbox(
                "Select a result to view",
                options=list(available_results.keys()),
                index=list(available_results.keys()).index(st.session_state.active_plot_key) if st.session_state.active_plot_key in available_results else 0
            )

            if 'view_direction_info' in st.session_state:
                st.info(st.session_state.view_direction_info)

            if active_plot_key == 'Emission_1D':
                st.subheader("Emission Analysis")
                data = st.session_state.results['Emission_1D']
                
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
            
            elif active_plot_key == 'RT_1D':
                st.subheader("R/T Spectrum Result")
                data = st.session_state.results['RT_1D']
                fig, ax = plt.subplots()
                ax.plot(data['wl'], data['R'] * 100, label='Reflectance')
                ax.plot(data['wl'], data['T'] * 100, label='Transmittance', linestyle='--')
                ax.set_title(f"Spectrum at {data['angle']}°"); ax.set_xlabel("Wavelength (nm)"); ax.set_ylabel("Percent (%)")
                ax.set_ylim(0, 100); ax.grid(True, alpha=0.5); ax.legend()
                st.pyplot(fig)

            elif active_plot_key in ['Emission_2D', 'RT_2D']:
                st.subheader("2D Angle-Dependent Result")
                if active_plot_key == 'Emission_2D':
                    plot_data = st.session_state.results['Emission_2D']
                    data_to_plot = plot_data['map']
                    plot_title = "Angle-Resolved Emission"
                    cbar_label = "Normalized Intensity"
                    is_percentage = False
                else: # RT_2D
                    plot_data = st.session_state.results['RT_2D']
                    plot_type = st.radio("Plot Type", ["Reflectance", "Transmittance"], horizontal=True, key="2d_rt_type")
                    data_to_plot = plot_data['R_map'] if plot_type == "Reflectance" else plot_data['T_map']
                    plot_title = f"Angle-Resolved {plot_type}"
                    cbar_label = f"{plot_type} (%)"
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
                
            elif active_plot_key == 'Sweep':
                data = st.session_state.results['Sweep']
                if data['type'] == '1D':
                    st.subheader("1D Sweep Result")
                    fig, ax = plt.subplots()
                    ax.plot(data['df']['val'], data['df']['radiance'], label='Radiance', marker='o')
                    ax.plot(data['df']['val'], data['df']['luminance'], label='Luminance', marker='x')
                    ax.set_xlabel(f"{data['param_name']}"); ax.set_ylabel("Value (arb. units)")
                    ax.set_title(f"Performance vs. {data['param_name']}"); ax.grid(True, alpha=0.5); ax.legend()
                    st.pyplot(fig)
                elif data['type'] == '2D':
                    st.subheader("2D Sweep Result")
                    with st.container(border=True):
                        st.write("**Plot Controls**")
                        c1, c2 = st.columns(2)
                        colormap = c1.selectbox("Colormap", ('viridis', 'plasma', 'inferno', 'magma', 'jet', 'rainbow'), key="sweep_2d_cmap")
                        use_manual_range = c2.checkbox("Set Manual Color Bar Range", key="sweep_2d_manual_range")
                        if use_manual_range:
                            c1, c2 = st.columns(2)
                            cbar_min = c1.number_input("Min", value=float(data['map'].min()), format="%.3f", key="sweep_2d_min")
                            cbar_max = c2.number_input("Max", value=float(data['map'].max()), format="%.3f", key="sweep_2d_max")
                        else:
                            cbar_min, cbar_max = None, None
                            
                    fig, ax = plt.subplots(figsize=(8, 6))
                    im = ax.pcolormesh(data['range1'], data['range2'], data['map'], shading='gouraud', cmap=colormap, vmin=cbar_min, vmax=cbar_max)
                    cbar = fig.colorbar(im, ax=ax); cbar.set_label("Radiance (arb. units)")
                    ax.set_xlabel(f"{data['name1']}"); ax.set_ylabel(f"{data['name2']}")
                    ax.set_title(f"Radiance vs. {data['name1']} and {data['name2']}")
                    st.pyplot(fig)

st.markdown("---")
st.write("Simulator v13.0 (Final Version) | Built with Streamlit & TMM")
