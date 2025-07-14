import os
import numpy as np
import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
from tmm import coh_tmm
import uvicorn

# ===== FastAPI 설정 =====
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

# ===== 데이터 모델 =====
class Layer(BaseModel):
    name: str
    material: str
    thickness: float

class SimRequest(BaseModel):
    stack: List[Layer]
    emitter_position: float
    view_angle: float

# ===== 재료 로딩 =====
MATERIALS_DIR = "materials"
materials_db = {}

def load_materials():
    global materials_db
    materials_db = {}
    for file in os.listdir(MATERIALS_DIR):
        if file.endswith(".csv"):
            name = file[:-4]
            df = pd.read_csv(os.path.join(MATERIALS_DIR, file))
            if all(col in df.columns for col in ['wavelength_nm', 'n', 'k']):
                materials_db[name] = df.sort_values("wavelength_nm").reset_index(drop=True)

def get_n_complex(material, wl):
    if material not in materials_db:
        raise ValueError(f"{material} not found in database.")
    df = materials_db[material]
    n = np.interp(wl, df['wavelength_nm'], df['n'])
    k = np.interp(wl, df['wavelength_nm'], df['k'])
    return n + 1j * k

# ===== EMISSION =====
@app.post("/emission")
async def emission(data: SimRequest):
    try:
        wavelengths = np.linspace(400, 800, 100)
        th_0 = np.deg2rad(data.view_angle)
        stack = data.stack
        d_list = [np.inf] + [l.thickness for l in stack] + [np.inf]
        materials = [l.material for l in stack]
        eml_index = len(stack) // 2
        eml_thickness = stack[eml_index].thickness
        emitter_depth = sum(d_list[1:eml_index+1]) + data.emitter_position * eml_thickness

        intensity_list = []
        for wl in wavelengths:
            n_list = [get_n_complex("Air", wl)] + [get_n_complex(m, wl) for m in materials] + [get_n_complex("Glass", wl)]
            th_list = [np.arcsin(n_list[0] / n * np.sin(th_0)) for n in n_list]
            kz_list = 2 * np.pi * np.array(n_list) * np.cos(th_list) / wl

            M_list = []
            for i in range(len(n_list)-1):
                Y_i = n_list[i] * np.cos(th_list[i])
                Y_j = n_list[i+1] * np.cos(th_list[i+1])
                Y_i = Y_i if abs(Y_i) > 1e-9 else 1e-9
                m11 = 0.5 * (1 + Y_j/Y_i)
                m12 = 0.5 * (1 - Y_j/Y_i)
                m21 = 0.5 * (1 - Y_j/Y_i)
                m22 = 0.5 * (1 + Y_j/Y_i)
                M_list.append(np.array([[m11, m12], [m21, m22]]))

            P_list = [np.array([
                [np.exp(1j * kz_list[i] * d_list[i]), 0],
                [0, np.exp(-1j * kz_list[i] * d_list[i])]
            ]) for i in range(1, len(n_list)-1)]

            T_total = M_list[0]
            for i in range(len(P_list)):
                T_total = T_total @ P_list[i] @ M_list[i+1]

            coeffs = [np.array([[1/T_total[0,0]], [0]])]
            for i in range(len(P_list)-1, -1, -1):
                coeffs.insert(0, P_list[i] @ M_list[i+1] @ coeffs[0])
            coeffs.insert(0, M_list[0] @ coeffs[0])

            # Field 계산
            z_pts, I_pts = [], []
            current_z = 0
            for i in range(1, len(n_list)-1):
                A, B = coeffs[i][0,0], coeffs[i][1,0]
                kz = kz_list[i]
                z_in = np.linspace(0, d_list[i], 30)
                for z in z_in:
                    val = np.abs(A * np.exp(1j * kz * z) + B * np.exp(-1j * kz * z))**2
                    z_pts.append(current_z + z)
                    I_pts.append(val)
                current_z += d_list[i]

            I_at_emitter = np.interp(emitter_depth, z_pts, I_pts)
            intensity_list.append(I_at_emitter)

        intensity = np.array(intensity_list)
        if intensity.max() > 0:
            intensity /= intensity.max()

        return {
            "wavelength_nm": wavelengths.tolist(),
            "intensity": intensity.tolist()
        }
    except Exception as e:
        return {"error": str(e)}

# ===== RT SPECTRUM =====
@app.post("/rt")
async def rt(data: SimRequest):
    try:
        wavelengths = np.linspace(400, 800, 100)
        th_0 = np.deg2rad(data.view_angle)
        stack = data.stack
        d_list = [np.inf] + [l.thickness for l in stack] + [np.inf]
        materials = [l.material for l in stack]

        R_list, T_list = [], []

        for wl in wavelengths:
            n_list = [get_n_complex("Air", wl)] + [get_n_complex(m, wl) for m in materials] + [get_n_complex("Glass", wl)]
            r_s = coh_tmm('s', n_list, d_list, th_0, wl)
            r_p = coh_tmm('p', n_list, d_list, th_0, wl)
            R_list.append((r_s['R'] + r_p['R']) / 2)
            T_list.append((r_s['T'] + r_p['T']) / 2)

        return {
            "wavelength_nm": wavelengths.tolist(),
            "R": R_list,
            "T": T_list
        }
    except Exception as e:
        return {"error": str(e)}

# ===== 실행 시작 =====
if __name__ == "__main__":
    load_materials()
    print("📦 material DB 로딩 완료")
    uvicorn.run(app, host="0.0.0.0", port=8000)
