import os
import numpy as np
import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
from tmm import coh_tmm
import uvicorn

# ===== FastAPI 기본 설정 =====
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 개발용으로 전체 허용
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===== 데이터 모델 =====
class Layer(BaseModel):
    name: str
    material: str
    thickness: float

class EmissionRequest(BaseModel):
    stack: List[Layer]
    emitter_position: float
    view_angle: float

# ===== CSV로부터 재료 데이터 로드 =====
MATERIALS_DIR = "materials"
materials_db = {}

def load_materials():
    global materials_db
    materials_db = {}
    if not os.path.exists(MATERIALS_DIR):
        raise FileNotFoundError(f"'{MATERIALS_DIR}' 디렉토리를 찾을 수 없습니다.")

    for filename in os.listdir(MATERIALS_DIR):
        if filename.endswith(".csv"):
            material_name = filename[:-4]
            path = os.path.join(MATERIALS_DIR, filename)
            try:
                df = pd.read_csv(path)
                if all(col in df.columns for col in ['wavelength_nm', 'n', 'k']):
                    df = df.sort_values('wavelength_nm')
                    materials_db[material_name] = df.reset_index(drop=True)
                    print(f"✅ {material_name} 로드 완료")
                else:
                    print(f"⚠️ {filename} 형식 오류: 'wavelength_nm', 'n', 'k' 필드가 필요합니다.")
            except Exception as e:
                print(f"❌ {filename} 읽기 실패: {e}")

def get_n_complex(material: str, wl: float):
    if material not in materials_db:
        raise ValueError(f"'{material}' 데이터를 찾을 수 없습니다.")
    df = materials_db[material]
    n = np.interp(wl, df['wavelength_nm'], df['n'])
    k = np.interp(wl, df['wavelength_nm'], df['k'])
    return n + 1j * k

# ===== 메인 API =====
@app.post("/emission")
async def simulate_emission(data: EmissionRequest):
    try:
        wavelengths = np.linspace(400, 800, 100)
        th_0 = np.deg2rad(data.view_angle)
        emitter_pos = data.emitter_position
        materials_list = [layer.material for layer in data.stack]
        thicknesses = [layer.thickness for layer in data.stack]

        # EML 레이어는 마지막에서 두 번째라고 가정
        eml_index = len(materials_list) // 2
        eml_thickness = thicknesses[eml_index]
        emitter_depth = sum(thicknesses[:eml_index]) + emitter_pos * eml_thickness

        incident_medium = "Air"
        exit_medium = "Glass"
        d_list = [np.inf] + thicknesses + [np.inf]

        intensities = []

        for wl in wavelengths:
            n_list = [get_n_complex(incident_medium, wl)] + \
                     [get_n_complex(m, wl) for m in materials_list] + \
                     [get_n_complex(exit_medium, wl)]
            
            # kz (z 방향 파동수)
            th_list = [np.arcsin(n_list[0] / n * np.sin(th_0)) for n in n_list]
            kz_list = 2 * np.pi * np.array(n_list) * np.cos(th_list) / wl

            # field 계산
            M_list = []
            for i in range(len(n_list) - 1):
                # s-편광 기준
                Y_i = n_list[i] * np.cos(th_list[i])
                Y_j = n_list[i+1] * np.cos(th_list[i+1])
                if abs(Y_i) < 1e-9: Y_i = 1e-9
                m11 = 0.5 * (1 + Y_j/Y_i)
                m12 = 0.5 * (1 - Y_j/Y_i)
                m21 = 0.5 * (1 - Y_j/Y_i)
                m22 = 0.5 * (1 + Y_j/Y_i)
                M_list.append(np.array([[m11, m12], [m21, m22]]))

            P_list = [np.array([
                [np.exp(1j*kz_list[i]*d_list[i]), 0],
                [0, np.exp(-1j*kz_list[i]*d_list[i])]
            ]) for i in range(1, len(n_list)-1)]

            T_total = M_list[0]
            for i in range(len(P_list)):
                T_total = T_total @ P_list[i] @ M_list[i+1]

            coeffs = [np.array([[1/T_total[0,0]], [0]])]
            for i in range(len(P_list)-1, -1, -1):
                coeffs.insert(0, P_list[i] @ M_list[i+1] @ coeffs[0])
            coeffs.insert(0, M_list[0] @ coeffs[0])

            # 특정 위치에서 field intensity 계산
            z_pts = []
            I_pts = []
            current_z = 0
            for i in range(1, len(n_list)-1):
                d = d_list[i]
                A, B = coeffs[i][0,0], coeffs[i][1,0]
                kz = kz_list[i]
                z_in_layer = np.linspace(0, d, 30)
                for z in z_in_layer:
                    intensity = np.abs(A * np.exp(1j * kz * z) + B * np.exp(-1j * kz * z))**2
                    I_pts.append(intensity)
                    z_pts.append(current_z + z)
                current_z += d

            I_at_emitter = np.interp(emitter_depth, z_pts, I_pts)
            intensities.append(I_at_emitter)

        intensities = np.array(intensities)
        if intensities.max() > 0:
            intensities /= intensities.max()

        return {
            "wavelength_nm": wavelengths.tolist(),
            "intensity": intensities.tolist()
        }
    except Exception as e:
        return {"error": str(e)}

# ===== 시작점 =====
if __name__ == "__main__":
    load_materials()
    print("📦 material DB 로딩 완료")
    uvicorn.run(app, host="0.0.0.0", port=8000)
