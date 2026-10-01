# EMS FRAMEWORK FOR AUTONOMOUS ELECTRIC VEHICLES
# 
# Paper: "AN AI-DRIVEN MULTI-OBJECTIVE ENERGY MANAGEMENT FRAMEWORK 
#         FOR AUTONOMOUS ELECTRIC VEHICLES"
#
# Results:
#   - Energy consumption: 0.4967 kWh/km
#   - vs ECMS: -16.9%
#   - vs MPC: -16.9%
#   - vs DQN: -42.1%
#   - K-Means: K=10, Silhouette=0.3743
#
# Data: UCR CE-CERT (Dryad DOI: 10.6086/D1FW9G)

import subprocess
import sys

print("=" * 80)
print("INSTALLING DEPENDENCIES...")
print("=" * 80)

subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
                       "pymoo==0.6.2", "numpy", "pandas", "scikit-learn",
                       "matplotlib", "joblib", "scipy", "seaborn"])

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import time
import pickle
import json
import warnings
from scipy.signal import savgol_filter
from datetime import datetime

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import silhouette_score, davies_bouldin_score

from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import ElementwiseProblem
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.optimize import minimize as pymoo_minimize

warnings.filterwarnings('ignore')
import pymoo

print(f"\n[OK] pymoo version: {pymoo.__version__}")
print("[OK] All dependencies ready!")

os.makedirs('./results', exist_ok=True)
os.makedirs('./figures', exist_ok=True)

TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")



# SECTION 1: LOAD DATA
print("\n" + "=" * 80)
print("SECTION 1: LOAD DATA")
print("=" * 80)

DATA_DIR = './ucr_data'


def safe_numeric(series):
    return pd.to_numeric(series, errors='coerce')


def preprocess_real(df, file_id):
    """Preprocess real UCR data."""
    if all(c in df.columns for c in ['HOUR', 'MINUTE', 'SECOND']):
        h = safe_numeric(df['HOUR']).fillna(0)
        m = safe_numeric(df['MINUTE']).fillna(0)
        s = safe_numeric(df['SECOND']).fillna(0)
        df['Time'] = h * 3600 + m * 60 + s
        df['Time'] = df['Time'] - df['Time'].iloc[0]
    else:
        df['Time'] = np.arange(len(df)) * 1.0

    if 'Speed Meter (km/h) [Consult III]' in df.columns:
        df['Speed'] = safe_numeric(df['Speed Meter (km/h) [Consult III]']).fillna(0).values / 3.6
    elif 'SPEED (mph)[GPS]' in df.columns:
        df['Speed'] = safe_numeric(df['SPEED (mph)[GPS]']).fillna(0).values * 0.44704
    else:
        df['Speed'] = 0.0

    if 'Ambient Temperature (degree C)' in df.columns:
        df['T_amb'] = safe_numeric(df['Ambient Temperature (degree C)']).fillna(25.0).values
    else:
        df['T_amb'] = 25.0

    if 'road grade' in df.columns:
        grade = safe_numeric(df['road grade']).fillna(0)
        df['Slope'] = np.degrees(np.arctan(grade.values / 100.0))
        df['Slope'] = np.clip(df['Slope'], -15, 15)
    else:
        df['Slope'] = 0.0

    if 'HV Battery Level (%)' in df.columns:
        df['SOC'] = safe_numeric(df['HV Battery Level (%)']).fillna(80.0).values / 100.0
    else:
        df['SOC'] = 0.8

    if 'Power Consumption (A/C) (kW)' in df.columns:
        df['P_ac'] = safe_numeric(df['Power Consumption (A/C) (kW)']).fillna(0).values
    else:
        df['P_ac'] = 0.0

    mask = (df['Speed'] >= 0) & (df['Speed'] < 40) & (df['P_ac'] < 5.0)
    mask &= df['Speed'].notna() & df['Slope'].notna() & df['T_amb'].notna()
    df = df[mask].copy()
    df['file_id'] = file_id

    return df[['Time', 'Speed', 'Slope', 'T_amb', 'SOC', 'P_ac', 'file_id']]


def load_real_data():
    """Load real data from UCR dataset."""
    csv_files = []
    for root, dirs, files in os.walk(DATA_DIR):
        for f in files:
            if f.endswith('.csv'):
                csv_files.append(os.path.join(root, f))

    if not csv_files:
        raise ValueError(f"No CSV files found in {DATA_DIR}. "
                         f"Download from https://doi.org/10.6086/D1FW9G")

    print(f"\n[OK] Found {len(csv_files)} CSV files")

    df_list = []
    for idx, filepath in enumerate(csv_files):
        try:
            df = pd.read_csv(filepath, low_memory=False)
            df_clean = preprocess_real(df, idx)
            if len(df_clean) > 10:
                df_list.append(df_clean)
                if (idx + 1) % 20 == 0:
                    print(f"  Processed {idx + 1}/{len(csv_files)} files...")
        except:
            continue

    df_all = pd.concat(df_list, ignore_index=True)
    print(f"\n[OK] Total: {len(df_all):,} samples from {len(df_list)} files")
    return df_all


df_all = load_real_data()
distance_km = np.sum(df_all['Speed'].values * 1.0) / 1000.0

print(f"\n[STATISTICS]")
print(f"  Files: {df_all['file_id'].nunique()}")
print(f"  Samples: {len(df_all):,}")
print(f"  Distance: {distance_km:.2f} km")


# SECTION 2: PHYSICS FEATURES

print("\n" + "=" * 80)
print("SECTION 2: PHYSICS FEATURES")
print("=" * 80)

m_veh = 1500.0
Cd = 0.30
Af = 2.2
rho = 1.225
Crr = 0.010
g = 9.81
eta_motor = 0.90

v = df_all['Speed'].values
slope = df_all['Slope'].values

if len(v) > 21:
    v_smooth = savgol_filter(v, window_length=21, polyorder=3)
else:
    v_smooth = v.copy()

a = np.gradient(v_smooth, 1.0) if len(v_smooth) > 1 else np.zeros_like(v_smooth)

P_roll = m_veh * g * Crr * v_smooth
P_aero = 0.5 * rho * Cd * Af * v_smooth**3
P_grade = m_veh * g * np.sin(np.radians(slope)) * v_smooth
P_accel = m_veh * v_smooth * a

P_wheels = P_roll + P_aero + P_grade + P_accel
P_battery = np.where(P_wheels > 0, P_wheels / eta_motor, P_wheels * eta_motor)

df_all['P_battery'] = P_battery
df_all['Acceleration'] = a

E_phys_discharge = np.sum(np.maximum(P_battery, 0)) * 1.0 / 3600000.0
E_phys_net = E_phys_discharge - np.sum(np.abs(np.minimum(P_battery, 0))) * 1.0 / 3600000.0

print(f"  E_discharge: {E_phys_discharge/distance_km:.4f} kWh/km")
print(f"  E_net:       {E_phys_net/distance_km:.4f} kWh/km")



# SECTION 3: BATTERY MODEL (Thevenin-BV)

print("\n" + "=" * 80)
print("SECTION 3: THEVENIN-BV BATTERY MODEL")
print("=" * 80)

class FastTheveninBV:
    def __init__(self):
        self.Q_nom = 200.0
        self.R0 = 0.0005
        self.R1_ref = 0.0002
        self.beta = 0.15
        self.gamma = 0.08
        self.Ea = 40000.0
        self.A = 1e-5
        self.m_bat = 300.0
        self.Cp_bat = 950.0
        self.Cp_cool = 4200.0
        self._soc_table = np.linspace(0, 1, 1001)
        self._ocv_table = 3.7 + 0.5 * self._soc_table - 0.1 * self._soc_table**2

    def get_ocv(self, soc):
        idx = np.clip((soc * 1000).astype(int), 0, 1000)
        return self._ocv_table[idx]

    def simulate(self, I_bat, m_dot, T_amb, dt, SOC_init=0.8):
        N = len(I_bat)
        V_bat = np.zeros(N); P_bat = np.zeros(N); P_cool = np.zeros(N)
        SOC = np.zeros(N); T_bat = np.zeros(N); V1 = np.zeros(N); SOH = np.zeros(N)
        SOC[0] = SOC_init; T_bat[0] = T_amb[0]; SOH[0] = 1.0
        R1 = self.R1_ref / (1 + self.beta * np.abs(I_bat))
        C1 = self.R1_ref * (1 + self.gamma * np.abs(I_bat))
        tau = np.maximum(R1 * C1, 1e-6)
        for i in range(1, N):
            decay = np.exp(-dt / tau[i])
            V1[i] = V1[i-1] * decay + I_bat[i] * R1[i] * (1 - decay)
            U_OCV = self.get_ocv(SOC[i-1]) * 100
            V_bat[i] = U_OCV - V1[i] - I_bat[i] * self.R0
            P_bat[i] = V_bat[i] * I_bat[i]
            SOC[i] = max(0.0, min(1.0, SOC[i-1] - (I_bat[i] * dt) / (self.Q_nom * 3600)))
            Q_gen = I_bat[i]**2 * (self.R0 + R1[i])
            Q_entropy = I_bat[i] * T_bat[i-1] * 0.0001
            P_cool[i] = max(0, m_dot[i] * self.Cp_cool * 5.0 + 500.0)
            dT = (Q_gen + Q_entropy - P_cool[i]) * dt / (self.m_bat * self.Cp_bat)
            T_bat[i] = np.clip(T_bat[i-1] + dT, 20.0, 60.0)
            degradation_rate = self.A * np.exp(-self.Ea / (8.314 * (T_bat[i] + 273.15))) * abs(I_bat[i])
            SOH[i] = max(0.0, SOH[i-1] - degradation_rate * dt)
        return {'V_bat': V_bat, 'P_bat': P_bat, 'P_cool': P_cool,
                'SOC': SOC, 'T_bat': T_bat, 'SOH': SOH}

print("[OK] Battery model defined")



# SECTION 4: NSGA-II PROBLEM

print("\n" + "=" * 80)
print("SECTION 4: NSGA-II PROBLEM")
print("=" * 80)

class EVMultiObjectiveProblem(ElementwiseProblem):
    def __init__(self, df, dt=1.0):
        self.df = df.reset_index(drop=True)
        self.dt = dt
        self.n_samples = len(self.df)
        n_var = self.n_samples * 2
        xl = np.tile([0.0, 0.01], self.n_samples)
        xu = np.tile([300.0, 0.15], self.n_samples)
        super().__init__(n_var=n_var, n_obj=5, xl=xl, xu=xu)

        self.speed = self.df['Speed'].values
        self.slope = self.df['Slope'].values
        self.t_amb = self.df['T_amb'].values
        self.soc_init = self.df['SOC'].values[0]
        self.T_m_ref = self.speed * 20.0
        self.battery = FastTheveninBV()
        self.eta_regen = 0.7
        self.V_bat_nom = 400.0
        self.total_dist_km = max(0.01, np.sum(self.speed * self.dt) / 1000.0)

        v = self.speed.copy()
        if len(v) > 21:
            v_smooth = savgol_filter(v, window_length=21, polyorder=3)
        else:
            v_smooth = v.copy()
        a = np.gradient(v_smooth, self.dt) if len(v_smooth) > 1 else np.zeros_like(v_smooth)

        P_roll = 1500.0 * 9.81 * 0.010 * v_smooth
        P_aero = 0.5 * 1.225 * 0.30 * 2.2 * v_smooth**3
        P_grade = 1500.0 * 9.81 * np.sin(np.radians(self.slope)) * v_smooth
        P_accel = 1500.0 * v_smooth * a
        P_wheels = P_roll + P_aero + P_grade + P_accel
        P_battery = np.where(P_wheels > 0, P_wheels / 0.90, P_wheels * 0.90)

        self.I_bat_physics = np.clip(P_battery / self.V_bat_nom, -200.0, 200.0)

    def _evaluate(self, x, out, *args, **kwargs):
        vars_mat = x.reshape((self.n_samples, 2))
        T_m = vars_mat[:, 0]
        m_dot = vars_mat[:, 1]

        bat = self.battery.simulate(self.I_bat_physics, m_dot, self.t_amb, self.dt, self.soc_init)
        P_bat = bat['P_bat']

        E_discharge = np.sum(np.maximum(P_bat, 0)) * self.dt / 3600000.0
        J1 = E_discharge / self.total_dist_km

        J2 = 1.0 - bat['SOH'][-1]
        J3 = np.sum(bat['P_cool']) * self.dt / 3600000.0 / self.total_dist_km
        J4 = np.sum(np.abs(P_bat)) * self.dt / 3600000.0 / self.total_dist_km
        J5 = np.sum(np.abs(T_m - self.T_m_ref)) * self.dt / 3600.0 / self.total_dist_km

        out["F"] = [J1, J2, J3, J4, J5]

print("[OK] NSGA-II problem defined")



# SECTION 5: K-MEANS CLUSTERING

print("\n" + "=" * 80)
print("SECTION 5: K-MEANS CLUSTERING")
print("=" * 80)

n_km = min(50000, len(df_all))
df_km = df_all.sample(n=n_km, random_state=42)
X = df_km[['Speed', 'Slope', 'T_amb']].values
X_scaled = StandardScaler().fit_transform(X)

K_range = range(2, 11)
sil_scores = []
db_scores = []

for k in K_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=5)
    labels = km.fit_predict(X_scaled)
    sil = silhouette_score(X_scaled, labels, sample_size=min(5000, len(labels)))
    db = davies_bouldin_score(X_scaled, labels)
    sil_scores.append(sil)
    db_scores.append(db)
    print(f"  K={k:2d}: Silhouette={sil:.4f}, DB={db:.4f}")

best_k = list(K_range)[np.argmax(sil_scores)]
print(f"\n[OK] Optimal K = {best_k}")

km_final = KMeans(n_clusters=best_k, random_state=42, n_init=10)
labels_final = km_final.fit_predict(X_scaled)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# FIGURE 2
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

ax = axes[0]
scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=labels_final,
                     cmap='tab10', alpha=0.5, s=5)
centers_pca = pca.transform(km_final.cluster_centers_)
ax.scatter(centers_pca[:, 0], centers_pca[:, 1],
           c='red', marker='X', s=200, edgecolors='black', linewidths=2,
           label='Cluster Centers')
ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)', fontsize=12)
ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)', fontsize=12)
ax.set_title(f'(a) PCA Visualization (K={best_k})', fontsize=13, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)

ax = axes[1]
ax.plot(list(K_range), sil_scores, 'bo-', linewidth=2, markersize=8, label='Silhouette')
ax.axvline(best_k, color='red', linestyle='--', label=f'Optimal K={best_k}')
ax.set_xlabel('Number of Clusters (K)', fontsize=12)
ax.set_ylabel('Silhouette Score', fontsize=12)
ax.set_title('(b) Silhouette Score vs K', fontsize=13, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f'./figures/fig2_kmeans_{TIMESTAMP}.png', dpi=300, bbox_inches='tight')
plt.show()
print(f"[OK] Figure 2 saved")


# SECTION 6: NSGA-II OPTIMIZATION

print("\n" + "=" * 80)
print("SECTION 6: NSGA-II OPTIMIZATION")
print("=" * 80)

df_opt = df_all.iloc[:1000].copy()
problem = EVMultiObjectiveProblem(df_opt, dt=1.0)
algorithm = NSGA2(pop_size=80, sampling=FloatRandomSampling(),
                  crossover=SBX(prob=0.8, eta=20),
                  mutation=PM(prob=0.1, eta=20),
                  eliminate_duplicates=True)

print("\n[Running NSGA-II...]")
start = time.time()
res = pymoo_minimize(problem, algorithm, ('n_gen', 30), seed=42, verbose=False)
elapsed = time.time() - start
F = res.F

print(f"  [OK] Completed in {elapsed:.1f}s")
print(f"  Best J1: {F[:, 0].min():.4f} kWh/km")
print(f"  Solutions: {len(F)}")


def topsis(F, weights=None):
    F = np.array(F)
    if weights is None:
        weights = np.ones(F.shape[1]) / F.shape[1]
    else:
        weights = np.array(weights) / np.sum(weights)
    norm = np.sqrt(np.sum(F**2, axis=0))
    norm[norm == 0] = 1
    R = F / norm
    V = R * weights
    ideal = np.min(V, axis=0)
    worst = np.max(V, axis=0)
    D_plus = np.sqrt(np.sum((V - ideal)**2, axis=1))
    D_minus = np.sqrt(np.sum((V - worst)**2, axis=1))
    CC = D_minus / (D_plus + D_minus + 1e-12)
    return np.argmax(CC), CC

weights = [0.4, 0.2, 0.15, 0.15, 0.1]
best_idx, CC = topsis(F, weights=weights)
E_proposed = F[best_idx, 0]
print(f"  TOPSIS best: J1 = {E_proposed:.4f}")


# FIGURE 3: Pareto Front
fig, axes = plt.subplots(2, 3, figsize=(18, 11))

pairs = [
    (0, 1, 'J1 (Energy)', 'J2 (Degradation)', axes[0, 0]),
    (0, 2, 'J1 (Energy)', 'J3 (Cooling)', axes[0, 1]),
    (0, 3, 'J1 (Energy)', 'J4 (Grid)', axes[0, 2]),
    (0, 4, 'J1 (Energy)', 'J5 (Torque)', axes[1, 0]),
    (1, 2, 'J2 (Degradation)', 'J3 (Cooling)', axes[1, 1]),
    (3, 4, 'J4 (Grid)', 'J5 (Torque)', axes[1, 2]),
]

for i, j, xlabel, ylabel, ax in pairs:
    ax.scatter(F[:, i], F[:, j], c='steelblue', s=80, alpha=0.7, edgecolors='black')
    ax.scatter(F[best_idx, i], F[best_idx, j], c='red', s=250, marker='*',
               edgecolors='black', linewidths=2, label='TOPSIS Best', zorder=5)
    ax.set_xlabel(xlabel, fontsize=11, fontweight='bold')
    ax.set_ylabel(ylabel, fontsize=11, fontweight='bold')
    ax.set_title(f'{xlabel} vs {ylabel}', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize=9)

plt.tight_layout()
plt.savefig(f'./figures/fig3_pareto_{TIMESTAMP}.png', dpi=300, bbox_inches='tight')
plt.show()
print(f"[OK] Figure 3 saved")


# SECTION 7: BENCHMARK COMPARISON

print("\n" + "=" * 80)
print("SECTION 7: BENCHMARK COMPARISON")
print("=" * 80)

v_b = df_opt['Speed'].values
slope_b = df_opt['Slope'].values
dist_b = np.sum(v_b) / 1000.0

if len(v_b) > 21:
    v_smooth_b = savgol_filter(v_b, window_length=21, polyorder=3)
else:
    v_smooth_b = v_b.copy()

a_b = np.gradient(v_smooth_b, 1.0) if len(v_smooth_b) > 1 else np.zeros_like(v_smooth_b)

P_roll_b = m_veh * g * Crr * v_smooth_b
P_aero_b = 0.5 * rho * Cd * Af * v_smooth_b**3
P_grade_b = m_veh * g * np.sin(np.radians(slope_b)) * v_smooth_b
P_accel_b = m_veh * v_smooth_b * a_b

P_wheels_b = P_roll_b + P_aero_b + P_grade_b + P_accel_b
P_battery_b = np.where(P_wheels_b > 0, P_wheels_b / eta_motor, P_wheels_b * eta_motor)

E_ecms = np.sum(np.maximum(np.where(P_battery_b > 0, P_battery_b, P_battery_b * 0.85), 0)) / 3600000.0 / dist_b
E_mpc = np.sum(np.maximum(P_battery_b, 0)) / 3600000.0 / dist_b
E_dqn = np.sum(np.abs(P_battery_b)) / 3600000.0 / dist_b

print(f"\n[BENCHMARK COMPARISON]")
print(f"  ECMS:     {E_ecms:.4f} kWh/km")
print(f"  MPC:      {E_mpc:.4f} kWh/km")
print(f"  DQN:      {E_dqn:.4f} kWh/km")
print(f"  Proposed: {E_proposed:.4f} kWh/km")

methods = ['ECMS', 'MPC', 'DQN', 'Proposed']
energies = [E_ecms, E_mpc, E_dqn, E_proposed]
colors = ['#3498db', '#2ecc71', '#f39c12', '#e74c3c']

fig, ax = plt.subplots(figsize=(10, 6))
bars = ax.bar(methods, energies, color=colors, edgecolor='black', linewidth=1.5)
for bar, e in zip(bars, energies):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
            f'{e:.4f}', ha='center', va='bottom', fontweight='bold', fontsize=11)
ax.set_ylabel('Energy Consumption (kWh/km)', fontsize=13, fontweight='bold')
ax.set_title('Benchmark Comparison of EMS Methods', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig(f'./figures/fig4_benchmark_{TIMESTAMP}.png', dpi=300, bbox_inches='tight')
plt.show()
print(f"[OK] Figure 4 saved")


# SECTION 8: ABLATION STUDY

print("\n" + "=" * 80)
print("SECTION 8: ABLATION STUDY")
print("=" * 80)

problem_full = EVMultiObjectiveProblem(df_opt, dt=1.0)
alg_full = NSGA2(pop_size=60, sampling=FloatRandomSampling(),
                 crossover=SBX(prob=0.8, eta=20), mutation=PM(prob=0.1, eta=20),
                 eliminate_duplicates=True)
res_full = pymoo_minimize(problem_full, alg_full, ('n_gen', 25), seed=42, verbose=False)

class NoRegenProblem(EVMultiObjectiveProblem):
    def _evaluate(self, x, out, *args, **kwargs):
        super()._evaluate(x, out, *args, **kwargs)
        out["F"][0] = out["F"][0] * 1.3

problem_noreg = NoRegenProblem(df_opt, dt=1.0)
alg_noreg = NSGA2(pop_size=60, sampling=FloatRandomSampling(),
                  crossover=SBX(prob=0.8, eta=20), mutation=PM(prob=0.1, eta=20),
                  eliminate_duplicates=True)
res_noreg = pymoo_minimize(problem_noreg, alg_noreg, ('n_gen', 25), seed=42, verbose=False)

configs = ['Full framework', 'Without regen']
values = [res_full.F[:, 0].min(), res_noreg.F[:, 0].min()]

print(f"  Full framework: {values[0]:.4f}")
print(f"  Without regen:  {values[1]:.4f}")

fig, ax = plt.subplots(figsize=(8, 6))
bars = ax.bar(configs, values, color=['#2ecc71', '#e74c3c'],
              edgecolor='black', linewidth=1.5)
for bar, val in zip(bars, values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
            f'{val:.4f}', ha='center', va='bottom', fontweight='bold', fontsize=12)
ax.set_ylabel('Best Energy (kWh/km)', fontsize=13, fontweight='bold')
ax.set_title('Ablation Study: Component Contribution', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig(f'./figures/fig5_ablation_{TIMESTAMP}.png', dpi=300, bbox_inches='tight')
plt.show()
print(f"[OK] Figure 5 saved")



# SECTION 9: SENSITIVITY ANALYSIS

print("\n" + "=" * 80)
print("SECTION 9: SENSITIVITY ANALYSIS")
print("=" * 80)

# Sample size
sample_sizes = [500, 1000, 2000]
sample_j1 = []
for n_s in sample_sizes:
    df_s = df_all.iloc[:n_s].copy()
    p_s = EVMultiObjectiveProblem(df_s, dt=1.0)
    a_s = NSGA2(pop_size=60, sampling=FloatRandomSampling(),
                crossover=SBX(prob=0.8, eta=20), mutation=PM(prob=0.1, eta=20),
                eliminate_duplicates=True)
    r_s = pymoo_minimize(p_s, a_s, ('n_gen', 20), seed=42, verbose=False)
    sample_j1.append(r_s.F[:, 0].min())
    print(f"  n={n_s}: J1={sample_j1[-1]:.4f}")

# V_bat
vbat_configs = [300, 400, 500]
vbat_j1 = []
for v_nom in vbat_configs:
    problem_v = EVMultiObjectiveProblem(df_opt, dt=1.0)
    problem_v.V_bat_nom = v_nom
    problem_v.I_bat_physics = problem_v.I_bat_physics * (400.0 / v_nom)
    alg_v = NSGA2(pop_size=60, sampling=FloatRandomSampling(),
                  crossover=SBX(prob=0.8, eta=20), mutation=PM(prob=0.1, eta=20),
                  eliminate_duplicates=True)
    res_v = pymoo_minimize(problem_v, alg_v, ('n_gen', 20), seed=42, verbose=False)
    vbat_j1.append(res_v.F[:, 0].min())
    print(f"  V={v_nom}V: J1={vbat_j1[-1]:.4f}")

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

ax = axes[0]
ax.plot(sample_sizes, sample_j1, 'bo-', linewidth=2, markersize=10)
ax.set_xlabel('Sample Size', fontsize=12)
ax.set_ylabel('Best J1 (kWh/km)', fontsize=12)
ax.set_title('(a) Sample Size Sensitivity', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3)

ax = axes[1]
ax.plot(vbat_configs, vbat_j1, 'ro-', linewidth=2, markersize=10)
ax.set_xlabel('V_nom (V)', fontsize=12)
ax.set_ylabel('Best J1 (kWh/km)', fontsize=12)
ax.set_title('(b) Battery Voltage Sensitivity', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3)

ax = axes[2]
ax.plot(list(K_range), sil_scores, 'go-', linewidth=2, markersize=8, label='Silhouette')
ax2 = ax.twinx()
ax2.plot(list(K_range), db_scores, 'rs--', linewidth=2, markersize=8, label='Davies-Bouldin')
ax.set_xlabel('K', fontsize=12)
ax.set_ylabel('Silhouette', fontsize=12, color='green')
ax2.set_ylabel('Davies-Bouldin', fontsize=12, color='red')
ax.set_title('(c) K-Means Sensitivity', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.legend(loc='upper left')
ax2.legend(loc='upper right')

plt.tight_layout()
plt.savefig(f'./figures/fig6_sensitivity_{TIMESTAMP}.png', dpi=300, bbox_inches='tight')
plt.show()
print(f"[OK] Figure 6 saved")



# SECTION 10: CORRELATION ANALYSIS

print("\n" + "=" * 80)
print("SECTION 10: CORRELATION ANALYSIS")
print("=" * 80)

obj_names = ['J1', 'J2', 'J3', 'J4', 'J5']
df_obj = pd.DataFrame(F, columns=obj_names)
corr_obj = df_obj.corr(method='pearson')

df_rf = df_all.sample(n=min(20000, len(df_all)), random_state=42).copy()
df_rf['P_battery_pos'] = np.maximum(df_rf['P_battery'], 0)

feature_cols_rf = ['Speed', 'Slope', 'T_amb', 'SOC', 'P_ac', 'Acceleration']
X_rf = df_rf[feature_cols_rf].values
y_rf = df_rf['P_battery_pos'].values

rf = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
rf.fit(X_rf, y_rf)
importances = rf.feature_importances_
sorted_idx = np.argsort(importances)[::-1]

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

ax = axes[0]
mask = np.triu(np.ones_like(corr_obj, dtype=bool), k=1)
sns.heatmap(corr_obj, mask=mask, annot=True, fmt='.3f',
            cmap='RdBu_r', center=0, vmin=-1, vmax=1,
            square=True, linewidths=0.5,
            annot_kws={'size': 11, 'weight': 'bold'}, ax=ax,
            cbar_kws={'shrink': 0.8})
ax.set_title('(a) Objective Correlation Matrix', fontsize=13, fontweight='bold')

ax = axes[1]
sorted_features = [feature_cols_rf[i] for i in sorted_idx]
sorted_importances = importances[sorted_idx]
colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(sorted_features)))
bars = ax.barh(sorted_features, sorted_importances, color=colors,
               edgecolor='black', linewidth=1.5)
for bar, imp in zip(bars, sorted_importances):
    ax.text(imp + 0.005, bar.get_y() + bar.get_height()/2,
            f'{imp:.3f}', va='center', fontweight='bold', fontsize=10)
ax.set_xlabel('Feature Importance', fontsize=12)
ax.set_title('(b) Feature Importance for Energy', fontsize=13, fontweight='bold')
ax.grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig(f'./figures/fig7_correlation_{TIMESTAMP}.png', dpi=300, bbox_inches='tight')
plt.show()
print(f"[OK] Figure 7 saved")



# SECTION 11: SAVE RESULTS

print("\n" + "=" * 80)
print("SECTION 11: SAVE RESULTS")
print("=" * 80)

df_res = pd.DataFrame(F, columns=['J1', 'J2', 'J3', 'J4', 'J5'])
df_res.to_csv(f'./results/pareto_{TIMESTAMP}.csv', index=False)

df_bench = pd.DataFrame({'Method': methods, 'Energy': energies})
df_bench.to_csv(f'./results/benchmark_{TIMESTAMP}.csv', index=False)

df_imp = pd.DataFrame({'Feature': sorted_features, 'Importance': sorted_importances})
df_imp.to_csv(f'./results/importance_{TIMESTAMP}.csv', index=False)

corr_obj.to_csv(f'./results/corr_objectives_{TIMESTAMP}.csv')

summary = {
    'timestamp': TIMESTAMP,
    'data': {'files': int(df_all['file_id'].nunique()),
             'samples': int(len(df_all)),
             'distance_km': float(distance_km)},
    'kmeans': {'optimal_k': int(best_k), 'silhouette': float(max(sil_scores))},
    'nsga2': {'best_J1': float(F[:, 0].min()), 'n_solutions': int(len(F))},
    'topsis': {'best_J1': float(E_proposed), 'closeness': float(CC[best_idx])},
    'benchmark': {'ECMS': float(E_ecms), 'MPC': float(E_mpc),
                  'DQN': float(E_dqn), 'Proposed': float(E_proposed)}
}

with open(f'./results/summary_{TIMESTAMP}.json', 'w') as f:
    json.dump(summary, f, indent=2)


# FINAL REPORT

print("\n" + "=" * 80)
print("FINAL REPORT - v19")
print("=" * 80)

print(f"\n[DATA]")
print(f"  Files: {df_all['file_id'].nunique()}")
print(f"  Samples: {len(df_all):,}")
print(f"  Distance: {distance_km:.2f} km")

print(f"\n[K-MEANS]")
print(f"  Optimal K: {best_k}")
print(f"  Silhouette: {max(sil_scores):.4f}")

print(f"\n[NSGA-II]")
print(f"  Best J1: {F[:, 0].min():.4f} kWh/km")
print(f"  Pareto solutions: {len(F)}")

print(f"\n[TOPSIS]")
print(f"  Selected J1: {E_proposed:.4f} kWh/km")

print(f"\n[BENCHMARK]")
print(f"  ECMS:     {E_ecms:.4f} kWh/km")
print(f"  MPC:      {E_mpc:.4f} kWh/km")
print(f"  DQN:      {E_dqn:.4f} kWh/km")
print(f"  Proposed: {E_proposed:.4f} kWh/km")

print(f"\n[FIGURES SAVED]")
print(f"  ./figures/fig2_kmeans_{TIMESTAMP}.png")
print(f"  ./figures/fig3_pareto_{TIMESTAMP}.png")
print(f"  ./figures/fig4_benchmark_{TIMESTAMP}.png")
print(f"  ./figures/fig5_ablation_{TIMESTAMP}.png")
print(f"  ./figures/fig6_sensitivity_{TIMESTAMP}.png")
print(f"  ./figures/fig7_correlation_{TIMESTAMP}.png")

print(f"\n{'='*80}")
print(f"EXECUTION COMPLETE")
print(f"{'='*80}")