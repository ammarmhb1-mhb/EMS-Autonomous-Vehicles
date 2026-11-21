EMS Framework for Autonomous Electric Vehicles
Source code for the paper: "An AI-Driven Multi-Objective Energy Management Framework for Autonomous Electric Vehicles" (IJICT, 2026).

Final Verified Results (V3.6 - 2026-11-21)
Dataset: UCR Driving Dataset - 80 unique files (287,962 samples) after deduplication from 320 files including (1) copies.

K-Means Clustering
Best K=10: Silhouette 0.3411, Inertia 16730.25
Second K=6: Silhouette 0.3381
Energy Management (5 Objectives)
Scenario	J1 (kWh/km)	HV
Baseline (Idealized)	0.2827	-
+ HVAC Load	0.2933	-
+ Sensor Noise	0.1848	-
+ Traffic Variability	0.2624	-
Full Real-World (+ Road Grade)	0.2802 avg, 0.2617 best	60.51 -> 62.74 (+3.7%)
Without CO-PSO	0.4293	60.51
Full (NSGA-II+CO-PSO)	0.2617 (39% improvement)	62.74
Hybrid Model Weights
XGBoost: 0.3131
LightGBM: 0.3122
MLP: 0.2866 (replaces LSTM)
DecisionTree: 0.0879
Battery Model (400V System)
U_ocv = 400V, Q = 200Ah, 80kWh
R0 = 0.15 ?
R1 = R1_ref / (1 + ?*|I|) = 0.10 / (1 + 0.15*|I|)
C1 = C1_ref * (1 + ?*|I|) = 2000 * (1 + 0.08*|I|)
C_th = 1500*3600 J/K
Thermal: dT = (Q_gen - P_cool)*dt / C_th
SOC Constraint: G = sum((SOC<0.2)|(SOC>0.9)) - 0.5
Objectives Normalization
J1 = Energy [kWh/km] / 1.0
J2 = Aging * 1e9 (J2 × 10?)
J3 = Thermal / 0.2
J4 = Grid / 1.0
J5 = Comfort / 100 (J5 / 10²)
Runtime
0.83s total: 0.15s NSGA-II + 0.25s CO-PSO + 0.25s Hybrid + 0.18s TOPSIS
Files
EMS_FINAL_V36_GitHub_WithUpload.py - Final documented code (V3.6) with files.upload() prompt for GitHub/Colab
EMS_FINAL_V36_Complete_Documented.py - Same without upload (local execution)
EMS_UCR_Analysis.py - Legacy version
data/ - UCR Driving Dataset (80 files)
requirements.txt - Dependencies
Usage
Colab (with upload prompt - for documentation)
from google.colab import files
# Code will automatically prompt if no CSVs found
# Upload 80 UCR CSV files when prompted
python EMS_FINAL_V36_GitHub_WithUpload.py
Local / GitHub Clone
git clone https://github.com/ammarmhb1-mhb/EMS-Autonomous-Vehicles.git
cd EMS-Autonomous-Vehicles
# Place 80 UCR CSVs in current dir or data/ folder
pip install -r requirements.txt
python EMS_FINAL_V36_Complete_Documented.py
Key Fixes from Previous Versions
Deduplication: re.sub(r' \(\d+\)', '', base) removes (1) copies - 320 -> 80 files
Realistic Battery: R0=0.15? not 0.0005?, 400V not 48V
Differential Thermal: dT=(Q_gen-P_cool)*dt/C_th
SOC Constraint: Implemented via G
Dynamic HV Reference: ref = max(F)*1.2 + 0.1
No files.upload() in local version, with upload in GitHub version
MLP instead of LSTM
Citation
If you use this code, please cite the IJICT 2026 paper.

License
MIT

