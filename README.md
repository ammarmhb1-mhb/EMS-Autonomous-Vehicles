# EMS Framework for Autonomous Electric Vehicles

Source code for the paper: **"An AI-Driven Multi-Objective Energy Management Framework for Autonomous Electric Vehicles"** (IJICT, 2026).

## Results

| Metric | Value |
|--------|-------|
| Energy Consumption | **0.4967 kWh/km** |
| vs ECMS | -16.9% |
| vs MPC | -16.9% |
| vs DQN | -42.1% |
| K-Means Optimal K | 10 (Silhouette: 0.3743) |
| Pareto Solutions | 80 |

## Dataset

- **Source:** Dryad Digital Repository
- **DOI:** https://doi.org/10.6086/D1FW9G
- **Size:** 80 CSV files, 287,962 samples, 4,030 km

Download and extract to ./ucr_data/

## Requirements

pip install -r requirements.txt

## Usage

python EMS_UCR_Analysis.py

Output: ./figures/ (7 figures) and ./results/ (CSV + JSON)

## Methodology

1. **K-Means Clustering** — Driving pattern recognition (K=10)
2. **Thevenin-BV Model** — Battery state estimation (SOC error: ±2.8%)
3. **NSGA-II** — Multi-objective optimization (5 objectives: energy, battery, thermal, grid, torque)
4. **TOPSIS** — Decision-making (weights: [0.4, 0.2, 0.15, 0.15, 0.1])
5. **Control Signals** — I_bat, T_m, m_cool, theta_gear

## Benchmark Models

| Method | Type | Key Parameter |
|--------|------|---------------|
| ECMS | Rule-based | s_factor = 0.85 |
| MPC | Optimization | horizon = 10 |
| DQN | Learning | discharge-only |
| Proposed | Hybrid AI | 5 objectives |

## Citation

@article{bedear2026ems,
  title={An AI-Driven Multi-Objective Energy Management Framework for Autonomous Electric Vehicles},
  author={Bedear, Ammar Majeed Hameed},
  journal={Iraqi Journal of Information and Communication Technology},
  year={2026}
}

## Contact

**Ammar Majeed Hameed Bedear** — ammarmhb1@gmail.com
Karbala Education Directorate, Karbala, Iraq

## License

MIT License
