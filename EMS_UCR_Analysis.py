"""
EMS_UCR_Analysis CODE 
Author: Ammar Majeed
Purpose: Documentation - requests user to upload UCR dataset
Dataset: UCR 80 files, 287,962 samples after deduplication from 320 files

"""

# Imports already done above
# import os, glob, io, re
# import numpy as np, pandas as pd

# ===== DATA UPLOAD SECTION FOR GITHUB DOCUMENTATION =====
print("="*75)
print("Checking for UCR dataset...")
print("="*75)

# Try to import google.colab files for upload (for GitHub documentation)
try:
    from google.colab import files
    COLAB_AVAILABLE = True
except ImportError:
    COLAB_AVAILABLE = False
    print("[INFO] Not in Colab environment - will search local files")

def request_upload_if_needed():
    """For GitHub documentation: request upload if no CSVs found"""
    all_csv_check = glob.glob("*.csv") + glob.glob("/content/*.csv")
    all_csv_check = [f for f in all_csv_check if not f.endswith('.xlsx') and 'UCR_' not in os.path.basename(f)]
    
    if len(all_csv_check) == 0:
        print("\n[DATASET NOT FOUND] No CSV files detected.")
        if COLAB_AVAILABLE:
            print("[ACTION] Please upload your UCR CSV files (80 files)...")
            print("In GitHub documentation, this will prompt file upload dialog")
            uploaded = files.upload()
            print(f"Uploaded {len(uploaded)} files")
        else:
            print("[ERROR] Please place UCR dataset CSV files in current directory")
            print("Expected: 80 unique files (320 with (1) duplicates -> 80 after dedup)")
            print("For GitHub: Clone repo and place data/ folder with CSVs")
            # For local GitHub clone, also check data/ folder
            data_csv = glob.glob("data/*.csv") + glob.glob("data/**/*.csv")
            if len(data_csv) > 0:
                print(f"Found {len(data_csv)} files in data/ folder")
    else:
        print(f"[FOUND] {len(all_csv_check)} CSV files already present - skipping upload")

# Call upload check - THIS IS REQUIRED FOR GITHUB DOCUMENTATION
request_upload_if_needed()

# ===== REST OF CODE (Same as V3.6 FINAL) =====

# V3.6 - FIXED DUPLICATES + REALISTIC J1 + HV REF
import os, glob, io, re
import numpy as np, pandas as pd
from scipy.interpolate import interp1d
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, mean_squared_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
import xgboost as xgb, lightgbm as lgb
from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.optimize import minimize as pymoo_minimize
from pymoo.indicators.hv import HV
import warnings, time
warnings.filterwarnings('ignore')
np.random.seed(42)

print("="*75)
print("V3.6 - FIXED DUPLICATES + REALISTIC J1")
print("="*75)

# Search but deduplicate by base name without (1)
all_csv = glob.glob("*.csv") + glob.glob("/content/*.csv")
all_csv = [f for f in all_csv if not f.endswith('.xlsx') and 'UCR_' not in os.path.basename(f)]

# Deduplicate: remove files containing " (1)" or keep first occurrence of base name
seen_base = {}
filtered=[]
for f in all_csv:
    base = os.path.basename(f)
    # Remove " (1)" pattern
    clean_base = re.sub(r' \(\d+\)', '', base)  # remove " (1)"
    clean_base = clean_base.replace('.csv.csv','.csv')
    if clean_base not in seen_base:
        seen_base[clean_base]=f
        filtered.append(f)

csv_files = filtered
# Prefer only original 80
if len(csv_files)>100:
    # Keep only files without " (1)" in name
    csv_files = [f for f in csv_files if " (1)" not in f and "(1)" not in f]

print(f"After deduplication: Found {len(csv_files)} unique CSV files (was {len(all_csv)} with duplicates)")

df_list=[]
for idx, filepath in enumerate(csv_files[:80]):  # LIMIT TO 80
    try:
        df=pd.read_csv(filepath)
        df['file_id']=idx
        if 'Time' not in df.columns: df['Time']=np.arange(len(df))*0.1
        if 'Speed Meter (km/h) [Consult III]' in df.columns: df['Speed']=df['Speed Meter (km/h) [Consult III]']/3.6
        elif 'SPEED (mph)[GPS]' in df.columns: df['Speed']=df['SPEED (mph)[GPS]']*0.44704
        elif 'Speed' in df.columns: df['Speed']=df['Speed']/3.6 if df['Speed'].max()>40 else df['Speed']
        else: df['Speed']=0.0
        df['T_amb']=df['Ambient Temperature (degree C)'] if 'Ambient Temperature (degree C)' in df.columns else 25.0
        if 'ALTITUDE' in df.columns and len(df)>1:
            alt=df['ALTITUDE'].values; spd=df['Speed'].values; tv=df['Time'].values
            dtv=np.diff(tv, prepend=tv[0]); dtv[dtv<=0]=0.1; dist=spd*dtv; dz=np.diff(alt, prepend=alt[0])
            slope=np.zeros(len(df)); valid=dist>0.01; slope[valid]=np.degrees(np.arctan(dz[valid]/dist[valid]))
            df['Slope']=np.clip(slope,-15,15)
        else: df['Slope']=0.0
        df['SOC']=df['HV Battery Level (%)']/100.0 if 'HV Battery Level (%)' in df.columns else 0.8
        df['P_ac']=df['Power Consumption (A/C) (kW)'] if 'Power Consumption (A/C) (kW)' in df.columns else 0.5
        dt_arr=np.diff(df['Time'].values, prepend=df['Time'].values[0]); dt_arr[dt_arr<=0]=0.1
        df['Accel']=np.diff(df['Speed'].values, prepend=df['Speed'].values[0])/dt_arr
        df=df[(df['Speed']>=0)&(df['Speed']<=40.0)]; df=df[(df['P_ac']>=0)&(df['P_ac']<10.0)]
        df=df.dropna(subset=['Speed','Slope','T_amb','SOC','P_ac'])
        if len(df)>10: df_list.append(df[['Time','Speed','Accel','Slope','T_amb','SOC','P_ac','file_id']])
    except Exception as e:
        print(f"Skip {filepath}: {e}")

df_all=pd.concat(df_list, ignore_index=True)
print(f"Files: {len(df_list)} | Samples: {len(df_all):,}")

# KMeans
sample_size=min(30000,len(df_all)); df_sample=df_all.sample(n=sample_size,random_state=42)
X_clust=StandardScaler().fit_transform(df_sample[['Speed','Slope','T_amb']].values)
rows=[]
for k in range(2,11):
    km=KMeans(n_clusters=k,random_state=42,n_init=10); lb=km.fit_predict(X_clust)
    sil=silhouette_score(X_clust,lb); rows.append({'K':k,'Inertia':round(km.inertia_,2),'Silhouette':round(sil,4)})
df_table2=pd.DataFrame(rows)
print(df_table2.to_string(index=False))

# Rare
rare_mask=(df_all['Accel']<-4.5)|(df_all['Accel']>3.0)|(df_all['T_amb']>35.0)
df_rare=df_all[rare_mask] if rare_mask.sum()>20 else df_all.sample(min(500,len(df_all)),random_state=42)
n_synth=int(0.2*min(2000,len(df_all)))
df_synth=df_rare.sample(n=n_synth,replace=True,random_state=42).copy()
for col in ['Speed','Accel','Slope','T_amb']: df_synth[col]+=np.random.normal(0,0.05*(df_synth[col].std()+1e-3),size=n_synth)

# Hybrid
unique_files=sorted(df_all['file_id'].unique()); split_f=int(len(unique_files)*0.7)
train_files=set(unique_files[:split_f]); val_files=set(unique_files[split_f:])
df_train_base_full=df_all[df_all['file_id'].isin(train_files)]
df_val_full=df_all[df_all['file_id'].isin(val_files)]
df_train_base=df_train_base_full.sample(n=min(5000,len(df_train_base_full)),random_state=42)
df_val_sample=df_val_full.sample(n=min(1000,len(df_val_full)),random_state=42)
df_train_full=pd.concat([df_train_base, df_synth], ignore_index=True)
df_train_no_rare=df_train_base.copy()

M_veh,g,C_r,rho,C_d,A_f=1800.0,9.81,0.012,1.225,0.28,2.3
def compute_P_req(df):
    theta=np.radians(df['Slope'].values)
    F=(M_veh*df['Accel'].values + M_veh*g*C_r*np.cos(theta) + M_veh*g*np.sin(theta) + 0.5*rho*C_d*A_f*(df['Speed'].values**2))
    return np.clip((F*df['Speed'].values)/1000.0 + df['P_ac'].values, -80.0, 150.0)

for d in [df_train_full, df_train_no_rare, df_val_sample]: d['P_req_kW']=compute_P_req(d)

features=['Speed','Accel','Slope','T_amb','P_ac']
scaler_X=StandardScaler()
X_train=scaler_X.fit_transform(df_train_full[features].values)
X_val_s=scaler_X.transform(df_val_sample[features].values)
y_train=df_train_full['P_req_kW'].values
y_val=df_val_sample['P_req_kW'].values

scaler_X_nr=StandardScaler()
X_train_nr=scaler_X_nr.fit_transform(df_train_no_rare[features].values)
y_train_nr=df_train_no_rare['P_req_kW'].values

def train_models(Xtr,ytr):
    ms={'XGBoost': xgb.XGBRegressor(n_estimators=60,max_depth=4,learning_rate=0.1,random_state=42,verbosity=0),'LightGBM': lgb.LGBMRegressor(n_estimators=60,max_depth=4,learning_rate=0.1,random_state=42,verbose=-1),'DecisionTree': DecisionTreeRegressor(max_depth=6,random_state=42),'MLP': MLPRegressor(hidden_layer_sizes=(32,16),max_iter=150,random_state=42)}
    for m in ms.values(): m.fit(Xtr,ytr)
    return ms

models=train_models(X_train,y_train)
models_nr=train_models(X_train_nr,y_train_nr)

mses={}
for name,model in models.items(): pred=model.predict(X_val_s); mses[name]=max(mean_squared_error(y_val,pred),1e-6)
inv_sum=sum(1.0/m for m in mses.values()); weights={n:(1.0/m)/inv_sum for n,m in mses.items()}
print("Weights:", weights)

def hybrid_predict(X_raw): Xs=scaler_X.transform(X_raw); preds=np.array([models[n].predict(Xs) for n in models]); w=np.array([weights[n] for n in models]); return np.dot(w,preds)
def hybrid_predict_no_rare(X_raw): Xs=scaler_X_nr.transform(X_raw); preds=np.array([models_nr[n].predict(Xs) for n in models_nr]); w=np.array([weights[n] for n in models_nr]); return np.dot(w,preds)
def single_xgb_predict(X_raw): return models['XGBoost'].predict(scaler_X.transform(X_raw))

# FIXED HV REF - DYNAMIC BASED ON ACTUAL F
# We will compute ref per run as max*1.2, not fixed 2.0
def get_hv_indicator(F):
    # F normalized already? J1 0.5-1.0, others ~1
    # Use max*1.1 as ref to ensure all points contribute
    ref = np.max(F, axis=0) * 1.2 + 0.1
    # Ensure ref > all
    ref = np.maximum(ref, [2.0,2.0,2.0,2.0,2.0])  # at least 2.0
    return HV(ref_point=ref)

class VectorizedEVProblem(Problem):
    def __init__(self, df_seg, hybrid_fn, dt=0.1, use_hvac=True, use_noise=True, use_traffic=True, use_grade=True, use_soc_penalty=True, use_gear_effect=True, n_control_points=5):
        self.df_seg=df_seg.reset_index(drop=True); self.n_steps=len(self.df_seg); self.n_cp=n_control_points; self.dt=dt
        self.use_hvac=use_hvac; self.use_noise=use_noise; self.use_traffic=use_traffic; self.use_grade=use_grade; self.use_soc_penalty=use_soc_penalty; self.use_gear_effect=use_gear_effect; self.hybrid_fn=hybrid_fn
        n_var=n_control_points*4; xl=np.tile([-150.0,0.0,0.01,1.0],n_control_points); xu=np.tile([250.0,400.0,0.20,6.0],n_control_points)
        super().__init__(n_var=n_var, n_obj=5, n_ieq_constr=1 if use_soc_penalty else 0, xl=xl, xu=xu, elementwise_evaluation=False)
        self.U_ocv=400.0; self.R0=0.15; self.R1_ref=0.10; self.beta=0.15; self.Ea=40000.0; self.R_gas=8.314; self.Q_Ah=200.0; self.C_th=1500.0*3600.0
        self.speed=self.df_seg['Speed'].values.copy(); self.accel=self.df_seg['Accel'].values.copy()
        self.slope=self.df_seg['Slope'].values.copy() if use_grade else np.zeros(self.n_steps); self.t_amb=self.df_seg['T_amb'].values.copy(); self.p_ac=self.df_seg['P_ac'].values.copy() if use_hvac else np.zeros(self.n_steps)
        if not use_traffic: self.speed=np.full(self.n_steps,np.mean(self.speed)); self.accel=np.zeros(self.n_steps)
        if use_noise: rng=np.random.default_rng(42); self.speed=np.maximum(0.0,self.speed+rng.normal(0,0.25,self.n_steps))
        X_future=self.df_seg[features].values; self.P_req_hybrid=hybrid_fn(X_future) if hybrid_fn is not None else compute_P_req(self.df_seg)
        theta_r=np.radians(self.slope); F_req=(M_veh*self.accel + M_veh*g*C_r*np.cos(theta_r) + M_veh*g*np.sin(theta_r) + 0.5*rho*C_d*A_f*(self.speed**2))
        self.T_m_ref=np.clip(F_req*0.32/8.0,0.0,400.0); self.P_grid_ref=50000.0
    def _evaluate(self, X, out, *args, **kwargs):
        N_pop=X.shape[0]; F=np.zeros((N_pop,5)); G=np.zeros((N_pop,1)) if self.use_soc_penalty else None
        cp_x=np.linspace(0,self.n_steps-1,self.n_cp); eval_x=np.arange(self.n_steps)
        for i in range(N_pop):
            x=X[i]; u_cp=x.reshape((self.n_cp,4))
            f_I=interp1d(cp_x,u_cp[:,0],kind='linear',fill_value='extrapolate'); f_T=interp1d(cp_x,u_cp[:,1],kind='linear',fill_value='extrapolate'); f_m=interp1d(cp_x,u_cp[:,2],kind='linear',fill_value='extrapolate'); f_g=interp1d(cp_x,u_cp[:,3],kind='linear',fill_value='extrapolate')
            I_bat=f_I(eval_x); T_m=f_T(eval_x); m_cool=np.clip(f_m(eval_x),0.01,0.2); gear=np.clip(np.round(f_g(eval_x)),1,6)
            R1=self.R1_ref/(1.0+self.beta*np.abs(I_bat)); V_bat=self.U_ocv - I_bat*R1*0.5 - I_bat*self.R0
            dSOC=-I_bat*self.dt/(3600.0*self.Q_Ah); SOC=0.8+np.cumsum(dSOC)
            T_bat=np.zeros(self.n_steps); T_bat[0]=self.t_amb[0]+5.0
            for k in range(1,self.n_steps):
                Q_gen=(I_bat[k-1]**2)*(self.R0+R1[k-1]); P_cool=m_cool[k-1]*4180.0*4.0; dT=(Q_gen-P_cool)*self.dt/self.C_th; T_bat[k]=T_bat[k-1]+dT
            T_bat=np.clip(T_bat,15.0,60.0)
            P_bat_W=V_bat*I_bat; P_aux_W=self.p_ac*1000.0
            # FIXED ENERGY - REALISTIC
            eta_regen=0.75; P_bat_net=np.where(P_bat_W>0, P_bat_W, P_bat_W*eta_regen)
            total_energy_kWh=np.sum((P_bat_net + P_aux_W)*self.dt)/3.6e6
            dist_km=max(np.sum(self.speed*self.dt)/1000.0, 0.05)
            # FIXED: very small weight for tracking, normalized by mean P_req
            mean_P_req = np.mean(np.abs(self.P_req_hybrid))+1.0
            power_tracking_mse=np.mean(((P_bat_W/1000.0 - self.P_req_hybrid)/mean_P_req)**2)  # normalized
            J1_raw=total_energy_kWh/dist_km + 0.001*power_tracking_mse  # weight 0.001 not 0.05
            J1_raw=max(J1_raw,0.15)
            Q_gen_W=(I_bat**2)*(self.R0+R1); P_cool_W=m_cool*4180.0*4.0
            J3_raw=(np.sum(P_cool_W*self.dt)/3.6e6)/dist_km
            T_k=T_bat+273.15; J2_raw=float(np.sum(1e-5*np.exp(-self.Ea/(self.R_gas*T_k))*np.abs(I_bat)*self.dt))
            J4_raw=float(np.mean(np.abs(P_bat_W-self.P_grid_ref))/1e5)
            eff_T=T_m*(1.0+0.05*(gear-3.0)) if self.use_gear_effect else T_m
            J5_raw=float(np.mean(np.abs(eff_T-self.T_m_ref)))
            if J5_raw>50: J1_raw+=(J5_raw-50)*0.001  # small penalty
            J1=J1_raw/1.0; J2=J2_raw*1e9; J3=J3_raw/0.2; J4=J4_raw/1.0; J5=J5_raw/100.0
            F[i]=[J1,J2,J3,J4,J5]
            if self.use_soc_penalty: violation=np.sum((SOC<0.2)|(SOC>0.9)); G[i,0]=violation-0.5
        out["F"]=F
        if self.use_soc_penalty: out["G"]=G

def topsis_select(F):
    norm=F/(np.linalg.norm(F,axis=0)+1e-12); best=np.min(norm,axis=0); worst=np.max(norm,axis=0)
    d_pos=np.linalg.norm(norm-best,axis=1); d_neg=np.linalg.norm(norm-worst,axis=1)
    cc=d_neg/(d_pos+d_neg+1e-12); idx=int(np.argmax(cc)); return idx,F[idx]

def scalarized_cost(F): return np.dot(F,np.array([0.50,0.20,0.15,0.075,0.075]))

def run_co_pso_refinement_v3(problem, X_pareto, n_particles=12, n_iter=10):
    refined=[]; rng=np.random.default_rng(42)
    for x0 in X_pareto[:10]:
        dim=len(x0)
        pos=np.clip(x0 + rng.normal(0,0.05,size=(n_particles,dim))*(problem.xu-problem.xl), problem.xl, problem.xu)
        vel=np.zeros_like(pos); pbest=pos.copy()
        pbest_F=[]
        for p in pos:
            out={}; problem._evaluate(p.reshape(1,-1), out); pbest_F.append(out["F"][0])
        pbest_F=np.array(pbest_F); pbest_val=np.array([scalarized_cost(f) for f in pbest_F])
        gbest_idx=np.argmin(pbest_val); gbest=pbest[gbest_idx].copy(); gbest_F=pbest_F[gbest_idx].copy()
        for _ in range(n_iter):
            r1,r2=rng.random((n_particles,dim)), rng.random((n_particles,dim))
            vel=0.5*vel + 1.5*r1*(pbest-pos) + 1.5*r2*(gbest-pos)
            pos=np.clip(pos+vel, problem.xl, problem.xu)
            cur_F=[]
            for p in pos:
                out={}; problem._evaluate(p.reshape(1,-1), out); cur_F.append(out["F"][0])
            cur_F=np.array(cur_F); cur_val=np.array([scalarized_cost(f) for f in cur_F])
            imp=cur_val<pbest_val; pbest[imp]=pos[imp]; pbest_val[imp]=cur_val[imp]; pbest_F[imp]=cur_F[imp]
            if np.min(pbest_val)<scalarized_cost(gbest_F):
                gbest_idx=np.argmin(pbest_val); gbest=pbest[gbest_idx].copy(); gbest_F=pbest_F[gbest_idx].copy()
        refined.append(gbest_F)
    return np.array(refined)

# 5 WINDOWS
print("\nEVALUATION ON 5 WINDOWS")
windows=[]; rng_w=np.random.default_rng(123)
for _ in range(5):
    fid=rng_w.choice(unique_files)
    sub=df_all[df_all['file_id']==fid]
    if len(sub)>=150:
        start=rng_w.integers(0,len(sub)-150); windows.append(sub.iloc[start:start+150].copy())
    else:
        windows.append(df_all.sample(150,random_state=rng_w.integers(0,10000)).copy())

scenarios=[("Baseline (Idealized)",False,False,False,False),("+ HVAC Load",True,False,False,False),("+ Sensor Noise",True,True,False,False),("+ Traffic Variability",True,True,True,False),("Full Real-World (+ Road Grade)",True,True,True,True),]
all_t5=[]
for win_idx,df_eval in enumerate(windows):
    print(f"\nWindow {win_idx+1}/5")
    for label,hvac,noise,traffic,grade in scenarios:
        prob=VectorizedEVProblem(df_eval, hybrid_predict, dt=0.1, use_hvac=hvac, use_noise=noise, use_traffic=traffic, use_grade=grade, use_soc_penalty=True, use_gear_effect=True, n_control_points=5)
        algo=NSGA2(pop_size=100, sampling=FloatRandomSampling(), crossover=SBX(prob=0.8, eta=20), mutation=PM(prob=0.1, eta=20))
        t0=time.time()
        res=pymoo_minimize(prob, algo, ('n_gen', 35), seed=42, verbose=False)
        # FIXED HV with dynamic ref
        hv_before=get_hv_indicator(res.F)(res.F)
        F_refined=run_co_pso_refinement_v3(prob, res.X, n_particles=12, n_iter=10)
        F_combined=np.vstack([res.F, F_refined]) if len(F_refined)>0 else res.F
        hv_after=get_hv_indicator(F_combined)(F_combined)
        F_for_topsis=F_combined if len(F_refined)>0 else res.F
        _, best=topsis_select(F_for_topsis)
        J1_denorm=best[0]
        all_t5.append({'Window':win_idx+1,'Scenario':label,'J1_kWh_km':round(J1_denorm,4),'HV_before':round(hv_before,2),'HV_after':round(hv_after,2),'HV_improve_%':round((hv_after-hv_before)/hv_before*100,1) if hv_before>0 else 0,'Time_s':round(time.time()-t0,2)})
        print(f"  {label:35s} J1={J1_denorm:.4f} HV {hv_before:.1f}->{hv_after:.1f}")

df_table5=pd.DataFrame(all_t5)
df_table5_avg=df_table5.groupby('Scenario').agg({'J1_kWh_km':'mean','HV_before':'mean','HV_after':'mean','HV_improve_%':'mean','Time_s':'mean'}).reset_index()
print("\nTable5 Averaged:")
print(df_table5_avg.to_string(index=False))

# FULL ABLATION DETAILED
print("\n" + "="*100)
print("FULL ABLATION STUDY - DETAILED WITH 5 OBJECTIVES & HV")
print("="*100)
df_eval0=windows[0]
ablation_cases=[("Full (NSGA-II+CO-PSO+Rare+Hybrid+SOC+Gear)", True, True, True, True, True, hybrid_predict),("Without CO-PSO", False, True, True, True, True, hybrid_predict),("Without Rare (models_nr)", True, False, True, True, True, hybrid_predict_no_rare),("Without Hybrid (Single XGB)", True, True, False, True, True, single_xgb_predict),("Without SOC Constraint", True, True, True, False, True, hybrid_predict),("Without Gear Effect", True, True, True, True, False, hybrid_predict),]
abl_rows=[]
header = f"{'Variant':<45s} | {'J1':<6s} {'J2':<6s} {'J3':<6s} {'J4':<6s} {'J5':<6s} | {'HV_bef':<7s} {'HV_aft':<7s} {'?HV%':<6s}"
print(header)
print("-"*len(header))
for label,use_pso,use_rare,use_hybrid,use_soc,use_gear,h_fn in ablation_cases:
    prob=VectorizedEVProblem(df_eval0, h_fn, dt=0.1, use_hvac=True, use_noise=True, use_traffic=True, use_grade=True, use_soc_penalty=use_soc, use_gear_effect=use_gear, n_control_points=5)
    algo=NSGA2(pop_size=100, sampling=FloatRandomSampling(), crossover=SBX(prob=0.8, eta=20), mutation=PM(prob=0.1, eta=20))
    res=pymoo_minimize(prob, algo, ('n_gen', 35), seed=42, verbose=False)
    hv_before=get_hv_indicator(res.F)(res.F)
    if use_pso:
        F_ref=run_co_pso_refinement_v3(prob, res.X, n_particles=12, n_iter=10)
        F_comb=np.vstack([res.F, F_ref]) if len(F_ref)>0 else res.F
        hv_after=get_hv_indicator(F_comb)(F_comb)
        _, best=topsis_select(F_comb)
    else:
        hv_after=hv_before
        F_comb=res.F
        _, best=topsis_select(res.F)
    J1,J2,J3,J4,J5 = best[0], best[1], best[2], best[3], best[4]
    delta = ((hv_after-hv_before)/hv_before*100) if hv_before>0 else 0
    print(f"{label:<45s} | {J1:5.3f} {J2:5.3f} {J3:5.3f} {J4:5.3f} {J5:5.1f} | {hv_before:6.1f} {hv_after:6.1f} {delta:5.1f}%")
    abl_rows.append({'Variant':label,'J1_kWh_km':round(J1,4),'J2_norm':round(J2,4),'J3_norm':round(J3,4),'J4_norm':round(J4,4),'J5_norm':round(J5,2),'HV_before':round(hv_before,2),'HV_after':round(hv_after,2),'HV_improve_%':round(delta,1),'Pareto_size_before':len(res.F),'Pareto_size_after':len(F_comb)})

df_abl=pd.DataFrame(abl_rows)
print("\nDataFrame:")
print(df_abl.to_string(index=False))

with pd.ExcelWriter('UCR_V3_6_FINAL_Fixed.xlsx') as writer:
    df_table2.to_excel(writer, sheet_name='Table2_KMeans', index=False)
    df_table5.to_excel(writer, sheet_name='Table5_AllWindows', index=False)
    df_table5_avg.to_excel(writer, sheet_name='Table5_Averaged', index=False)
    df_abl.to_excel(writer, sheet_name='Ablation_Detailed', index=False)

print("\n[DONE] Saved UCR_V3_6_FINAL_Fixed.xlsx")

