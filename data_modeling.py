import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
import warnings

# Basic settings
warnings.filterwarnings('ignore')
plt.rcParams['axes.grid'] = False
sns.set_style("white")

# =========================================================
# 1. Load Data & Feature Engineering
# =========================================================
try:
    df_raw = pd.read_csv('merged_data.csv').dropna(subset=['Day'])
except FileNotFoundError:
    print("Error: 'merged_data.csv' not found.")
    exit()

def time_to_min(ts):
    try: h, m = map(int, str(ts).split(':')); return h * 60 + m
    except: return 0

def numeric_to_time(n):
    return f"{int(n)//60:02d}:{int(n)%60:02d}"

df_raw['Time_Numeric'] = df_raw['Time_Slot'].apply(time_to_min)

def get_minutes_to_class(current_mins):
    class_times = [h * 60 + m for h in range(8, 20) for m in (0, 30)]
    closest_class = min(class_times, key=lambda x: abs(x - current_mins))
    return current_mins - closest_class

df_raw['Minutes_to_Class'] = df_raw['Time_Numeric'].apply(get_minutes_to_class)
df_raw['Min_to_Class_Sq'] = df_raw['Minutes_to_Class']**2

OFFICIAL_SLOTS = [540, 630, 720, 810, 900, 990, 1080, 1170]
offsets = []
for day in df_raw['Day'].unique():
    day_df = df_raw[df_raw['Day'] == day]
    if not day_df.empty:
        peak_row = day_df.loc[day_df['Floating_Population'].idxmax()]
        closest_slot = min(OFFICIAL_SLOTS, key=lambda x: abs(x - peak_row['Time_Numeric']))
        offsets.append(peak_row['Time_Numeric'] - closest_slot)

VERIFIED_OFFSET = int(np.mean(offsets)) if offsets else -5

def get_calibrated_density(day_name, time_numeric):
    DENSITY_MAP_VALS = [[108, 138, 122, 137, 94], [144, 128, 148, 124, 41], [74, 86, 79, 85, 25], 
                        [153, 153, 167, 149, 57], [151, 152, 155, 140, 53], [68, 63, 70, 54, 8], 
                        [24, 32, 20, 17, 6], [10, 4, 4, 4, 2]]
    day_idx = {"Monday":0, "Tuesday":1, "Wednesday":2, "Thursday":3, "Friday":4}.get(day_name, 0)
    max_d = 0
    for i, densities in enumerate(DENSITY_MAP_VALS):
        cal_peak = OFFICIAL_SLOTS[i] + VERIFIED_OFFSET
        if cal_peak - 30 <= time_numeric <= cal_peak:
            max_d = max(max_d, densities[day_idx] * (time_numeric - (cal_peak - 30)) / 30)
        elif cal_peak < time_numeric <= cal_peak + 10:
            max_d = max(max_d, densities[day_idx] * ((cal_peak + 10) - time_numeric) / 10)
    return max_d

df_raw['Class_Density'] = df_raw.apply(lambda r: get_calibrated_density(r['Day'], r['Time_Numeric']), axis=1)

y = df_raw['Floating_Population'].astype(float)

# Only numerical features to prevent flat lines (removing Day OneHotEncoder)
X_final = df_raw[['Time_Numeric', 'Minutes_to_Class', 'Min_to_Class_Sq', 'Class_Density']]

X_train, X_test, y_train, y_test = train_test_split(X_final, y, test_size=0.2, random_state=42)

# =========================================================
# 2. Modeling
# =========================================================
# [Parametric]
parametric_results = []
glm_families = {"Poisson": sm.families.Poisson(), "Negative Binomial": sm.families.NegativeBinomial(), "Gaussian (Linear)": sm.families.Gaussian()}
X_tr_c = sm.add_constant(X_train); X_te_c = sm.add_constant(X_test)
best_para_model_obj, best_para_aic = None, float('inf')

for name, family in glm_families.items():
    try:
        model = sm.GLM(y_train, X_tr_c, family=family).fit()
        pred = model.predict(X_te_c)
        aic = model.aic
        parametric_results.append({"Model": name, "AIC": aic, "R2": r2_score(y_test, pred)})
        if aic < best_para_aic:
            best_para_aic = aic
            best_para_model_obj = model
    except Exception: pass

para_df = pd.DataFrame(parametric_results).sort_values("AIC")
best_para = para_df.iloc[0]
best_para_name = best_para['Model']

p_values = best_para_model_obj.pvalues
main_pvals = p_values[['Minutes_to_Class', 'Min_to_Class_Sq', 'Class_Density']]

# [Non-Parametric]
non_parametric_results = []
ml_models = {"Random Forest": RandomForestRegressor(n_estimators=150, max_depth=6, random_state=42), 
             "Gradient Boosting": GradientBoostingRegressor(n_estimators=150, max_depth=5, random_state=42), 
             "KNN": KNeighborsRegressor(n_neighbors=5)}
sc = StandardScaler()
X_tr_s = sc.fit_transform(X_train); X_te_s = sc.transform(X_test)

for name, model in ml_models.items():
    if name == "KNN":
        model.fit(X_tr_s, y_train)
        y_pred = model.predict(X_te_s)
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
    non_parametric_results.append({"Model": name, "R2": r2_score(y_test, y_pred)})

non_para_df = pd.DataFrame(non_parametric_results).sort_values("R2", ascending=False)
best_nonpara = non_para_df.iloc[0]
best_nonpara_name = best_nonpara['Model']
best_nonpara_model_obj = ml_models[best_nonpara_name]

# Quantile Models for CI
q_lower = GradientBoostingRegressor(loss='quantile', alpha=0.05, max_depth=5, random_state=42).fit(X_train, y_train)
q_upper = GradientBoostingRegressor(loss='quantile', alpha=0.95, max_depth=5, random_state=42).fit(X_train, y_train)

# =========================================================
# 3. Visualization Dashboard
# =========================================================
fig = plt.figure(figsize=(20, 24), facecolor='white')

# [1] Target Distribution 
plt.subplot(3, 2, 1)
sns.histplot(y, kde=True, color='skyblue')
plt.title(f"1. Target Distribution (Skewness: {y.skew():.3f})", fontweight='bold', fontsize=15)
plt.xlabel("Floating Population")

# [2] Parametric Comparison 
plt.subplot(3, 2, 2)
sns.barplot(x='AIC', y='Model', data=para_df, palette='flare')
plt.title("2. Parametric Model Comparison (Lower AIC Better)", fontweight='bold', fontsize=15)

# [3] Non-Parametric Comparison 
plt.subplot(3, 2, 3)
sns.barplot(x='R2', y='Model', data=non_para_df, palette='viridis')
plt.title("3. Non-Parametric Model Comparison (Higher R2 Better)", fontweight='bold', fontsize=15)

# [4] Hypothesis Testing (p-value Visualization)
plt.subplot(3, 2, 4)
colors = ['coral' if p < 0.05 else 'lightgray' for p in main_pvals.values]
ax = sns.barplot(x=main_pvals.values, y=main_pvals.index, palette=colors)
plt.axvline(0.05, color='red', linestyle='--', linewidth=2, label='Significance Threshold (p=0.05)')
plt.title("4. Hypothesis Testing: p-values (Parametric Model)", fontweight='bold', fontsize=15)
plt.xlabel("p-value (Bars crossing the red line are NOT significant)")
for i, v in enumerate(main_pvals.values):
    ax.text(v + (max(main_pvals.values)*0.02), i, f"p = {v:.4f}", color='black', va='center', fontweight='bold', fontsize=12)
plt.legend(loc='lower right', fontsize=12)

# =========================================================
# [5 & 6] Simulation vs Actual 
# =========================================================
sample_day = "Tuesday"
time_range = np.arange(480, 1261, 1) 

s_df = pd.DataFrame({'Time_Numeric': time_range, 'Day': sample_day})
s_df['Minutes_to_Class'] = s_df['Time_Numeric'].apply(get_minutes_to_class)
s_df['Min_to_Class_Sq'] = s_df['Minutes_to_Class']**2
s_df['Class_Density'] = s_df['Time_Numeric'].apply(lambda x: get_calibrated_density(sample_day, x))

X_s_final = s_df[['Time_Numeric', 'Minutes_to_Class', 'Min_to_Class_Sq', 'Class_Density']]
real_pts = df_raw[(df_raw['Day'] == sample_day)]

# [5] Parametric vs Actual
plt.subplot(3, 2, 5)
X_s_const = sm.add_constant(X_s_final, has_constant='add')
para_preds = best_para_model_obj.predict(X_s_const)
plt.plot(time_range, para_preds, label=f"Parametric ({best_para_name})", color='black', linestyle='--', linewidth=2.5)
plt.scatter(real_pts['Time_Numeric'], real_pts['Floating_Population'], color='gray', s=50, alpha=0.6, label='Actual Data', zorder=5)

# 🌟 Add R2 and AIC to graph
para_text = f"★ Best Model: {best_para_name}\n► Test R²: {best_para['R2']:.3f}\n► AIC: {best_para['AIC']:.1f}"
plt.text(0.03, 0.85, para_text, transform=plt.gca().transAxes, fontsize=14, fontweight='bold', 
         bbox=dict(facecolor='white', alpha=0.9, edgecolor='black', boxstyle='round,pad=0.5'))

plt.title(f"5. Parametric Model vs Actual ({sample_day})", fontweight='bold', fontsize=15)
plt.xlabel("Time"); plt.ylabel("Floating Population")
plt.xticks(np.arange(480, 1261, 120), [numeric_to_time(t) for t in np.arange(480, 1261, 120)])
plt.legend(loc='upper right')

# [6] Non-Parametric vs Actual
plt.subplot(3, 2, 6)
if best_nonpara_name == "KNN":
    X_s_final_input = sc.transform(X_s_final)
else:
    X_s_final_input = X_s_final

non_para_preds = best_nonpara_model_obj.predict(X_s_final_input)
lower_preds = q_lower.predict(X_s_final)
upper_preds = q_upper.predict(X_s_final)

plt.fill_between(time_range, lower_preds, upper_preds, color='red', alpha=0.15, label='90% Interval')
plt.plot(time_range, non_para_preds, label=f"Non-Parametric ({best_nonpara_name})", color='red', linewidth=2.5)
plt.scatter(real_pts['Time_Numeric'], real_pts['Floating_Population'], color='blue', s=50, alpha=0.4, label='Actual Data', zorder=5)

# 🌟 Add R2 and AIC to graph
nonpara_text = f"★ Best Model: {best_nonpara_name}\n► Test R²: {best_nonpara['R2']:.3f}\n► AIC: N/A"
plt.text(0.03, 0.85, nonpara_text, transform=plt.gca().transAxes, fontsize=14, fontweight='bold', 
         bbox=dict(facecolor='white', alpha=0.9, edgecolor='black', boxstyle='round,pad=0.5'))

plt.title(f"6. Non-Parametric Model vs Actual ({sample_day})", fontweight='bold', fontsize=15)
plt.xlabel("Time"); plt.ylabel("Floating Population")
plt.xticks(np.arange(480, 1261, 120), [numeric_to_time(t) for t in np.arange(480, 1261, 120)])
plt.legend(loc='upper right')

plt.tight_layout()
plt.savefig('data_modeling_results.png', dpi=300, bbox_inches='tight')
