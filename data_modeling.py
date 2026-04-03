import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
import warnings

# 0. 기초 설정
warnings.filterwarnings('ignore')
plt.rcParams['axes.grid'] = False
sns.set_style("white")

# 보조 함수
def time_to_min(ts):
    try: h, m = map(int, str(ts).split(':')); return h * 60 + m
    except: return 0
def numeric_to_time(n): return f"{int(n)//60:02d}:{int(n)%60:02d}"

# 1. 데이터 로드 및 시차 보정 (Offset)
df_raw = pd.read_csv('merged_data.csv').dropna(subset=['Day'])
df_raw['Time_Numeric'] = df_raw['Time_Slot'].apply(time_to_min)
OFFICIAL_SLOTS = [540, 630, 720, 810, 900, 990, 1080, 1170]

# 피크 시차 자동 계산
offsets = []
for day in df_raw['Day'].unique():
    day_df = df_raw[df_raw['Day'] == day]
    peak_row = day_df.loc[day_df['Floating_Population'].idxmax()]
    offsets.append(peak_row['Time_Numeric'] - min(OFFICIAL_SLOTS, key=lambda x: abs(x - peak_row['Time_Numeric'])))
VERIFIED_OFFSET = int(np.mean(offsets))

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

# 데이터셋 구성
df_raw['Class_Density'] = df_raw.apply(lambda r: get_calibrated_density(r['Day'], r['Time_Numeric']), axis=1)
df_analysis = df_raw[df_raw['Class_Density'] > 5].reset_index(drop=True)

encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore').fit(pd.DataFrame({'Day': ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]}))
X_raw = df_analysis[['Time_Numeric', 'Class_Density']]
X_day = pd.DataFrame(encoder.transform(df_analysis[['Day']]), columns=encoder.get_feature_names_out(['Day']), index=df_analysis.index)
X_combined = pd.concat([X_raw, X_day], axis=1)
y = df_analysis['Floating_Population']

X_train, X_test, y_train, y_test = train_test_split(X_combined, y, test_size=0.2, random_state=42)

# ---------------------------------------------------------
# 단계 2 & 3. 모수 및 비모수 모델 성능 측정
# ---------------------------------------------------------

# 모수 결과 집계 (AIC, R2)
parametric_results = []
glm_families = {"Poisson": sm.families.Poisson(), "Neg-Binomial": sm.families.NegativeBinomial(), "Linear": sm.families.Gaussian()}
X_tr_c = sm.add_constant(X_train); X_te_c = sm.add_constant(X_test)

for name, family in glm_families.items():
    model = sm.GLM(y_train, X_tr_c, family=family).fit()
    parametric_results.append({"Model": name, "Type": "Parametric", "AIC": model.aic, "R2": r2_score(y_test, model.predict(X_te_c)), "MAE": mean_absolute_error(y_test, model.predict(X_te_c))})

# 비모수 결과 집계 (R2, MAE)
non_parametric_results = []
ml_models = {
    "Random Forest": RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, max_depth=4, random_state=42),
    "KNN": KNeighborsRegressor(n_neighbors=5)
}

# KNN용 스케일링
sc = StandardScaler().fit(X_train); X_tr_s = sc.transform(X_train); X_te_s = sc.transform(X_test)

for name, model in ml_models.items():
    xtr = X_tr_s if name == "KNN" else X_train; xte = X_te_s if name == "KNN" else X_test
    model.fit(xtr, y_train)
    non_parametric_results.append({"Model": name, "Type": "Non-Parametric", "AIC": np.nan, "R2": r2_score(y_test, model.predict(xte)), "MAE": mean_absolute_error(y_test, model.predict(xte))})

# ---------------------------------------------------------
# 단계 4, 5, 6. 최상위 모델 선정 및 시각화 리포트
# ---------------------------------------------------------

# 최상위 모델 추출
best_para = pd.DataFrame(parametric_results).sort_values("R2", ascending=False).iloc[0]
best_nonpara = pd.DataFrame(non_parametric_results).sort_values("R2", ascending=False).iloc[0]

fig = plt.figure(figsize=(22, 28), facecolor='white')

# 1. 분포 분석 (왜도)
plt.subplot(3, 2, 1)
sns.histplot(y, kde=True, color='skyblue')
plt.title(f"1. Distribution Analysis (Skewness: {y.skew():.3f})", fontweight='bold', fontsize=15)

# 2. 모수 모델 비교 (AIC 기준)
plt.subplot(3, 2, 2)
sns.barplot(x='AIC', y='Model', data=pd.DataFrame(parametric_results).sort_values("AIC"), palette='flare')
plt.title("2. Parametric Model Comparison (Lower AIC is Better)", fontweight='bold', fontsize=15)

# 3. 비모수 모델 비교 (R2 기준)
plt.subplot(3, 2, 3)
sns.barplot(x='R2', y='Model', data=pd.DataFrame(non_parametric_results).sort_values("R2", ascending=False), palette='viridis')
plt.title("3. Non-Parametric Model Comparison (Higher R2 is Better)", fontweight='bold', fontsize=15)

# 4 & 5. 최상위 모델 선정 결과 표
plt.subplot(3, 2, 4)
plt.axis('off')
best_summary = f"""
[4. Top Parametric Model]
- Name: {best_para['Model']}
- Test R2: {best_para['R2']:.3f}
- MAE: {best_para['MAE']:.2f}명

[5. Top Non-Parametric Model]
- Name: {best_nonpara['Model']}
- Test R2: {best_nonpara['R2']:.3f}
- MAE: {best_nonpara['MAE']:.2f}명
"""
plt.text(0.1, 0.5, best_summary, fontsize=16, family='monospace', fontweight='bold', verticalalignment='center')

# 6. 모수 vs 비모수 최종 비교 (화요일 타임라인)
plt.subplot(3, 1, 3)
sample_day = "Tuesday"
time_range = np.arange(480, 1261, 1)
s_df = pd.DataFrame({'Day': [sample_day]*len(time_range), 'Time_Numeric': time_range})
s_df['Class_Density'] = s_df['Time_Numeric'].apply(lambda x: get_calibrated_density(sample_day, x))
X_s_raw = s_df[['Time_Numeric', 'Class_Density']]
X_s_day = pd.DataFrame(encoder.transform(s_df[['Day']]), columns=encoder.get_feature_names_out(['Day']), index=s_df.index)
X_s_combined = pd.concat([X_s_raw, X_s_day], axis=1)

# 모수 최상위 예측
m_para_final = sm.GLM(y, sm.add_constant(X_combined), family=glm_families[best_para['Model']]).fit()
# 상수항 수동 추가로 에러 방지
X_s_const = X_s_combined.copy(); X_s_const.insert(0, 'const', 1.0)
plt.plot(time_range, m_para_final.predict(X_s_const), label=f"TOP Parametric: {best_para['Model']}", ls='--', color='black', alpha=0.7, lw=2)

# 비모수 최상위 예측
m_np_final = ml_models[best_nonpara['Model']].fit(X_train, y_train)
plt.plot(time_range, m_np_final.predict(X_s_combined), label=f"TOP Non-Parametric: {best_nonpara['Model']}", color='red', lw=3)

# 실제 데이터
real_pts = df_raw[df_raw['Day'] == sample_day]
plt.scatter(real_pts['Time_Numeric'], real_pts['Floating_Population'], color='blue', s=60, label='Actual Data', zorder=5, alpha=0.6)

plt.title(f"6. Final Battle: {best_para['Model']} vs {best_nonpara['Model']} ({sample_day})", fontweight='bold', fontsize=18)
plt.xticks(np.arange(480, 1261, 60), [numeric_to_time(t) for t in np.arange(480, 1261, 60)])
plt.legend(loc='upper right', fontsize=12)
plt.xlabel("Time"); plt.ylabel("Floating Population")

plt.tight_layout()
plt.savefig('final_top_model_report.png', dpi=300, bbox_inches='tight')
plt.show()

print("--- Analysis Completed ---")