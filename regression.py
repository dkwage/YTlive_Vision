import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.linear_model import LinearRegression
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.preprocessing import OneHotEncoder

# ==========================================
# 1. 환경 설정 및 보간 로직 (Helper Functions)
# ==========================================
DENSITY_MAP = {
    "09:00": [108, 138, 122, 137, 94], "10:30": [144, 128, 148, 124, 41],
    "12:00": [74, 86, 79, 85, 25],     "13:30": [153, 153, 167, 149, 57],
    "15:00": [151, 152, 155, 140, 53]
}
DAY_LIST = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
DAY_INDEX = {day: i for i, day in enumerate(DAY_LIST)}

def get_interpolated_density(day_name, time_numeric):
    day_idx = DAY_INDEX.get(day_name, 0)
    max_d = 0
    for slot_str, densities in DENSITY_MAP.items():
        h, m = map(int, slot_str.split(':'))
        slot_min = h * 60 + m
        peak = densities[day_idx]
        if slot_min - 30 <= time_numeric <= slot_min:
            max_d = max(max_d, peak * (time_numeric - (slot_min - 30)) / 30)
        elif slot_min < time_numeric <= slot_min + 10:
            max_d = max(max_d, peak * ((slot_min + 10) - time_numeric) / 10)
    return max_d

def numeric_to_time(n):
    return f"{int(n)//60:02d}:{int(n)%60:02d}"

# 데이터 로드 및 전처리
df = pd.read_csv('test_data.csv')
df['Time_Numeric'] = df['Time_Slot'].apply(lambda x: int(x.split(':')[0])*60 + int(x.split(':')[1]))

train_df = df[df['Data_Type'] == 'Train'].copy()
val_df = df[df['Data_Type'] == 'Validation'].copy()

# ==========================================
# 2. 모델 학습 (MLR & GPR)
# ==========================================
# Encoder 설정 (모든 요일 대응)
encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
encoder.fit(pd.DataFrame({'Day': DAY_LIST}))
day_cols = encoder.get_feature_names_out(['Day'])

# 학습 데이터 Feature 생성
train_day_enc = encoder.transform(train_df[['Day']])
X_train = pd.concat([train_df[['Time_Numeric', 'Class_Density']].reset_index(drop=True), 
                     pd.DataFrame(train_day_enc, columns=day_cols)], axis=1)
y_train = train_df['In_Count_Delta']

# 모델 1: 다중 선형 회귀 (MLR)
mlr = LinearRegression().fit(X_train, y_train)

# 모델 2: 가우시안 과정 회귀 (GPR) - 최적화 경고 방지 설정
kernel = C(1.0, (1e-3, 1e9)) * RBF(length_scale=45, length_scale_bounds=(1, 1e4))
gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, alpha=2.0).fit(X_train, y_train)

# ==========================================
# 3. 통합 시각화 리포트 (4 Sections)
# ==========================================
fig = plt.figure(figsize=(18, 30))
gs = gridspec.GridSpec(9, 1, figure=fig)

# [SECTION 1] Train Data (요일별 개별 시각화 + 단절 구간 반영)
train_days = ["Monday", "Tuesday", "Thursday"]
colors = {"Monday": "#3498db", "Tuesday": "#e67e22", "Thursday": "#9b59b6"}

for i, day in enumerate(train_days):
    ax = fig.add_subplot(gs[i, 0])
    day_data = train_df[train_df['Day'] == day].sort_values('Time_Numeric')
    day_data['group'] = (day_data['Time_Numeric'].diff() > 1).cumsum() # 단절 감지
    
    ax_dens = ax.twinx()
    for _, g_df in day_data.groupby('group'):
        ax.bar(g_df['Time_Numeric'], g_df['In_Count_Delta'], color=colors[day], alpha=0.5)
        ax_dens.plot(g_df['Time_Numeric'], g_df['Class_Density'], color='red', linewidth=2.5)
    
    ax.set_title(f"1-{i+1}. Training Samples: {day}", fontsize=15, fontweight='bold')
    win_ticks = [day_data['Time_Numeric'].min(), day_data['Time_Numeric'].max()]
    ax.set_xticks(win_ticks); ax.set_xticklabels([numeric_to_time(t) for t in win_ticks])
    ax.set_ylabel("Traffic"); ax_dens.set_ylabel("Density", color='red')

# [SECTION 2] Relationship Analysis
ax_rel = fig.add_subplot(gs[3, 0])
ax_rel.scatter(train_df['Class_Density'], train_df['In_Count_Delta'], color='blue', alpha=0.4, label='Train (Mon/Tue/Thu)')
ax_rel.scatter(val_df['Class_Density'], val_df['In_Count_Delta'], color='red', alpha=0.4, label='Validation (Fri)')
ax_rel.set_title("2. Correlation Analysis: Class Density vs Traffic", fontsize=15, fontweight='bold')
ax_rel.set_xlabel("Class Density"); ax_rel.set_ylabel("Traffic"); ax_rel.legend(); ax_rel.grid(True, alpha=0.2)

# [SECTION 3] Friday Validation (Actual = Red Dots)
ax_val = fig.add_subplot(gs[4:6, 0])
val_day_enc = encoder.transform(val_df[['Day']])
X_val = pd.concat([val_df[['Time_Numeric', 'Class_Density']].reset_index(drop=True), 
                   pd.DataFrame(val_day_enc, columns=day_cols)], axis=1)

y_val_mlr = mlr.predict(X_val)
y_val_gpr, sigma_val = gpr.predict(X_val, return_std=True)

ax_val.scatter(val_df['Time_Numeric'], val_df['In_Count_Delta'], color='red', s=15, alpha=0.6, label='Friday Actual')
ax_val.plot(val_df['Time_Numeric'], y_val_mlr, color='blue', linestyle='--', label='MLR Prediction')
ax_val.plot(val_df['Time_Numeric'], y_val_gpr, color='green', linewidth=2.5, label='GPR Prediction')
ax_val.fill_between(val_df['Time_Numeric'], y_val_gpr - 1.96*sigma_val, y_val_gpr + 1.96*sigma_val, alpha=0.2, color='green')

ax_val.set_title("3. Friday Validation: Actual vs Model Predictions", fontsize=15, fontweight='bold')
v_ticks = np.arange(val_df['Time_Numeric'].min(), val_df['Time_Numeric'].max()+1, 30)
ax_val.set_xticks(v_ticks); ax_val.set_xticklabels([numeric_to_time(t) for t in v_ticks]); ax_val.legend()

# [SECTION 4] Full-Day Simulation (Monday 08:00 - 21:00)
ax_full = fig.add_subplot(gs[6:9, 0])
full_times = np.arange(8*60, 21*60 + 1)
full_dens = [get_interpolated_density("Monday", t) for t in full_times]
monday_df = pd.DataFrame(['Monday'] * len(full_times), columns=['Day'])
monday_enc = encoder.transform(monday_df)

X_full = pd.concat([pd.DataFrame({'Time_Numeric': full_times, 'Class_Density': full_dens}), 
                    pd.DataFrame(monday_enc, columns=day_cols)], axis=1)
y_full_gpr, sigma_full = gpr.predict(X_full, return_std=True)

ax_full.plot(full_times, y_full_gpr, color='navy', linewidth=3, label='Monday Estimation')
ax_full.fill_between(full_times, y_full_gpr - 1.96*sigma_full, y_full_gpr + 1.96*sigma_full, alpha=0.15, color='navy')
ax_full.set_title("4. Business Insight: Full-Day Traffic Estimation (Monday)", fontsize=16, fontweight='bold', color='navy')
f_ticks = np.arange(8*60, 21*60 + 1, 60)
ax_full.set_xticks(f_ticks); ax_full.set_xticklabels([numeric_to_time(t) for t in f_ticks]); ax_full.legend()

plt.tight_layout()
plt.savefig('comprehensive_analysis_report.png')
plt.show()

# ==========================================
# 4. 주간 통합 예측 데이터 CSV 저장 (Mon~Fri)
# ==========================================
weekly_forecast = []
for day in DAY_LIST:
    day_dens = [get_interpolated_density(day, t) for t in full_times]
    d_df = pd.DataFrame({'Day': [day] * len(full_times)})
    d_enc = encoder.transform(d_df)
    X_f = pd.concat([pd.DataFrame({'Time_Numeric': full_times, 'Class_Density': day_dens}), 
                     pd.DataFrame(d_enc, columns=day_cols)], axis=1)
    preds, sigmas = gpr.predict(X_f, return_std=True)
    
    for t, den, p, s in zip(full_times, day_dens, preds, sigmas):
        weekly_forecast.append({
            "Day": day, "Time": numeric_to_time(t), "Density": round(den, 2),
            "Predicted_Traffic": round(p, 2), "95_CI": round(1.96 * s, 2)
        })

pd.DataFrame(weekly_forecast).to_csv('weekly_traffic_forecast.csv', index=False, encoding='utf-8-sig')
print("분석 완료: 'comprehensive_analysis_report.png' 및 'weekly_traffic_forecast.csv'가 저장되었습니다.")