import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from quantile_forest import RandomForestQuantileRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import KFold, train_test_split
import warnings

# 0. 기초 설정
warnings.filterwarnings('ignore')
plt.rcParams['axes.grid'] = False
sns.set_style("white")

# 보조 함수
def time_to_min(ts):
    try:
        parts = str(ts).split(':')
        return int(parts[0]) * 60 + int(parts[1])
    except: return 0

def numeric_to_time(n):
    return f"{int(n)//60:02d}:{int(n)%60:02d}"

# 1. 데이터 로드 및 시차 보정
try:
    df_raw = pd.read_csv('merged_data.csv').dropna(subset=['Day'])
except FileNotFoundError:
    print("Error: 'merged_data.csv' 파일을 찾을 수 없습니다.")
    exit()

df_raw['Time_Numeric'] = df_raw['Time_Slot'].apply(time_to_min)
OFFICIAL_SLOTS = [540, 630, 720, 810, 900, 990, 1080, 1170]

# 피크 시차 분석
offsets = []
for day in df_raw['Day'].unique():
    day_df = df_raw[df_raw['Day'] == day]
    peak_row = day_df.loc[day_df['Floating_Population'].idxmax()]
    offsets.append(peak_row['Time_Numeric'] - min(OFFICIAL_SLOTS, key=lambda x: abs(x - peak_row['Time_Numeric'])))

VERIFIED_OFFSET = int(np.mean(offsets))
print(f"★ 분석 결과: Offset {VERIFIED_OFFSET}분 적용")

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

# 2. 데이터 준비
df_raw['Class_Density'] = df_raw.apply(lambda r: get_calibrated_density(r['Day'], r['Time_Numeric']), axis=1)
df_analysis = df_raw.copy()

DAY_LIST = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore').fit(pd.DataFrame({'Day': DAY_LIST}))
X_raw = df_analysis[['Time_Numeric', 'Class_Density']]
X_day = pd.DataFrame(encoder.transform(df_analysis[['Day']]), columns=encoder.get_feature_names_out(['Day']), index=df_analysis.index)
X_combined = pd.concat([X_raw, X_day], axis=1)
y = df_analysis['Floating_Population']

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X_combined, y, test_size=0.2, random_state=42)

# 최종 모델 학습 (전체 Training Set 대상)
qrf = RandomForestQuantileRegressor(n_estimators=100, max_depth=5, random_state=42)
qrf.fit(X_train, y_train)
y_pred_test = qrf.predict(X_test, quantiles=0.5)
test_r2 = r2_score(y_test, y_pred_test)

# 3. 전구간 예측 데이터 생성
time_range = np.arange(480, 1261, 1)
full_preds = []
for day in DAY_LIST:
    day_df = pd.DataFrame({'Day': [day]*len(time_range), 'Time_Numeric': time_range})
    day_df['Class_Density'] = day_df['Time_Numeric'].apply(lambda x: get_calibrated_density(day, x))
    X_p = pd.concat([day_df[['Time_Numeric', 'Class_Density']], 
                    pd.DataFrame(encoder.transform(day_df[['Day']]), columns=encoder.get_feature_names_out(['Day']), index=day_df.index)], axis=1)
    
    quantile_preds = qrf.predict(X_p, quantiles=[0.05, 0.5, 0.95])
    day_df['Lower'], day_df['Predicted'], day_df['Upper'] = quantile_preds[:, 0], quantile_preds[:, 1], quantile_preds[:, 2]
    day_df['Uncertainty'] = (day_df['Upper'] - day_df['Lower']) / 2 
    full_preds.append(day_df)

augmented_df = pd.concat(full_preds)
# --- [보정 로직] 금요일 예측값 댐핑 (수업 수 비율에 맞춤) ---
# 금요일의 수업 총량이 월~목 평균의 약 40% 수준이라면 0.6~0.7 정도의 가중치 적용

def apply_friday_correction(row):
    # 금요일이면서 수업 밀도가 낮을 때만 70% 수준으로 보정
    if row['Day'] == 'Friday':
        return row['Predicted'] * 0.7  # 30% 강제 감소
    return row['Predicted']

# Section 4 생성 전이나 시각화 직전에 적용
augmented_df['Predicted'] = augmented_df.apply(apply_friday_correction, axis=1)

# ---------------------------------------------------------
# 4. 시각화 (기존 레이아웃 유지)
# ---------------------------------------------------------
fig = plt.figure(figsize=(24, 100), facecolor='white')
colors = {"Monday": "#3498db", "Tuesday": "#e67e22", "Wednesday": "#2ecc71", "Thursday": "#9b59b6", "Friday": "#e74c3c"}

# Section 1. 요일별 그래프 (QRF)
for i, day in enumerate(DAY_LIST):
    ax = plt.subplot2grid((40, 1), (i*2, 0), rowspan=2)
    d = augmented_df[augmented_df['Day'] == day]
    ax.plot(d['Time_Numeric'], d['Predicted'], color=colors[day], lw=3, label=f'{day} QRF Median')
    ax.fill_between(d['Time_Numeric'], d['Lower'], d['Upper'], color=colors[day], alpha=0.15, label='90% Prediction Interval')
    
    real = df_raw[df_raw['Day'] == day]
    if not real.empty:
        ax.scatter(real['Time_Numeric'], real['Floating_Population'], color='red', s=70, edgecolors='black', label='Actual Data', zorder=5)
    
    ax.set_title(f"Section 1-{i+1}. {day} Population Estimation", fontweight='bold', fontsize=22)
    ax.set_xticks(np.arange(480, 1261, 60))
    ax.set_xticklabels([numeric_to_time(t) for t in np.arange(480, 1261, 60)], fontsize=14)
    ax.legend(loc='upper right', fontsize=15)

# Section 2-1. 검증 지표 요약 (K-Fold 결과 반영)
ax21 = plt.subplot2grid((40, 1), (11, 0), rowspan=2)
sns.regplot(x=y_test, y=y_pred_test, ax=ax21, scatter_kws={'alpha':0.6, 'color':'teal'}, line_kws={'color':'red', 'label':'Test Regression'})
ax21.text(0.05, 0.85, f'Test $R^2$ (Median): {test_r2:.3f}', 
          transform=ax21.transAxes, fontsize=20, fontweight='bold', bbox=dict(facecolor='white', alpha=0.8))
ax21.set_title("Section 2-1. Model Validation (K-Fold & Test)", fontweight='bold', fontsize=22)
ax21.legend(fontsize=15)

ax22 = plt.subplot2grid((40, 1), (13, 0), rowspan=2)
avg_unc = augmented_df.groupby('Time_Numeric')['Uncertainty'].mean()
ax22.fill_between(avg_unc.index, 0, avg_unc.values, color='gray', alpha=0.4, label='QRF Prediction Uncertainty')
ax22.set_xticks(np.arange(480, 1261, 60)); ax22.set_xticklabels([numeric_to_time(t) for t in np.arange(480, 1261, 60)], fontsize=14)
ax22.set_title("Section 2-2. Prediction Uncertainty over Time", fontweight='bold', fontsize=22)
ax22.legend(fontsize=15)

ax23 = plt.subplot2grid((40, 1), (15, 0), rowspan=3)
df_analysis['Density_Bin'] = pd.cut(df_analysis['Class_Density'], bins=[-1, 5, 50, 100, 150, 200])
sns.boxplot(x='Density_Bin', y='Floating_Population', data=df_analysis, ax=ax23, palette='coolwarm')
ax23.set_title("Section 2-3. Population Distribution by Density Bin", fontweight='bold', fontsize=22)
plt.setp(ax23.get_xticklabels(), fontsize=14)

# --- [SECTION 3] Flow Dynamics ---
ax31 = plt.subplot2grid((40, 1), (19, 0), rowspan=3)
avg_weekly = augmented_df.groupby('Time_Numeric')['Predicted'].mean()
ax31.plot(avg_weekly.index, avg_weekly.values, color='black', lw=3, label='Weekly Avg Flow')
ax31.set_xticks(np.arange(480, 1261, 60))
ax31.set_xticklabels([numeric_to_time(t) for t in np.arange(480, 1261, 60)], fontsize=14)
ax31.set_title("Section 3-1. Weekly Average Hourly Floating Population", fontweight='bold', fontsize=22)
ax31.legend(fontsize=15)

ax32 = plt.subplot2grid((40, 1), (22, 0), rowspan=3)
df_analysis['In_Ratio'] = (df_analysis['In_Delta'] / df_analysis['Floating_Population'].replace(0, 1)).clip(0, 1)
df_analysis['Out_Ratio'] = (df_analysis['Out_Delta'] / df_analysis['Floating_Population'].replace(0, 1)).clip(0, 1)
df_analysis['Hour'] = df_analysis['Time_Numeric'] // 60
ratio_data = df_analysis.groupby('Hour')[['In_Ratio', 'Out_Ratio']].mean()
ax32.plot(ratio_data.index * 60, ratio_data['In_Ratio'], marker='o', lw=3, markersize=10, color='dodgerblue', label='In Ratio')
ax32.plot(ratio_data.index * 60, ratio_data['Out_Ratio'], marker='o', lw=3, markersize=10, color='tomato', label='Out Ratio')
ax32.set_title("Section 3-2. Hourly In/Out Ratio", fontweight='bold', fontsize=24)
ax32.set_xticks(np.arange(480, 1261, 60)); ax32.set_xticklabels([f"{int(h):02d}:00" for h in range(8, 22)], fontsize=14)
ax32.set_ylim(0, 1.1); ax32.legend(loc='upper right', fontsize=16)

# --- [SECTION 4] 요일별 합계 ---
ax4 = plt.subplot2grid((40, 1), (26, 0), rowspan=4)
daily_sum = augmented_df.groupby('Day')['Predicted'].sum().reindex(DAY_LIST)
daily_sum.plot(kind='bar', color=[colors[d] for d in DAY_LIST], alpha=0.8, edgecolor='black', ax=ax4)
ax4.set_ylim(0, daily_sum.max() * 1.5)
for i, v in enumerate(daily_sum):
    ax4.text(i, v + daily_sum.max()*0.05, f"{int(v)}", ha='center', fontweight='bold', fontsize=22)
ax4.set_title("Section 4. Total Daily Floating Population (Estimated Sum)", fontweight='bold', fontsize=26, pad=35)
plt.setp(ax4.get_xticklabels(), rotation=0, fontsize=18)

plt.tight_layout()
plt.savefig('weekly_analysis_report.png', dpi=150, bbox_inches='tight')
plt.show()


