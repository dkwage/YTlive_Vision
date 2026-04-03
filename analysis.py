import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold, cross_val_score
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
df_raw = pd.read_csv('merged_data.csv').dropna(subset=['Day'])
df_raw['Time_Numeric'] = df_raw['Time_Slot'].apply(time_to_min)
OFFICIAL_SLOTS = [540, 630, 720, 810, 900, 990, 1080, 1170]

# 피크 시차 분석
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

# 2. 데이터 준비 (0 포함)
df_raw['Class_Density'] = df_raw.apply(lambda r: get_calibrated_density(r['Day'], r['Time_Numeric']), axis=1)
df_analysis = df_raw.copy()

DAY_LIST = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore').fit(pd.DataFrame({'Day': DAY_LIST}))
X_raw = df_analysis[['Time_Numeric', 'Class_Density']]
X_day = pd.DataFrame(encoder.transform(df_analysis[['Day']]), columns=encoder.get_feature_names_out(['Day']), index=df_analysis.index)
X_combined = pd.concat([X_raw, X_day], axis=1)
y = df_analysis['Floating_Population']

# K-Fold 검증 및 최종 학습
kf = KFold(n_splits=5, shuffle=True, random_state=42)
rf_model = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)
cv_scores = cross_val_score(rf_model, X_combined, y, cv=kf, scoring='r2')
avg_cv_r2 = cv_scores.mean()
rf_model.fit(X_combined, y)

# 3. 전구간 예측 데이터 생성
time_range = np.arange(480, 1261, 1)
full_preds = []
for day in DAY_LIST:
    day_df = pd.DataFrame({'Day': [day]*len(time_range), 'Time_Numeric': time_range})
    day_df['Class_Density'] = day_df['Time_Numeric'].apply(lambda x: get_calibrated_density(day, x))
    X_p = pd.concat([day_df[['Time_Numeric', 'Class_Density']], 
                    pd.DataFrame(encoder.transform(day_df[['Day']]), columns=encoder.get_feature_names_out(['Day']), index=day_df.index)], axis=1)
    day_df['Predicted'] = rf_model.predict(X_p)
    tree_preds = np.array([tree.predict(X_p) for tree in rf_model.estimators_])
    std = np.std(tree_preds, axis=0)
    day_df['Lower'] = np.maximum(0, day_df['Predicted'] - 1.645 * std)
    day_df['Upper'] = day_df['Predicted'] + 1.645 * std
    day_df['Uncertainty'] = std 
    full_preds.append(day_df)

augmented_df = pd.concat(full_preds)

# ---------------------------------------------------------
# 4. 시각화 (레이아웃 정정 반영)
# ---------------------------------------------------------
fig = plt.figure(figsize=(24, 85), facecolor='white') # 수직 배치를 위해 높이 확장
colors = {"Monday": "#3498db", "Tuesday": "#e67e22", "Wednesday": "#2ecc71", "Thursday": "#9b59b6", "Friday": "#e74c3c"}

# --- [SECTION 1] 요일별 예상 추정치 ---
for i, day in enumerate(DAY_LIST):
    ax = plt.subplot2grid((40, 1), (i*2, 0), rowspan=2)
    d = augmented_df[augmented_df['Day'] == day]
    ax.plot(d['Time_Numeric'], d['Predicted'], color=colors[day], lw=3, label=f'{day} RF Prediction')
    ax.fill_between(d['Time_Numeric'], d['Lower'], d['Upper'], color=colors[day], alpha=0.15, label='90% Confidence Interval')
    
    real = df_raw[df_raw['Day'] == day]
    if not real.empty:
        ax.scatter(real['Time_Numeric'], real['Floating_Population'], color='red', s=70, edgecolors='black', label='Actual Data', zorder=5)
    
    ax.set_title(f"Section 1-{i+1}. {day} Population Estimation", fontweight='bold', fontsize=22)
    ax.set_xticks(np.arange(480, 1261, 60))
    ax.set_xticklabels([numeric_to_time(t) for t in np.arange(480, 1261, 60)], fontsize=14)
    ax.legend(loc='upper right', fontsize=15)

# --- [SECTION 2] 분석 지표 (수직 1열 배치) ---
# 2-1. 모델 vs 실제 회귀선
ax21 = plt.subplot2grid((40, 1), (11, 0), rowspan=2)
sns.regplot(x=y, y=rf_model.predict(X_combined), ax=ax21, scatter_kws={'alpha':0.4, 'color':'teal'}, line_kws={'color':'red', 'label':'Regression Line'})
ax21.text(0.05, 0.9, f'K-Fold Avg $R^2$: {avg_cv_r2:.3f}', transform=ax21.transAxes, fontsize=18, fontweight='bold', bbox=dict(facecolor='white', alpha=0.8))
ax21.set_title("Section 2-1. Actual vs. Predicted Correlation (Regression)", fontweight='bold', fontsize=22)
ax21.legend(fontsize=15)

# 2-2. 모델 Uncertainty (시간별)
ax22 = plt.subplot2grid((40, 1), (13, 0), rowspan=2)
avg_unc = augmented_df.groupby('Time_Numeric')['Uncertainty'].mean()
ax22.fill_between(avg_unc.index, 0, avg_unc.values, color='gray', alpha=0.4, label='Prediction Uncertainty (Std)')
ax22.set_xticks(np.arange(480, 1261, 60)); ax22.set_xticklabels([numeric_to_time(t) for t in np.arange(480, 1261, 60)], fontsize=14)
ax22.set_title("Section 2-2. Prediction Uncertainty over Time", fontweight='bold', fontsize=22)
ax22.legend(fontsize=15)

# 2-3. 트래픽 vs 밀집도 박스 플롯
ax23 = plt.subplot2grid((40, 1), (15, 0), rowspan=3)
df_analysis['Density_Bin'] = pd.cut(df_analysis['Class_Density'], bins=[-1, 5, 50, 100, 150, 200])
sns.boxplot(x='Density_Bin', y='Floating_Population', data=df_analysis, ax=ax23, palette='coolwarm')
ax23.set_title("Section 2-4. Floating Population Distribution by Density Bin", fontweight='bold', fontsize=22)
plt.setp(ax23.get_xticklabels(), fontsize=14)

# --- [SECTION 3] ---
# 3-1. 일주일 평균 예상 유동인구
ax31 = plt.subplot2grid((40, 1), (19, 0), rowspan=3)
avg_weekly = augmented_df.groupby('Time_Numeric')['Predicted'].mean()
ax31.plot(avg_weekly.index, avg_weekly.values, color='black', lw=3, label='Weekly Average Flow')
ax31.set_xticks(np.arange(480, 1261, 60))
ax31.set_xticklabels([numeric_to_time(t) for t in np.arange(480, 1261, 60)], fontsize=14)
ax31.set_title("Section 3-1. Weekly Average Estimated Population Flow", fontweight='bold', fontsize=22)
ax31.legend(fontsize=15)

# 3-2. 시간별 In/Out 비율
ax32 = plt.subplot2grid((40, 1), (22, 0), rowspan=3)
df_analysis['In_Ratio'] = (df_analysis['In_Delta'] / df_analysis['Floating_Population'].replace(0, 1)).clip(0, 1)
df_analysis['Out_Ratio'] = (df_analysis['Out_Delta'] / df_analysis['Floating_Population'].replace(0, 1)).clip(0, 1)
df_analysis['Hour'] = df_analysis['Time_Numeric'] // 60
ratio_data = df_analysis.groupby('Hour')[['In_Ratio', 'Out_Ratio']].mean()

ax32.plot(ratio_data.index * 60, ratio_data['In_Ratio'], marker='o', lw=3, markersize=10, color='dodgerblue', label='Entering Ratio (In)')
ax32.plot(ratio_data.index * 60, ratio_data['Out_Ratio'], marker='o', lw=3, markersize=10, color='tomato', label='Exiting Ratio (Out)')
ax32.set_title("Section 3-2. Hourly In/Out Ratio relative to Total Population", fontweight='bold', fontsize=24)
ax32.set_xticks(np.arange(480, 1261, 60)); ax32.set_xticklabels([f"{int(h):02d}:00" for h in range(8, 22)], fontsize=14)
ax32.set_ylim(0, 1.1); ax32.legend(loc='upper right', fontsize=16); ax32.set_ylabel("Ratio", fontsize=16)

# --- [SECTION 4] 요일별 예상 유동인구 합계 (상단 여백 확장) ---
ax4 = plt.subplot2grid((40, 1), (26, 0), rowspan=4)
daily_sum = augmented_df.groupby('Day')['Predicted'].sum().reindex(DAY_LIST)
daily_sum.plot(kind='bar', color=[colors[d] for d in DAY_LIST], alpha=0.8, edgecolor='black', ax=ax4)

# Y축 상한선을 최대값의 1.4배로 늘려 숫자 공간 확보
ax4.set_ylim(0, daily_sum.max() * 1.4) 

for i, v in enumerate(daily_sum):
    ax4.text(i, v + daily_sum.max()*0.05, f"{int(v)}", ha='center', fontweight='bold', fontsize=22)

ax4.set_title("Section 4. Total Daily Floating Population (Estimated Sum)", fontweight='bold', fontsize=26, pad=35)
ax4.set_ylabel("Daily Total Sum", fontsize=18)
plt.setp(ax4.get_xticklabels(), rotation=0, fontsize=18)

plt.tight_layout()
plt.savefig('weekly_floating_population.png', dpi=150, bbox_inches='tight')
plt.show()