#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
2023/7/1 00:00時点でのハイブリッドモデル適用分析
水位予測デモと統合
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from water_level_prediction_model_v2 import WaterLevelPredictorV2
from discharge_prediction_model_v3_hybrid import DischargePredictionModelV3Hybrid

# 日本語フォント設定
plt.rcParams['font.sans-serif'] = ['Yu Gothic', 'Hiragino Sans', 'Meirio', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# データ読み込み
df = pd.read_csv('統合データ_水位ダム_20250730_142903.csv', encoding='utf-8')
df['時刻'] = pd.to_datetime(df['時刻'])

# 予測時点
current_time = pd.to_datetime('2023-07-01 00:00')

print(f"=== 2023/7/1 00:00時点のハイブリッドモデル分析 ===")

# 過去データの準備
historical_mask = df['時刻'] <= current_time
historical_data = df[historical_mask].copy()

# 現在の状態
current_water_level = historical_data.iloc[-1]['水位_水位']
current_discharge = historical_data.iloc[-1]['ダム_全放流量']
current_rainfall = historical_data.iloc[-1]['ダム_60分雨量']

print(f"\n現在の状態:")
print(f"  水位: {current_water_level:.2f} m")
print(f"  放流量: {current_discharge:.1f} m³/s")
print(f"  降雨強度: {current_rainfall:.1f} mm/h")

# 降雨予測（1時間先まで）
future_1h_mask = (df['時刻'] > current_time) & (df['時刻'] <= current_time + timedelta(hours=1))
future_1h_data = df[future_1h_mask]

rainfall_forecast = pd.DataFrame({
    '時刻': future_1h_data['時刻'],
    '降雨強度': future_1h_data['ダム_60分雨量']
})

print("\n降雨予測（1時間先まで）:")
for _, row in rainfall_forecast.iterrows():
    print(f"  {row['時刻'].strftime('%H:%M')}: {row['降雨強度']:.1f} mm/h")

# ハイブリッドモデルによる放流量予測
print("\n放流量予測（ハイブリッドモデル）...")
discharge_model = DischargePredictionModelV3Hybrid()
discharge_model.load_model('discharge_prediction_model_v3_hybrid_20250801_231150.pkl')

discharge_predictions = discharge_model.predict(
    current_time, 
    historical_data, 
    prediction_hours=3,
    rainfall_forecast=rainfall_forecast
)

# 水位予測モデル
print("水位予測...")
water_model = WaterLevelPredictorV2()
water_model.load_model('water_level_predictor_v2_20250731_143150.pkl')

# 予測データの準備
future_data = []
for i, row in discharge_predictions.iterrows():
    future_data.append({
        '時刻': row['時刻'],
        'ダム_全放流量': row['予測放流量'],
        'ダム_60分雨量': row['使用降雨強度']  # 遅延考慮後の降雨
    })

future_df = pd.DataFrame(future_data)

# 水位予測（各時点で予測）
water_predictions = []
current_level = current_water_level

for i in range(len(future_df)):
    # 予測用のデータを準備
    prediction_data = {
        '時刻': future_df.iloc[i]['時刻'],
        '水位_水位': current_level,  # 前回の予測値を使用
        'ダム_全放流量': future_df.iloc[i]['ダム_全放流量'],
        'ダム_60分雨量': future_df.iloc[i]['ダム_60分雨量']
    }
    
    # 過去データと結合
    temp_df = pd.concat([historical_data.iloc[-50:], pd.DataFrame([prediction_data])], ignore_index=True)
    
    # 予測実行
    pred_level = water_model.predict_water_level(temp_df, len(temp_df)-1)
    water_predictions.append(pred_level)
    current_level = pred_level  # 次の予測用に更新

# 結果の分析
print("\n=== 予測結果の詳細分析 ===")
print("\n最初の1時間の予測:")
print("時刻     放流量   変化   遅延   ML調整(遅延) ML調整(率)  水位")
print("-" * 70)

for i in range(min(6, len(discharge_predictions))):
    d_row = discharge_predictions.iloc[i]
    w_pred = water_predictions[i] if i < len(water_predictions) else np.nan
    
    change = d_row['予測放流量'] - (discharge_predictions.iloc[i-1]['予測放流量'] if i > 0 else current_discharge)
    
    print(f"{d_row['時刻'].strftime('%H:%M')}  {d_row['予測放流量']:6.1f}  {change:+5.1f}  {d_row['適用遅延']:3.0f}分    {d_row['ML調整係数_遅延']:5.2f}     {d_row['ML調整係数_変化率']:5.2f}    {w_pred:5.2f}")

# 重要な洞察
print("\n=== ハイブリッドモデルの特徴 ===")
print("\n1. 遅延時間の動的調整:")
unique_delays = discharge_predictions['適用遅延'].unique()
for delay in sorted(unique_delays):
    count = (discharge_predictions['適用遅延'] == delay).sum()
    print(f"   {delay:3.0f}分: {count}回")

print("\n2. ML調整係数の統計:")
print(f"   遅延調整: 平均 {discharge_predictions['ML調整係数_遅延'].mean():.2f}倍, 範囲 {discharge_predictions['ML調整係数_遅延'].min():.2f}～{discharge_predictions['ML調整係数_遅延'].max():.2f}")
print(f"   変化率調整: 平均 {discharge_predictions['ML調整係数_変化率'].mean():.2f}倍, 範囲 {discharge_predictions['ML調整係数_変化率'].min():.2f}～{discharge_predictions['ML調整係数_変化率'].max():.2f}")

# 実績データとの比較
future_3h_mask = (df['時刻'] > current_time) & (df['時刻'] <= current_time + timedelta(hours=3))
actual_future = df[future_3h_mask]

# グラフ作成
fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)

# 表示範囲
plot_start = current_time - timedelta(hours=1)
plot_end = current_time + timedelta(hours=3)

# 1. 水位
ax1 = axes[0]
# 過去と実績
past_mask = (df['時刻'] >= plot_start) & (df['時刻'] <= current_time)
past_data = df[past_mask]
ax1.plot(past_data['時刻'], past_data['水位_水位'], 'k-', label='実績', linewidth=2)
ax1.plot(actual_future['時刻'], actual_future['水位_水位'], 'k--', label='実績（予測期間）', linewidth=2)

# 予測
pred_times = [current_time + timedelta(minutes=i*10) for i in range(1, len(water_predictions)+1)]
ax1.plot(pred_times, water_predictions, 'r-', label='予測（ハイブリッド）', linewidth=2, marker='o', markersize=4)

ax1.axvline(x=current_time, color='gray', linestyle='--', alpha=0.5)
ax1.axhline(y=5.5, color='red', linestyle=':', alpha=0.5, label='避難判断水位')
ax1.set_ylabel('水位 (m)')
ax1.set_title(f'ハイブリッドモデルによる水位・放流量予測 - {current_time.strftime("%Y年%m月%d日 %H:%M")}時点')
ax1.legend(loc='upper left')
ax1.grid(True, alpha=0.3)

# 2. 放流量
ax2 = axes[1]
ax2.plot(past_data['時刻'], past_data['ダム_全放流量'], 'k-', label='実績', linewidth=2)
ax2.plot(actual_future['時刻'], actual_future['ダム_全放流量'], 'k--', label='実績（予測期間）', linewidth=2)
ax2.plot(discharge_predictions['時刻'], discharge_predictions['予測放流量'], 'r-', 
         label='予測（ハイブリッド）', linewidth=2, marker='o', markersize=4)
ax2.axvline(x=current_time, color='gray', linestyle='--', alpha=0.5)
ax2.set_ylabel('放流量 (m³/s)')
ax2.legend(loc='upper left')
ax2.grid(True, alpha=0.3)

# 3. 遅延時間とML調整
ax3 = axes[2]
ax3.plot(discharge_predictions['時刻'], discharge_predictions['適用遅延'], 
         'g-', label='適用遅延時間', linewidth=2, marker='s', markersize=6)
ax3.axvline(x=current_time, color='gray', linestyle='--', alpha=0.5)
ax3.set_ylabel('遅延時間 (分)', color='green')
ax3.tick_params(axis='y', labelcolor='green')
ax3.set_ylim(0, 130)

# ML調整係数（右軸）
ax3_twin = ax3.twinx()
ax3_twin.plot(discharge_predictions['時刻'], discharge_predictions['ML調整係数_変化率'], 
              'm-', label='ML変化率調整', linewidth=2, marker='^', markersize=6)
ax3_twin.set_ylabel('ML調整係数', color='magenta')
ax3_twin.tick_params(axis='y', labelcolor='magenta')
ax3_twin.set_ylim(0.5, 1.5)

ax3.legend(loc='upper left')
ax3_twin.legend(loc='upper right')
ax3.grid(True, alpha=0.3)

# 4. 降雨強度
ax4 = axes[3]
# 実績降雨
all_data_mask = (df['時刻'] >= plot_start) & (df['時刻'] <= plot_end)
all_data = df[all_data_mask]
ax4.bar(all_data['時刻'], all_data['ダム_60分雨量'], 
        width=0.007, alpha=0.7, color='blue', label='降雨強度')

# 使用降雨（遅延考慮後）
ax4.scatter(discharge_predictions['時刻'], discharge_predictions['使用降雨強度'], 
           color='red', s=50, zorder=5, label='モデル使用降雨')

ax4.axvline(x=current_time, color='gray', linestyle='--', alpha=0.5)
ax4.set_xlabel('時刻')
ax4.set_ylabel('降雨強度 (mm/h)')
ax4.legend()
ax4.grid(True, alpha=0.3)

plt.tight_layout()
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
filename = f'hybrid_model_analysis_20230701_0000_{timestamp}.png'
plt.savefig(filename, dpi=150, bbox_inches='tight')
plt.close()

print(f"\n分析グラフを保存: {filename}")

# 予測精度の評価（可能な範囲で）
if len(actual_future) > 0:
    print("\n=== 予測精度評価 ===")
    
    # 放流量の精度
    common_times = []
    pred_values = []
    actual_values = []
    
    for _, pred_row in discharge_predictions.iterrows():
        pred_time = pred_row['時刻']
        actual_row = actual_future[actual_future['時刻'] == pred_time]
        if not actual_row.empty:
            common_times.append(pred_time)
            pred_values.append(pred_row['予測放流量'])
            actual_values.append(actual_row.iloc[0]['ダム_全放流量'])
    
    if common_times:
        pred_array = np.array(pred_values)
        actual_array = np.array(actual_values)
        
        mae = np.mean(np.abs(pred_array - actual_array))
        rmse = np.sqrt(np.mean((pred_array - actual_array)**2))
        
        print(f"\n放流量予測精度（{len(common_times)}点）:")
        print(f"  MAE: {mae:.1f} m³/s")
        print(f"  RMSE: {rmse:.1f} m³/s")
        
        # 時点別の誤差
        print("\n時点別の誤差:")
        for i, time in enumerate(common_times[:6]):
            error = pred_values[i] - actual_values[i]
            print(f"  {time.strftime('%H:%M')}: 予測 {pred_values[i]:.1f}, 実績 {actual_values[i]:.1f}, 誤差 {error:+.1f} m³/s")