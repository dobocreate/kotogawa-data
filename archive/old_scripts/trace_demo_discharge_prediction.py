#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
デモプログラムの放流量予測をトレース
"""

import pandas as pd
from datetime import datetime, timedelta
from discharge_prediction_model_v2 import DischargePredictionModelV2
import matplotlib.pyplot as plt

# データ読み込み（デモと同じ統合データを使用）
full_data = pd.read_csv('統合データ_水位ダム_20250730_142903.csv', encoding='utf-8')
full_data['時刻'] = pd.to_datetime(full_data['時刻'])

current_time = pd.to_datetime('2023-07-01 00:00')
full_idx = full_data['時刻'].searchsorted(current_time)

# 降雨予測データの準備（デモと同じ）
future_1h_mask = (full_data['時刻'] > current_time) & \
               (full_data['時刻'] <= current_time + pd.Timedelta(hours=1))
future_1h_data = full_data[future_1h_mask]

rainfall_forecast = pd.DataFrame({
    '時刻': future_1h_data['時刻'],
    '降雨強度': future_1h_data['ダム_60分雨量']
})

print("=== デモプログラムの放流量予測をトレース ===")
print(f"\n使用するデータファイル: 統合データ_水位ダム_20250730_142903.csv")
print(f"現在時刻: {current_time}")
print(f"現在のインデックス: {full_idx}")

# モデル読み込み（最新モデル）
model = DischargePredictionModelV2()
model.load_model('discharge_prediction_model_v2_20250801_120003.pkl')

# 放流量予測を実行（デモと同じ）
predicted_discharge = model.predict(
    current_time, 
    full_data.iloc[:full_idx+1],
    prediction_hours=3,
    rainfall_forecast=rainfall_forecast
)

print("\n最初の1時間の予測結果:")
print("時刻     放流量  変化率  使用降雨  適用遅延")
print("-" * 45)
for i in range(min(6, len(predicted_discharge))):
    row = predicted_discharge.iloc[i]
    time_str = row['時刻'].strftime('%H:%M')
    discharge = row['予測放流量']
    change = row['変化率']
    rainfall = row['使用降雨強度']
    delay = row['適用遅延']
    
    print(f"{time_str}  {discharge:6.1f}  {change:+6.1f}  {rainfall:6.1f}    {delay:3.0f}分")

# グラフ作成
plt.figure(figsize=(12, 6))
actual_mask = (full_data['時刻'] >= current_time - timedelta(hours=3)) & \
              (full_data['時刻'] <= current_time + timedelta(hours=3))
actual_data = full_data[actual_mask]

plt.plot(actual_data['時刻'], actual_data['ダム_全放流量'], 'k-', linewidth=2, label='実績')
plt.plot(predicted_discharge['時刻'], predicted_discharge['予測放流量'], 'b--', linewidth=2, label='予測')
plt.axvline(x=current_time, color='red', linestyle='--', alpha=0.7, label='現在時刻')
plt.xlabel('時刻')
plt.ylabel('放流量 (m³/s)')
plt.title('デモと同じ条件での放流量予測')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('trace_demo_discharge_20250801.png', dpi=150, bbox_inches='tight')
plt.close()

print("\nグラフを保存: trace_demo_discharge_20250801.png")