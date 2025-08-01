#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
放流量予測の詳細確認
"""

import pandas as pd
from datetime import datetime, timedelta
from discharge_prediction_model_v2 import DischargePredictionModelV2

# データ読み込み
df = pd.read_csv('統合データ_水位ダム_20250730_205325.csv', encoding='utf-8')
df['時刻'] = pd.to_datetime(df['時刻'])

# テスト時刻
test_time = pd.to_datetime('2023-06-30 22:00')

# 過去データ
historical_mask = df['時刻'] <= test_time
historical_data = df[historical_mask].copy()

# 1時間先までの降雨予測
future_1h_mask = (df['時刻'] > test_time) & \
                 (df['時刻'] <= test_time + timedelta(hours=1))
future_1h_data = df[future_1h_mask]

rainfall_forecast = pd.DataFrame({
    '時刻': future_1h_data['時刻'],
    '降雨強度': future_1h_data['ダム_60分雨量']
})

print("1時間先までの降雨予測:")
for _, row in rainfall_forecast.iterrows():
    print(f"  {row['時刻'].strftime('%H:%M')} - {row['降雨強度']:.1f} mm/h")

# モデル予測
model = DischargePredictionModelV2()
predictions = model.predict(
    test_time, 
    historical_data,
    prediction_hours=3,
    rainfall_forecast=rainfall_forecast
)

print("\n放流量予測結果:")
print("時刻     放流量  変化率  使用降雨  状態")
print("-" * 45)
for i, row in predictions.iterrows():
    time_str = row['時刻'].strftime('%H:%M')
    discharge = row['予測放流量']
    change = row['変化率']
    rainfall = row['使用降雨強度']
    state = row['状態']
    
    state_str = {1: "増加", 0: "定常", -1: "減少"}[state]
    
    print(f"{time_str}  {discharge:6.1f}  {change:+6.1f}  {rainfall:6.1f}  {state_str}")
    
    if i == 5:  # 1時間後
        print("-" * 45 + " 1時間経過")

# 変化の傾向を分析
print("\n放流量の変化傾向:")
initial_discharge = historical_data.iloc[-1]['ダム_全放流量']
print(f"現在: {initial_discharge:.1f} m³/s")

for hours in [1, 2, 3]:
    idx = hours * 6 - 1
    if idx < len(predictions):
        discharge = predictions.iloc[idx]['予測放流量']
        change = discharge - initial_discharge
        print(f"{hours}時間後: {discharge:.1f} m³/s (変化: {change:+.1f})")

# 使用された降雨強度の分析
print("\n使用降雨強度の分析:")
rainfall_values = predictions['使用降雨強度'].unique()
print(f"ユニークな値: {sorted(rainfall_values)}")

# 1時間後以降の降雨
after_1h = predictions.iloc[6:]
print(f"\n1時間後以降の使用降雨強度:")
print(f"最小: {after_1h['使用降雨強度'].min():.1f} mm/h")
print(f"最大: {after_1h['使用降雨強度'].max():.1f} mm/h")
print(f"平均: {after_1h['使用降雨強度'].mean():.1f} mm/h")