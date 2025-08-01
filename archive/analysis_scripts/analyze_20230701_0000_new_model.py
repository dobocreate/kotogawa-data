#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
2023-07-01 00:00の新モデルでの詳細分析
"""

import pandas as pd
from datetime import datetime, timedelta
from discharge_prediction_model_v2 import DischargePredictionModelV2

# データ読み込み
df = pd.read_csv('統合データ_水位ダム_20250730_205325.csv', encoding='utf-8')
df['時刻'] = pd.to_datetime(df['時刻'])

test_time = pd.to_datetime('2023-07-01 00:00')

# 過去データの準備
historical_mask = df['時刻'] <= test_time
historical_data = df[historical_mask].copy()

print("=== 2023-07-01 00:00 新モデルの分析 ===")
print(f"\n現在時刻: {test_time}")
print(f"現在の放流量: {historical_data.iloc[-1]['ダム_全放流量']:.1f} m³/s")
print(f"現在の降雨: {historical_data.iloc[-1]['ダム_60分雨量']:.1f} mm/h")

# 過去の放流量・降雨履歴
print("\n過去1時間の履歴:")
for i in range(7):
    idx = -7 + i
    time = historical_data.iloc[idx]['時刻']
    discharge = historical_data.iloc[idx]['ダム_全放流量']
    rainfall = historical_data.iloc[idx]['ダム_60分雨量']
    print(f"  {time.strftime('%H:%M')} - 放流量: {discharge:6.1f} m³/s, 降雨: {rainfall:5.1f} mm/h")

# 将来の降雨予測（実データ）
future_1h_mask = (df['時刻'] > test_time) & (df['時刻'] <= test_time + timedelta(hours=1))
future_1h_data = df[future_1h_mask]

print("\n1時間先までの降雨予測（実データ）:")
for _, row in future_1h_data.iterrows():
    print(f"  {row['時刻'].strftime('%H:%M')} - {row['ダム_60分雨量']:.1f} mm/h")

# モデルで状態判定
model = DischargePredictionModelV2()
discharge_history = historical_data.iloc[-7:]['ダム_全放流量']
rainfall_history = historical_data.iloc[-7:]['ダム_60分雨量']

state, rate = model.identify_discharge_state(discharge_history)
print(f"\n状態判定:")
print(f"  過去30分の変化: {discharge_history.iloc[-1] - discharge_history.iloc[-4]:.1f} m³/s")
print(f"  判定結果: {['減少', '定常', '増加'][state+1]} (変化率: {rate:.1f} m³/s/10min)")

# 30分先までの簡易予測
print("\n30分先までの予測の分析:")
print("  30分先までの降雨平均:")
rainfall_30min = []
for i in range(1, 4):
    future_time = test_time + timedelta(minutes=i * 10)
    idx = future_1h_data['時刻'].searchsorted(future_time)
    if idx < len(future_1h_data):
        rain = future_1h_data.iloc[idx]['ダム_60分雨量']
        rainfall_30min.append(rain)
        print(f"    {future_time.strftime('%H:%M')} - {rain:.1f} mm/h")

if rainfall_30min:
    avg_rainfall_30min = sum(rainfall_30min) / len(rainfall_30min)
    print(f"  30分平均: {avg_rainfall_30min:.1f} mm/h")
    
    # 30分先の放流量変化予測
    print(f"\n  現在の降雨: {historical_data.iloc[-1]['ダム_60分雨量']:.1f} mm/h")
    print(f"  30分先平均: {avg_rainfall_30min:.1f} mm/h")
    
    if avg_rainfall_30min > 40:
        print("  → 高降雨が続くため、増加傾向が予測される")
    elif avg_rainfall_30min > 20:
        print("  → 中程度の降雨が続くため、緩やかな増加が予測される")
    else:
        print("  → 低降雨のため、現状維持または減少が予測される")

# 実際の放流量変化を確認
print("\n実際の放流量変化（答え合わせ）:")
actual_future = df[(df['時刻'] > test_time) & (df['時刻'] <= test_time + timedelta(hours=3))]
for _, row in actual_future.iloc[::6].iterrows():  # 1時間ごと
    time_diff = (row['時刻'] - test_time).total_seconds() / 3600
    print(f"  +{time_diff:.0f}時間後: {row['ダム_全放流量']:.1f} m³/s")