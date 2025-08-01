#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
30分先までの予測を考慮した遅延時間決定のテスト
"""

import pandas as pd
from datetime import datetime, timedelta
from discharge_prediction_model_v2 import DischargePredictionModelV2

# データ読み込み
df = pd.read_csv('統合データ_水位ダム_20250730_205325.csv', encoding='utf-8')
df['時刻'] = pd.to_datetime(df['時刻'])

# テストケース1: 2023-07-01 02:00（高放流量、降雨減少中）
print("=== テストケース1: 2023-07-01 02:00 ===")
test_time1 = pd.to_datetime('2023-07-01 02:00')

historical_mask1 = df['時刻'] <= test_time1
historical_data1 = df[historical_mask1].copy()

# 1時間先までの降雨予測
future_1h_mask1 = (df['時刻'] > test_time1) & (df['時刻'] <= test_time1 + timedelta(hours=1))
future_1h_data1 = df[future_1h_mask1]

rainfall_forecast1 = pd.DataFrame({
    '時刻': future_1h_data1['時刻'],
    '降雨強度': future_1h_data1['ダム_60分雨量']
})

print(f"現在時刻: {test_time1}")
print(f"現在の放流量: {historical_data1.iloc[-1]['ダム_全放流量']:.1f} m³/s")
print(f"現在の降雨: {historical_data1.iloc[-1]['ダム_60分雨量']:.1f} mm/h")

print("\n30分先までの降雨予測:")
for i, row in rainfall_forecast1.iloc[:3].iterrows():
    print(f"  {row['時刻'].strftime('%H:%M')} - {row['降雨強度']:.1f} mm/h")

# モデル初期化と予測
model = DischargePredictionModelV2()
predictions1 = model.predict(test_time1, historical_data1, 
                           prediction_hours=1, 
                           rainfall_forecast=rainfall_forecast1)

print("\n予測結果（最初の30分）:")
print("時刻     放流量  変化率  適用遅延")
print("-" * 35)
for _, row in predictions1.iloc[:3].iterrows():
    print(f"{row['時刻'].strftime('%H:%M')}  {row['予測放流量']:6.1f}  {row['変化率']:+6.1f}  {row['適用遅延']:4.0f}分")

# テストケース2: 2023-06-30 22:00（中放流量、低降雨）
print("\n\n=== テストケース2: 2023-06-30 22:00 ===")
test_time2 = pd.to_datetime('2023-06-30 22:00')

historical_mask2 = df['時刻'] <= test_time2
historical_data2 = df[historical_mask2].copy()

# 1時間先までの降雨予測
future_1h_mask2 = (df['時刻'] > test_time2) & (df['時刻'] <= test_time2 + timedelta(hours=1))
future_1h_data2 = df[future_1h_mask2]

rainfall_forecast2 = pd.DataFrame({
    '時刻': future_1h_data2['時刻'],
    '降雨強度': future_1h_data2['ダム_60分雨量']
})

print(f"現在時刻: {test_time2}")
print(f"現在の放流量: {historical_data2.iloc[-1]['ダム_全放流量']:.1f} m³/s")
print(f"現在の降雨: {historical_data2.iloc[-1]['ダム_60分雨量']:.1f} mm/h")

print("\n30分先までの降雨予測:")
for i, row in rainfall_forecast2.iloc[:3].iterrows():
    print(f"  {row['時刻'].strftime('%H:%M')} - {row['降雨強度']:.1f} mm/h")

# 予測
predictions2 = model.predict(test_time2, historical_data2, 
                           prediction_hours=1, 
                           rainfall_forecast=rainfall_forecast2)

print("\n予測結果（最初の30分）:")
print("時刻     放流量  変化率  適用遅延")
print("-" * 35)
for _, row in predictions2.iloc[:3].iterrows():
    print(f"{row['時刻'].strftime('%H:%M')}  {row['予測放流量']:6.1f}  {row['変化率']:+6.1f}  {row['適用遅延']:4.0f}分")

# テストケース3: 2023-06-30 23:30（降雨急増開始時）
print("\n\n=== テストケース3: 2023-06-30 23:30 ===")
test_time3 = pd.to_datetime('2023-06-30 23:30')

historical_mask3 = df['時刻'] <= test_time3
historical_data3 = df[historical_mask3].copy()

# 1時間先までの降雨予測
future_1h_mask3 = (df['時刻'] > test_time3) & (df['時刻'] <= test_time3 + timedelta(hours=1))
future_1h_data3 = df[future_1h_mask3]

rainfall_forecast3 = pd.DataFrame({
    '時刻': future_1h_data3['時刻'],
    '降雨強度': future_1h_data3['ダム_60分雨量']
})

print(f"現在時刻: {test_time3}")
print(f"現在の放流量: {historical_data3.iloc[-1]['ダム_全放流量']:.1f} m³/s")
print(f"現在の降雨: {historical_data3.iloc[-1]['ダム_60分雨量']:.1f} mm/h")

print("\n30分先までの降雨予測:")
for i, row in rainfall_forecast3.iloc[:3].iterrows():
    print(f"  {row['時刻'].strftime('%H:%M')} - {row['降雨強度']:.1f} mm/h")

# 予測
predictions3 = model.predict(test_time3, historical_data3, 
                           prediction_hours=1, 
                           rainfall_forecast=rainfall_forecast3)

print("\n予測結果（最初の30分）:")
print("時刻     放流量  変化率  適用遅延")
print("-" * 35)
for _, row in predictions3.iloc[:3].iterrows():
    print(f"{row['時刻'].strftime('%H:%M')}  {row['予測放流量']:6.1f}  {row['変化率']:+6.1f}  {row['適用遅延']:4.0f}分")

print("\n\n=== まとめ ===")
print("30分先の予測を考慮することで：")
print("1. 降雨が今後増加する場合 → 遅延時間を短く（0分）")
print("2. 降雨が継続的に低い場合 → 減少の遅延時間（60分）")
print("3. 状態遷移が予測される場合 → 適切な遅延時間を選択")