#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
モデル統合の確認
"""

import os
import pandas as pd
from datetime import datetime, timedelta
from discharge_prediction_model_v2 import DischargePredictionModelV2

# 最新のモデルファイルを確認
model_files = [f for f in os.listdir('.') if f.startswith('discharge_prediction_model_v2_') and f.endswith('.pkl')]
model_files.sort(key=lambda x: os.path.getmtime(x))

print("=== 利用可能なモデルファイル ===")
for f in model_files[-5:]:
    mtime = os.path.getmtime(f)
    print(f"{f} - {datetime.fromtimestamp(mtime).strftime('%Y-%m-%d %H:%M:%S')}")

print(f"\n最新モデル: {model_files[-1]}")

# データ読み込み
df = pd.read_csv('統合データ_水位ダム_20250730_205325.csv', encoding='utf-8')
df['時刻'] = pd.to_datetime(df['時刻'])

test_time = pd.to_datetime('2023-07-01 00:00')

# 過去データの準備
historical_mask = df['時刻'] <= test_time
historical_data = df[historical_mask].copy()

# 降雨予測
future_1h_mask = (df['時刻'] > test_time) & (df['時刻'] <= test_time + timedelta(hours=1))
future_1h_data = df[future_1h_mask]

rainfall_forecast = pd.DataFrame({
    '時刻': future_1h_data['時刻'],
    '降雨強度': future_1h_data['ダム_60分雨量']
})

# 最新モデルで予測
print("\n=== 最新モデルでの予測 ===")
model = DischargePredictionModelV2()
model.load_model(model_files[-1])

predictions = model.predict(test_time, historical_data, 
                          prediction_hours=0.5,
                          rainfall_forecast=rainfall_forecast)

print("\n最初の30分の予測:")
print("時刻     放流量  変化率  使用降雨  適用遅延")
print("-" * 45)
for _, row in predictions.iterrows():
    time_str = row['時刻'].strftime('%H:%M')
    discharge = row['予測放流量']
    change = row['変化率']
    rainfall = row['使用降雨強度']
    delay = row['適用遅延']
    
    print(f"{time_str}  {discharge:6.1f}  {change:+6.1f}  {rainfall:6.1f}    {delay:3.0f}分")