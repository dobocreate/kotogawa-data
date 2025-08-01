#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
10分先予測版モデルのテスト
"""

import pandas as pd
from datetime import datetime, timedelta
from discharge_prediction_model_v2 import DischargePredictionModelV2

# データ読み込み
df = pd.read_csv('統合データ_水位ダム_20250730_142903.csv', encoding='utf-8')
df['時刻'] = pd.to_datetime(df['時刻'])

# テストケース: 2023-07-01 00:00
test_time = pd.to_datetime('2023-07-01 00:00')

# 過去データの準備
historical_mask = df['時刻'] <= test_time
historical_data = df[historical_mask].copy()

# 1時間先までの降雨予測
future_1h_mask = (df['時刻'] > test_time) & (df['時刻'] <= test_time + timedelta(hours=1))
future_1h_data = df[future_1h_mask]

rainfall_forecast = pd.DataFrame({
    '時刻': future_1h_data['時刻'],
    '降雨強度': future_1h_data['ダム_60分雨量']
})

print("=== 10分先予測版モデルのテスト ===")
print(f"\n現在時刻: {test_time}")
print(f"現在の放流量: {historical_data.iloc[-1]['ダム_全放流量']:.1f} m³/s")
print(f"現在の降雨: {historical_data.iloc[-1]['ダム_60分雨量']:.1f} mm/h")

print("\n降雨予測（最初の30分）:")
for i, row in rainfall_forecast.iloc[:3].iterrows():
    print(f"  {row['時刻'].strftime('%H:%M')} - {row['降雨強度']:.1f} mm/h")

# モデル初期化と予測
model = DischargePredictionModelV2()
model.load_model('discharge_prediction_model_v2_20250801_121601.pkl')

# 予測実行（デバッグ情報を含む）
predictions = model.predict(test_time, historical_data, 
                           prediction_hours=1, 
                           rainfall_forecast=rainfall_forecast)

print("\n予測結果（最初の30分）:")
print("時刻     放流量  変化率  使用降雨  適用遅延")
print("-" * 45)
for i in range(min(3, len(predictions))):
    row = predictions.iloc[i]
    print(f"{row['時刻'].strftime('%H:%M')}  {row['予測放流量']:6.1f}  {row['変化率']:+6.1f}  {row['使用降雨強度']:6.1f}    {row['適用遅延']:3.0f}分")

# 10分先予測の詳細確認
print("\n10分先予測の詳細:")
print(f"  現在の降雨: {historical_data.iloc[-1]['ダム_60分雨量']:.1f} mm/h")
print(f"  10分先の降雨: {rainfall_forecast.iloc[0]['降雨強度']:.1f} mm/h")

# 30分先予測版との比較
print("\n30分先予測版との違い:")
print("  30分先版: 30分間の平均降雨（46+52+60）/3 = 52.7 mm/h")
print("  10分先版: 10分先の降雨のみ = 46.0 mm/h")
print("  → より保守的な予測となります")