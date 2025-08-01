#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
00:00の最初の数ステップの詳細デバッグ
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

# 1時間先までの降雨予測
future_1h_mask = (df['時刻'] > test_time) & (df['時刻'] <= test_time + timedelta(hours=1))
future_1h_data = df[future_1h_mask]

rainfall_forecast = pd.DataFrame({
    '時刻': future_1h_data['時刻'],
    '降雨強度': future_1h_data['ダム_60分雨量']
})

# モデル初期化
model = DischargePredictionModelV2()

# 予測の最初の3ステップを手動で実行
predictions = model.predict(test_time, historical_data, 
                          prediction_hours=0.5,  # 30分のみ
                          rainfall_forecast=rainfall_forecast)

print("=== 00:00の予測詳細（最初の30分） ===")
print("\n予測結果:")
print("時刻     放流量  変化率  使用降雨  適用遅延  状態")
print("-" * 50)
for _, row in predictions.iterrows():
    time_str = row['時刻'].strftime('%H:%M')
    discharge = row['予測放流量']
    change = row['変化率']
    rainfall = row['使用降雨強度']
    delay = row['適用遅延']
    state = row['状態']
    
    state_str = {-1: "減少", 0: "定常", 1: "増加"}[state]
    print(f"{time_str}  {discharge:6.1f}  {change:+6.1f}  {rainfall:6.1f}    {delay:3.0f}分  {state_str}")

print("\n\n問題の分析:")
print("1. 最初のステップで遅延時間0分が適用されているか？")
print(f"   → {predictions.iloc[0]['適用遅延']}分")

if predictions.iloc[0]['適用遅延'] == 0:
    print("   ✓ 正しく0分が適用されています")
    print(f"   使用降雨強度: {predictions.iloc[0]['使用降雨強度']:.1f} mm/h")
    print("   この降雨強度での変化率の計算...")
    
    # 変化率の再計算
    state = 1  # 増加中
    rainfall = predictions.iloc[0]['使用降雨強度']
    if rainfall >= 20:
        expected_rate = model.params['rate_increase_high']
        category = "高降雨"
    elif rainfall >= 10:
        expected_rate = model.params['rate_increase_low']
        category = "中降雨"
    else:
        expected_rate = -model.params['rate_decrease_low']
        category = "低降雨"
    
    print(f"   降雨カテゴリ: {category}")
    print(f"   期待される変化率: {expected_rate:+.1f} m³/s/10min")
    print(f"   実際の変化率: {predictions.iloc[0]['変化率']:+.1f} m³/s/10min")
else:
    print("   ✗ 遅延時間が正しく適用されていません")