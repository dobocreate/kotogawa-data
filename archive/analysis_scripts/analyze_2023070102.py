#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
2023-07-01 02:00の状況分析
"""

import pandas as pd
from datetime import datetime, timedelta
from discharge_prediction_model_v2 import DischargePredictionModelV2

# データ読み込み
df = pd.read_csv('統合データ_水位ダム_20250730_205325.csv', encoding='utf-8')
df['時刻'] = pd.to_datetime(df['時刻'])

test_time = pd.to_datetime('2023-07-01 02:00')

# 過去データの準備
historical_mask = df['時刻'] <= test_time
historical_data = df[historical_mask].copy()

print("=== 2023-07-01 02:00 の状況分析 ===")
print(f"\n現在時刻: {test_time}")
print(f"現在の放流量: {historical_data.iloc[-1]['ダム_全放流量']:.1f} m³/s")
print(f"現在の降雨: {historical_data.iloc[-1]['ダム_60分雨量']:.1f} mm/h")

# 過去の放流量変化を確認
print("\n過去1時間の放流量変化:")
for i in range(7):
    idx = -7 + i
    time = historical_data.iloc[idx]['時刻']
    discharge = historical_data.iloc[idx]['ダム_全放流量']
    rainfall = historical_data.iloc[idx]['ダム_60分雨量']
    print(f"  {time.strftime('%H:%M')} - 放流量: {discharge:6.1f} m³/s, 降雨: {rainfall:5.1f} mm/h")

# モデルで状態判定
model = DischargePredictionModelV2()
discharge_history = historical_data.iloc[-7:]['ダム_全放流量']
rainfall_history = historical_data.iloc[-7:]['ダム_60分雨量']

state, rate = model.identify_discharge_state(discharge_history)
print(f"\n状態判定:")
print(f"  過去30分の変化: {discharge_history.iloc[-1] - discharge_history.iloc[-4]:.1f} m³/s")
print(f"  判定結果: {['減少', '定常', '増加'][state+1]} (変化率: {rate:.1f} m³/s/10min)")

# 降雨の増加検出
current_rainfall = historical_data.iloc[-1]['ダム_60分雨量']
rain_change = current_rainfall - rainfall_history.iloc[-4]
rain_increasing = rain_change > 10

print(f"\n降雨の変化:")
print(f"  30分前の降雨: {rainfall_history.iloc[-4]:.1f} mm/h")
print(f"  現在の降雨: {current_rainfall:.1f} mm/h")
print(f"  変化量: {rain_change:.1f} mm/h")
print(f"  急増判定: {rain_increasing}")

# 遅延時間の決定
delay = model.get_dynamic_delay(state, current_rainfall, rainfall_history)
print(f"\n遅延時間: {delay}分")

# 遅延時間決定ロジックの詳細
print("\n遅延時間決定ロジックの詳細:")
print(f"  1. 降雨急増({rain_increasing}) かつ 非増加状態({state <= 0}): {rain_increasing and (state <= 0)}")
print(f"     → delay_onset(0分)を適用: {rain_increasing and (state <= 0)}")
print(f"  2. 増加継続中(state=1): {state == 1}")
print(f"     → delay_increase(120分)を適用")
print(f"  3. 低降雨({current_rainfall < 10}) かつ 非増加状態: {current_rainfall < 10 and state <= 0}")
print(f"     → delay_decrease(60分)を適用")

# 予測1ステップ目の実効降雨
print(f"\n予測1ステップ目(02:10)の実効降雨計算:")
delay_steps = delay // 10
print(f"  遅延ステップ数: {delay_steps}")

if delay_steps == 0:
    print(f"  → 遅延0分なので、現在(02:00)の降雨 {current_rainfall:.1f} mm/h を使用")
else:
    target_time = test_time - timedelta(minutes=delay)
    print(f"  → {delay}分前({target_time.strftime('%H:%M')})の降雨を使用")