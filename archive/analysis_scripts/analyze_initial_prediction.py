#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
初期予測ステップの詳細分析
"""

import pandas as pd
from datetime import datetime, timedelta
from discharge_prediction_model_v2 import DischargePredictionModelV2

# データとモデルの準備
df = pd.read_csv('統合データ_水位ダム_20250730_205325.csv', encoding='utf-8')
df['時刻'] = pd.to_datetime(df['時刻'])
test_time = pd.to_datetime('2023-06-30 22:00')

historical_mask = df['時刻'] <= test_time
historical_data = df[historical_mask].copy()

# モデル初期化
model = DischargePredictionModelV2()

# 初期状態の確認
discharge_history = historical_data.iloc[-7:]['ダム_全放流量']
rainfall_history = historical_data.iloc[-7:]['ダム_60分雨量']

current_state, current_rate = model.identify_discharge_state(discharge_history)
current_discharge = historical_data.iloc[-1]['ダム_全放流量']
current_rainfall = historical_data.iloc[-1]['ダム_60分雨量']

print("初期状態:")
print(f"  現在の放流量: {current_discharge:.1f} m³/s")
print(f"  現在の降雨: {current_rainfall:.1f} mm/h")
print(f"  判定状態: {current_state} ({['減少', '定常', '増加'][current_state+1]})")
print(f"  変化率: {current_rate:.1f} m³/s/10min")

# 初期の予測ステップを詳細に追跡
print("\n\n初期予測ステップの詳細:")
print("="*70)

predicted_discharge = current_discharge
predicted_state = current_state

for step in range(1, 4):  # 最初の3ステップ
    pred_time = test_time + timedelta(minutes=step * 10)
    
    print(f"\nステップ {step} ({pred_time.strftime('%H:%M')}):")
    
    # 遅延時間の取得
    delay = model.get_dynamic_delay(predicted_state, current_rainfall, rainfall_history)
    print(f"  適用遅延: {delay}分")
    
    # 遅延を考慮した降雨強度
    delay_steps = delay // 10
    if step > delay_steps:
        past_idx = step - delay_steps - 1
        if past_idx < len(rainfall_history):
            effective_rainfall = rainfall_history.iloc[past_idx]
        else:
            effective_rainfall = 0
    else:
        hist_idx = delay_steps - step
        if hist_idx < len(rainfall_history):
            effective_rainfall = rainfall_history.iloc[-(hist_idx + 1)]
        else:
            effective_rainfall = rainfall_history.iloc[0]
    
    print(f"  実効降雨強度: {effective_rainfall:.1f} mm/h")
    
    # 変化率の計算
    change_rate = model.predict_discharge_change(
        predicted_state, effective_rainfall, predicted_discharge
    )
    print(f"  計算変化率: {change_rate:+.1f} m³/s/10min")
    
    # 放流量の更新
    predicted_discharge += change_rate
    print(f"  予測放流量: {predicted_discharge:.1f} m³/s")
    
    # 状態の更新
    if change_rate > 5:
        predicted_state = 1
        print(f"  新状態: 増加（変化率 > 5）")
    elif change_rate < -5:
        predicted_state = -1
        print(f"  新状態: 減少（変化率 < -5）")
    else:
        predicted_state = 0
        print(f"  新状態: 定常")

print("\n\n問題の要約:")
print("1. 初期状態は「定常」（閾値50m³/sが大きすぎるため）")
print("2. ステップ1で60分前の高降雨（10mm/h）が使用される")
print("3. 定常＋中降雨 → +27.1の増加")
print("4. 変化率+27.1 > 5 → 状態が「増加」に変更")
print("5. 以降、増加状態が維持され続ける")