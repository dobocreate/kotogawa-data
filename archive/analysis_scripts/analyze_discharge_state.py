#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
放流量の減少が状態判定に反映されない問題の分析
"""

import pandas as pd
from datetime import datetime, timedelta
from discharge_prediction_model_v2 import DischargePredictionModelV2

# データ読み込み
df = pd.read_csv('統合データ_水位ダム_20250730_205325.csv', encoding='utf-8')
df['時刻'] = pd.to_datetime(df['時刻'])

# テスト時刻（2023-06-30 22:00）
test_time = pd.to_datetime('2023-06-30 22:00')

# 過去1時間のデータを詳細に確認
start_time = test_time - timedelta(hours=1)
end_time = test_time

time_mask = (df['時刻'] >= start_time) & (df['時刻'] <= end_time)
recent_data = df[time_mask].copy()

print("過去1時間の放流量の推移:")
print("時刻     放流量  前回差  30分差  降雨")
print("-" * 45)

for i, row in recent_data.iterrows():
    time_str = row['時刻'].strftime('%H:%M')
    discharge = row['ダム_全放流量']
    rainfall = row['ダム_60分雨量']
    
    # 10分前との差
    if i > recent_data.index[0]:
        prev_discharge = recent_data.loc[i-1, 'ダム_全放流量']
        diff_10min = discharge - prev_discharge
    else:
        diff_10min = 0
    
    # 30分前との差
    idx_30min = i - 3
    if idx_30min in recent_data.index:
        discharge_30min = recent_data.loc[idx_30min, 'ダム_全放流量']
        diff_30min = discharge - discharge_30min
    else:
        diff_30min = None
    
    if diff_30min is not None:
        print(f"{time_str}  {discharge:6.1f}  {diff_10min:+6.1f}  {diff_30min:+6.1f}  {rainfall:4.1f}")
    else:
        print(f"{time_str}  {discharge:6.1f}  {diff_10min:+6.1f}      -    {rainfall:4.1f}")

# モデルによる状態判定を確認
print("\n\nモデルによる状態判定の分析:")
print("-" * 60)

model = DischargePredictionModelV2()

# 各時点での状態判定
for i in range(len(recent_data) - 3):  # 30分のデータが必要
    current_idx = recent_data.index[i + 3]
    current_time = recent_data.loc[current_idx, '時刻']
    
    # 過去30分のデータ（4点）
    discharge_history = recent_data.iloc[i:i+4]['ダム_全放流量']
    
    # 状態判定
    state, rate = model.identify_discharge_state(discharge_history)
    
    # 30分の変化量
    change_30min = discharge_history.iloc[-1] - discharge_history.iloc[0]
    
    state_str = {1: "増加", 0: "定常", -1: "減少"}[state]
    
    print(f"\n時刻: {current_time.strftime('%H:%M')}")
    print(f"  過去30分の放流量: {list(discharge_history.values)}")
    print(f"  30分変化量: {change_30min:.1f} m³/s")
    print(f"  判定状態: {state_str} (変化率: {rate:.1f} m³/s/10min)")
    print(f"  閾値: {model.params['state_threshold']} m³/s")

# 22:00時点の詳細分析
print("\n\n=== 22:00時点の詳細分析 ===")

# 22:00時点のデータ
idx_22 = recent_data[recent_data['時刻'] == test_time].index[0]

# 過去7点のデータ（1時間）
discharge_history_7 = recent_data.iloc[-7:]['ダム_全放流量']

print(f"\n過去1時間の放流量履歴:")
for i, (time, discharge) in enumerate(zip(recent_data.iloc[-7:]['時刻'], discharge_history_7)):
    print(f"  [{i}] {time.strftime('%H:%M')} - {discharge:.1f} m³/s")

# 各区間での変化
print(f"\n変化量の分析:")
print(f"  60分前→現在: {discharge_history_7.iloc[-1] - discharge_history_7.iloc[0]:.1f} m³/s")
print(f"  50分前→現在: {discharge_history_7.iloc[-1] - discharge_history_7.iloc[1]:.1f} m³/s")
print(f"  40分前→現在: {discharge_history_7.iloc[-1] - discharge_history_7.iloc[2]:.1f} m³/s")
print(f"  30分前→現在: {discharge_history_7.iloc[-1] - discharge_history_7.iloc[3]:.1f} m³/s")
print(f"  20分前→現在: {discharge_history_7.iloc[-1] - discharge_history_7.iloc[4]:.1f} m³/s")
print(f"  10分前→現在: {discharge_history_7.iloc[-1] - discharge_history_7.iloc[5]:.1f} m³/s")

# モデルの状態判定
state, rate = model.identify_discharge_state(discharge_history_7.iloc[-4:])

print(f"\nモデルの判定:")
print(f"  使用データ: 過去30分（4点）")
print(f"  30分変化量: {discharge_history_7.iloc[-1] - discharge_history_7.iloc[-4]:.1f} m³/s")
print(f"  判定閾値: {model.params['state_threshold']} m³/s")
print(f"  判定結果: {state} ({['減少', '定常', '増加'][state+1]})")

print("\n\n問題の特定:")
print("identify_discharge_stateメソッドは過去30分（最後の4点）のみを見ている")
print("50分前からの減少傾向は考慮されていない")