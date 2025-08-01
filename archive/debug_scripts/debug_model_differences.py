#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
10分統一モデルと以前のモデルの違いを詳細に分析
"""

import pandas as pd
from datetime import datetime, timedelta
from discharge_prediction_model_v2 import DischargePredictionModelV2

# より詳細なデバッグ
df = pd.read_csv('統合データ_水位ダム_20250730_142903.csv', encoding='utf-8')
df['時刻'] = pd.to_datetime(df['時刻'])

current_time = pd.to_datetime('2023-07-01 00:00')
historical_mask = df['時刻'] <= current_time
historical_data = df[historical_mask].copy()

# テストケース: 状態判定の閾値が異なるケースを探す
print('=== 10分統一モデルの内部動作確認 ===')

model = DischargePredictionModelV2()
model.load_model('discharge_prediction_model_v2_20250801_221633.pkl')

# 過去データ
discharge_history = historical_data.iloc[-7:]['ダム_全放流量']
print('\n過去の放流量履歴:')
for i in range(len(discharge_history)):
    print(f'  {i*10}分前: {discharge_history.iloc[-(i+1)]:.1f} m³/s')

# 状態判定の詳細確認
print('\n状態判定の詳細:')

# 過去10分のみでの判定
state_past_only, rate_past_only = model.identify_discharge_state(discharge_history)
print(f'\n過去10分のみ:')
print(f'  変化量: {discharge_history.iloc[-1] - discharge_history.iloc[-2]:.1f} m³/s')
print(f'  10分用しきい値: {model.params["state_threshold"] / 3:.1f} m³/s')
print(f'  判定: {["減少", "定常", "増加"][state_past_only+1]}')

# 10分先を含めた判定
future_discharge_10min = 421.9  # 計算済みの値
state_with_future, rate_with_future = model.identify_discharge_state(discharge_history, future_discharge_10min)
print(f'\n過去10分+10分先:')
print(f'  20分間の総変化: {future_discharge_10min - discharge_history.iloc[-2]:.1f} m³/s')
print(f'  20分用しきい値: {model.params["state_threshold"] * 2 / 3:.1f} m³/s')
print(f'  判定: {["減少", "定常", "増加"][state_with_future+1]}')

# 異なる時点でのテスト
print('\n\n=== 他の時点での比較 ===')
test_times = ['2023-06-30 22:00', '2023-07-01 02:00', '2023-07-01 06:00']

for test_time_str in test_times:
    test_time = pd.to_datetime(test_time_str)
    hist_mask = df['時刻'] <= test_time
    hist_data = df[hist_mask].copy()
    
    if len(hist_data) < 7:
        continue
        
    discharge_hist = hist_data.iloc[-7:]['ダム_全放流量']
    
    # 過去30分での変化（3ステップ）
    change_30min = discharge_hist.iloc[-1] - discharge_hist.iloc[-4]
    # 過去10分での変化
    change_10min = discharge_hist.iloc[-1] - discharge_hist.iloc[-2]
    
    # 30分用と10分用で判定が異なるケース
    state_30min = 1 if change_30min > 10 else (-1 if change_30min < -10 else 0)
    state_10min = 1 if change_10min > 3.33 else (-1 if change_10min < -3.33 else 0)
    
    print(f'\n{test_time_str}:')
    print(f'  過去30分変化: {change_30min:.1f} m³/s → 状態: {["減少", "定常", "増加"][state_30min+1]}')
    print(f'  過去10分変化: {change_10min:.1f} m³/s → 状態: {["減少", "定常", "増加"][state_10min+1]}')
    if state_30min != state_10min:
        print('  → 判定が異なる！')

# 降雨増加検出の違いも確認
print('\n\n=== 降雨増加検出の違い ===')
rainfall_history = historical_data.iloc[-7:]['ダム_60分雨量']
current_rainfall = rainfall_history.iloc[-1]

print(f'現在の降雨: {current_rainfall:.1f} mm/h')
print(f'10分前の降雨: {rainfall_history.iloc[-2]:.1f} mm/h')
print(f'30分前の降雨: {rainfall_history.iloc[-4]:.1f} mm/h')

# 10分での判定
rain_change_10min = current_rainfall - rainfall_history.iloc[-2]
rain_increasing_10min = rain_change_10min > 5

# 30分での判定（以前の方式）
rain_change_30min = current_rainfall - rainfall_history.iloc[-4]
rain_increasing_30min = rain_change_30min > 10

print(f'\n10分判定: 変化 {rain_change_10min:.1f} mm/h → 急増: {rain_increasing_10min}')
print(f'30分判定: 変化 {rain_change_30min:.1f} mm/h → 急増: {rain_increasing_30min}')

# 低降雨継続の判定の違い
print('\n\n=== 低降雨継続判定の違い ===')
print('10分統一モデル: 過去10分の降雨で判定')
print('以前のモデル: 過去30分（3ステップ）の平均で判定')

recent_rainfall_10min = rainfall_history.iloc[-1]
recent_rainfall_avg_30min = rainfall_history[-3:].mean()

print(f'\n過去10分の降雨: {recent_rainfall_10min:.1f} mm/h')
print(f'過去30分の平均降雨: {recent_rainfall_avg_30min:.1f} mm/h')

# 実際の予測フローでの違いを確認
print('\n\n=== 予測フローでの実際の違い ===')

# 降雨予測
future_1h_mask = (df['時刻'] > current_time) & (df['時刻'] <= current_time + timedelta(hours=1))
future_1h_data = df[future_1h_mask]
rainfall_forecast = pd.DataFrame({
    '時刻': future_1h_data['時刻'],
    '降雨強度': future_1h_data['ダム_60分雨量']
})

# 最初のステップでの遅延時間判定の違いを確認
print('\n最初の予測ステップ（00:10）での遅延時間判定:')

# 10分先の降雨
future_rainfall_10min = rainfall_forecast.iloc[0]['降雨強度']
print(f'  10分先の降雨: {future_rainfall_10min:.1f} mm/h')

# 10分先の放流量変化予測
if future_rainfall_10min >= 20:
    future_discharge_trend = model.params['rate_increase_high'] * 0.8
elif future_rainfall_10min >= 10:
    future_discharge_trend = model.params['rate_increase_low'] * 0.8
else:
    future_discharge_trend = -model.params['rate_decrease_low'] * 0.5

print(f'  10分先の放流量変化予測: {future_discharge_trend:.1f} m³/s')

# この時点での状態（10分先考慮）
state_with_lookahead, _ = model.identify_discharge_state(discharge_history, 
                                                         discharge_history.iloc[-1] + future_discharge_trend)
print(f'  10分先を考慮した状態: {["減少", "定常", "増加"][state_with_lookahead+1]}')

# 遅延時間の決定
delay = model.get_dynamic_delay(state_with_lookahead, current_rainfall, rainfall_history,
                               future_discharge_trend, future_rainfall_10min)
print(f'  適用される遅延時間: {delay}分')