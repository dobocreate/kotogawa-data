#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
2023-07-01 00:00の予測ロジックの詳細確認
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

print("=== 2023-07-01 00:00 予測ロジックの詳細確認 ===")
print(f"\n現在時刻: {test_time}")
print(f"現在の放流量: {historical_data.iloc[-1]['ダム_全放流量']:.1f} m³/s")
print(f"現在の降雨: {historical_data.iloc[-1]['ダム_60分雨量']:.1f} mm/h")

# モデル初期化と状態判定
model = DischargePredictionModelV2()
discharge_history = historical_data.iloc[-7:]['ダム_全放流量']
rainfall_history = historical_data.iloc[-7:]['ダム_60分雨量']

state, rate = model.identify_discharge_state(discharge_history)
print(f"\n状態判定:")
print(f"  過去30分の変化: {discharge_history.iloc[-1] - discharge_history.iloc[-4]:.1f} m³/s")
print(f"  判定結果: {['減少', '定常', '増加'][state+1]}")

# 30分先までの簡易予測（モデル内部と同じロジック）
print("\n30分先までの予測ロジック:")
temp_discharge = historical_data.iloc[-1]['ダム_全放流量']
future_discharge_sum = 0
future_rainfall_sum = 0

for look_ahead in range(1, 4):  # 10分、20分、30分先
    look_ahead_time = test_time + timedelta(minutes=look_ahead * 10)
    look_ahead_idx = rainfall_forecast['時刻'].searchsorted(look_ahead_time)
    if look_ahead_idx < len(rainfall_forecast):
        look_ahead_rain = rainfall_forecast.iloc[look_ahead_idx]['降雨強度']
    else:
        look_ahead_rain = historical_data.iloc[-1]['ダム_60分雨量']
    
    future_rainfall_sum += look_ahead_rain
    print(f"  {look_ahead_time.strftime('%H:%M')} - 降雨: {look_ahead_rain:.1f} mm/h")
    
    # 簡易的な変化予測
    if look_ahead_rain >= 20:
        temp_change = model.params['rate_increase_high'] * 0.8
    elif look_ahead_rain >= 10:
        temp_change = model.params['rate_increase_low'] * 0.8
    else:
        temp_change = -model.params['rate_decrease_low'] * 0.5
    
    temp_discharge += temp_change
    future_discharge_sum += temp_change
    print(f"    → 変化: {temp_change:+.1f} m³/s, 累計放流量: {temp_discharge:.1f} m³/s")

future_discharge_trend = future_discharge_sum
future_rainfall_30min = future_rainfall_sum / 3

print(f"\n30分先までの予測結果:")
print(f"  放流量変化予測: {future_discharge_trend:+.1f} m³/s")
print(f"  平均降雨予測: {future_rainfall_30min:.1f} mm/h")

# 遅延時間の決定
current_rainfall = historical_data.iloc[-1]['ダム_60分雨量']
delay = model.get_dynamic_delay(state, current_rainfall, rainfall_history,
                               future_discharge_trend, future_rainfall_30min)

print(f"\n遅延時間の決定:")
print(f"  現在の状態: {['減少', '定常', '増加'][state+1]}")
print(f"  将来の増加予測: {future_discharge_trend > model.params['state_threshold']}")
print(f"  将来の降雨増加: {future_rainfall_30min > current_rainfall + 5}")
print(f"  → 適用される遅延時間: {delay}分")

if delay == 0:
    print("\n【結論】遅延時間0分が適用されるため、降雨の影響が即座に反映されます")
else:
    print(f"\n【結論】遅延時間{delay}分が適用されるため、影響が遅れて現れます")