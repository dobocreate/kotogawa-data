#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
10分単位統一モデルのテスト
"""

import pandas as pd
from datetime import datetime, timedelta
from discharge_prediction_model_v2 import DischargePredictionModelV2

# データ読み込み
df = pd.read_csv('統合データ_水位ダム_20250730_142903.csv', encoding='utf-8')
df['時刻'] = pd.to_datetime(df['時刻'])

print("=== 10分単位統一モデルのテスト ===")
print("\n統一された変更点:")
print("1. 状態判定: 過去30分 → 過去10分+10分先予測")
print("2. 降雨増加検出: 過去30分 → 過去10分")
print("3. 将来の放流量傾向: 30分用しきい値 → 10分用しきい値")
print("4. 低降雨継続判定: 過去30分平均 → 過去10分")

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

print(f"\n現在時刻: {test_time}")
print(f"現在の放流量: {historical_data.iloc[-1]['ダム_全放流量']:.1f} m³/s")
print(f"現在の降雨: {historical_data.iloc[-1]['ダム_60分雨量']:.1f} mm/h")

# モデル初期化と状態判定テスト
model = DischargePredictionModelV2()
model.load_model('discharge_prediction_model_v2_20250801_221633.pkl')

# 過去データから状態判定に必要なデータを抽出
discharge_history = historical_data.iloc[-7:]['ダム_全放流量']
rainfall_history = historical_data.iloc[-7:]['ダム_60分雨量']

# 10分先の簡易予測（モデル内部と同じロジック）
future_rainfall_10min = rainfall_forecast.iloc[0]['降雨強度']
if future_rainfall_10min >= 20:
    future_discharge_10min = historical_data.iloc[-1]['ダム_全放流量'] + model.params['rate_increase_high'] * 0.8
elif future_rainfall_10min >= 10:
    future_discharge_10min = historical_data.iloc[-1]['ダム_全放流量'] + model.params['rate_increase_low'] * 0.8
else:
    future_discharge_10min = historical_data.iloc[-1]['ダム_全放流量'] - model.params['rate_decrease_low'] * 0.5

# 状態判定（10分先予測あり）
state_with_future, rate_with_future = model.identify_discharge_state(discharge_history, future_discharge_10min)

# 状態判定（10分先予測なし）
state_without_future, rate_without_future = model.identify_discharge_state(discharge_history)

print("\n状態判定の比較:")
print(f"  過去10分の変化: {discharge_history.iloc[-1] - discharge_history.iloc[-2]:.1f} m³/s")
print(f"  10分先の降雨: {future_rainfall_10min:.1f} mm/h")
print(f"  10分先の予測放流量: {future_discharge_10min:.1f} m³/s")
print(f"  \n  10分先予測なしの判定: {['減少', '定常', '増加'][state_without_future+1]} (変化率: {rate_without_future:.1f})")
print(f"  10分先予測ありの判定: {['減少', '定常', '増加'][state_with_future+1]} (変化率: {rate_with_future:.1f})")

# 遅延時間の判定テスト
current_rainfall = historical_data.iloc[-1]['ダム_60分雨量']
delay = model.get_dynamic_delay(state_with_future, current_rainfall, rainfall_history,
                               future_discharge_10min - discharge_history.iloc[-1], future_rainfall_10min)

print(f"\n遅延時間の判定:")
print(f"  過去10分の降雨変化: {current_rainfall - rainfall_history.iloc[-2]:.1f} mm/h")
print(f"  降雨急増判定: {current_rainfall - rainfall_history.iloc[-2] > 5}")
print(f"  適用される遅延時間: {delay}分")

# 予測実行
predictions = model.predict(test_time, historical_data, 
                           prediction_hours=1, 
                           rainfall_forecast=rainfall_forecast)

print("\n予測結果（最初の30分）:")
print("時刻     放流量  変化率  使用降雨  適用遅延  状態")
print("-" * 50)
for i in range(min(3, len(predictions))):
    row = predictions.iloc[i]
    state_str = {-1: "減少", 0: "定常", 1: "増加"}[row['状態']]
    print(f"{row['時刻'].strftime('%H:%M')}  {row['予測放流量']:6.1f}  {row['変化率']:+6.1f}  {row['使用降雨強度']:6.1f}    {row['適用遅延']:3.0f}分  {state_str}")

print("\n【まとめ】")
print("10分単位への統一により：")
print("- より短期的で敏感な状態判定")
print("- 降雨変化への素早い反応")
print("- 一貫性のある時間スケール")