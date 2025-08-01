#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
予測で使用されるデータの範囲を詳細に分析
"""

import pandas as pd
from datetime import datetime, timedelta

# データ読み込み
df = pd.read_csv('統合データ_水位ダム_20250730_205325.csv', encoding='utf-8')
df['時刻'] = pd.to_datetime(df['時刻'])

test_time = pd.to_datetime('2023-06-30 22:00')

print("=== データ使用範囲の分析 ===")
print(f"\n現在時刻: {test_time}")

# 1. 過去データ（historical_data）の範囲
historical_mask = df['時刻'] <= test_time
historical_data = df[historical_mask].copy()

print(f"\n1. 過去データ（historical_data）:")
print(f"   範囲: {historical_data['時刻'].min()} ～ {historical_data['時刻'].max()}")
print(f"   件数: {len(historical_data)}件")

# 2. 降雨履歴（rainfall_history）の初期値
rainfall_history_initial = historical_data.iloc[-7:]['ダム_60分雨量']
print(f"\n2. 降雨履歴の初期値（過去1時間）:")
for i, (time, rain) in enumerate(zip(historical_data.iloc[-7:]['時刻'], rainfall_history_initial)):
    print(f"   [{i}] {time.strftime('%H:%M')} - {rain:.1f} mm/h")

# 3. 降雨予測（rainfall_forecast）の範囲
future_1h_mask = (df['時刻'] > test_time) & \
                 (df['時刻'] <= test_time + timedelta(hours=1))
future_1h_data = df[future_1h_mask]

print(f"\n3. 降雨予測データ（1時間先まで）:")
for _, row in future_1h_data.iterrows():
    print(f"   {row['時刻'].strftime('%H:%M')} - {row['ダム_60分雨量']:.1f} mm/h")

# 4. 遅延時間と実効降雨の関係
print(f"\n4. 遅延時間による実効降雨の決定:")
print("\n【重要な点】")
print("- 遅延時間は「過去のどの時点の降雨が現在の放流量に影響するか」を示す")
print("- 例：遅延60分 → 60分前の降雨が現在の放流量に影響")
print("")
print("【予測での使用】")
print("- ステップ1（22:10）で遅延60分の場合:")
print("  → 21:10の降雨（10mm/h）を使用")
print("- ステップ1（22:10）で遅延120分の場合:")
print("  → 20:10の降雨を使用すべきだが、データがないため代替処理")

# 5. 実際の遅延計算をシミュレート
print(f"\n5. 各予測ステップでの実効降雨（シミュレーション）:")

for step in range(1, 7):
    pred_time = test_time + timedelta(minutes=step * 10)
    
    # 60分遅延の例
    delay_60 = 60
    delay_steps_60 = delay_60 // 10  # 6ステップ
    
    print(f"\nステップ{step} ({pred_time.strftime('%H:%M')}) - 遅延60分:")
    
    if step <= delay_steps_60:
        # まだ遅延時間内 → 過去データを使用
        hist_idx = delay_steps_60 - step  # 6-1=5, 6-2=4, ...
        if hist_idx < 7:  # rainfall_historyの長さ
            target_time = historical_data.iloc[-(hist_idx + 1)]['時刻']
            print(f"  → 過去データ使用: {target_time.strftime('%H:%M')}のデータ")
        else:
            print(f"  → データ不足")
    else:
        # 遅延時間を超えた → 予測期間のデータ
        future_idx = step - delay_steps_60 - 1
        future_time = test_time + timedelta(minutes=(future_idx + 1) * 10)
        print(f"  → 予測データ使用: {future_time.strftime('%H:%M')}のデータ")

print("\n\n【結論】")
print("- 現時点（22:00）より前の実測値を使用")
print("- 1時間先（23:00）までの予測値も使用")
print("- 遅延時間により、どの時点のデータを使うかが決まる")
print("- 遅延時間前からスタートするわけではない")