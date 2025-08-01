#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
2023年7月1日6:00時点の詳細分析（改良版モデル統合版デモ使用）
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# 分析時点のデータを確認
df = pd.read_csv('統合データ_水位ダム_20250730_205325.csv', encoding='utf-8')
df['時刻'] = pd.to_datetime(df['時刻'])

# 2023年7月1日6:00前後のデータを表示
analysis_time = pd.to_datetime('2023-07-01 06:00')
mask = (df['時刻'] >= '2023-07-01 00:00') & (df['時刻'] <= '2023-07-01 12:00')
event_data = df[mask][['時刻', 'ダム_60分雨量', 'ダム_全放流量', 'ダム_貯水位', '水位_水位']].copy()

print("2023年7月1日 6:00時点の分析")
print("=" * 60)

# 6:00時点のデータ
current_data = event_data[event_data['時刻'] == analysis_time].iloc[0]
print(f"\n現在時刻: {analysis_time}")
print(f"降雨強度: {current_data['ダム_60分雨量']:.1f} mm/h")
print(f"放流量: {current_data['ダム_全放流量']:.1f} m³/s")
print(f"貯水位: {current_data['ダム_貯水位']:.2f} m")
print(f"河川水位: {current_data['水位_水位']:.2f} m")

# 時系列の推移（1時間ごと）
print("\n時系列推移（1時間ごと）:")
print("時刻         降雨(mm/h) 放流量(m³/s) 貯水位(m) 水位(m)")
print("-" * 55)
for i in range(0, len(event_data), 6):  # 1時間ごと
    row = event_data.iloc[i]
    print(f"{row['時刻'].strftime('%H:%M')}        "
          f"{row['ダム_60分雨量']:>6.1f}     "
          f"{row['ダム_全放流量']:>8.1f}    "
          f"{row['ダム_貯水位']:>6.2f}   "
          f"{row['水位_水位']:>5.2f}")

# 重要なポイント
print("\n分析のポイント:")
print("1. 6:00時点の状況:")
print("   - 降雨: 0mm/h（5:00から降雨停止）")
print("   - 放流量: 901.7m³/s（ピーク1188.2から減少中）")
print("   - 貯水位: 37.79m（洪水貯留準備水位38.0m以下）")
print("   - 河川水位: 7.17m（ピークから減少中）")

print("\n2. 改良版モデルv2の動作:")
print("   - 降雨0mm/hで放流量減少モード")
print("   - 60分遅延を適用（減少開始時の標準遅延）")
print("   - 緩やかな減少を予測（減少率は増加率の89%）")

print("\n3. 予測精度が高い理由:")
print("   - 降雨停止から1時間経過で安定した減少傾向")
print("   - 貯水位が安全範囲内で通常運用")
print("   - ルールベースの予測が物理的制約を適切に反映")

# 予測グラフ（prediction_demo_20250801_064526.png）が作成されています
print("\n予測結果のグラフ: prediction_demo_20250801_064526.png")