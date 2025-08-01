#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
デモの放流量予測値を抽出
"""

import pandas as pd
from datetime import datetime, timedelta
from water_level_prediction_demo_interactive import WaterLevelPredictionDemo

# デモインスタンスを作成
demo = WaterLevelPredictionDemo()

# 予測を実行（内部的な予測データを取得）
current_time = pd.to_datetime('2023-07-01 00:00')
demo.run_prediction(current_time_str='2023-07-01 00:00', show_actual=True, use_discharge_model=True)

print("\n=== 放流量予測の詳細 ===")
if hasattr(demo, 'last_discharge_predictions'):
    predictions = demo.last_discharge_predictions
    print("\n最初の1時間の放流量予測:")
    print("時刻     予測放流量  変化率  使用降雨  適用遅延")
    print("-" * 50)
    for i in range(min(6, len(predictions))):
        row = predictions.iloc[i]
        time_str = row['時刻'].strftime('%H:%M')
        discharge = row['予測放流量']
        if '変化率' in row:
            change = row['変化率']
            rainfall = row.get('使用降雨強度', 0)
            delay = row.get('適用遅延', 0)
            print(f"{time_str}    {discharge:7.1f}  {change:+6.1f}  {rainfall:6.1f}    {delay:3.0f}分")
        else:
            print(f"{time_str}    {discharge:7.1f}")
else:
    print("放流量予測データが見つかりません")