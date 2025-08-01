#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
放流量予測ロジックのデバッグ
"""

import pandas as pd
from datetime import datetime, timedelta
from discharge_prediction_model_v2 import DischargePredictionModelV2

# モデルの内部動作を確認
model = DischargePredictionModelV2()

print("モデルパラメータ:")
print(f"  低降雨閾値: {model.params['rain_threshold_low']} mm/h")
print(f"  低降雨時の減少率: {model.params['rate_decrease_low']} m³/s/10min")

# 状態ごとの変化率計算を確認
print("\n変化率計算のテスト:")

# ケース1: 増加状態、低降雨
state = 1  # 増加中
rainfall = 1.0  # 低降雨
discharge = 500  # 現在の放流量

change_rate = model.predict_discharge_change(state, rainfall, discharge)
print(f"\nケース1: 増加状態 + 低降雨({rainfall}mm/h)")
print(f"  変化率: {change_rate:.1f} m³/s/10min")

# ケース2: 定常状態、低降雨
state = 0  # 定常
change_rate = model.predict_discharge_change(state, rainfall, discharge)
print(f"\nケース2: 定常状態 + 低降雨({rainfall}mm/h)")
print(f"  変化率: {change_rate:.1f} m³/s/10min")

# ケース3: 減少状態、低降雨
state = -1  # 減少中
change_rate = model.predict_discharge_change(state, rainfall, discharge)
print(f"\nケース3: 減少状態 + 低降雨({rainfall}mm/h)")
print(f"  変化率: {change_rate:.1f} m³/s/10min")

# predict_discharge_changeメソッドのロジックを確認
print("\n\npredict_discharge_changeメソッドの内部ロジック:")
print("```python")
print("if rain_category == 'low':")
print("    if state <= 0:  # 減少中または定常")
print("        base_rate = -self.params['rate_decrease_low']  # -24.3")
print("    else:  # 増加中")
print("        base_rate = self.params['rate_increase_low'] * 0.3  # 27.1 * 0.3 = 8.13")
print("```")
print("\n問題: 増加状態で低降雨でも +8.13 m³/s/10min の増加が続く")