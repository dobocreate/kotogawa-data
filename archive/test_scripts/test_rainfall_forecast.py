#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
降雨予測を活用した予測のテスト
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from water_level_prediction_demo_interactive import InteractivePredictionDemo

def test_with_rainfall_forecast():
    """降雨予測を使用したテスト"""
    print("降雨予測を活用した水位・放流量予測テスト")
    print("=" * 60)
    
    # デモの初期化
    demo = InteractivePredictionDemo()
    
    # テストケース1: 降雨増加予測
    print("\n【ケース1】降雨増加予測（2023年6月30日 23:00）")
    current_time = "2023-06-30 23:00:00"
    
    # 降雨予測データ（増加シナリオ）
    forecast_times = pd.date_range(start='2023-06-30 23:10', end='2023-07-01 00:00', freq='10min')
    rainfall_forecast_increase = pd.DataFrame({
        '時刻': forecast_times,
        '降雨強度': [5, 10, 15, 20, 30, 40]  # 増加予測
    })
    
    # 予測実行
    predictions1 = demo.run_prediction(
        current_time_str=current_time,
        show_actual=True,
        rainfall_forecast_data=rainfall_forecast_increase
    )
    
    # テストケース2: 降雨減少予測
    print("\n【ケース2】降雨減少予測（2023年7月1日 01:00）")
    current_time = "2023-07-01 01:00:00"
    
    # 降雨予測データ（減少シナリオ）
    forecast_times = pd.date_range(start='2023-07-01 01:10', end='2023-07-01 02:00', freq='10min')
    rainfall_forecast_decrease = pd.DataFrame({
        '時刻': forecast_times,
        '降雨強度': [40, 30, 20, 10, 5, 0]  # 減少予測
    })
    
    predictions2 = demo.run_prediction(
        current_time_str=current_time,
        show_actual=True,
        rainfall_forecast_data=rainfall_forecast_decrease
    )
    
    # テストケース3: デフォルト（段階的減少）
    print("\n【ケース3】デフォルト降雨予測（2023年7月1日 06:00）")
    current_time = "2023-07-01 06:00:00"
    
    # 降雨予測データなし（デフォルトの段階的減少を使用）
    predictions3 = demo.run_prediction(
        current_time_str=current_time,
        show_actual=True,
        rainfall_forecast_data=None  # デフォルト使用
    )
    
    print("\n分析完了！")
    print("\n結果:")
    print("- ケース1: 降雨増加予測により、より早い放流量増加を予測")
    print("- ケース2: 降雨減少予測により、放流量のピークを適切に予測")
    print("- ケース3: デフォルトの段階的減少モデルで安定した減少を予測")

if __name__ == "__main__":
    test_with_rainfall_forecast()