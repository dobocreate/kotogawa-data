#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
v2モデルのデバッグ - 降雨履歴の問題を調査
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from discharge_prediction_model_v2 import DischargePredictionModelV2

def debug_model():
    """モデルの動作をデバッグ"""
    print("放流量予測モデルv2のデバッグ")
    print("=" * 60)
    
    # データ読み込み
    df = pd.read_csv('統合データ_水位ダム_20250730_205325.csv', encoding='utf-8')
    df['時刻'] = pd.to_datetime(df['時刻'])
    
    # テストケース：2023年6月30日 22:00
    test_time = pd.to_datetime('2023-06-30 22:00')
    
    # 過去データの準備
    historical_mask = df['時刻'] <= test_time
    historical_data = df[historical_mask].copy()
    
    # 初期データの確認
    current_idx = len(historical_data) - 1
    print(f"\n現在時刻: {test_time}")
    print(f"現在の降雨: {historical_data.iloc[current_idx]['ダム_60分雨量']:.1f} mm/h")
    
    # 過去1時間の降雨履歴
    print("\n初期の降雨履歴（過去1時間）:")
    rainfall_history = historical_data.iloc[-7:]['ダム_60分雨量']
    for i, (time, rain) in enumerate(zip(historical_data.iloc[-7:]['時刻'], rainfall_history)):
        print(f"  [{i}] {time.strftime('%H:%M')} - {rain:.1f} mm/h")
    
    # 120分前のデータ
    delay_120min_idx = current_idx - 12  # 120分 = 12ステップ
    if delay_120min_idx >= 0:
        print(f"\n120分前（20:00）の降雨: {historical_data.iloc[delay_120min_idx]['ダム_60分雨量']:.1f} mm/h")
    
    # 60分前のデータ
    delay_60min_idx = current_idx - 6  # 60分 = 6ステップ
    if delay_60min_idx >= 0:
        print(f"60分前（21:00）の降雨: {historical_data.iloc[delay_60min_idx]['ダム_60分雨量']:.1f} mm/h")
    
    # 使用される実効降雨を確認
    print("\n予測における実効降雨の計算:")
    
    # step 1での計算を模擬
    delay_minutes = 60  # 減少開始の遅延
    delay_steps = delay_minutes // 10  # 6ステップ
    step = 1
    
    print(f"\nStep {step}:")
    print(f"  遅延: {delay_minutes}分 = {delay_steps}ステップ")
    
    if step > delay_steps:
        print(f"  条件: step({step}) > delay_steps({delay_steps}) → False")
    else:
        print(f"  条件: step({step}) <= delay_steps({delay_steps}) → True")
        hist_idx = delay_steps - step  # 6 - 1 = 5
        print(f"  hist_idx = {hist_idx}")
        print(f"  rainfall_history.iloc[-{hist_idx + 1}] を使用")
        effective_idx = -(hist_idx + 1)
        print(f"  実際のインデックス: {effective_idx} (末尾から{hist_idx + 1}番目)")
        
        if effective_idx >= -len(rainfall_history):
            effective_rainfall = rainfall_history.iloc[effective_idx]
            time_used = historical_data.iloc[-7:]['時刻'].iloc[effective_idx]
            print(f"  使用される降雨: {effective_rainfall:.1f} mm/h ({time_used.strftime('%H:%M')})")
        else:
            print(f"  エラー: インデックスが範囲外")

if __name__ == "__main__":
    debug_model()