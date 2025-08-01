#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
1時間後以降の降雨予測継続のテスト
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from discharge_prediction_model_v2 import DischargePredictionModelV2
import matplotlib.pyplot as plt

# 日本語フォント設定
plt.rcParams['font.sans-serif'] = ['Yu Gothic', 'Hiragino Sans', 'Meirio', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def test_rainfall_continuation():
    """降雨予測継続のテスト"""
    print("降雨予測継続機能のテスト")
    print("=" * 60)
    
    # データ読み込み
    df = pd.read_csv('統合データ_水位ダム_20250730_205325.csv', encoding='utf-8')
    df['時刻'] = pd.to_datetime(df['時刻'])
    
    # テストケース：2023年6月30日 22:00
    test_time = pd.to_datetime('2023-06-30 22:00')
    
    # 過去データの準備
    historical_mask = df['時刻'] <= test_time
    historical_data = df[historical_mask].copy()
    
    # 1時間先までの降雨予測（実データ）
    future_1h_mask = (df['時刻'] > test_time) & \
                     (df['時刻'] <= test_time + timedelta(hours=1))
    future_1h_data = df[future_1h_mask]
    
    rainfall_forecast = pd.DataFrame({
        '時刻': future_1h_data['時刻'],
        '降雨強度': future_1h_data['ダム_60分雨量']
    })
    
    print(f"\nテスト時刻: {test_time}")
    print(f"現在の降雨: {historical_data.iloc[-1]['ダム_60分雨量']:.1f} mm/h")
    
    print("\n1時間先までの降雨予測:")
    for _, row in rainfall_forecast.iterrows():
        print(f"  {row['時刻'].strftime('%H:%M')} - {row['降雨強度']:.1f} mm/h")
    
    # 最後の予測値
    last_forecast = rainfall_forecast.iloc[-1]['降雨強度']
    print(f"\n1時間後の予測値（継続される値）: {last_forecast:.1f} mm/h")
    
    # モデル初期化と予測
    model = DischargePredictionModelV2()
    predictions = model.predict(
        test_time, 
        historical_data,
        prediction_hours=3,
        rainfall_forecast=rainfall_forecast
    )
    
    # 使用された降雨強度を確認
    print("\n予測で使用された降雨強度:")
    print("時刻               使用降雨強度  備考")
    print("-" * 50)
    for i, row in predictions.iterrows():
        time_str = row['時刻'].strftime('%H:%M')
        rainfall = row['使用降雨強度']
        
        if i < 6:
            note = "予測データ"
        else:
            if abs(rainfall - last_forecast) < 0.1:
                note = "1時間後の値を継続"
            else:
                note = "遅延による過去データ"
        
        print(f"{time_str}           {rainfall:6.1f}        {note}")
    
    # 実際のデータと比較
    print("\n実際の降雨データとの比較:")
    actual_mask = (df['時刻'] > test_time) & \
                  (df['時刻'] <= test_time + timedelta(hours=3))
    actual_data = df[actual_mask]
    
    print("時刻     実際の降雨  モデルが想定した降雨")
    print("-" * 40)
    for _, actual_row in actual_data.iterrows():
        time = actual_row['時刻']
        actual_rain = actual_row['ダム_60分雨量']
        
        # 予測での想定値を取得
        pred_idx = predictions[predictions['時刻'] == time].index
        if len(pred_idx) > 0:
            assumed_rain = predictions.loc[pred_idx[0], '使用降雨強度']
            print(f"{time.strftime('%H:%M')}    {actual_rain:6.1f}         {assumed_rain:6.1f}")
    
    # 可視化
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    
    # 放流量予測
    ax1.plot(predictions['時刻'], predictions['予測放流量'], 
            'r-', linewidth=2, label='予測')
    ax1.axvline(x=test_time, color='green', linestyle='--', alpha=0.7, label='現在時刻')
    ax1.axvline(x=test_time + timedelta(hours=1), color='orange', linestyle='--', 
               alpha=0.7, label='1時間後')
    ax1.set_ylabel('予測放流量 (m³/s)')
    ax1.set_title('降雨予測継続機能のテスト結果')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 降雨強度（実際と想定）
    # 実際の降雨
    actual_times = actual_data['時刻']
    actual_rains = actual_data['ダム_60分雨量']
    ax2.bar(actual_times, actual_rains, width=0.007, 
           color='lightblue', alpha=0.7, label='実際の降雨')
    
    # モデルが使用した降雨
    ax2.plot(predictions['時刻'], predictions['使用降雨強度'], 
            'ro-', markersize=4, label='モデルが使用した値')
    
    ax2.axvline(x=test_time, color='green', linestyle='--', alpha=0.7)
    ax2.axvline(x=test_time + timedelta(hours=1), color='orange', linestyle='--', alpha=0.7)
    ax2.axhline(y=last_forecast, color='red', linestyle=':', alpha=0.5, 
               label=f'継続値 ({last_forecast:.1f}mm/h)')
    
    ax2.set_xlabel('時刻')
    ax2.set_ylabel('降雨強度 (mm/h)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f'rainfall_continuation_test_{timestamp}.png'
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\nグラフを保存: {filename}")
    
    # モデルを保存
    model.save_model(f'discharge_model_v2_with_continuation_{timestamp}.pkl')
    print(f"モデルを保存: discharge_model_v2_with_continuation_{timestamp}.pkl")

if __name__ == "__main__":
    test_rainfall_continuation()