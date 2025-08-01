#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
最終版v2モデルのテスト
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from discharge_prediction_model_v2_final import DischargePredictionModelV2
import matplotlib.pyplot as plt

# 日本語フォント設定
plt.rcParams['font.sans-serif'] = ['Yu Gothic', 'Hiragino Sans', 'Meirio', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def test_final_model():
    """最終版モデルのテスト"""
    print("最終版放流量予測モデルv2のテスト")
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
    print(f"現在の放流量: {historical_data.iloc[-1]['ダム_全放流量']:.1f} m³/s")
    
    # 過去の降雨状況を確認
    print("\n過去の降雨状況:")
    print(f"  60分前（21:00）: {historical_data.iloc[-7]['ダム_60分雨量']:.1f} mm/h")
    print(f"  120分前（20:00）: {historical_data.iloc[-13]['ダム_60分雨量']:.1f} mm/h")
    
    print("\n1時間先までの降雨予測:")
    for _, row in rainfall_forecast.iterrows():
        print(f"  {row['時刻'].strftime('%H:%M')} - {row['降雨強度']:.1f} mm/h")
    
    # モデル初期化と予測
    model = DischargePredictionModelV2()
    predictions = model.predict(
        test_time, 
        historical_data,
        prediction_hours=3,
        rainfall_forecast=rainfall_forecast
    )
    
    # 結果の確認
    print("\n予測結果（最初の1時間）:")
    print(predictions[['時刻', '予測放流量', '使用降雨強度', '適用遅延', '状態', '変化率']].head(6))
    
    # 問題の確認
    initial_change = predictions.iloc[0]['変化率']
    print(f"\n初期変化率: {initial_change:.1f} m³/s/10min")
    
    if initial_change > 10:
        print("状態: 放流量が増加する予測")
    elif initial_change < -10:
        print("状態: 放流量が減少する予測")
    else:
        print("状態: 放流量がほぼ一定の予測")
    
    # 可視化
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    
    # 放流量
    display_start = test_time - timedelta(hours=3)
    display_end = test_time + timedelta(hours=3)
    display_mask = (df['時刻'] >= display_start) & (df['時刻'] <= display_end)
    
    ax1.plot(df.loc[display_mask, '時刻'], df.loc[display_mask, 'ダム_全放流量'], 
            'k-', linewidth=2, label='実績')
    ax1.plot(predictions['時刻'], predictions['予測放流量'], 
            'r--', linewidth=2, label='予測')
    ax1.axvline(x=test_time, color='green', linestyle='--', alpha=0.7, label='現在時刻')
    ax1.set_ylabel('放流量 (m³/s)')
    ax1.set_title('最終版モデルの予測結果（2023年6月30日 22:00）')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 降雨強度
    ax2.bar(df.loc[display_mask, '時刻'], df.loc[display_mask, 'ダム_60分雨量'], 
           width=0.007, color='blue', alpha=0.7, label='実績')
    ax2.bar(rainfall_forecast['時刻'], rainfall_forecast['降雨強度'], 
           width=0.007, color='orange', alpha=0.7, label='予測（1時間）')
    ax2.axvline(x=test_time, color='green', linestyle='--', alpha=0.7)
    ax2.axvline(x=test_time - timedelta(minutes=60), color='red', linestyle=':', 
               alpha=0.5, label='60分前')
    ax2.axvline(x=test_time - timedelta(minutes=120), color='purple', linestyle=':', 
               alpha=0.5, label='120分前')
    ax2.set_xlabel('時刻')
    ax2.set_ylabel('降雨強度 (mm/h)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f'v2_final_test_{timestamp}.png'
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\nグラフを保存: {filename}")
    
    # モデルを保存
    model.save_model(f'discharge_prediction_model_v2_final_{timestamp}.pkl')
    print(f"最終版モデルを保存: discharge_prediction_model_v2_final_{timestamp}.pkl")
    
    # 7月1日のケースもテスト
    print("\n" + "="*60)
    print("7月1日 00:00のケースもテスト")
    
    test_time2 = pd.to_datetime('2023-07-01 00:00')
    historical_mask2 = df['時刻'] <= test_time2
    historical_data2 = df[historical_mask2].copy()
    
    predictions2 = model.predict(
        test_time2, 
        historical_data2,
        prediction_hours=2
    )
    
    print(f"\nテスト時刻: {test_time2}")
    print(f"現在の降雨: {historical_data2.iloc[-1]['ダム_60分雨量']:.1f} mm/h")
    print("\n予測結果（最初の30分）:")
    print(predictions2[['時刻', '予測放流量', '使用降雨強度', '変化率']].head(3))

if __name__ == "__main__":
    test_final_model()