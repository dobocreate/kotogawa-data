#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
最新モデルでの動作確認
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from discharge_prediction_model_v2 import DischargePredictionModelV2
import matplotlib.pyplot as plt

# 日本語フォント設定
plt.rcParams['font.sans-serif'] = ['Yu Gothic', 'Hiragino Sans', 'Meirio', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def test_latest_model():
    """最新モデルのテスト"""
    print("最新の放流量予測モデルv2のテスト")
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
    
    print("\n1時間先までの降雨予測:")
    for _, row in rainfall_forecast.iterrows():
        print(f"  {row['時刻'].strftime('%H:%M')} - {row['降雨強度']:.1f} mm/h")
    
    # 最新モデルを使用（修正版）
    model = DischargePredictionModelV2()
    # model.load_model('discharge_prediction_model_v2_20250801_101908.pkl')
    
    predictions = model.predict(
        test_time, 
        historical_data,
        prediction_hours=3,
        rainfall_forecast=rainfall_forecast
    )
    
    # 結果の確認
    print("\n予測結果（最初の1時間）:")
    print(predictions[['時刻', '予測放流量', '使用降雨強度', '適用遅延', '状態', '変化率']].head(6))
    
    print("\n1時間後以降の使用降雨強度:")
    for i in range(6, min(12, len(predictions))):
        row = predictions.iloc[i]
        print(f"  {row['時刻'].strftime('%H:%M')} - {row['使用降雨強度']:.1f} mm/h")
    
    # 変化の傾向を確認
    initial_discharge = historical_data.iloc[-1]['ダム_全放流量']
    hour1_discharge = predictions.iloc[5]['予測放流量']  # 1時間後
    hour2_discharge = predictions.iloc[11]['予測放流量']  # 2時間後
    hour3_discharge = predictions.iloc[17]['予測放流量']  # 3時間後
    
    print(f"\n放流量の変化:")
    print(f"  現在: {initial_discharge:.1f} m³/s")
    print(f"  1時間後: {hour1_discharge:.1f} m³/s ({hour1_discharge - initial_discharge:+.1f})")
    print(f"  2時間後: {hour2_discharge:.1f} m³/s ({hour2_discharge - initial_discharge:+.1f})")
    print(f"  3時間後: {hour3_discharge:.1f} m³/s ({hour3_discharge - initial_discharge:+.1f})")
    
    # 可視化
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    
    # 放流量
    display_start = test_time - timedelta(hours=3)
    display_end = test_time + timedelta(hours=3)
    display_mask = (df['時刻'] >= display_start) & (df['時刻'] <= display_end)
    
    ax1.plot(df.loc[display_mask, '時刻'], df.loc[display_mask, 'ダム_全放流量'], 
            'k-', linewidth=2, label='実績')
    ax1.plot(predictions['時刻'], predictions['予測放流量'], 
            'r--', linewidth=2, label='予測（継続版）')
    ax1.axvline(x=test_time, color='green', linestyle='--', alpha=0.7, label='現在時刻')
    ax1.axvline(x=test_time + timedelta(hours=1), color='orange', linestyle='--', 
               alpha=0.7, label='1時間後')
    ax1.set_ylabel('放流量 (m³/s)')
    ax1.set_title('最新モデル（1時間後の値を継続）の予測結果')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 降雨強度と使用値
    ax2.bar(df.loc[display_mask, '時刻'], df.loc[display_mask, 'ダム_60分雨量'], 
           width=0.007, color='blue', alpha=0.7, label='実績降雨')
    ax2.plot(predictions['時刻'], predictions['使用降雨強度'], 
            'ro-', markersize=3, label='モデル使用値')
    ax2.axvline(x=test_time, color='green', linestyle='--', alpha=0.7)
    ax2.axvline(x=test_time + timedelta(hours=1), color='orange', linestyle='--', alpha=0.7)
    ax2.set_xlabel('時刻')
    ax2.set_ylabel('降雨強度 (mm/h)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f'latest_model_test_{timestamp}.png'
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\nグラフを保存: {filename}")

if __name__ == "__main__":
    test_latest_model()