#\!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
改善されたモデルのテスト
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from discharge_prediction_model_v2 import DischargePredictionModelV2
import matplotlib.pyplot as plt

# 日本語フォント設定
plt.rcParams['font.sans-serif'] = ['Yu Gothic', 'Hiragino Sans', 'Meirio', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def test_improved_model():
    """改善版モデルのテスト"""
    print("改善版放流量予測モデルv2のテスト")
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
    
    # 過去の放流量変化を確認
    past_30min = historical_data.iloc[-4]['ダム_全放流量']
    change_30min = historical_data.iloc[-1]['ダム_全放流量'] - past_30min
    print(f"\n過去30分の放流量変化: {change_30min:.1f} m³/s")
    
    # モデル初期化と状態判定
    model = DischargePredictionModelV2()
    discharge_history = historical_data.iloc[-7:]['ダム_全放流量']
    state, rate = model.identify_discharge_state(discharge_history)
    
    print(f"\n改善後のモデル:")
    print(f"  状態判定しきい値: {model.params['state_threshold']} m³/s")
    print(f"  判定結果: {['減少', '定常', '増加'][state+1]} (変化率: {rate:.1f} m³/s/10min)")
    
    # 予測実行
    predictions = model.predict(
        test_time, 
        historical_data,
        prediction_hours=3,
        rainfall_forecast=rainfall_forecast
    )
    
    # 結果の確認
    print("\n予測結果（最初の1時間）:")
    print("時刻     放流量  変化率  使用降雨  状態")
    print("-" * 45)
    for i in range(6):
        row = predictions.iloc[i]
        time_str = row['時刻'].strftime('%H:%M')
        discharge = row['予測放流量']
        change = row['変化率']
        rainfall = row['使用降雨強度']
        state = row['状態']
        
        state_str = {1: "増加", 0: "定常", -1: "減少"}[state]
        print(f"{time_str}  {discharge:6.1f}  {change:+6.1f}  {rainfall:6.1f}  {state_str}")
    
    print("\n1時間後以降の傾向:")
    for i in range(6, min(18, len(predictions)), 3):
        row = predictions.iloc[i]
        time_str = row['時刻'].strftime('%H:%M')
        discharge = row['予測放流量']
        change = row['変化率']
        state = row['状態']
        
        state_str = {1: "増加", 0: "定常", -1: "減少"}[state]
        print(f"{time_str}  {discharge:6.1f}  {change:+6.1f}  {state_str}")
    
    # 改善効果の確認
    print("\n改善効果の確認:")
    initial_discharge = historical_data.iloc[-1]['ダム_全放流量']
    final_discharge = predictions.iloc[-1]['予測放流量']
    total_change = final_discharge - initial_discharge
    
    print(f"  初期放流量: {initial_discharge:.1f} m³/s")
    print(f"  3時間後: {final_discharge:.1f} m³/s")
    print(f"  総変化量: {total_change:+.1f} m³/s")
    
    if total_change < 50:
        print("  → 改善成功：低降雨時の過度な増加が抑制されています")
    else:
        print("  → さらなる改善が必要かもしれません")
    
    # 可視化
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    
    # 放流量
    display_start = test_time - timedelta(hours=3)
    display_end = test_time + timedelta(hours=3)
    display_mask = (df['時刻'] >= display_start) & (df['時刻'] <= display_end)
    
    ax1.plot(df.loc[display_mask, '時刻'], df.loc[display_mask, 'ダム_全放流量'], 
            'k-', linewidth=2, label='実績')
    ax1.plot(predictions['時刻'], predictions['予測放流量'], 
            'r--', linewidth=2, label='予測（改善版）')
    ax1.axvline(x=test_time, color='green', linestyle='--', alpha=0.7, label='現在時刻')
    ax1.axhline(y=initial_discharge, color='gray', linestyle=':', alpha=0.5)
    ax1.set_ylabel('放流量 (m³/s)')
    ax1.set_title('改善版モデルの予測結果')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 降雨強度と状態
    ax2.bar(df.loc[display_mask, '時刻'], df.loc[display_mask, 'ダム_60分雨量'], 
           width=0.007, color='blue', alpha=0.7, label='実績降雨')
    
    # 状態を色分けして表示
    colors = {-1: 'red', 0: 'orange', 1: 'green'}
    for _, row in predictions.iterrows():
        ax2.scatter(row['時刻'], -2, color=colors[row['状態']], s=20)
    
    ax2.axvline(x=test_time, color='green', linestyle='--', alpha=0.7)
    ax2.set_xlabel('時刻')
    ax2.set_ylabel('降雨強度 (mm/h)')
    ax2.set_ylim(bottom=-5)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 状態の凡例
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='green', label='増加'),
        Patch(facecolor='orange', label='定常'),
        Patch(facecolor='red', label='減少')
    ]
    ax2.legend(handles=legend_elements, loc='upper right', title='状態')
    
    plt.tight_layout()
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f'improved_model_test_{timestamp}.png'
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\nグラフを保存: {filename}")
    
    # モデルを保存
    model.save_model(f'discharge_prediction_model_v2_improved_{timestamp}.pkl')
    print(f"改善版モデルを保存: discharge_prediction_model_v2_improved_{timestamp}.pkl")

if __name__ == "__main__":
    test_improved_model()
EOF < /dev/null
