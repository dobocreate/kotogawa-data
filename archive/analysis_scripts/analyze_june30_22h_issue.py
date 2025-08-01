#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
2023年6月30日22:00時点の予測分析
降雨予測と放流量予測の不整合を調査
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import matplotlib.dates as mdates

# 日本語フォント設定
plt.rcParams['font.sans-serif'] = ['Yu Gothic', 'Hiragino Sans', 'Meirio', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def analyze_prediction_issue():
    """予測の問題を分析"""
    print("2023年6月30日 22:00時点の予測分析")
    print("=" * 60)
    
    # データ読み込み
    df = pd.read_csv('統合データ_水位ダム_20250730_205325.csv', encoding='utf-8')
    df['時刻'] = pd.to_datetime(df['時刻'])
    
    # 分析時点
    analysis_time = pd.to_datetime('2023-06-30 22:00')
    
    # 前後のデータを抽出
    start_time = analysis_time - timedelta(hours=3)
    end_time = analysis_time + timedelta(hours=3)
    mask = (df['時刻'] >= start_time) & (df['時刻'] <= end_time)
    event_data = df[mask].copy()
    
    # 22:00時点のデータ
    current_idx = event_data[event_data['時刻'] == analysis_time].index[0]
    current_data = event_data.loc[current_idx]
    
    print(f"\n現在時刻: {analysis_time}")
    print(f"現在の降雨: {current_data['ダム_60分雨量']:.1f} mm/h")
    print(f"現在の放流量: {current_data['ダム_全放流量']:.1f} m³/s")
    print(f"現在の貯水位: {current_data['ダム_貯水位']:.2f} m")
    
    # 1時間先までの降雨予測（実データ）
    future_1h_mask = (event_data['時刻'] > analysis_time) & \
                     (event_data['時刻'] <= analysis_time + timedelta(hours=1))
    future_1h = event_data[future_1h_mask]
    
    print("\n1時間先までの降雨予測（実データ）:")
    for _, row in future_1h.iterrows():
        print(f"  {row['時刻'].strftime('%H:%M')} - {row['ダム_60分雨量']:.1f} mm/h")
    
    # 120分前の降雨データ（遅延考慮）
    delay_time = analysis_time - timedelta(minutes=120)
    delay_mask = (event_data['時刻'] >= delay_time - timedelta(minutes=30)) & \
                 (event_data['時刻'] <= delay_time + timedelta(minutes=30))
    delay_data = event_data[delay_mask]
    
    print("\n120分前の降雨データ（放流量に影響）:")
    for _, row in delay_data.iterrows():
        print(f"  {row['時刻'].strftime('%H:%M')} - {row['ダム_60分雨量']:.1f} mm/h")
    
    # 状態判定
    recent_30min = event_data[
        (event_data['時刻'] > analysis_time - timedelta(minutes=30)) & 
        (event_data['時刻'] <= analysis_time)
    ]
    
    if len(recent_30min) >= 4:
        discharge_change = current_data['ダム_全放流量'] - recent_30min.iloc[0]['ダム_全放流量']
        print(f"\n過去30分の放流量変化: {discharge_change:.1f} m³/s")
        
        if discharge_change > 50:
            state = "増加中"
        elif discharge_change < -50:
            state = "減少中"
        else:
            state = "定常"
        print(f"現在の状態: {state}")
    
    # 可視化
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    
    # 1. 降雨強度
    ax = axes[0]
    ax.bar(event_data['時刻'], event_data['ダム_60分雨量'], 
           width=0.007, color='blue', alpha=0.7)
    ax.axvline(x=analysis_time, color='red', linestyle='--', label='現在時刻')
    ax.axvline(x=delay_time, color='orange', linestyle='--', label='120分前')
    
    # 1時間予測範囲
    ax.axvspan(analysis_time, analysis_time + timedelta(hours=1), 
              alpha=0.2, color='yellow', label='予測範囲')
    
    ax.set_ylabel('降雨強度 (mm/h)')
    ax.set_title('降雨強度の推移と予測')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. 放流量
    ax = axes[1]
    ax.plot(event_data['時刻'], event_data['ダム_全放流量'], 
           'g-', linewidth=2)
    ax.axvline(x=analysis_time, color='red', linestyle='--')
    ax.axhline(y=current_data['ダム_全放流量'], color='gray', 
              linestyle=':', alpha=0.5)
    
    ax.set_ylabel('放流量 (m³/s)')
    ax.set_title('放流量の推移')
    ax.grid(True, alpha=0.3)
    
    # 3. 貯水位
    ax = axes[2]
    ax.plot(event_data['時刻'], event_data['ダム_貯水位'], 
           'm-', linewidth=2)
    ax.axvline(x=analysis_time, color='red', linestyle='--')
    ax.axhline(y=38.0, color='orange', linestyle='--', 
              alpha=0.5, label='洪水貯留準備水位')
    
    ax.set_xlabel('時刻')
    ax.set_ylabel('貯水位 (m)')
    ax.set_title('貯水位の推移')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    
    plt.tight_layout()
    
    # 保存
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f'june30_22h_analysis_{timestamp}.png'
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\n分析グラフを保存: {filename}")
    
    # 問題の分析
    print("\n=== 問題の分析 ===")
    print("1. 降雨予測:")
    print("   - 22:10-22:30: 1-2mm/h（ほぼ降雨なし）")
    print("   - 22:40以降: 29mm/h以上に急増")
    
    print("\n2. 放流量が増加する理由:")
    print("   - 120分前（20:00）の降雨: 1mm/h")
    print("   - 現在の放流量状態: 定常または微増")
    print("   - モデルの問題: 降雨増加の検出が早すぎる可能性")
    
    print("\n3. 改良版モデルv2の動作:")
    print("   - 遅延時間の決定ロジックに問題がある可能性")
    print("   - 将来の降雨増加を先読みしている可能性")

if __name__ == "__main__":
    analyze_prediction_issue()