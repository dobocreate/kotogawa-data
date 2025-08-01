#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
重要時点の詳細分析
2023-06-30 23:00前後の状況を詳しく分析
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# 日本語フォント設定
plt.rcParams['font.sans-serif'] = ['Yu Gothic', 'Hiragino Sans', 'Meirio', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def analyze_critical_timepoint():
    """重要時点の分析"""
    
    # データ読み込み
    print("データ読み込み中...")
    data = pd.read_csv('統合データ_水位ダム_20250730_142903.csv', encoding='utf-8')
    data['時刻'] = pd.to_datetime(data['時刻'])
    
    # カラム名を整理
    data = data[['時刻', '水位_水位', 'ダム_全放流量', 'ダム_流入量']].copy()
    data.columns = ['時刻', '水位', '放流量', '流入量']
    
    # 2023-06-30 20:00 から 2023-07-01 04:00 のデータを抽出
    start_time = pd.to_datetime('2023-06-30 20:00')
    end_time = pd.to_datetime('2023-07-01 04:00')
    mask = (data['時刻'] >= start_time) & (data['時刻'] <= end_time)
    period_data = data[mask].copy()
    
    # 派生特徴量を計算
    period_data['流入放流差'] = period_data['流入量'] - period_data['放流量']
    period_data['流入量変化'] = period_data['流入量'].diff()
    period_data['放流量変化'] = period_data['放流量'].diff()
    period_data['水位変化'] = period_data['水位'].diff()
    
    # 累積値
    period_data['流入放流差累積_1h'] = period_data['流入放流差'].rolling(6, min_periods=1).sum()
    period_data['流入放流差累積_2h'] = period_data['流入放流差'].rolling(12, min_periods=1).sum()
    
    # 2023-06-30 23:00の時点を特定
    critical_time = pd.to_datetime('2023-06-30 23:00')
    critical_idx = period_data[period_data['時刻'] == critical_time].index[0]
    
    print(f"\n=== {critical_time} 時点の状況 ===")
    critical_row = period_data.loc[critical_idx]
    print(f"水位: {critical_row['水位']:.2f} m")
    print(f"放流量: {critical_row['放流量']:.0f} m³/s")
    print(f"流入量: {critical_row['流入量']:.0f} m³/s")
    print(f"流入放流差: {critical_row['流入放流差']:.0f} m³/s")
    print(f"過去1時間の流入放流差累積: {critical_row['流入放流差累積_1h']:.0f} m³/s")
    print(f"過去2時間の流入放流差累積: {critical_row['流入放流差累積_2h']:.0f} m³/s")
    
    # 前後1時間の変化を確認
    print("\n=== 前後の変化 ===")
    
    # 1時間前
    if critical_idx - 6 >= period_data.index[0]:
        past_1h = period_data.loc[critical_idx - 6]
        print(f"\n1時間前 ({past_1h['時刻']}):")
        print(f"  水位: {past_1h['水位']:.2f} m → {critical_row['水位']:.2f} m (変化: {critical_row['水位'] - past_1h['水位']:.2f} m)")
        print(f"  放流量: {past_1h['放流量']:.0f} m³/s → {critical_row['放流量']:.0f} m³/s")
        print(f"  流入量: {past_1h['流入量']:.0f} m³/s → {critical_row['流入量']:.0f} m³/s")
    
    # 1時間後の実際の変化
    if critical_idx + 6 < period_data.index[-1]:
        future_1h = period_data.loc[critical_idx + 6]
        print(f"\n1時間後 ({future_1h['時刻']}) の実際:")
        print(f"  水位: {critical_row['水位']:.2f} m → {future_1h['水位']:.2f} m (変化: {future_1h['水位'] - critical_row['水位']:.2f} m)")
        print(f"  放流量: {critical_row['放流量']:.0f} m³/s → {future_1h['放流量']:.0f} m³/s (変化: {future_1h['放流量'] - critical_row['放流量']:.0f} m³/s)")
        print(f"  流入量: {critical_row['流入量']:.0f} m³/s → {future_1h['流入量']:.0f} m³/s")
    
    # 放流量が急増したタイミングを特定
    discharge_increase_mask = period_data['放流量変化'] > 100
    discharge_increases = period_data[discharge_increase_mask]
    
    print("\n=== 放流量急増のタイミング ===")
    for idx, row in discharge_increases.iterrows():
        print(f"{row['時刻']}: 放流量 +{row['放流量変化']:.0f} m³/s (総量: {row['放流量']:.0f} m³/s)")
        
        # その時点の状況
        if idx >= 6:
            past_1h_sum = period_data.loc[idx-6:idx, '流入放流差'].sum()
            past_2h_sum = period_data.loc[max(period_data.index[0], idx-12):idx, '流入放流差'].sum()
            water_change_1h = period_data.loc[idx, '水位'] - period_data.loc[idx-6, '水位']
            
            print(f"  過去1時間の流入放流差累積: {past_1h_sum:.0f} m³/s")
            print(f"  過去2時間の流入放流差累積: {past_2h_sum:.0f} m³/s")
            print(f"  過去1時間の水位変化: {water_change_1h:.2f} m")
            print(f"  流入量: {row['流入量']:.0f} m³/s")
    
    # 可視化
    fig, axes = plt.subplots(4, 1, figsize=(14, 16), sharex=True)
    
    # 1. 流入量と放流量
    ax1 = axes[0]
    ax1.plot(period_data['時刻'], period_data['流入量'], 'g-', linewidth=2, label='流入量')
    ax1.plot(period_data['時刻'], period_data['放流量'], 'b-', linewidth=2, label='放流量')
    ax1.axvline(x=critical_time, color='red', linestyle='--', alpha=0.5, label='23:00')
    
    # 放流量急増点をマーク
    for idx, row in discharge_increases.iterrows():
        ax1.axvline(x=row['時刻'], color='orange', linestyle=':', alpha=0.5)
        ax1.text(row['時刻'], ax1.get_ylim()[1]*0.9, f"+{row['放流量変化']:.0f}", 
                rotation=90, ha='right', va='top', fontsize=8)
    
    ax1.set_ylabel('流量 (m³/s)')
    ax1.set_title('流入量と放流量の推移')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. 水位
    ax2 = axes[1]
    ax2.plot(period_data['時刻'], period_data['水位'], 'r-', linewidth=2)
    ax2.axvline(x=critical_time, color='red', linestyle='--', alpha=0.5)
    ax2.axhline(y=5.5, color='red', linestyle='--', alpha=0.3, label='氾濫危険水位')
    ax2.set_ylabel('水位 (m)')
    ax2.set_title('水位の推移')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. 流入放流差と累積
    ax3 = axes[2]
    ax3.plot(period_data['時刻'], period_data['流入放流差'], 'k-', linewidth=2, label='流入放流差')
    ax3.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax3.fill_between(period_data['時刻'], 0, period_data['流入放流差'],
                    where=(period_data['流入放流差'] > 0), alpha=0.3, color='red', label='流入超過')
    
    ax3_twin = ax3.twinx()
    ax3_twin.plot(period_data['時刻'], period_data['流入放流差累積_1h'], 
                 'r--', linewidth=2, alpha=0.7, label='1時間累積')
    ax3_twin.plot(period_data['時刻'], period_data['流入放流差累積_2h'], 
                 'r:', linewidth=2, alpha=0.7, label='2時間累積')
    
    ax3.axvline(x=critical_time, color='red', linestyle='--', alpha=0.5)
    ax3.set_ylabel('流入放流差 (m³/s)')
    ax3_twin.set_ylabel('累積 (m³/s)')
    ax3.set_title('流入放流差とその累積')
    
    # 凡例
    lines1, labels1 = ax3.get_legend_handles_labels()
    lines2, labels2 = ax3_twin.get_legend_handles_labels()
    ax3.legend(lines1 + lines2, labels1 + labels2, loc='best')
    ax3.grid(True, alpha=0.3)
    
    # 4. 変化率
    ax4 = axes[3]
    ax4.plot(period_data['時刻'], period_data['流入量変化'], 'g-', linewidth=2, label='流入量変化', alpha=0.7)
    ax4.plot(period_data['時刻'], period_data['放流量変化'], 'b-', linewidth=2, label='放流量変化')
    ax4.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax4.axhline(y=100, color='orange', linestyle='--', alpha=0.3, label='急増閾値(100)')
    ax4.axvline(x=critical_time, color='red', linestyle='--', alpha=0.5)
    
    ax4.set_xlabel('時刻')
    ax4.set_ylabel('変化量 (m³/s/10分)')
    ax4.set_title('流入量と放流量の変化率')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # x軸のフォーマット
    import matplotlib.dates as mdates
    ax4.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d %H:%M'))
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    
    # 保存
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"critical_timepoint_analysis_{timestamp}.png"
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"\n分析結果を保存: {filename}")
    
    plt.show()
    
    # 予測ルールの提案
    print("\n=== 予測ルール改善の提案 ===")
    
    # 23:00時点の条件を確認
    print(f"\n23:00時点の条件:")
    print(f"  過去1時間の流入放流差累積: {critical_row['流入放流差累積_1h']:.0f} m³/s")
    print(f"  過去2時間の流入放流差累積: {critical_row['流入放流差累積_2h']:.0f} m³/s")
    
    # 実際に放流量が増加した時点での条件
    if len(discharge_increases) > 0:
        first_increase = discharge_increases.iloc[0]
        first_idx = discharge_increases.index[0]
        
        if first_idx >= 12:
            conditions = period_data.loc[first_idx]
            print(f"\n実際に放流量が増加した時点 ({first_increase['時刻']}) の条件:")
            print(f"  過去1時間の流入放流差累積: {conditions['流入放流差累積_1h']:.0f} m³/s")
            print(f"  過去2時間の流入放流差累積: {conditions['流入放流差累積_2h']:.0f} m³/s")
            print(f"  水位: {conditions['水位']:.2f} m")
            print(f"  流入量: {conditions['流入量']:.0f} m³/s")
    
    print("\n改善案:")
    print("1. 流入放流差累積の閾値を下げる（500 → 100-200 m³/s）")
    print("2. 2時間累積も考慮に入れる")
    print("3. 流入量の絶対値だけでなく、増加傾向も考慮")
    print("4. 水位が4.5m以上の場合は、より積極的に放流量を増加")
    
    return period_data

if __name__ == "__main__":
    period_data = analyze_critical_timepoint()