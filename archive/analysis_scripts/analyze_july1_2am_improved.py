#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
改良版モデルを使用した2023年7月1日2:00時点の分析
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from discharge_prediction_model_v2 import DischargePredictionModelV2
import matplotlib.dates as mdates

# 日本語フォント設定
plt.rcParams['font.sans-serif'] = ['Yu Gothic', 'Hiragino Sans', 'Meirio', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def analyze_july1_2am():
    """2023年7月1日2:00時点の詳細分析"""
    print("2023年7月1日 2:00時点の分析（改良版モデル）")
    print("=" * 60)
    
    # データ読み込み
    df = pd.read_csv('統合データ_水位ダム_20250730_205325.csv', encoding='utf-8')
    df['時刻'] = pd.to_datetime(df['時刻'])
    
    # 改良版モデルの読み込み
    model = DischargePredictionModelV2()
    
    # 分析時点
    analysis_time = pd.to_datetime('2023-07-01 02:00')
    
    # 表示期間（前後6時間）
    display_start = analysis_time - timedelta(hours=6)
    display_end = analysis_time + timedelta(hours=6)
    
    # データ抽出
    mask = (df['時刻'] >= display_start) & (df['時刻'] <= display_end)
    display_data = df[mask].copy()
    
    # 現在時刻までのデータ
    historical_mask = df['時刻'] <= analysis_time
    historical_data = df[historical_mask].copy()
    
    # 現在時点の状況を表示
    current_idx = df[df['時刻'] == analysis_time].index[0]
    current_data = df.iloc[current_idx]
    
    print(f"\n現在時刻: {analysis_time}")
    print(f"降雨強度: {current_data['ダム_60分雨量']:.1f} mm/h")
    print(f"放流量: {current_data['ダム_全放流量']:.1f} m³/s")
    print(f"貯水位: {current_data['ダム_貯水位']:.2f} m")
    print(f"河川水位: {current_data['水位_水位']:.2f} m")
    
    # 過去1時間の状況
    past_1h_mask = (df['時刻'] > analysis_time - timedelta(hours=1)) & (df['時刻'] <= analysis_time)
    past_1h = df[past_1h_mask]
    
    print(f"\n過去1時間の推移:")
    print(f"降雨強度: {past_1h['ダム_60分雨量'].min():.1f} → {past_1h['ダム_60分雨量'].max():.1f} mm/h")
    print(f"放流量: {past_1h['ダム_全放流量'].min():.1f} → {past_1h['ダム_全放流量'].max():.1f} m³/s")
    
    # 予測実行
    print("\n改良版モデルによる予測を実行...")
    predictions = model.predict(analysis_time, historical_data, prediction_hours=4)
    
    # 詳細な予測情報
    print("\n予測の詳細（最初の1時間）:")
    print(predictions[['時刻', '予測放流量', '使用降雨強度', '適用遅延', '状態', '変化率']].head(6))
    
    # 可視化
    fig, axes = plt.subplots(4, 1, figsize=(14, 16), sharex=True)
    
    # 1. 放流量（予測と実績）
    ax = axes[0]
    # 実績データ
    ax.plot(display_data['時刻'], display_data['ダム_全放流量'], 
           'k-', linewidth=2, label='実績', zorder=3)
    # 予測データ
    ax.plot(predictions['時刻'], predictions['予測放流量'], 
           'r--', linewidth=2, label='予測（改良版）', zorder=2)
    
    # 現在時刻と重要な時刻をマーク
    ax.axvline(x=analysis_time, color='green', linestyle='--', alpha=0.7, 
              label='分析時点 (2:00)', linewidth=2)
    ax.axvline(x=pd.to_datetime('2023-07-01 00:00'), color='blue', 
              linestyle=':', alpha=0.5, label='降雨増加開始')
    ax.axvline(x=pd.to_datetime('2023-07-01 03:00'), color='orange', 
              linestyle=':', alpha=0.5, label='放流量ピーク')
    
    # 予測の特徴点を注釈
    peak_idx = predictions['予測放流量'].idxmax()
    peak_time = predictions.iloc[peak_idx]['時刻']
    peak_value = predictions.iloc[peak_idx]['予測放流量']
    ax.annotate(f'予測ピーク\n{peak_value:.0f} m³/s', 
               xy=(peak_time, peak_value), xytext=(peak_time + timedelta(hours=0.5), peak_value + 100),
               arrowprops=dict(arrowstyle='->', color='red', alpha=0.7),
               fontsize=10, ha='left')
    
    ax.set_ylabel('放流量 (m³/s)', fontsize=12)
    ax.set_title('放流量の実績と予測（2023年7月1日 2:00時点）', fontsize=14)
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)
    
    # 2. 降雨強度と使用降雨強度
    ax = axes[1]
    # 実際の降雨
    ax.bar(display_data['時刻'], display_data['ダム_60分雨量'], 
          width=0.007, color='blue', alpha=0.7, label='実際の降雨強度')
    # 予測で使用した降雨
    ax.bar(predictions['時刻'], predictions['使用降雨強度'], 
          width=0.007, color='cyan', alpha=0.5, label='予測で使用した降雨')
    
    ax.axvline(x=analysis_time, color='green', linestyle='--', alpha=0.7, linewidth=2)
    ax.set_ylabel('降雨強度 (mm/h)', fontsize=12)
    ax.set_title('降雨強度の推移', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)
    
    # 3. 適用遅延時間と状態
    ax = axes[2]
    # 遅延時間
    ax.plot(predictions['時刻'], predictions['適用遅延'], 
           'g-', linewidth=2, label='適用遅延時間')
    ax.set_ylabel('遅延時間 (分)', fontsize=12, color='green')
    ax.tick_params(axis='y', labelcolor='green')
    
    # 状態を背景色で表示
    ax2 = ax.twinx()
    states = predictions['状態'].values
    times = predictions['時刻'].values
    
    # 状態ごとに色分け
    for i in range(len(states)):
        if states[i] == 1:
            ax2.axvspan(times[i], times[min(i+1, len(times)-1)], 
                       alpha=0.2, color='red', label='増加中' if i == 0 else '')
        elif states[i] == -1:
            ax2.axvspan(times[i], times[min(i+1, len(times)-1)], 
                       alpha=0.2, color='blue', label='減少中' if i == 0 else '')
        else:
            ax2.axvspan(times[i], times[min(i+1, len(times)-1)], 
                       alpha=0.2, color='gray', label='定常' if i == 0 else '')
    
    ax2.set_ylabel('状態', fontsize=12)
    ax2.set_ylim(-2, 2)
    ax2.set_yticks([-1, 0, 1])
    ax2.set_yticklabels(['減少', '定常', '増加'])
    
    ax.axvline(x=analysis_time, color='green', linestyle='--', alpha=0.7, linewidth=2)
    ax.set_title('適用遅延時間と放流状態', fontsize=14)
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    
    # 4. 変化率の推移
    ax = axes[3]
    ax.plot(predictions['時刻'], predictions['変化率'], 
           'm-', linewidth=2, label='予測変化率')
    ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax.fill_between(predictions['時刻'], 0, predictions['変化率'], 
                   where=(predictions['変化率'] > 0), alpha=0.3, color='red', 
                   label='増加')
    ax.fill_between(predictions['時刻'], 0, predictions['変化率'], 
                   where=(predictions['変化率'] < 0), alpha=0.3, color='blue', 
                   label='減少')
    
    ax.axvline(x=analysis_time, color='green', linestyle='--', alpha=0.7, linewidth=2)
    ax.set_xlabel('時刻', fontsize=12)
    ax.set_ylabel('変化率 (m³/s/10min)', fontsize=12)
    ax.set_title('放流量変化率の推移', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # x軸のフォーマット
    for ax in axes:
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d %H:%M'))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
    
    plt.tight_layout()
    
    # 保存
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f'july1_2am_analysis_improved_{timestamp}.png'
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\n分析結果を保存: {filename}")
    
    # 実績との比較（予測精度）
    future_mask = (df['時刻'] > analysis_time) & (df['時刻'] <= analysis_time + timedelta(hours=4))
    actual_future = df[future_mask]
    
    # 予測と実績のマージ
    merged = pd.merge_asof(
        predictions[['時刻', '予測放流量']],
        actual_future[['時刻', 'ダム_全放流量']],
        on='時刻',
        direction='nearest',
        tolerance=pd.Timedelta('10min')
    )
    
    if len(merged) > 0:
        print("\n予測精度評価:")
        for hours in [1, 2, 3, 4]:
            hour_mask = merged['時刻'] <= analysis_time + timedelta(hours=hours)
            if hour_mask.sum() > 0:
                mae = np.mean(np.abs(merged.loc[hour_mask, '予測放流量'] - 
                                   merged.loc[hour_mask, 'ダム_全放流量']))
                rmse = np.sqrt(np.mean((merged.loc[hour_mask, '予測放流量'] - 
                                      merged.loc[hour_mask, 'ダム_全放流量'])**2))
                print(f"  {hours}時間先まで - MAE: {mae:.1f} m³/s, RMSE: {rmse:.1f} m³/s")
    
    # 分析のポイント
    print("\n分析のポイント:")
    print("1. 2:00時点は降雨ピーク（54mm/h）から1時間経過")
    print("2. 放流量は増加継続中（1078.5 m³/s）")
    print("3. 改良版モデルは120分遅延を適用し、さらなる増加を予測")
    print("4. 実際のピーク（3:00の1188.2 m³/s）に向けて増加を継続")
    
    return predictions

if __name__ == "__main__":
    predictions = analyze_july1_2am()