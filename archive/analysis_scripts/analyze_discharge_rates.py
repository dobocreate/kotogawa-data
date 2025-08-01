#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
放流量の増加率・減少率の詳細分析
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# 日本語フォント設定
plt.rcParams['font.sans-serif'] = ['Yu Gothic', 'Hiragino Sans', 'Meirio', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def analyze_discharge_rates():
    """放流量の増加率・減少率を詳細分析"""
    print("放流量の増加率・減少率分析")
    print("=" * 60)
    
    # データ読み込み
    df = pd.read_csv('統合データ_水位ダム_20250730_205325.csv', encoding='utf-8')
    df['時刻'] = pd.to_datetime(df['時刻'])
    
    # 必要なカラムを抽出
    df = df[['時刻', 'ダム_60分雨量', 'ダム_全放流量']].copy()
    df.columns = ['時刻', '降雨強度', '放流量']
    
    # 有効なデータのみ（放流量 >= 150）
    df = df[df['放流量'] >= 150].copy()
    
    # 変化率の計算（10分, 30分, 60分）
    for interval in [1, 3, 6]:  # 10分, 30分, 60分
        df[f'放流量変化_{interval*10}分'] = df['放流量'].diff(interval)
        df[f'放流量変化率_{interval*10}分'] = df[f'放流量変化_{interval*10}分'] / interval
    
    # 降雨強度の移動平均（ノイズ除去）
    df['降雨強度_MA30'] = df['降雨強度'].rolling(3, center=True).mean()
    
    # 状態の分類
    threshold = 50  # m³/s per 30min
    df['状態'] = 'stable'
    df.loc[df['放流量変化_30分'] > threshold, '状態'] = 'increasing'
    df.loc[df['放流量変化_30分'] < -threshold, '状態'] = 'decreasing'
    
    # 降雨強度別の分析
    print("\n=== 降雨強度別の放流量変化率 ===")
    
    rainfall_bins = [(0, 10), (10, 20), (20, 50), (50, 100), (100, 300)]
    results = []
    
    for rain_min, rain_max in rainfall_bins:
        mask = (df['降雨強度'] >= rain_min) & (df['降雨強度'] < rain_max)
        
        # 各状態での統計
        for state in ['increasing', 'decreasing', 'stable']:
            state_mask = mask & (df['状態'] == state)
            if state_mask.sum() > 10:  # 十分なサンプル数がある場合
                rate_30 = df.loc[state_mask, '放流量変化率_30分'].abs()
                
                result = {
                    '降雨強度範囲': f'{rain_min}-{rain_max}',
                    '状態': state,
                    'サンプル数': state_mask.sum(),
                    '平均変化率_30分': rate_30.mean(),
                    '中央値変化率_30分': rate_30.median(),
                    '95%ile変化率_30分': rate_30.quantile(0.95)
                }
                results.append(result)
                
                print(f"{rain_min}-{rain_max}mm/h, {state}: "
                      f"平均{rate_30.mean():.1f}, 中央値{rate_30.median():.1f}, "
                      f"95%ile{rate_30.quantile(0.95):.1f} m³/s/10min")
    
    results_df = pd.DataFrame(results)
    
    # 可視化
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. 状態別の変化率分布
    ax = axes[0, 0]
    for state, color in [('increasing', 'red'), ('decreasing', 'blue'), ('stable', 'gray')]:
        state_data = df[df['状態'] == state]['放流量変化率_30分'].abs()
        if len(state_data) > 0:
            ax.hist(state_data, bins=50, alpha=0.5, label=state, color=color, 
                   range=(0, 200))
    ax.set_xlabel('放流量変化率の絶対値 (m³/s/10min)')
    ax.set_ylabel('頻度')
    ax.set_title('状態別の放流量変化率分布')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. 降雨強度と放流量変化率の関係
    ax = axes[0, 1]
    for state, color in [('increasing', 'red'), ('decreasing', 'blue')]:
        state_mask = df['状態'] == state
        if state_mask.sum() > 0:
            ax.scatter(df.loc[state_mask, '降雨強度'], 
                      df.loc[state_mask, '放流量変化率_30分'].abs(),
                      alpha=0.3, s=10, label=state, color=color)
    ax.set_xlabel('降雨強度 (mm/h)')
    ax.set_ylabel('放流量変化率の絶対値 (m³/s/10min)')
    ax.set_title('降雨強度と放流量変化率の関係')
    ax.set_xlim(0, 200)
    ax.set_ylim(0, 300)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. 増加率と減少率の比較（降雨強度別）
    ax = axes[1, 0]
    rain_ranges = ['0-10', '10-20', '20-50', '50-100', '100-300']
    increase_rates = []
    decrease_rates = []
    
    for rain_range in rain_ranges:
        inc_data = results_df[(results_df['降雨強度範囲'] == rain_range) & 
                             (results_df['状態'] == 'increasing')]
        dec_data = results_df[(results_df['降雨強度範囲'] == rain_range) & 
                             (results_df['状態'] == 'decreasing')]
        
        increase_rates.append(inc_data['平均変化率_30分'].values[0] if len(inc_data) > 0 else 0)
        decrease_rates.append(dec_data['平均変化率_30分'].values[0] if len(dec_data) > 0 else 0)
    
    x = np.arange(len(rain_ranges))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, increase_rates, width, label='増加率', color='red', alpha=0.7)
    bars2 = ax.bar(x + width/2, decrease_rates, width, label='減少率', color='blue', alpha=0.7)
    
    ax.set_xlabel('降雨強度範囲 (mm/h)')
    ax.set_ylabel('平均変化率 (m³/s/10min)')
    ax.set_title('降雨強度別の増加率・減少率比較')
    ax.set_xticks(x)
    ax.set_xticklabels(rain_ranges)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # 4. 時間遅れ相関分析
    ax = axes[1, 1]
    
    # 降雨強度の変化と放流量変化の相関を時間遅れで分析
    max_lag = 18  # 3時間
    correlations = []
    
    for lag in range(-6, max_lag + 1):  # -1時間から3時間
        if lag < 0:
            corr = df['降雨強度'].iloc[-lag:].corr(
                df['放流量変化率_30分'].iloc[:lag].abs())
        elif lag > 0:
            corr = df['降雨強度'].iloc[:-lag].corr(
                df['放流量変化率_30分'].iloc[lag:].abs())
        else:
            corr = df['降雨強度'].corr(df['放流量変化率_30分'].abs())
        
        correlations.append({
            'lag': lag * 10,
            'correlation': corr
        })
    
    corr_df = pd.DataFrame(correlations)
    ax.plot(corr_df['lag'], corr_df['correlation'], 'g-', linewidth=2)
    ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('時間遅れ (分)')
    ax.set_ylabel('相関係数')
    ax.set_title('降雨強度と放流量変化率の時間遅れ相関')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 保存
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f'discharge_rate_analysis_{timestamp}.png'
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\nグラフを保存: {filename}")
    
    # 分析結果をCSVで保存
    results_df.to_csv(f'discharge_rate_statistics_{timestamp}.csv', 
                      index=False, encoding='utf-8')
    
    # 推奨パラメータの計算
    print("\n=== 推奨パラメータ ===")
    
    # 降雨強度閾値での典型的な変化率
    low_rain_inc = results_df[(results_df['降雨強度範囲'] == '0-10') & 
                              (results_df['状態'] == 'increasing')]['平均変化率_30分'].values
    low_rain_dec = results_df[(results_df['降雨強度範囲'] == '0-10') & 
                              (results_df['状態'] == 'decreasing')]['平均変化率_30分'].values
    
    high_rain_inc = results_df[(results_df['降雨強度範囲'] == '20-50') & 
                               (results_df['状態'] == 'increasing')]['平均変化率_30分'].values
    
    print(f"低降雨時（<10mm/h）の増加率: {low_rain_inc[0] if len(low_rain_inc) > 0 else 'N/A':.1f} m³/s/10min")
    print(f"低降雨時（<10mm/h）の減少率: {low_rain_dec[0] if len(low_rain_dec) > 0 else 'N/A':.1f} m³/s/10min")
    print(f"中降雨時（20-50mm/h）の増加率: {high_rain_inc[0] if len(high_rain_inc) > 0 else 'N/A':.1f} m³/s/10min")
    
    # 減少率/増加率の比率
    if len(low_rain_inc) > 0 and len(low_rain_dec) > 0:
        ratio = low_rain_dec[0] / low_rain_inc[0]
        print(f"\n減少率/増加率の比率: {ratio:.2%}")
    
    return results_df

if __name__ == "__main__":
    results = analyze_discharge_rates()
    print("\n分析完了！")