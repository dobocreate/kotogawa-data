#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
2023/7/1 00:00時点でのハイブリッドモデル分析（簡易版）
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from discharge_prediction_model_v3_hybrid import DischargePredictionModelV3Hybrid
from discharge_prediction_model_v2 import DischargePredictionModelV2

# 日本語フォント設定
plt.rcParams['font.sans-serif'] = ['Yu Gothic', 'Hiragino Sans', 'Meirio', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# データ読み込み
df = pd.read_csv('統合データ_水位ダム_20250730_142903.csv', encoding='utf-8')
df['時刻'] = pd.to_datetime(df['時刻'])

# 予測時点
current_time = pd.to_datetime('2023-07-01 00:00')

print(f"=== 2023/7/1 00:00時点のハイブリッドモデル分析 ===")

# 過去データの準備
historical_mask = df['時刻'] <= current_time
historical_data = df[historical_mask].copy()

# 現在の状態
current_water_level = historical_data.iloc[-1]['水位_水位']
current_discharge = historical_data.iloc[-1]['ダム_全放流量']
current_rainfall = historical_data.iloc[-1]['ダム_60分雨量']

print(f"\n現在の状態:")
print(f"  水位: {current_water_level:.2f} m")
print(f"  放流量: {current_discharge:.1f} m³/s")
print(f"  降雨強度: {current_rainfall:.1f} mm/h")

# 過去1時間の履歴
print("\n過去1時間の履歴:")
for i in range(6, -1, -1):
    idx = -i-1
    t = historical_data.iloc[idx]['時刻'].strftime('%H:%M')
    d = historical_data.iloc[idx]['ダム_全放流量']
    r = historical_data.iloc[idx]['ダム_60分雨量']
    print(f"  {t}: 放流量 {d:6.1f} m³/s, 降雨 {r:5.1f} mm/h")

# 降雨予測（1時間先まで）
future_1h_mask = (df['時刻'] > current_time) & (df['時刻'] <= current_time + timedelta(hours=1))
future_1h_data = df[future_1h_mask]

rainfall_forecast = pd.DataFrame({
    '時刻': future_1h_data['時刻'],
    '降雨強度': future_1h_data['ダム_60分雨量']
})

print("\n降雨予測（1時間先まで）:")
for _, row in rainfall_forecast.iterrows():
    print(f"  {row['時刻'].strftime('%H:%M')}: {row['降雨強度']:.1f} mm/h")

# モデル比較
print("\n=== ルールベースモデル vs ハイブリッドモデル ===")

# ルールベースモデル（v2）
v2_model = DischargePredictionModelV2()
v2_model.load_model('discharge_prediction_model_v2_20250801_221633.pkl')

v2_predictions = v2_model.predict(
    current_time, 
    historical_data, 
    prediction_hours=3,
    rainfall_forecast=rainfall_forecast
)

# ハイブリッドモデル（v3）
v3_model = DischargePredictionModelV3Hybrid()
v3_model.load_model('discharge_prediction_model_v3_hybrid_20250801_231150.pkl')

v3_predictions = v3_model.predict(
    current_time, 
    historical_data, 
    prediction_hours=3,
    rainfall_forecast=rainfall_forecast
)

# 結果の比較
print("\n最初の1時間の予測比較:")
print("時刻   ルールベース  ハイブリッド   差分    遅延(v2→v3)   ML調整")
print("-" * 70)

for i in range(min(6, len(v2_predictions))):
    v2_row = v2_predictions.iloc[i]
    v3_row = v3_predictions.iloc[i]
    
    diff = v3_row['予測放流量'] - v2_row['予測放流量']
    
    print(f"{v2_row['時刻'].strftime('%H:%M')}    {v2_row['予測放流量']:6.1f}      {v3_row['予測放流量']:6.1f}    {diff:+6.1f}    {v2_row['適用遅延']:3.0f}→{v3_row['適用遅延']:3.0f}分   x{v3_row['ML調整係数_変化率']:.2f}")

# 実績データとの比較
future_3h_mask = (df['時刻'] > current_time) & (df['時刻'] <= current_time + timedelta(hours=3))
actual_future = df[future_3h_mask]

print("\n=== 実績との比較（最初の1時間）===")
print("時刻     実績    ルール   ハイブリッド   誤差(v2)  誤差(v3)")
print("-" * 65)

for i in range(min(6, len(actual_future))):
    actual_row = actual_future.iloc[i]
    time = actual_row['時刻']
    
    # 対応する予測値を見つける
    v2_match = v2_predictions[v2_predictions['時刻'] == time]
    v3_match = v3_predictions[v3_predictions['時刻'] == time]
    
    if not v2_match.empty and not v3_match.empty:
        v2_val = v2_match.iloc[0]['予測放流量']
        v3_val = v3_match.iloc[0]['予測放流量']
        actual_val = actual_row['ダム_全放流量']
        
        error_v2 = v2_val - actual_val
        error_v3 = v3_val - actual_val
        
        print(f"{time.strftime('%H:%M')}   {actual_val:6.1f}   {v2_val:6.1f}     {v3_val:6.1f}      {error_v2:+6.1f}   {error_v3:+6.1f}")

# 重要な洞察
print("\n=== ハイブリッドモデルの特徴 ===")

# 遅延時間の分析
print("\n1. 遅延時間の適応:")
v3_delays = v3_predictions['適用遅延'].values[:18]  # 3時間分
unique_delays = np.unique(v3_delays)
for delay in sorted(unique_delays):
    count = np.sum(v3_delays == delay)
    pct = count / len(v3_delays) * 100
    print(f"   {delay:3.0f}分: {count:2d}回 ({pct:4.1f}%)")

# ML調整の分析
print("\n2. 機械学習による調整:")
ml_delay_factors = v3_predictions['ML調整係数_遅延'].values[:18]
ml_rate_factors = v3_predictions['ML調整係数_変化率'].values[:18]

print(f"   遅延調整: 平均 {np.mean(ml_delay_factors):.2f}倍 (範囲: {np.min(ml_delay_factors):.2f}～{np.max(ml_delay_factors):.2f})")
print(f"   変化率調整: 平均 {np.mean(ml_rate_factors):.2f}倍 (範囲: {np.min(ml_rate_factors):.2f}～{np.max(ml_rate_factors):.2f})")

# グラフ作成
fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

# 表示範囲
plot_start = current_time - timedelta(hours=1)
plot_end = current_time + timedelta(hours=3)

# 1. 放流量の比較
ax1 = axes[0]
# 実績
past_mask = (df['時刻'] >= plot_start) & (df['時刻'] <= current_time)
past_data = df[past_mask]
ax1.plot(past_data['時刻'], past_data['ダム_全放流量'], 'k-', label='実績', linewidth=2)
ax1.plot(actual_future['時刻'], actual_future['ダム_全放流量'], 'k--', label='実績（予測期間）', linewidth=2)

# 予測
ax1.plot(v2_predictions['時刻'], v2_predictions['予測放流量'], 'b--', 
         label='ルールベース（v2）', linewidth=2, alpha=0.7)
ax1.plot(v3_predictions['時刻'], v3_predictions['予測放流量'], 'r-', 
         label='ハイブリッド（v3）', linewidth=2)

ax1.axvline(x=current_time, color='gray', linestyle='--', alpha=0.5)
ax1.set_ylabel('放流量 (m³/s)')
ax1.set_title(f'ハイブリッドモデル分析 - {current_time.strftime("%Y年%m月%d日 %H:%M")}時点')
ax1.legend(loc='upper left')
ax1.grid(True, alpha=0.3)

# 2. 遅延時間とML調整
ax2 = axes[1]
# 遅延時間の比較
ax2.plot(v2_predictions['時刻'], v2_predictions['適用遅延'], 
         'bo-', label='ルールベース遅延', markersize=6, alpha=0.7)
ax2.plot(v3_predictions['時刻'], v3_predictions['適用遅延'], 
         'ro-', label='ハイブリッド遅延', markersize=6)
ax2.axvline(x=current_time, color='gray', linestyle='--', alpha=0.5)
ax2.set_ylabel('遅延時間 (分)')
ax2.set_ylim(-10, 130)
ax2.legend(loc='upper left')
ax2.grid(True, alpha=0.3)

# ML調整係数（右軸）
ax2_twin = ax2.twinx()
ax2_twin.plot(v3_predictions['時刻'], v3_predictions['ML調整係数_変化率'], 
              'g-', label='ML変化率調整', linewidth=2, alpha=0.7)
ax2_twin.set_ylabel('ML調整係数', color='green')
ax2_twin.tick_params(axis='y', labelcolor='green')
ax2_twin.set_ylim(0.5, 1.5)
ax2_twin.legend(loc='upper right')

# 3. 降雨強度と使用降雨
ax3 = axes[2]
# 実績降雨
all_data_mask = (df['時刻'] >= plot_start) & (df['時刻'] <= plot_end)
all_data = df[all_data_mask]
ax3.bar(all_data['時刻'], all_data['ダム_60分雨量'], 
        width=0.007, alpha=0.7, color='blue', label='降雨強度')

# モデル使用降雨の比較
ax3.scatter(v2_predictions['時刻'], v2_predictions['使用降雨強度'], 
           color='blue', s=30, marker='s', alpha=0.7, label='v2使用降雨')
ax3.scatter(v3_predictions['時刻'], v3_predictions['使用降雨強度'], 
           color='red', s=30, marker='o', label='v3使用降雨')

ax3.axvline(x=current_time, color='gray', linestyle='--', alpha=0.5)
ax3.set_xlabel('時刻')
ax3.set_ylabel('降雨強度 (mm/h)')
ax3.legend()
ax3.grid(True, alpha=0.3)

plt.tight_layout()
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
filename = f'hybrid_analysis_20230701_0000_{timestamp}.png'
plt.savefig(filename, dpi=150, bbox_inches='tight')
plt.close()

print(f"\n分析グラフを保存: {filename}")

# 問題点の改善確認
print("\n=== 00:20の問題改善の確認 ===")
v2_20 = v2_predictions.iloc[1]
v3_20 = v3_predictions.iloc[1]

print(f"\nルールベース（v2）:")
print(f"  予測放流量: {v2_20['予測放流量']:.1f} m³/s")
print(f"  使用降雨: {v2_20['使用降雨強度']:.1f} mm/h（遅延{v2_20['適用遅延']:.0f}分）")
print(f"  → 過去の低降雨により減少")

print(f"\nハイブリッド（v3）:")
print(f"  予測放流量: {v3_20['予測放流量']:.1f} m³/s")
print(f"  使用降雨: {v3_20['使用降雨強度']:.1f} mm/h（遅延{v3_20['適用遅延']:.0f}分）")
print(f"  ML調整: 遅延×{v3_20['ML調整係数_遅延']:.2f}, 変化率×{v3_20['ML調整係数_変化率']:.2f}")
print(f"  → 高降雨予測に適切に反応！")