#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ハイブリッドモデルのテスト
2023-07-01 00:00のケースで改善を確認
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from discharge_prediction_model_v2 import DischargePredictionModelV2
from discharge_prediction_model_v3_hybrid import DischargePredictionModelV3Hybrid
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

# 日本語フォント設定
plt.rcParams['font.sans-serif'] = ['Yu Gothic', 'Hiragino Sans', 'Meirio', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# データ読み込み
df = pd.read_csv('統合データ_水位ダム_20250730_142903.csv', encoding='utf-8')
df['時刻'] = pd.to_datetime(df['時刻'])

# テスト時点
test_time = pd.to_datetime('2023-07-01 00:00')

# 過去データと降雨予測の準備
historical_mask = df['時刻'] <= test_time
historical_data = df[historical_mask].copy()

future_1h_mask = (df['時刻'] > test_time) & (df['時刻'] <= test_time + timedelta(hours=1))
future_1h_data = df[future_1h_mask]

rainfall_forecast = pd.DataFrame({
    '時刻': future_1h_data['時刻'],
    '降雨強度': future_1h_data['ダム_60分雨量']
})

print("=== ハイブリッドモデルのテスト ===")
print(f"テスト時刻: {test_time}")
print(f"現在の放流量: {historical_data.iloc[-1]['ダム_全放流量']:.1f} m³/s")
print(f"現在の降雨: {historical_data.iloc[-1]['ダム_60分雨量']:.1f} mm/h")

print("\n降雨予測（1時間先まで）:")
for _, row in rainfall_forecast.iterrows():
    print(f"  {row['時刻'].strftime('%H:%M')}: {row['降雨強度']:.1f} mm/h")

# モデルの比較
models = {
    'ルールベース（v2）': DischargePredictionModelV2(),
    'ハイブリッド（v3）': DischargePredictionModelV3Hybrid()
}

# モデルの読み込み
models['ルールベース（v2）'].load_model('discharge_prediction_model_v2_20250801_221633.pkl')
models['ハイブリッド（v3）'].load_model('discharge_prediction_model_v3_hybrid_20250801_231150.pkl')

# 予測実行
predictions = {}
for name, model in models.items():
    predictions[name] = model.predict(test_time, historical_data, 
                                     prediction_hours=1, 
                                     rainfall_forecast=rainfall_forecast)

# 結果の比較
print("\n=== 予測結果の比較 ===")
print("時刻   ルールベース  ハイブリッド  差分   遅延(v2) 遅延(v3)")
print("-" * 60)

for i in range(6):
    time_str = predictions['ルールベース（v2）'].iloc[i]['時刻'].strftime('%H:%M')
    val_v2 = predictions['ルールベース（v2）'].iloc[i]['予測放流量']
    val_v3 = predictions['ハイブリッド（v3）'].iloc[i]['予測放流量']
    delay_v2 = predictions['ルールベース（v2）'].iloc[i]['適用遅延']
    delay_v3 = predictions['ハイブリッド（v3）'].iloc[i]['適用遅延']
    diff = val_v3 - val_v2
    
    print(f"{time_str}    {val_v2:6.1f}      {val_v3:6.1f}    {diff:+6.1f}    {delay_v2:3.0f}分   {delay_v3:3.0f}分")

# 詳細分析
print("\n=== 問題点（00:20）の詳細分析 ===")
for name in ['ルールベース（v2）', 'ハイブリッド（v3）']:
    row = predictions[name].iloc[1]  # 00:20
    print(f"\n{name}:")
    print(f"  予測放流量: {row['予測放流量']:.1f} m³/s")
    print(f"  使用降雨強度: {row['使用降雨強度']:.1f} mm/h")
    print(f"  適用遅延: {row['適用遅延']:.0f}分")
    print(f"  変化率: {row['変化率']:.1f} m³/s/10min")
    
    if 'ML調整係数_遅延' in row:
        print(f"  ML調整係数（遅延）: {row['ML調整係数_遅延']:.2f}")
        print(f"  ML調整係数（変化率）: {row['ML調整係数_変化率']:.2f}")

# グラフ作成
fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

# 実績データの準備
actual_mask = (df['時刻'] >= test_time - timedelta(hours=1)) & (df['時刻'] <= test_time + timedelta(hours=1))
actual_data = df[actual_mask]

# 1. 放流量の比較
ax1 = axes[0]
ax1.plot(actual_data['時刻'], actual_data['ダム_全放流量'], 'ko-', label='実績', markersize=6)
ax1.plot(predictions['ルールベース（v2）']['時刻'], predictions['ルールベース（v2）']['予測放流量'], 
         'b--', label='ルールベース（v2）', linewidth=2)
ax1.plot(predictions['ハイブリッド（v3）']['時刻'], predictions['ハイブリッド（v3）']['予測放流量'], 
         'r-', label='ハイブリッド（v3）', linewidth=2)
ax1.axvline(x=test_time, color='gray', linestyle='--', alpha=0.5)
ax1.set_ylabel('放流量 (m³/s)')
ax1.set_title('ハイブリッドモデルによる改善効果')
ax1.legend()
ax1.grid(True, alpha=0.3)

# 2. 適用遅延時間の比較
ax2 = axes[1]
ax2.plot(predictions['ルールベース（v2）']['時刻'], predictions['ルールベース（v2）']['適用遅延'], 
         'bo-', label='ルールベース（v2）', markersize=8)
ax2.plot(predictions['ハイブリッド（v3）']['時刻'], predictions['ハイブリッド（v3）']['適用遅延'], 
         'ro-', label='ハイブリッド（v3）', markersize=8)
ax2.axvline(x=test_time, color='gray', linestyle='--', alpha=0.5)
ax2.set_ylabel('遅延時間 (分)')
ax2.set_ylim(-10, 130)
ax2.legend()
ax2.grid(True, alpha=0.3)

# 3. 降雨強度
ax3 = axes[2]
ax3.bar(actual_data['時刻'], actual_data['ダム_60分雨量'], width=0.007, 
        color='blue', alpha=0.7, label='降雨強度')
ax3.axvline(x=test_time, color='gray', linestyle='--', alpha=0.5)
ax3.set_xlabel('時刻')
ax3.set_ylabel('降雨強度 (mm/h)')
ax3.legend()
ax3.grid(True, alpha=0.3)

plt.tight_layout()
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
filename = f'hybrid_model_comparison_{timestamp}.png'
plt.savefig(filename, dpi=150, bbox_inches='tight')
plt.close()

print(f"\n比較グラフを保存: {filename}")

# 機械学習コンポーネントの影響度分析
print("\n=== 機械学習コンポーネントの影響度 ===")
v3_pred = predictions['ハイブリッド（v3）']
for i in range(min(6, len(v3_pred))):
    row = v3_pred.iloc[i]
    if 'ML調整係数_遅延' in row:
        print(f"{row['時刻'].strftime('%H:%M')}: 遅延調整 {row['ML調整係数_遅延']:.2f}x, 変化率調整 {row['ML調整係数_変化率']:.2f}x")