#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
データファイルの違いを確認
"""

import pandas as pd

# 両方のデータファイルを読み込み
df1 = pd.read_csv('統合データ_水位ダム_20250730_142903.csv', encoding='utf-8')
df1['時刻'] = pd.to_datetime(df1['時刻'])

df2 = pd.read_csv('統合データ_水位ダム_20250730_205325.csv', encoding='utf-8')
df2['時刻'] = pd.to_datetime(df2['時刻'])

# 2023-07-01 00:00前後のデータを比較
test_time = pd.to_datetime('2023-07-01 00:00')

print("=== データファイルの比較 ===")
print(f"\nファイル1: 統合データ_水位ダム_20250730_142903.csv")
print(f"ファイル2: 統合データ_水位ダム_20250730_205325.csv")

# 23:00から00:00までのデータを比較
mask = (df1['時刻'] >= '2023-06-30 23:00') & (df1['時刻'] <= '2023-07-01 00:00')

print("\n23:00-00:00のデータ比較:")
print("\nファイル1（デモ使用）:")
for _, row in df1[mask].iterrows():
    print(f"  {row['時刻'].strftime('%H:%M')} - 放流量: {row['ダム_全放流量']:6.1f} m³/s, 降雨: {row['ダム_60分雨量']:5.1f} mm/h")

print("\nファイル2（私のテスト）:")
for _, row in df2[mask].iterrows():
    print(f"  {row['時刻'].strftime('%H:%M')} - 放流量: {row['ダム_全放流量']:6.1f} m³/s, 降雨: {row['ダム_60分雨量']:5.1f} mm/h")

# 00:00時点の違いを確認
idx1 = df1['時刻'].searchsorted(test_time)
idx2 = df2['時刻'].searchsorted(test_time)

print(f"\n00:00時点のデータ:")
print(f"ファイル1: 放流量={df1.iloc[idx1]['ダム_全放流量']:.1f}, 降雨={df1.iloc[idx1]['ダム_60分雨量']:.1f}")
print(f"ファイル2: 放流量={df2.iloc[idx2]['ダム_全放流量']:.1f}, 降雨={df2.iloc[idx2]['ダム_60分雨量']:.1f}")