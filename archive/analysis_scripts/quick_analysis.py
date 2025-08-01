#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
軽量な分析スクリプト - 水位予測に必要な基本的な関係性のみを抽出
"""

import pandas as pd
import numpy as np
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

def main():
    print("=== 軽量分析開始 ===")
    
    # データ読み込み
    file_path = "/mnt/c/users/kishida/cursorproject/kotogawa-data/統合データ_水位ダム_20250730_205325.csv"
    df = pd.read_csv(file_path, encoding='utf-8')
    print(f"データ読み込み完了: {len(df)}行")
    
    # 時刻をdatetimeに変換
    df['時刻'] = pd.to_datetime(df['時刻'])
    
    # 欠損値を除去
    df_clean = df.dropna(subset=['水位_水位', 'ダム_全放流量'])
    print(f"欠損値除去後: {len(df_clean)}行")
    
    # 基本統計
    print("\n【基本統計】")
    print(f"放流量: {df_clean['ダム_全放流量'].min():.1f} ～ {df_clean['ダム_全放流量'].max():.1f} m³/s")
    print(f"水位: {df_clean['水位_水位'].min():.2f} ～ {df_clean['水位_水位'].max():.2f} m")
    
    # 1. 放流量と水位の関係
    print("\n【放流量と水位の関係】")
    discharge = df_clean['ダム_全放流量'].values
    water_level = df_clean['水位_水位'].values
    
    # 線形回帰
    slope, intercept, r_value, p_value, std_err = stats.linregress(discharge, water_level)
    print(f"線形回帰: 水位 = {slope:.6f} × 放流量 + {intercept:.3f}")
    print(f"相関係数 R = {r_value:.4f}")
    print(f"決定係数 R² = {r_value**2:.4f}")
    
    # 2. 遅延時間の簡易計算（相互相関）
    print("\n【遅延時間分析（相互相関）】")
    max_lag = 30  # 最大300分
    correlations = []
    lags = range(0, max_lag + 1)
    
    for lag in lags:
        if lag > 0:
            corr = np.corrcoef(discharge[:-lag], water_level[lag:])[0, 1]
        else:
            corr = np.corrcoef(discharge, water_level)[0, 1]
        correlations.append(corr)
    
    max_corr_idx = np.argmax(correlations)
    print(f"最適遅延時間: {max_corr_idx * 10}分")
    print(f"最大相関係数: {correlations[max_corr_idx]:.4f}")
    
    # 3. 放流量レベル別の分析
    print("\n【放流量レベル別分析】")
    discharge_ranges = [(0, 100), (100, 200), (200, 300), (300, 400), (400, 500), (500, 1000)]
    
    for min_d, max_d in discharge_ranges:
        mask = (discharge >= min_d) & (discharge < max_d)
        if mask.sum() >= 100:  # 十分なデータがある場合
            d_range = discharge[mask]
            w_range = water_level[mask]
            
            if len(d_range) > 0:
                # この範囲での線形回帰
                slope_r, intercept_r, r_value_r, _, _ = stats.linregress(d_range, w_range)
                
                # 遅延時間計算
                best_lag = 0
                best_corr = 0
                for lag in range(0, 16):  # 0-150分
                    if lag > 0 and len(d_range) > lag:
                        corr = np.corrcoef(d_range[:-lag], w_range[lag:])[0, 1]
                        if not np.isnan(corr) and corr > best_corr:
                            best_corr = corr
                            best_lag = lag
                
                print(f"\n{min_d}-{max_d} m³/s (n={mask.sum()}):")
                print(f"  回帰式: 水位 = {slope_r:.6f} × 放流量 + {intercept_r:.3f} (R={r_value_r:.3f})")
                print(f"  最適遅延: {best_lag * 10}分 (r={best_corr:.3f})")
    
    # 4. 変化率の分析
    print("\n【変化率分析】")
    # 放流量変化率と水位変化率
    discharge_diff = np.diff(discharge)
    water_diff = np.diff(water_level)
    
    # ゼロ除算を避ける
    valid_mask = discharge[:-1] > 10  # 放流量が10以上
    if valid_mask.sum() > 0:
        discharge_rate = discharge_diff[valid_mask] / discharge[:-1][valid_mask]
        water_rate = water_diff[valid_mask]
        
        if len(discharge_rate) > 0:
            slope_rate, intercept_rate, r_value_rate, _, _ = stats.linregress(discharge_rate, water_rate)
            print(f"水位変化 = {slope_rate:.3f} × 放流量変化率 + {intercept_rate:.4f} (R={r_value_rate:.3f})")
    
    # 5. 高放流量域での特性
    print("\n【高放流量域（≥300 m³/s）の特性】")
    high_mask = discharge >= 300
    if high_mask.sum() > 0:
        high_discharge = discharge[high_mask]
        high_water = water_level[high_mask]
        
        # 2次多項式フィット
        if len(high_discharge) >= 10:
            z = np.polyfit(high_discharge, high_water, 2)
            print(f"2次多項式: 水位 = {z[0]:.8f}×放流量² + {z[1]:.6f}×放流量 + {z[2]:.3f}")
            
            # 線形モデルとの比較
            water_pred_linear = slope * high_discharge + intercept
            water_pred_poly = np.polyval(z, high_discharge)
            
            rmse_linear = np.sqrt(np.mean((high_water - water_pred_linear)**2))
            rmse_poly = np.sqrt(np.mean((high_water - water_pred_poly)**2))
            
            print(f"RMSE（線形）: {rmse_linear:.3f}m")
            print(f"RMSE（2次）: {rmse_poly:.3f}m")
            print(f"改善率: {(1 - rmse_poly/rmse_linear)*100:.1f}%")
    
    print("\n=== 分析完了 ===")

if __name__ == "__main__":
    main()