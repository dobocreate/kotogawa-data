#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
直接CSVファイルを指定して分析を実行するスクリプト
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal, stats
from scipy.optimize import curve_fit
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# 日本語フォント設定
plt.rcParams['font.sans-serif'] = ['Yu Gothic', 'Hiragino Sans', 'Meirio', 'DejaVu Sans']
plt.rcParams['font.size'] = 10
plt.rcParams['axes.unicode_minus'] = False

# analyze_water_level_delay.pyからRiverDelayAnalysisクラスをインポート
from analyze_water_level_delay import RiverDelayAnalysis

class DirectAnalysis(RiverDelayAnalysis):
    """直接ファイルパスを指定して分析を行うクラス"""
    
    def load_data_direct(self, file_path):
        """データの直接読み込み"""
        print("=== データ読み込み ===")
        print(f"ファイル: {file_path}")
        
        self.df = pd.read_csv(file_path, encoding='utf-8')
        print(f"読み込み完了: {len(self.df)}行")
        print(f"カラム: {self.df.columns.tolist()}")
        
        # 時刻をdatetimeに変換
        self.df['時刻'] = pd.to_datetime(self.df['時刻'])
        
        # 必要なカラムの確認
        required_cols = ['時刻', '水位_水位', 'ダム_全放流量']
        for col in required_cols:
            if col not in self.df.columns:
                raise ValueError(f"必要なカラム '{col}' が見つかりません")
        
        return self.df

def main():
    """メイン処理"""
    print("河川水位遅延時間分析プログラム（直接実行版）")
    print("=" * 50)
    
    analyzer = DirectAnalysis()
    
    try:
        # 1. データ読み込み（直接ファイルパスを指定）
        file_path = "/mnt/c/users/kishida/cursorproject/kotogawa-data/統合データ_水位ダム_20250730_205325.csv"
        analyzer.load_data_direct(file_path)
        
        # 2. 前処理
        analyzer.preprocess_data()
        
        # 3. 変化点検出
        analyzer.detect_change_points(ma_window='30')
        
        # 4. 変化点での遅延時間計算
        analyzer.calculate_change_point_delays()
        
        # 5. 相互相関分析
        analyzer.calculate_cross_correlation()
        
        # 5.5. イベント分離分析
        analyzer.event_separation_analysis()
        
        # 6. 非線形モデル構築
        analyzer.fit_nonlinear_model()
        
        # 7. 安定・変動期間の分類と分析
        analyzer.classify_stable_variable_periods(window_hours=2, cv_threshold=0.05)
        analyzer.classify_stable_variable_periods_detailed()
        analyzer.analyze_stable_periods()
        analyzer.analyze_stable_periods_detailed()
        analyzer.analyze_variable_periods()
        
        # 8. 結果のサマリー表示
        print("\n" + "="*60)
        print("=== 分析結果サマリー ===")
        print("="*60)
        
        # 基本統計
        print("\n【基本統計】")
        print(f"データ期間: {analyzer.df['時刻'].min()} ～ {analyzer.df['時刻'].max()}")
        print(f"データ点数: {len(analyzer.df)}点")
        print(f"放流量範囲: {analyzer.df['ダム_全放流量'].min():.1f} ～ {analyzer.df['ダム_全放流量'].max():.1f} m³/s")
        print(f"水位範囲: {analyzer.df['水位_水位'].min():.2f} ～ {analyzer.df['水位_水位'].max():.2f} m")
        
        # 相関分析結果
        if analyzer.correlation_results:
            print("\n【相関分析結果】")
            print(f"全体の最適遅延時間: {analyzer.correlation_results['overall']['optimal_lag_minutes']}分")
            print(f"最大相関係数: {analyzer.correlation_results['overall']['max_correlation']:.3f}")
            
            if not analyzer.correlation_results['by_level'].empty:
                print("\n放流量レベル別の遅延時間:")
                for _, row in analyzer.correlation_results['by_level'].iterrows():
                    print(f"  {row['discharge_range']} m³/s: {row['optimal_lag_minutes']}分 (r={row['max_correlation']:.3f}, n={row['data_points']})")
        
        # イベント分析結果
        if hasattr(analyzer, 'event_results') and analyzer.event_results:
            print("\n【イベント分析結果】")
            print(f"検出イベント数: {analyzer.event_results['summary']['total_events']}")
            print(f"有効イベント数: {analyzer.event_results['summary']['valid_events']}")
            print(f"平均遅延時間: {analyzer.event_results['summary']['mean_delay']:.1f}分")
            print(f"中央値遅延時間: {analyzer.event_results['summary']['median_delay']:.1f}分")
        
        # 安定期間分析結果
        if hasattr(analyzer, 'stable_analysis_results') and len(analyzer.stable_analysis_results) > 0:
            print("\n【安定期間分析結果】")
            print(f"分析された安定期間数: {len(analyzer.stable_analysis_results)}")
            print(f"平均遅延時間: {analyzer.stable_analysis_results['delay_minutes'].mean():.1f}分")
        
        # 変動期間分析結果
        if hasattr(analyzer, 'variable_analysis_results') and len(analyzer.variable_analysis_results) > 0:
            print("\n【変動期間分析結果】")
            print(f"分析された変動期間数: {len(analyzer.variable_analysis_results)}")
            print(f"平均遅延時間: {analyzer.variable_analysis_results['delay_minutes'].mean():.1f}分")
            
            # 増加期vs減少期
            increase_mask = analyzer.variable_analysis_results['direction'] == 'increase'
            decrease_mask = analyzer.variable_analysis_results['direction'] == 'decrease'
            
            if increase_mask.any():
                print(f"  増加期の平均遅延: {analyzer.variable_analysis_results[increase_mask]['delay_minutes'].mean():.1f}分")
            if decrease_mask.any():
                print(f"  減少期の平均遅延: {analyzer.variable_analysis_results[decrease_mask]['delay_minutes'].mean():.1f}分")
        
        # モデルパラメータ
        if analyzer.model_params:
            print("\n【遅延時間予測モデル】")
            print(f"べき乗則: {analyzer.model_params['power_law']['equation']}")
            print(f"  R²: {analyzer.model_params['power_law']['r2']:.3f}")
        
        print("\n分析完了！")
        
    except Exception as e:
        print(f"\nエラーが発生しました: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()