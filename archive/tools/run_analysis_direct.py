#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
直接ファイルを指定して分析を実行するスクリプト
"""

import sys
import os

# analyze_water_level_delay.pyのRiverDelayAnalysisクラスを使用
from analyze_water_level_delay import RiverDelayAnalysis

# ファイルダイアログを使わずに直接ファイルを読み込むようにオーバーライド
class DirectAnalysis(RiverDelayAnalysis):
    def load_data(self):
        """データの読み込み（直接ファイル指定）"""
        print("=== データ読み込み ===")
        
        # 直接ファイルパスを指定
        file_path = "統合データ_水位ダム_20250730_030746.csv"
        
        if not os.path.exists(file_path):
            raise ValueError(f"ファイルが見つかりません: {file_path}")
        
        import pandas as pd
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
    print("河川水位遅延時間分析プログラム（直接ファイル読み込み）")
    print("=" * 50)
    
    analyzer = DirectAnalysis()
    
    try:
        # 1. データ読み込み
        analyzer.load_data()
        
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
        
        # 8. 可視化処理
        print("\n=== 可視化処理 ===")
        
        # Figure 5: 安定期間の分析（水位と遅延時間の関係を含む）
        from create_figure5_with_water_level import UpdatedFigure5Analysis
        figure5_analyzer = UpdatedFigure5Analysis()
        # 既に読み込まれたデータを共有
        figure5_analyzer.df = analyzer.df
        figure5_analyzer.df_processed = analyzer.df_processed
        figure5_analyzer.stable_periods = analyzer.stable_periods
        figure5_analyzer.variable_periods = analyzer.variable_periods
        figure5_analyzer.detailed_stable_results = analyzer.detailed_stable_results
        # Figure 5の生成
        figure5_analyzer.unify_stable_periods()
        figure5_analyzer.visualize_updated_figure5()
        
        # Figure 6: 妥当な遅延時間範囲の分析
        from create_figure6_valid_delay_periods import Figure6ValidDelayAnalysis
        figure6_analyzer = Figure6ValidDelayAnalysis()
        # 既に読み込まれたデータを共有
        figure6_analyzer.df = analyzer.df
        figure6_analyzer.df_processed = analyzer.df_processed
        figure6_analyzer.stable_periods = analyzer.stable_periods
        figure6_analyzer.variable_periods = analyzer.variable_periods
        figure6_analyzer.detailed_stable_results = analyzer.detailed_stable_results
        # Figure 6の生成
        figure6_analyzer.create_figure6_valid_delays()
        
        # Figure 7: 変動期間の分析
        from create_figure7_variable_periods import Figure7VariablePeriodsAnalysis
        figure7_analyzer = Figure7VariablePeriodsAnalysis()
        # 既に読み込まれたデータを共有
        figure7_analyzer.df = analyzer.df
        figure7_analyzer.df_processed = analyzer.df_processed
        figure7_analyzer.stable_periods = analyzer.stable_periods
        figure7_analyzer.variable_periods = analyzer.variable_periods
        figure7_analyzer.variable_analysis_results = analyzer.variable_analysis_results
        # Figure 7の生成
        figure7_analyzer.visualize_figure7_variable_periods()
        
        # Figure 8: 生データ分析
        from create_figure8_raw_data import Figure8RawDataAnalysis
        figure8_analyzer = Figure8RawDataAnalysis()
        # 既に読み込まれたデータを共有
        figure8_analyzer.df = analyzer.df
        figure8_analyzer.df_processed = analyzer.df_processed
        figure8_analyzer.stable_periods = analyzer.stable_periods
        figure8_analyzer.variable_periods = analyzer.variable_periods
        figure8_analyzer.variable_analysis_results = analyzer.variable_analysis_results
        # Figure 8の生成
        figure8_analyzer.visualize_figure8_raw_data()
        
        # Figure 9: 変動期間グリッド表示
        from create_figure9_variable_periods_grid import Figure9VariablePeriodsGrid
        figure9_analyzer = Figure9VariablePeriodsGrid()
        # 既に読み込まれたデータを共有
        figure9_analyzer.df = analyzer.df
        figure9_analyzer.df_processed = analyzer.df_processed
        figure9_analyzer.stable_periods = analyzer.stable_periods
        figure9_analyzer.variable_periods = analyzer.variable_periods
        figure9_analyzer.variable_analysis_results = analyzer.variable_analysis_results
        # Figure 9の生成
        figure9_analyzer.visualize_figure9_variable_periods_grid()
        
        # Figure 10: ピーク・ボトム対応分析
        from create_figure10_peak_bottom_analysis import Figure10PeakBottomAnalysis
        figure10_analyzer = Figure10PeakBottomAnalysis()
        # 既に読み込まれたデータを共有
        figure10_analyzer.df = analyzer.df
        figure10_analyzer.df_processed = analyzer.df_processed
        figure10_analyzer.stable_periods = analyzer.stable_periods
        figure10_analyzer.variable_periods = analyzer.variable_periods
        figure10_analyzer.variable_analysis_results = analyzer.variable_analysis_results
        # Figure 10の生成
        figure10_analyzer.create_figure10_peak_bottom_analysis()
        
        # Figure 11: ヒステリシス分析
        from create_figure11_hysteresis_analysis import Figure11HysteresisAnalysis
        figure11_analyzer = Figure11HysteresisAnalysis()
        # 既に読み込まれたデータを共有
        figure11_analyzer.df = analyzer.df
        figure11_analyzer.df_processed = analyzer.df_processed
        figure11_analyzer.stable_periods = analyzer.stable_periods
        figure11_analyzer.variable_periods = analyzer.variable_periods
        # Figure 11の生成
        figure11_analyzer.create_figure11_hysteresis_analysis()
        
        # Figure 12: 変化量の関係性分析
        from create_figure12_change_amount_analysis import Figure12ChangeAmountAnalysis
        figure12_analyzer = Figure12ChangeAmountAnalysis()
        # 既に読み込まれたデータを共有
        figure12_analyzer.df = analyzer.df
        figure12_analyzer.df_processed = analyzer.df_processed
        figure12_analyzer.stable_periods = analyzer.stable_periods
        figure12_analyzer.variable_periods = analyzer.variable_periods
        # Figure 12の生成
        figure12_analyzer.create_figure12_change_amount_analysis()
        
        print("\n分析完了！")
        
    except Exception as e:
        print(f"\nエラーが発生しました: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()