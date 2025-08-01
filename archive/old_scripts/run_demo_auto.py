#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
水位予測デモの自動実行スクリプト
"""

import sys
import os

# water_level_prediction_demo.pyから必要な部分をインポート
from water_level_prediction_demo import WaterLevelPredictionDemo

def main(period_idx=None, target_time=None):
    """
    自動実行
    
    Parameters:
    -----------
    period_idx : int or None
        デモ期間のインデックス（1-5）。Noneの場合は期間1を自動選択
    target_time : str or None
        指定時刻（例: '2023-07-01 12:00:00'）。指定した場合はこの時刻を含む期間でデモを実行
    """
    print("=== 水位予測モデル デモンストレーション（自動実行） ===")
    
    # 最新のモデルファイルを使用
    model_path = "water_level_predictor_20250731_022436.pkl"
    
    if not os.path.exists(model_path):
        print(f"エラー: モデルファイル {model_path} が見つかりません")
        return
    
    print(f"使用するモデル: {model_path}")
    
    # デモの初期化
    demo = WaterLevelPredictionDemo(model_path)
    
    # 時刻指定がある場合
    if target_time:
        print(f"\n指定時刻でデモ期間を検索: {target_time}")
        selected_period = demo.find_period_by_time(target_time)
        print(f"選択された期間: {selected_period['start_time']} (指定時刻: {target_time})")
        print(f"放流量範囲: {selected_period['discharge_range'][0]:.0f}-{selected_period['discharge_range'][1]:.0f} m³/s")
        print(f"変動係数: {selected_period['cv']:.3f}")
    else:
        # デモ期間の検出
        print("\n適切なデモ期間を検索中...")
        demo_periods = demo.find_demo_periods(n_periods=5)
        
        if not demo_periods:
            print("エラー: 適切なデモ期間が見つかりません")
            return
        
        # デモ期間の表示
        print("\n=== デモ期間の候補 ===")
        for i, period in enumerate(demo_periods):
            print(f"{i+1}. {period['start_time']} - "
                  f"放流量: {period['discharge_range'][0]:.0f}-{period['discharge_range'][1]:.0f} m³/s, "
                  f"変動係数: {period['cv']:.3f}")
        
        # 期間の選択
        if period_idx is None:
            selected_idx = 0
            print(f"\n期間1を自動選択")
        else:
            # 1-5の範囲チェック
            if 1 <= period_idx <= len(demo_periods):
                selected_idx = period_idx - 1
                print(f"\n期間{period_idx}を選択")
            else:
                print(f"\n無効な期間番号: {period_idx}。期間1を使用します。")
                selected_idx = 0
        
        selected_period = demo_periods[selected_idx]
        print(f"選択された期間: {selected_period['start_time']}")
    
    # 静的デモを実行（CSV保存あり）
    print("\n静的デモを実行中...")
    demo.run_static_demo(selected_period, save_csv=True)
    
    print("\nデモ完了！")
    print("生成された画像とCSVファイルを確認してください。")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='水位予測デモの自動実行')
    parser.add_argument('--period', type=int, help='デモ期間番号（1-5）')
    parser.add_argument('--time', type=str, help='指定時刻（例: 2023-07-01 12:00:00）')
    
    args = parser.parse_args()
    
    # 引数の検証
    if args.period and args.time:
        print("エラー: --periodと--timeは同時に指定できません")
        sys.exit(1)
    
    # メイン処理実行
    main(period_idx=args.period, target_time=args.time)