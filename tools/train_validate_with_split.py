#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
初期学習と検証を一括実行するスクリプト（データ分割機能付き）
"""

import subprocess
import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime
import tkinter as tk
from tkinter import filedialog
import json

def select_data_file():
    """ファイルダイアログでデータファイルを選択"""
    print("\n学習用データファイルを選択してください...")
    
    # tkinterのルートウィンドウを作成（非表示）
    root = tk.Tk()
    root.withdraw()
    
    # ファイルダイアログを開く
    file_path = filedialog.askopenfilename(
        title="データファイルを選択",
        filetypes=[
            ("CSVファイル", "*.csv"),
            ("すべてのファイル", "*.*")
        ],
        initialdir=os.getcwd()
    )
    
    # ウィンドウを破棄
    root.destroy()
    
    if not file_path:
        print("ファイルが選択されませんでした。")
        return None
    
    print(f"選択されたファイル: {file_path}")
    return file_path

def split_data(file_path, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    """
    データを学習用、検証用、テスト用に分割
    
    Parameters:
    -----------
    file_path : str
        データファイルのパス
    train_ratio : float
        学習用データの割合（デフォルト: 70%）
    val_ratio : float
        検証用データの割合（デフォルト: 15%）
    test_ratio : float
        テスト用データの割合（デフォルト: 15%）
    
    Returns:
    --------
    dict : 分割されたデータファイルのパス
    """
    print("\n=== データ分割 ===")
    print(f"学習用: {train_ratio*100:.0f}%, 検証用: {val_ratio*100:.0f}%, テスト用: {test_ratio*100:.0f}%")
    
    # データ読み込み
    df = pd.read_csv(file_path, encoding='utf-8')
    print(f"総データ数: {len(df)}行")
    
    # 時系列データなので、時間順に分割
    total_len = len(df)
    train_end = int(total_len * train_ratio)
    val_end = train_end + int(total_len * val_ratio)
    
    # データ分割
    train_data = df[:train_end]
    val_data = df[train_end:val_end]
    test_data = df[val_end:]
    
    # ファイル保存
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    train_path = f"{base_name}_train.csv"
    val_path = f"{base_name}_val.csv"
    test_path = f"{base_name}_test.csv"
    
    train_data.to_csv(train_path, index=False, encoding='utf-8')
    val_data.to_csv(val_path, index=False, encoding='utf-8')
    test_data.to_csv(test_path, index=False, encoding='utf-8')
    
    print(f"\n分割結果:")
    print(f"  学習用: {len(train_data)}行 -> {train_path}")
    print(f"  検証用: {len(val_data)}行 -> {val_path}")
    print(f"  テスト用: {len(test_data)}行 -> {test_path}")
    
    # 期間情報を表示
    if '時刻' in df.columns:
        train_data['時刻'] = pd.to_datetime(train_data['時刻'])
        val_data['時刻'] = pd.to_datetime(val_data['時刻'])
        test_data['時刻'] = pd.to_datetime(test_data['時刻'])
        
        print(f"\n期間:")
        print(f"  学習用: {train_data['時刻'].min()} ～ {train_data['時刻'].max()}")
        print(f"  検証用: {val_data['時刻'].min()} ～ {val_data['時刻'].max()}")
        print(f"  テスト用: {test_data['時刻'].min()} ～ {test_data['時刻'].max()}")
    
    return {
        "train": train_path,
        "val": val_path,
        "test": test_path,
        "original": file_path
    }

def run_command(command, description):
    """コマンドを実行して結果を表示"""
    print(f"\n{'='*60}")
    print(f"{description}")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            encoding='utf-8'
        )
        
        if result.stdout:
            print(result.stdout)
        
        if result.stderr:
            print("エラー出力:")
            print(result.stderr)
        
        if result.returncode != 0:
            print(f"エラー: コマンドが失敗しました（終了コード: {result.returncode}）")
            return False
        
        return True
        
    except Exception as e:
        print(f"実行エラー: {e}")
        return False

def main():
    """メイン処理"""
    print("=== 予測システム初期学習と検証（データ自動分割版） ===")
    print(f"実行開始: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # データファイルの選択
    data_file = select_data_file()
    if not data_file:
        print("処理を中止します。")
        sys.exit(1)
    
    # データ分割の設定
    print("\nデータ分割の設定:")
    print("1. デフォルト (70% / 15% / 15%)")
    print("2. カスタム設定")
    choice = input("選択してください (1 or 2) [デフォルト: 1]: ").strip()
    
    if choice == "2":
        try:
            train_ratio = float(input("学習用データの割合 (0-1) [0.7]: ") or "0.7")
            val_ratio = float(input("検証用データの割合 (0-1) [0.15]: ") or "0.15")
            test_ratio = 1 - train_ratio - val_ratio
            
            if test_ratio < 0 or train_ratio < 0 or val_ratio < 0:
                print("エラー: 割合の合計が1を超えています。デフォルト設定を使用します。")
                train_ratio, val_ratio, test_ratio = 0.7, 0.15, 0.15
        except ValueError:
            print("エラー: 無効な入力です。デフォルト設定を使用します。")
            train_ratio, val_ratio, test_ratio = 0.7, 0.15, 0.15
    else:
        train_ratio, val_ratio, test_ratio = 0.7, 0.15, 0.15
    
    # データ分割
    split_files = split_data(data_file, train_ratio, val_ratio, test_ratio)
    
    # 1. 初期学習（学習用データを使用）
    print("\n" + "="*60)
    print("ステップ 1/4: 初期学習")
    print("="*60)
    
    training_script = f"""
import sys
sys.path.append('.')
from initial_training import InitialTraining

# 学習用データで学習
trainer = InitialTraining("{split_files['train']}")
trainer.load_data()
trainer.calculate_delay_parameters()
trainer.calculate_response_rates()
trainer.calculate_hysteresis_correction()
trainer.calculate_water_level_correction()
trainer.optimize_parameters()
config = trainer.save_results("core/configs/learned_config.json")
trainer.visualize_learning_results()

print("\\n学習完了サマリー:")
print(f"使用データ: {os.path.basename(split_files['train'])}")
print(f"遅延パラメータ: {{len(trainer.training_results['delay_params'])}}種類")
print(f"応答率: {{sum(len(v) for v in trainer.training_results['response_rates'].values())}}種類")
"""
    
    with open("temp_training.py", "w", encoding="utf-8") as f:
        f.write(training_script)
    
    if run_command("python temp_training.py", f"学習用データでパラメータを学習中..."):
        print("✓ 初期学習が完了しました")
    else:
        print("✗ 初期学習に失敗しました")
        sys.exit(1)
    
    if os.path.exists("temp_training.py"):
        os.remove("temp_training.py")
    
    # 設定ファイルの完全性を確保
    print("\n設定ファイルの完全性を確認中...")
    if run_command("python tools/ensure_complete_config.py", "不足パラメータの補完..."):
        print("✓ 設定ファイルの確認が完了しました")
    
    # 2. 検証用データでの評価
    print("\n" + "="*60)
    print("ステップ 2/4: 検証用データでの評価")
    print("="*60)
    
    validation_script = f"""
import sys
sys.path.append('.')
from validate_prediction import PredictionValidator

# 検証用データで評価
validator = PredictionValidator(
    data_path="{split_files['val']}",
    config_path="core/configs/learned_config.json"
)

# 全期間で検証
validator.run_full_validation(num_periods=3, period_hours=24)
"""
    
    with open("temp_validate.py", "w", encoding="utf-8") as f:
        f.write(validation_script)
    
    if run_command("python temp_validate.py", "検証用データで評価中..."):
        print("✓ 検証が完了しました")
    else:
        print("✗ 検証に失敗しました")
    
    if os.path.exists("temp_validate.py"):
        os.remove("temp_validate.py")
    
    # 3. テスト用データでの最終評価
    print("\n" + "="*60)
    print("ステップ 3/4: テスト用データでの最終評価")
    print("="*60)
    
    test_script = f"""
import sys
sys.path.append('.')
from validate_prediction import PredictionValidator

# テスト用データで最終評価
validator = PredictionValidator(
    data_path="{split_files['test']}",
    config_path="core/configs/learned_config.json"
)

# 全期間で評価
results = validator.run_full_validation(num_periods=2, period_hours=24)

# 最終評価結果を保存
import json
with open("tools/results/final_test_results.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
"""
    
    with open("temp_test.py", "w", encoding="utf-8") as f:
        f.write(test_script)
    
    if run_command("python temp_test.py", "テスト用データで最終評価中..."):
        print("✓ 最終評価が完了しました")
    else:
        print("✗ 最終評価に失敗しました")
    
    if os.path.exists("temp_test.py"):
        os.remove("temp_test.py")
    
    # 4. デモの実行
    print("\n" + "="*60)
    print("ステップ 4/4: 予測デモ")
    print("="*60)
    
    demo_script = """
import sys
sys.path.append('.')
from prediction_demo import PredictionDemo
from realtime_prediction_model import RealtimePredictionModel

# 学習済み設定でデモを実行
class LearnedConfigDemo(PredictionDemo):
    def __init__(self):
        super().__init__()
        self.model = RealtimePredictionModel("core/configs/learned_config.json")

demo = LearnedConfigDemo()
demo.run_static_demo()
"""
    
    with open("temp_demo.py", "w", encoding="utf-8") as f:
        f.write(demo_script)
    
    if run_command("python temp_demo.py", "予測デモを実行中..."):
        print("✓ デモが完了しました")
    else:
        print("✗ デモに失敗しました")
    
    if os.path.exists("temp_demo.py"):
        os.remove("temp_demo.py")
    
    # 完了サマリー
    print("\n" + "="*60)
    print("完了サマリー")
    print("="*60)
    print("✓ すべての処理が正常に完了しました")
    print("\n生成されたファイル:")
    print("  【データ分割】")
    print(f"  - {split_files['train']}: 学習用データ")
    print(f"  - {split_files['val']}: 検証用データ")
    print(f"  - {split_files['test']}: テスト用データ")
    print("\n  【学習結果】")
    print("  - core/configs/learned_config.json: 学習済みパラメータ")
    print("  - initial_training_results_*.png: 学習結果の可視化")
    print("\n  【評価結果】")
    print("  - validation_result_*.png: 検証結果")
    print("  - validation_summary_*.json: 検証サマリー")
    print("  - tools/results/final_test_results.json: 最終評価結果")
    print("  - prediction_demo_static_*.png: デモ結果")
    print(f"\n実行完了: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # データ分割情報を保存
    split_info = {
        "original_file": data_file,
        "split_ratio": {
            "train": train_ratio,
            "validation": val_ratio,
            "test": test_ratio
        },
        "split_files": split_files,
        "timestamp": datetime.now().isoformat()
    }
    
    with open("data_split_info.json", "w", encoding="utf-8") as f:
        json.dump(split_info, f, indent=2, ensure_ascii=False)
    
    print("\n次のステップ:")
    print("1. core/configs/learned_config.json を使用して本番環境でモデルを初期化")
    print("2. tools/results/final_test_results.json で最終的な予測精度を確認")
    print("3. 必要に応じて分割比率を変更して再学習")


if __name__ == "__main__":
    main()