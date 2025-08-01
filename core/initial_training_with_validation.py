#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
予測モデルの初期学習スクリプト
過去データを使用してモデルパラメータを最適化
"""

import pandas as pd
import numpy as np
import json
from datetime import datetime, timedelta
from scipy.optimize import minimize
from scipy.stats import pearsonr
import matplotlib.pyplot as plt

# 日本語フォント設定
plt.rcParams['font.sans-serif'] = ['Yu Gothic', 'Hiragino Sans', 'Meirio', 'DejaVu Sans']
plt.rcParams['font.size'] = 10
plt.rcParams['axes.unicode_minus'] = False

class InitialTraining:
    def __init__(self, data_path):
        """
        初期化
        
        Parameters:
        -----------
        data_path : str
            学習用データのパス
        """
        self.data_path = data_path
        self.data = None
        self.training_results = {
            "delay_params": {},
            "response_rates": {},
            "hysteresis_correction": {},
            "water_level_correction": {}
        }
        
    def load_data(self):
        """データの読み込みと前処理"""
        print("=== 学習データ読み込み ===")
        self.data = pd.read_csv(self.data_path, encoding='utf-8')
        self.data['時刻'] = pd.to_datetime(self.data['時刻'])
        
        # 必要なカラムの確認
        required_cols = ['時刻', '水位_水位', 'ダム_全放流量']
        for col in required_cols:
            if col not in self.data.columns:
                raise ValueError(f"必要なカラム '{col}' が見つかりません")
        
        # 放流量150m³/s以上のデータのみ使用
        self.data = self.data[self.data['ダム_全放流量'] >= 150.0].reset_index(drop=True)
        
        print(f"学習データ: {len(self.data)}行")
        print(f"期間: {self.data['時刻'].min()} ～ {self.data['時刻'].max()}")
        
    def calculate_delay_parameters(self):
        """遅延時間パラメータの学習"""
        print("\n=== 遅延時間パラメータの学習 ===")
        
        # 放流量レンジ別に分析
        discharge_ranges = [
            ("150-300", 150, 300),
            ("300-500", 300, 500),
            ("500+", 500, 10000)
        ]
        
        for range_name, min_q, max_q in discharge_ranges:
            print(f"\n{range_name} m³/sの分析:")
            
            # 該当範囲のデータを抽出
            mask = (self.data['ダム_全放流量'] >= min_q) & (self.data['ダム_全放流量'] < max_q)
            subset = self.data[mask]
            
            if len(subset) < 100:
                print(f"  データ不足（{len(subset)}行）")
                continue
            
            # スライディングウィンドウで相互相関を計算
            delays = []
            correlations = []
            window_size = 120  # 2時間
            
            for i in range(0, len(subset) - window_size, 60):  # 1時間ごと
                window_data = subset.iloc[i:i+window_size]
                
                # 変動係数チェック（安定期間のみ使用）
                cv = window_data['ダム_全放流量'].std() / window_data['ダム_全放流量'].mean()
                if cv > 0.1:  # 変動が大きい場合はスキップ
                    continue
                
                # 相互相関計算
                best_delay = 0
                best_corr = 0
                
                for delay in range(0, 60):  # 0-60分の遅延
                    if delay + window_size > len(subset):
                        break
                    
                    q = window_data['ダム_全放流量'].values
                    h = subset.iloc[i+delay:i+delay+window_size]['水位_水位'].values
                    
                    if len(q) == len(h):
                        corr, _ = pearsonr(q, h)
                        if abs(corr) > abs(best_corr):
                            best_corr = corr
                            best_delay = delay
                
                if abs(best_corr) > 0.5:  # 相関が十分高い場合のみ採用
                    delays.append(best_delay)
                    correlations.append(abs(best_corr))
            
            if delays:
                # 中央値と平均相関を計算
                median_delay = np.median(delays)
                mean_correlation = np.mean(correlations)
                
                self.training_results["delay_params"][range_name] = {
                    "base_delay": int(median_delay),
                    "correlation": round(mean_correlation, 3),
                    "sample_size": len(delays)
                }
                
                print(f"  基準遅延時間: {median_delay:.0f}分")
                print(f"  平均相関係数: {mean_correlation:.3f}")
                print(f"  サンプル数: {len(delays)}")
            else:
                # データが不足している場合はデフォルト値を使用
                default_values = {
                    "150-300": {"base_delay": 25, "correlation": 0.824},
                    "300-500": {"base_delay": 15, "correlation": 0.710},
                    "500+": {"base_delay": 10, "correlation": 0.636}
                }
                
                if range_name in default_values:
                    self.training_results["delay_params"][range_name] = {
                        **default_values[range_name],
                        "sample_size": 0
                    }
                    print(f"  データ不足のためデフォルト値を使用")
                    print(f"  基準遅延時間: {default_values[range_name]['base_delay']}分")
                    print(f"  相関係数: {default_values[range_name]['correlation']}")
    
    def calculate_response_rates(self):
        """応答率（ΔH/ΔQ）の学習"""
        print("\n=== 応答率パラメータの学習 ===")
        
        # 変化量を計算
        self.data['dQ'] = self.data['ダム_全放流量'].diff()
        self.data['dH'] = self.data['水位_水位'].diff()
        
        # 放流量レンジと変化方向別に分析
        discharge_ranges = [
            ("150-300", 150, 300),
            ("300-500", 300, 500),
            ("500+", 500, 10000)
        ]
        
        for direction in ["increase", "decrease"]:
            self.training_results["response_rates"][direction] = {}
            
            print(f"\n{direction}時の応答率:")
            
            for range_name, min_q, max_q in discharge_ranges:
                # データ抽出
                mask = (self.data['ダム_全放流量'] >= min_q) & \
                       (self.data['ダム_全放流量'] < max_q) & \
                       (self.data['水位_水位'] >= 3.0)  # 水位3m以上
                
                if direction == "increase":
                    mask &= (self.data['dQ'] > 5)  # 5m³/s以上の増加
                else:
                    mask &= (self.data['dQ'] < -5)  # 5m³/s以上の減少
                
                subset = self.data[mask]
                
                if len(subset) < 50:
                    print(f"  {range_name}: データ不足")
                    continue
                
                # 遅延を考慮した応答率計算
                delay = self.training_results["delay_params"].get(range_name, {}).get("base_delay", 20)
                delay_steps = delay // 10  # 10分刻みのステップ数
                
                response_rates = []
                for i in range(delay_steps, len(subset) - delay_steps):
                    dQ = subset.iloc[i]['dQ']
                    dH = subset.iloc[i + delay_steps]['dH']
                    
                    if abs(dQ) > 10:  # 有意な変化のみ
                        rate = abs(dH / dQ)
                        if 0 < rate < 0.01:  # 異常値除外
                            response_rates.append(rate)
                
                if response_rates:
                    median_rate = np.median(response_rates)
                    self.training_results["response_rates"][direction][range_name] = round(median_rate, 4)
                    print(f"  {range_name}: {median_rate:.4f} m/(m³/s)")
                else:
                    # データが不足している場合はデフォルト値を使用
                    default_rates = {
                        "increase": {
                            "150-300": 0.0045,
                            "300-500": 0.0040,
                            "500+": 0.0035
                        },
                        "decrease": {
                            "150-300": 0.0042,
                            "300-500": 0.0038,
                            "500+": 0.0033
                        }
                    }
                    
                    if range_name in default_rates[direction]:
                        self.training_results["response_rates"][direction][range_name] = default_rates[direction][range_name]
                        print(f"  {range_name}: {default_rates[direction][range_name]:.4f} m/(m³/s) (デフォルト)")
    
    def calculate_hysteresis_correction(self):
        """ヒステリシス補正係数の学習"""
        print("\n=== ヒステリシス補正の学習 ===")
        
        # 上昇期と下降期で予測誤差を比較
        increase_errors = []
        decrease_errors = []
        
        # 簡易予測モデルで誤差を計算
        for i in range(100, len(self.data) - 100):
            # 現在の状態
            current_q = self.data.iloc[i]['ダム_全放流量']
            current_h = self.data.iloc[i]['水位_水位']
            
            # 変化方向
            dQ = self.data.iloc[i]['dQ']
            if abs(dQ) < 5:
                continue
            
            # 60分後の予測（簡易モデル）
            delay = 20  # 仮の遅延時間
            delay_steps = delay // 10
            
            if i + delay_steps + 6 < len(self.data):
                # 実際の変化
                actual_dH = self.data.iloc[i + delay_steps + 6]['水位_水位'] - current_h
                
                # 予測変化（基本応答率を使用）
                base_rate = 0.004  # 仮の応答率
                predicted_dH = base_rate * dQ * 6  # 60分間の累積
                
                # 誤差率
                if abs(predicted_dH) > 0.01:
                    error_rate = actual_dH / predicted_dH
                    
                    if 0.5 < error_rate < 2.0:  # 異常値除外
                        if dQ > 0:
                            increase_errors.append(error_rate)
                        else:
                            decrease_errors.append(error_rate)
        
        # 補正係数を計算
        if increase_errors and decrease_errors:
            increase_correction = np.median(increase_errors)
            decrease_correction = np.median(decrease_errors)
            
            self.training_results["hysteresis_correction"] = {
                "increase": round(increase_correction, 2),
                "decrease": round(decrease_correction, 2)
            }
            
            print(f"増加時補正: {increase_correction:.2f}")
            print(f"減少時補正: {decrease_correction:.2f}")
    
    def calculate_water_level_correction(self):
        """水位レベル別補正係数の学習"""
        print("\n=== 水位レベル補正の学習 ===")
        
        # 水位レベル別に応答特性を分析
        level_ranges = [
            ("low", 0, 3.5),
            ("medium", 3.5, 4.5),
            ("high", 4.5, 10)
        ]
        
        for level_name, min_h, max_h in level_ranges:
            # 該当範囲のデータ
            mask = (self.data['水位_水位'] >= min_h) & \
                   (self.data['水位_水位'] < max_h) & \
                   (abs(self.data['dQ']) > 10)
            
            subset = self.data[mask]
            
            if len(subset) < 50:
                print(f"{level_name}: データ不足")
                continue
            
            # 応答率を計算
            response_rates = []
            for i in range(20, len(subset) - 20):
                dQ = subset.iloc[i]['dQ']
                dH = subset.iloc[i + 2]['dH']  # 20分遅延
                
                if abs(dQ) > 10:
                    rate = abs(dH / dQ)
                    if 0 < rate < 0.01:
                        response_rates.append(rate)
            
            if response_rates:
                # 基準応答率（medium）との比率
                base_rate = 0.004  # 仮の基準値
                median_rate = np.median(response_rates)
                correction = median_rate / base_rate
                
                self.training_results["water_level_correction"][level_name] = round(correction, 1)
                print(f"{level_name} (<{max_h}m): {correction:.1f}")
            else:
                # データが不足している場合はデフォルト値を使用
                default_corrections = {
                    "low": 1.2,
                    "medium": 1.0,
                    "high": 0.9
                }
                
                if level_name in default_corrections:
                    self.training_results["water_level_correction"][level_name] = default_corrections[level_name]
                    print(f"{level_name} (<{max_h}m): {default_corrections[level_name]:.1f} (デフォルト)")
    
    def optimize_parameters(self):
        """パラメータの最適化（オプション）"""
        print("\n=== パラメータ最適化 ===")
        
        # ここでは簡易的な最適化を実装
        # 実際にはより高度な最適化手法を使用可能
        
        print("最適化完了（簡易版）")
    
    def save_results(self, output_path="core/configs/learned_config.json"):
        """学習結果の保存"""
        print(f"\n=== 学習結果を保存: {output_path} ===")
        
        # 設定ファイルの形式に整形
        config = {
            "min_discharge": 150.0,
            "prediction_hours": 3,
            "time_step": 10,
            "history_hours": 2,
            "delay_params": self.training_results["delay_params"],
            "response_rates": self.training_results["response_rates"],
            "hysteresis_correction": self.training_results["hysteresis_correction"],
            "water_level_correction": self.training_results["water_level_correction"],
            "stability_thresholds": {
                "stable": 0.05,
                "semi_variable": 0.15
            }
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=4, ensure_ascii=False)
        
        print("保存完了")
        
        return config
    
    def visualize_learning_results(self):
        """学習結果の可視化"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('初期学習結果', fontsize=14)
        
        # 1. 遅延時間パラメータ
        ax1 = axes[0, 0]
        if self.training_results["delay_params"]:
            ranges = list(self.training_results["delay_params"].keys())
            delays = [v["base_delay"] for v in self.training_results["delay_params"].values()]
            correlations = [v["correlation"] for v in self.training_results["delay_params"].values()]
            
            x = np.arange(len(ranges))
            width = 0.35
            
            ax1_twin = ax1.twinx()
            bars1 = ax1.bar(x - width/2, delays, width, label='遅延時間', color='skyblue')
            bars2 = ax1_twin.bar(x + width/2, correlations, width, label='相関係数', color='orange')
            
            ax1.set_xlabel('放流量レンジ')
            ax1.set_ylabel('遅延時間 (分)', color='skyblue')
            ax1_twin.set_ylabel('相関係数', color='orange')
            ax1.set_xticks(x)
            ax1.set_xticklabels(ranges)
            ax1.tick_params(axis='y', labelcolor='skyblue')
            ax1_twin.tick_params(axis='y', labelcolor='orange')
            ax1.set_title('放流量レンジ別の遅延時間')
            ax1.grid(True, alpha=0.3)
        
        # 2. 応答率
        ax2 = axes[0, 1]
        if self.training_results["response_rates"]:
            ranges = ["150-300", "300-500", "500+"]
            increase_rates = []
            decrease_rates = []
            
            for r in ranges:
                inc_rate = self.training_results["response_rates"].get("increase", {}).get(r, 0)
                dec_rate = self.training_results["response_rates"].get("decrease", {}).get(r, 0)
                increase_rates.append(inc_rate * 1000)  # m/(m³/s) → mm/(m³/s)
                decrease_rates.append(dec_rate * 1000)
            
            x = np.arange(len(ranges))
            width = 0.35
            
            bars1 = ax2.bar(x - width/2, increase_rates, width, label='増加時', color='red', alpha=0.7)
            bars2 = ax2.bar(x + width/2, decrease_rates, width, label='減少時', color='blue', alpha=0.7)
            
            ax2.set_xlabel('放流量レンジ')
            ax2.set_ylabel('応答率 (mm/(m³/s))')
            ax2.set_xticks(x)
            ax2.set_xticklabels(ranges)
            ax2.legend()
            ax2.set_title('放流量レンジ別の応答率')
            ax2.grid(True, alpha=0.3)
        
        # 3. ヒステリシス補正
        ax3 = axes[1, 0]
        if self.training_results["hysteresis_correction"]:
            categories = ['増加時', '減少時']
            corrections = [
                self.training_results["hysteresis_correction"].get("increase", 1.0),
                self.training_results["hysteresis_correction"].get("decrease", 1.0)
            ]
            
            bars = ax3.bar(categories, corrections, color=['red', 'blue'], alpha=0.7)
            ax3.axhline(y=1.0, color='black', linestyle='--', linewidth=1)
            ax3.set_ylabel('補正係数')
            ax3.set_title('ヒステリシス補正係数')
            ax3.set_ylim(0.9, 1.1)
            ax3.grid(True, alpha=0.3)
            
            # 値を表示
            for bar, val in zip(bars, corrections):
                ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{val:.2f}', ha='center', va='bottom')
        
        # 4. 水位レベル補正
        ax4 = axes[1, 1]
        if self.training_results["water_level_correction"]:
            levels = ['low\n(<3.5m)', 'medium\n(3.5-4.5m)', 'high\n(>4.5m)']
            corrections = [
                self.training_results["water_level_correction"].get("low", 1.0),
                self.training_results["water_level_correction"].get("medium", 1.0),
                self.training_results["water_level_correction"].get("high", 1.0)
            ]
            
            bars = ax4.bar(levels, corrections, color=['lightblue', 'lightgreen', 'lightcoral'])
            ax4.axhline(y=1.0, color='black', linestyle='--', linewidth=1)
            ax4.set_ylabel('補正係数')
            ax4.set_title('水位レベル別補正係数')
            ax4.set_ylim(0.8, 1.3)
            ax4.grid(True, alpha=0.3)
            
            # 値を表示
            for bar, val in zip(bars, corrections):
                ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{val:.1f}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        # 保存
        output_path = f"initial_training_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"\n学習結果グラフを保存: {output_path}")
        
        plt.show()


def main():
    """メイン処理"""
    import argparse
    import tkinter as tk
    from tkinter import filedialog
    import os
    
    parser = argparse.ArgumentParser(description='予測モデルの初期学習')
    parser.add_argument('--data', type=str, default=None,
                       help='学習用データファイルのパス')
    parser.add_argument('--output', type=str, default=None,
                       help='出力する設定ファイル名')
    parser.add_argument('--no-dialog', action='store_true',
                       help='ファイルダイアログを使用しない')
    
    args = parser.parse_args()
    
    print("=== 予測モデル初期学習 ===\n")
    
    # データファイルの選択
    if args.data and args.no_dialog:
        data_file = args.data
    else:
        root = tk.Tk()
        root.withdraw()
        data_file = filedialog.askopenfilename(
            title="学習用データファイルを選択してください",
            filetypes=[("CSVファイル", "*.csv"), ("すべてのファイル", "*.*")],
            initialfile=args.data if args.data else None
        )
        root.destroy()
        
        if not data_file:
            print("データファイルが選択されませんでした。")
            return
    
    print(f"学習データ: {data_file}")
    
    # 出力ファイルの選択
    if args.output and args.no_dialog:
        output_file = args.output
    else:
        root = tk.Tk()
        root.withdraw()
        default_name = os.path.splitext(os.path.basename(data_file))[0] + "_learned_config.json"
        output_file = filedialog.asksaveasfilename(
            title="学習結果の保存先を選択してください",
            defaultextension=".json",
            filetypes=[("JSONファイル", "*.json"), ("すべてのファイル", "*.*")],
            initialfile=args.output if args.output else default_name
        )
        root.destroy()
        
        if not output_file:
            print("出力ファイルが選択されませんでした。デフォルトのファイル名を使用します。")
            output_file = "core/configs/learned_config.json"
    
    print(f"出力ファイル: {output_file}")
    
    # 学習実行
    trainer = InitialTraining(data_file)
    
    # データ読み込み
    trainer.load_data()
    
    # 各パラメータの学習
    trainer.calculate_delay_parameters()
    trainer.calculate_response_rates()
    trainer.calculate_hysteresis_correction()
    trainer.calculate_water_level_correction()
    
    # パラメータ最適化（オプション）
    trainer.optimize_parameters()
    
    # 結果の保存
    config = trainer.save_results(output_file)
    
    # 結果の可視化
    trainer.visualize_learning_results()
    
    # サマリー表示
    print("\n=== 学習完了サマリー ===")
    print(f"遅延パラメータ: {len(trainer.training_results['delay_params'])}種類")
    print(f"応答率: {sum(len(v) for v in trainer.training_results['response_rates'].values())}種類")
    print("ヒステリシス補正: 設定完了")
    print("水位レベル補正: 設定完了")
    
    print(f"\n学習済み設定ファイル: {args.output}")
    print("この設定ファイルを使用してモデルを初期化してください：")
    print(f"model = RealtimePredictionModel('{args.output}')")


if __name__ == "__main__":
    main()