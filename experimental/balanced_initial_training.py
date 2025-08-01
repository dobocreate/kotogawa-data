#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
バランスの取れた初期学習スクリプト
放流量レンジごとに適切にサンプリングして学習
"""

import pandas as pd
import numpy as np
import json
from datetime import datetime, timedelta
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
from initial_training import InitialTraining

# 日本語フォント設定
plt.rcParams['font.sans-serif'] = ['Yu Gothic', 'Hiragino Sans', 'Meirio', 'DejaVu Sans']
plt.rcParams['font.size'] = 10
plt.rcParams['axes.unicode_minus'] = False

class BalancedInitialTraining(InitialTraining):
    """バランスの取れた学習を行うクラス"""
    
    def __init__(self, data_path):
        """初期化（親クラスは呼ばない）"""
        self.data_path = data_path
        self.data = None
        self.balanced_data = None
        self.training_results = {
            "delay_params": {},
            "response_rates": {},
            "hysteresis_correction": {},
            "water_level_correction": {}
        }
    
    def load_and_balance_data(self, strategy='sampling', samples_per_range=1000):
        """
        データの読み込みとバランシング
        
        Parameters:
        -----------
        strategy : str
            'sampling': 各レンジから一定数サンプリング
            'weighting': 重み付けによるバランシング
            'augmentation': 少数データの拡張
        samples_per_range : int
            各レンジからサンプリングする最大数
        """
        print("=== バランス調整されたデータ読み込み ===")
        
        # データ読み込み
        self.data = pd.read_csv(self.data_path, encoding='utf-8')
        self.data['時刻'] = pd.to_datetime(self.data['時刻'])
        
        # 放流量レンジの定義
        ranges = [
            ("0-150", 0, 150),
            ("150-300", 150, 300),
            ("300-500", 300, 500),
            ("500-800", 500, 800),
            ("800-1000", 800, 1000),
            ("1000+", 1000, 10000)
        ]
        
        # 元のデータ分布を表示
        print("\n元のデータ分布:")
        for name, min_q, max_q in ranges:
            mask = (self.data['ダム_全放流量'] >= min_q) & (self.data['ダム_全放流量'] < max_q)
            count = mask.sum()
            print(f"  {name:10s}: {count:6d}件")
        
        if strategy == 'sampling':
            # 各レンジから均等にサンプリング
            balanced_dfs = []
            
            for name, min_q, max_q in ranges:
                if min_q == 0:  # 0-150は学習対象外
                    continue
                    
                mask = (self.data['ダム_全放流量'] >= min_q) & (self.data['ダム_全放流量'] < max_q)
                range_data = self.data[mask]
                
                if len(range_data) > 0:
                    # サンプル数を決定
                    n_samples = min(len(range_data), samples_per_range)
                    
                    if len(range_data) > samples_per_range:
                        # ランダムサンプリング
                        sampled = range_data.sample(n=n_samples, random_state=42)
                    else:
                        # 全データを使用
                        sampled = range_data
                    
                    balanced_dfs.append(sampled)
                    print(f"  {name}: {len(sampled)}件をサンプリング")
            
            # 結合
            self.balanced_data = pd.concat(balanced_dfs, ignore_index=True)
            self.balanced_data = self.balanced_data.sort_values('時刻').reset_index(drop=True)
            
        elif strategy == 'augmentation':
            # 少数データの拡張（時系列を考慮した補間）
            balanced_dfs = []
            
            for name, min_q, max_q in ranges:
                if min_q == 0:
                    continue
                    
                mask = (self.data['ダム_全放流量'] >= min_q) & (self.data['ダム_全放流量'] < max_q)
                range_data = self.data[mask]
                
                if len(range_data) > 0:
                    balanced_dfs.append(range_data)
                    
                    # データが少ない場合は近傍データで補強
                    if len(range_data) < samples_per_range / 2:
                        # 前後のデータも含める
                        extended_mask = (
                            (self.data['ダム_全放流量'] >= min_q * 0.9) & 
                            (self.data['ダム_全放流量'] < max_q * 1.1)
                        )
                        extended_data = self.data[extended_mask]
                        
                        # 時系列的に連続したデータを優先
                        for idx in range_data.index:
                            # 前後1時間のデータを追加
                            time_mask = (
                                (self.data['時刻'] >= self.data.loc[idx, '時刻'] - timedelta(hours=1)) &
                                (self.data['時刻'] <= self.data.loc[idx, '時刻'] + timedelta(hours=1))
                            )
                            nearby_data = self.data[time_mask & extended_mask]
                            balanced_dfs.append(nearby_data)
            
            # 結合と重複除去
            self.balanced_data = pd.concat(balanced_dfs, ignore_index=True)
            self.balanced_data = self.balanced_data.drop_duplicates().sort_values('時刻').reset_index(drop=True)
        
        else:  # weighting
            # 重み付けによるバランシング（全データを使用し、レアなデータに高い重みを付ける）
            self.balanced_data = self.data[self.data['ダム_全放流量'] >= 150].copy()
            
            # 各データポイントに重みを付ける
            weights = []
            for _, row in self.balanced_data.iterrows():
                q = row['ダム_全放流量']
                if q < 300:
                    weight = 1.0
                elif q < 500:
                    weight = 2.0
                elif q < 800:
                    weight = 5.0
                elif q < 1000:
                    weight = 10.0
                else:
                    weight = 20.0
                weights.append(weight)
            
            self.balanced_data['weight'] = weights
        
        print(f"\nバランス調整後のデータ数: {len(self.balanced_data)}件")
        
        # バランス調整後の分布を表示
        print("\nバランス調整後の分布:")
        for name, min_q, max_q in ranges:
            if min_q == 0:
                continue
            mask = (self.balanced_data['ダム_全放流量'] >= min_q) & (self.balanced_data['ダム_全放流量'] < max_q)
            count = mask.sum()
            ratio = count / len(self.balanced_data) * 100 if len(self.balanced_data) > 0 else 0
            print(f"  {name:10s}: {count:6d}件 ({ratio:5.1f}%)")
    
    def calculate_delay_parameters_balanced(self):
        """バランスの取れたデータで遅延時間パラメータを学習"""
        print("\n=== バランスの取れた遅延時間パラメータの学習 ===")
        
        # 元のメソッドを使用（ただしself.dataをself.balanced_dataに置き換え）
        original_data = self.data
        self.data = self.balanced_data
        
        # 親クラスのメソッドを呼び出す
        self.calculate_delay_parameters()
        
        # データを戻す
        self.data = original_data
    
    def visualize_balanced_distribution(self):
        """バランス調整前後の分布を可視化"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # 元のデータ分布
        ax1.hist(self.data[self.data['ダム_全放流量'] >= 150]['ダム_全放流量'], 
                bins=50, edgecolor='black', alpha=0.7)
        ax1.set_xlabel('放流量 (m³/s)')
        ax1.set_ylabel('頻度')
        ax1.set_title('元のデータ分布（150m³/s以上）')
        ax1.set_yscale('log')  # 対数スケール
        ax1.grid(True, alpha=0.3)
        
        # バランス調整後の分布
        ax2.hist(self.balanced_data['ダム_全放流量'], 
                bins=50, edgecolor='black', alpha=0.7, color='orange')
        ax2.set_xlabel('放流量 (m³/s)')
        ax2.set_ylabel('頻度')
        ax2.set_title('バランス調整後の分布')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        output_path = f"balanced_distribution_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"\n分布比較グラフを保存: {output_path}")
        plt.show()
    
    def run_balanced_training(self):
        """バランスの取れた学習を実行"""
        # データの読み込みとバランシング
        self.load_and_balance_data(strategy='sampling', samples_per_range=500)
        
        # 分布の可視化
        self.visualize_balanced_distribution()
        
        # 各パラメータの学習（バランスの取れたデータで）
        original_data = self.data
        self.data = self.balanced_data
        
        self.calculate_delay_parameters()
        self.calculate_response_rates()
        self.calculate_hysteresis_correction()
        self.calculate_water_level_correction()
        self.optimize_parameters()
        
        self.data = original_data
        
        # 結果の保存
        config = self.save_results("balanced_learned_config.json")
        
        # 学習結果の可視化
        self.visualize_learning_results()
        
        return config


def compare_training_methods(data_path):
    """通常の学習とバランスの取れた学習を比較"""
    print("=== 学習方法の比較 ===\n")
    
    # 通常の学習
    print("1. 通常の学習:")
    normal_trainer = InitialTraining(data_path)
    normal_trainer.load_data()
    normal_trainer.calculate_delay_parameters()
    normal_config = normal_trainer.training_results["delay_params"]
    
    # バランスの取れた学習
    print("\n2. バランスの取れた学習:")
    balanced_trainer = BalancedInitialTraining(data_path)
    balanced_trainer.run_balanced_training()
    balanced_config = balanced_trainer.training_results["delay_params"]
    
    # 結果の比較
    print("\n=== 学習結果の比較 ===")
    print("放流量レンジ | 通常の遅延時間 | バランス学習の遅延時間")
    print("-" * 60)
    
    for range_name in ["150-300", "300-500", "500-800", "800-1000", "1000+"]:
        normal_delay = normal_config.get(range_name, {}).get("base_delay", "N/A")
        balanced_delay = balanced_config.get(range_name, {}).get("base_delay", "N/A")
        
        normal_str = f"{normal_delay}分" if normal_delay != "N/A" else "データ不足"
        balanced_str = f"{balanced_delay}分" if balanced_delay != "N/A" else "データ不足"
        
        print(f"{range_name:12s} | {normal_str:14s} | {balanced_str}")


def main():
    """メイン処理"""
    import argparse
    
    parser = argparse.ArgumentParser(description='バランスの取れた初期学習')
    parser.add_argument('--data', type=str, default='統合データ_水位ダム_20250730_030746.csv',
                       help='学習データファイル')
    parser.add_argument('--strategy', type=str, default='sampling',
                       choices=['sampling', 'weighting', 'augmentation'],
                       help='バランシング戦略')
    parser.add_argument('--samples', type=int, default=500,
                       help='各レンジからのサンプル数')
    parser.add_argument('--compare', action='store_true',
                       help='通常学習との比較を実行')
    
    args = parser.parse_args()
    
    if args.compare:
        # 比較モード
        compare_training_methods(args.data)
    else:
        # バランス学習のみ
        trainer = BalancedInitialTraining(args.data)
        trainer.load_and_balance_data(strategy=args.strategy, samples_per_range=args.samples)
        trainer.visualize_balanced_distribution()
        trainer.run_balanced_training()
        
        print("\n学習完了！")
        print("生成されたファイル:")
        print("  - balanced_learned_config.json: バランス学習済みパラメータ")
        print("  - balanced_distribution_*.png: データ分布の比較")
        print("  - initial_training_results_*.png: 学習結果")


if __name__ == "__main__":
    main()