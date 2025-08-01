#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
放流量150m³/s以上のデータをフィルタリングし、
期間分類に基づいて学習用・検証用・テスト用に分割するスクリプト
"""

import pandas as pd
import numpy as np
import json
from datetime import datetime, timedelta
import os
import tkinter as tk
from tkinter import filedialog

class DataFilterAndSplitter:
    def __init__(self, input_file=None):
        """
        初期化
        
        Parameters:
        -----------
        input_file : str
            入力データファイルのパス
        """
        self.input_file = input_file
        self.data = None
        self.filtered_data = None
        self.periods = []
        self.metadata = {
            "input_file": None,
            "processing_date": datetime.now().isoformat(),
            "filter_conditions": {
                "min_discharge": 150.0,
                "exclude_low_discharge_and_low_level": True,
                "low_level_threshold": 3.0
            },
            "statistics": {},
            "periods": [],
            "split_info": {}
        }
        
    def select_input_file(self):
        """ファイルダイアログで入力ファイルを選択"""
        if not self.input_file:
            root = tk.Tk()
            root.withdraw()
            self.input_file = filedialog.askopenfilename(
                title="データファイルを選択",
                filetypes=[("CSVファイル", "*.csv"), ("すべてのファイル", "*.*")]
            )
            root.destroy()
            
            if not self.input_file:
                raise ValueError("ファイルが選択されませんでした")
        
        self.metadata["input_file"] = self.input_file
        return self.input_file
    
    def load_and_filter_data(self):
        """データの読み込みとフィルタリング"""
        print("=== データ読み込みとフィルタリング ===")
        
        # ファイル選択
        self.select_input_file()
        
        # データ読み込み
        print(f"読み込み中: {self.input_file}")
        self.data = pd.read_csv(self.input_file, encoding='utf-8')
        self.data['時刻'] = pd.to_datetime(self.data['時刻'])
        
        # 元のデータ統計
        self.metadata["statistics"]["original_count"] = len(self.data)
        self.metadata["statistics"]["original_period"] = {
            "start": str(self.data['時刻'].min()),
            "end": str(self.data['時刻'].max())
        }
        
        print(f"元のデータ数: {len(self.data)}行")
        
        # フィルタリング条件の適用
        # 放流量 < 150 または 水位 < 3 を除外
        exclude_mask = (self.data['ダム_全放流量'] < 150) | (self.data['水位_水位'] < 3)
        self.filtered_data = self.data[~exclude_mask].copy()
        
        # フィルタリング後の統計
        self.metadata["statistics"]["filtered_count"] = len(self.filtered_data)
        self.metadata["statistics"]["excluded_count"] = len(self.data) - len(self.filtered_data)
        self.metadata["statistics"]["filtered_period"] = {
            "start": str(self.filtered_data['時刻'].min()),
            "end": str(self.filtered_data['時刻'].max())
        }
        
        print(f"フィルタリング後: {len(self.filtered_data)}行")
        print(f"除外されたデータ: {len(self.data) - len(self.filtered_data)}行")
        
        # インデックスをリセット
        self.filtered_data = self.filtered_data.reset_index(drop=True)
        
    def identify_periods(self, method="time_based", window_hours=3):
        """期間の識別と分類
        
        Parameters:
        -----------
        method : str
            'event_based': イベントベースの期間分割（従来の方法）
            'time_based': 時間ベースの期間分割（3時間窓など）
        window_hours : int
            時間ベース分割の場合の窓サイズ（時間）
        """
        print(f"\n=== 期間の識別と分類 ({method}) ===")
        
        if method == "time_based":
            # 時間ベースの分割
            window_size = window_hours * 6  # 10分刻みなので6データ/時間
            
            # 連続したデータグループを識別
            time_diff = self.filtered_data['時刻'].diff()
            gap_indices = self.filtered_data[time_diff > timedelta(minutes=20)].index
            
            # 連続グループの開始と終了
            group_starts = [0] + list(gap_indices)
            group_ends = list(gap_indices - 1) + [len(self.filtered_data) - 1]
            
            period_id = 0
            
            # 各連続グループ内で時間窓分割
            for group_start, group_end in zip(group_starts, group_ends):
                group_data = self.filtered_data.iloc[group_start:group_end+1]
                
                # 窓をスライドさせながら期間を作成
                for i in range(0, len(group_data), window_size):
                    end_i = min(i + window_size, len(group_data))
                    
                    # 最小データ数チェック（2時間分以上）
                    if end_i - i < 12:
                        continue
                    
                    period_data = group_data.iloc[i:end_i]
                    
                    # グローバルインデックスを計算
                    global_start_idx = group_start + i
                    global_end_idx = group_start + end_i - 1
                    
                    # 期間の統計を計算
                    duration_hours = (period_data['時刻'].iloc[-1] - period_data['時刻'].iloc[0]).total_seconds() / 3600
                    cv_discharge = period_data['ダム_全放流量'].std() / period_data['ダム_全放流量'].mean()
                    cv_water_level = period_data['水位_水位'].std() / period_data['水位_水位'].mean()
                    
                    # 期間タイプの分類
                    if cv_discharge < 0.05:
                        period_type = "stable"
                    elif cv_discharge < 0.15:
                        period_type = "semi_variable"
                    else:
                        period_type = "variable"
                    
                    # 期間情報を保存
                    period_info = {
                        "period_id": period_id,
                        "start_idx": global_start_idx,
                        "end_idx": global_end_idx,
                        "start_time": str(period_data['時刻'].iloc[0]),
                        "end_time": str(period_data['時刻'].iloc[-1]),
                        "duration_hours": round(duration_hours, 2),
                        "data_points": len(period_data),
                        "period_type": period_type,
                        "cv_discharge": round(cv_discharge, 4),
                        "cv_water_level": round(cv_water_level, 4),
                        "mean_discharge": round(period_data['ダム_全放流量'].mean(), 1),
                        "mean_water_level": round(period_data['水位_水位'].mean(), 2),
                        "max_discharge": round(period_data['ダム_全放流量'].max(), 1),
                        "min_discharge": round(period_data['ダム_全放流量'].min(), 1)
                    }
                    
                    self.periods.append(period_info)
                    period_id += 1
        
        else:
            # イベントベースの分割（従来の方法）
            # 時間差を計算して連続性をチェック
            time_diff = self.filtered_data['時刻'].diff()
            
            # 20分以上の間隔がある場合は別期間とする
            gap_indices = self.filtered_data[time_diff > timedelta(minutes=20)].index
            
            # 期間の開始と終了インデックスを特定
            period_starts = [0] + list(gap_indices)
            period_ends = list(gap_indices - 1) + [len(self.filtered_data) - 1]
            
            # 各期間を分析
            for i, (start_idx, end_idx) in enumerate(zip(period_starts, period_ends)):
                if end_idx - start_idx < 6:  # 1時間未満の期間はスキップ
                    continue
                    
                period_data = self.filtered_data.iloc[start_idx:end_idx+1]
                
                # 期間の長さ（時間）
                duration_hours = (period_data['時刻'].iloc[-1] - period_data['時刻'].iloc[0]).total_seconds() / 3600
                
                if duration_hours < 1.0:  # 1時間未満はスキップ
                    continue
                
                # 変動係数を計算（放流量と水位の両方）
                cv_discharge = period_data['ダム_全放流量'].std() / period_data['ダム_全放流量'].mean()
                cv_water_level = period_data['水位_水位'].std() / period_data['水位_水位'].mean()
                
                # 期間タイプの分類
                if cv_discharge < 0.05:
                    period_type = "stable"
                elif cv_discharge < 0.15:
                    period_type = "semi_variable"
                else:
                    period_type = "variable"
                
                # 期間情報を保存
                period_info = {
                    "period_id": i,
                    "start_idx": start_idx,
                    "end_idx": end_idx,
                    "start_time": str(period_data['時刻'].iloc[0]),
                    "end_time": str(period_data['時刻'].iloc[-1]),
                    "duration_hours": round(duration_hours, 2),
                    "data_points": len(period_data),
                    "period_type": period_type,
                    "cv_discharge": round(cv_discharge, 4),
                    "cv_water_level": round(cv_water_level, 4),
                    "mean_discharge": round(period_data['ダム_全放流量'].mean(), 1),
                    "mean_water_level": round(period_data['水位_水位'].mean(), 2),
                    "max_discharge": round(period_data['ダム_全放流量'].max(), 1),
                    "min_discharge": round(period_data['ダム_全放流量'].min(), 1)
                }
                
                self.periods.append(period_info)
        
        # メタデータに保存
        self.metadata["periods"] = self.periods
        self.metadata["statistics"]["total_periods"] = len(self.periods)
        self.metadata["statistics"]["period_types"] = {
            "stable": sum(1 for p in self.periods if p["period_type"] == "stable"),
            "semi_variable": sum(1 for p in self.periods if p["period_type"] == "semi_variable"),
            "variable": sum(1 for p in self.periods if p["period_type"] == "variable")
        }
        self.metadata["statistics"]["segmentation_method"] = method
        if method == "time_based":
            self.metadata["statistics"]["window_hours"] = window_hours
        
        print(f"識別された期間数: {len(self.periods)}")
        print(f"  安定期間: {self.metadata['statistics']['period_types']['stable']}")
        print(f"  準変動期間: {self.metadata['statistics']['period_types']['semi_variable']}")
        print(f"  変動期間: {self.metadata['statistics']['period_types']['variable']}")
        
    def split_periods(self, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
        """期間単位でデータを分割（平均放流量のバランスを考慮）"""
        print("\n=== データ分割（放流量バランス考慮） ===")
        
        # 放流量レベル別に期間を分類
        discharge_levels = [
            ("150-300", 150, 300),
            ("300-500", 300, 500),
            ("500-800", 500, 800),
            ("800+", 800, 10000)
        ]
        
        # 各レベルでさらに期間タイプ別に分類
        categorized_periods = {}
        for level_name, min_q, max_q in discharge_levels:
            categorized_periods[level_name] = {
                "stable": [],
                "semi_variable": [],
                "variable": []
            }
            
            for period in self.periods:
                if min_q <= period["mean_discharge"] < max_q:
                    period_type = period["period_type"]
                    categorized_periods[level_name][period_type].append(period)
        
        # 各カテゴリの期間数を表示
        print("\n放流量レベル×期間タイプ別の期間数:")
        for level_name in discharge_levels:
            level_data = categorized_periods[level_name[0]]
            total = sum(len(periods) for periods in level_data.values())
            if total > 0:
                print(f"\n{level_name[0]}:")
                for period_type in ["stable", "semi_variable", "variable"]:
                    count = len(level_data[period_type])
                    if count > 0:
                        print(f"  {period_type}: {count}期間")
        
        # バランスの取れた分割
        train_periods = []
        val_periods = []
        test_periods = []
        
        # 各放流量レベル×期間タイプから比例配分
        np.random.seed(42)
        
        for level_name, _, _ in discharge_levels:
            for period_type in ["stable", "semi_variable", "variable"]:
                periods = categorized_periods[level_name][period_type]
                n_periods = len(periods)
                
                if n_periods == 0:
                    continue
                
                # ランダムに並び替え
                shuffled_indices = np.random.permutation(n_periods)
                shuffled_periods = [periods[i] for i in shuffled_indices]
                
                # 分割数を計算
                if n_periods == 1:
                    # 1つしかない場合は学習用に
                    train_periods.extend(shuffled_periods)
                elif n_periods == 2:
                    # 2つの場合は学習と検証またはテストに
                    train_periods.append(shuffled_periods[0])
                    # 検証データが少ない場合は優先的に検証に
                    if len(val_periods) < len(test_periods):
                        val_periods.append(shuffled_periods[1])
                    else:
                        test_periods.append(shuffled_periods[1])
                else:
                    # 3つ以上の場合は比例配分
                    n_train = max(1, int(n_periods * train_ratio))
                    n_val = max(1 if n_periods >= 3 else 0, int(n_periods * val_ratio))
                    n_test = n_periods - n_train - n_val
                    
                    if n_test < 0:
                        # テストデータが負になる場合は調整
                        if n_periods >= 3:
                            n_train = n_periods - 2
                            n_val = 1
                            n_test = 1
                        else:
                            n_val = n_periods - n_train
                            n_test = 0
                    
                    train_periods.extend(shuffled_periods[:n_train])
                    if n_val > 0:
                        val_periods.extend(shuffled_periods[n_train:n_train+n_val])
                    if n_test > 0:
                        test_periods.extend(shuffled_periods[n_train+n_val:])
        
        # 分割結果のバランスを表示
        def show_balance(periods, name):
            if not periods:
                return
            print(f"\n{name}データのバランス:")
            for level_name, min_q, max_q in discharge_levels:
                count = sum(1 for p in periods if min_q <= p["mean_discharge"] < max_q)
                if count > 0:
                    mean_discharge = np.mean([p["mean_discharge"] for p in periods 
                                            if min_q <= p["mean_discharge"] < max_q])
                    print(f"  {level_name}: {count}期間 (平均{mean_discharge:.1f} m³/s)")
        
        show_balance(train_periods, "学習")
        show_balance(val_periods, "検証")
        show_balance(test_periods, "テスト")
        
        # データの抽出
        train_data = self._extract_data_from_periods(train_periods)
        val_data = self._extract_data_from_periods(val_periods)
        test_data = self._extract_data_from_periods(test_periods)
        
        # 時間順にソート（空でない場合のみ）
        if not train_data.empty:
            train_data = train_data.sort_values('時刻').reset_index(drop=True)
        if not val_data.empty:
            val_data = val_data.sort_values('時刻').reset_index(drop=True)
        if not test_data.empty:
            test_data = test_data.sort_values('時刻').reset_index(drop=True)
        
        # 分割情報を保存
        self.metadata["split_info"] = {
            "train": {
                "periods": len(train_periods),
                "data_points": len(train_data),
                "period_types": self._count_period_types(train_periods)
            },
            "validation": {
                "periods": len(val_periods),
                "data_points": len(val_data),
                "period_types": self._count_period_types(val_periods)
            },
            "test": {
                "periods": len(test_periods),
                "data_points": len(test_data),
                "period_types": self._count_period_types(test_periods)
            }
        }
        
        print(f"\n分割結果:")
        print(f"  学習用: {len(train_data)}行 ({len(train_periods)}期間)")
        print(f"  検証用: {len(val_data)}行 ({len(val_periods)}期間)")
        print(f"  テスト用: {len(test_data)}行 ({len(test_periods)}期間)")
        
        return train_data, val_data, test_data
    
    def _extract_data_from_periods(self, periods):
        """期間リストから実際のデータを抽出"""
        data_frames = []
        for period in periods:
            start_idx = period["start_idx"]
            end_idx = period["end_idx"]
            period_data = self.filtered_data.iloc[start_idx:end_idx+1].copy()
            data_frames.append(period_data)
        
        if data_frames:
            return pd.concat(data_frames, ignore_index=True)
        else:
            return pd.DataFrame()
    
    def _count_period_types(self, periods):
        """期間タイプをカウント"""
        counts = {"stable": 0, "semi_variable": 0, "variable": 0}
        for period in periods:
            counts[period["period_type"]] += 1
        return counts
    
    def save_filtered_data(self, output_dir="."):
        """フィルタリングされたデータを保存"""
        base_name = os.path.splitext(os.path.basename(self.input_file))[0]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # フィルタリング済み全データ
        filtered_file = os.path.join(output_dir, f"{base_name}_filtered_{timestamp}.csv")
        self.filtered_data.to_csv(filtered_file, index=False, encoding='utf-8')
        print(f"\nフィルタリング済みデータを保存: {filtered_file}")
        
        return filtered_file
    
    def save_split_data(self, train_data, val_data, test_data, output_dir="."):
        """分割されたデータを保存"""
        base_name = os.path.splitext(os.path.basename(self.input_file))[0]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 各データセットを保存
        train_file = os.path.join(output_dir, f"{base_name}_train_{timestamp}.csv")
        val_file = os.path.join(output_dir, f"{base_name}_val_{timestamp}.csv")
        test_file = os.path.join(output_dir, f"{base_name}_test_{timestamp}.csv")
        
        train_data.to_csv(train_file, index=False, encoding='utf-8')
        val_data.to_csv(val_file, index=False, encoding='utf-8')
        test_data.to_csv(test_file, index=False, encoding='utf-8')
        
        print(f"\n分割データを保存:")
        print(f"  学習用: {train_file}")
        print(f"  検証用: {val_file}")
        print(f"  テスト用: {test_file}")
        
        # メタデータを保存
        metadata_file = os.path.join(output_dir, f"{base_name}_split_metadata_{timestamp}.json")
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(self.metadata, f, indent=2, ensure_ascii=False)
        print(f"  メタデータ: {metadata_file}")
        
        return train_file, val_file, test_file, metadata_file
    
    def visualize_split_result(self):
        """分割結果の可視化"""
        import matplotlib.pyplot as plt
        
        # 期間タイプの分布
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # 1. 期間タイプの分布（全体）
        ax1 = axes[0, 0]
        period_types = [p["period_type"] for p in self.periods]
        type_counts = pd.Series(period_types).value_counts()
        ax1.bar(type_counts.index, type_counts.values)
        ax1.set_title('期間タイプの分布（全体）')
        ax1.set_xlabel('期間タイプ')
        ax1.set_ylabel('期間数')
        
        # 2. 期間の長さ分布
        ax2 = axes[0, 1]
        durations = [p["duration_hours"] for p in self.periods]
        ax2.hist(durations, bins=20, edgecolor='black')
        ax2.set_title('期間長さの分布')
        ax2.set_xlabel('期間長さ（時間）')
        ax2.set_ylabel('頻度')
        
        # 3. 平均放流量の分布
        ax3 = axes[1, 0]
        mean_discharges = [p["mean_discharge"] for p in self.periods]
        ax3.hist(mean_discharges, bins=20, edgecolor='black')
        ax3.set_title('期間平均放流量の分布')
        ax3.set_xlabel('平均放流量 (m³/s)')
        ax3.set_ylabel('頻度')
        
        # 4. データ分割の結果
        ax4 = axes[1, 1]
        split_info = self.metadata["split_info"]
        datasets = ['Train', 'Val', 'Test']
        data_points = [
            split_info["train"]["data_points"],
            split_info["validation"]["data_points"],
            split_info["test"]["data_points"]
        ]
        ax4.bar(datasets, data_points)
        ax4.set_title('データ分割結果')
        ax4.set_xlabel('データセット')
        ax4.set_ylabel('データ点数')
        
        # 各バーの上に数値を表示
        for i, v in enumerate(data_points):
            ax4.text(i, v + max(data_points)*0.01, str(v), ha='center')
        
        plt.tight_layout()
        
        # 保存
        output_file = f"data_split_visualization_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"\n可視化結果を保存: {output_file}")
        plt.show()


def main():
    """メイン処理"""
    import argparse
    
    parser = argparse.ArgumentParser(description='データフィルタリングと分割')
    parser.add_argument('--method', type=str, default='time_based',
                       choices=['event_based', 'time_based'],
                       help='期間分割方法')
    parser.add_argument('--window-hours', type=int, default=3,
                       help='時間ベース分割の窓サイズ（時間）')
    parser.add_argument('--input', type=str, default=None,
                       help='入力データファイル')
    
    args = parser.parse_args()
    
    print("=== 放流量150m³/s以上データのフィルタリングと分割 ===\n")
    print(f"分割方法: {args.method}")
    if args.method == 'time_based':
        print(f"窓サイズ: {args.window_hours}時間")
    
    # 処理実行
    splitter = DataFilterAndSplitter(input_file=args.input)
    
    # 1. データ読み込みとフィルタリング
    splitter.load_and_filter_data()
    
    # 2. 期間の識別
    splitter.identify_periods(method=args.method, window_hours=args.window_hours)
    
    # 3. フィルタリング済みデータの保存
    filtered_file = splitter.save_filtered_data()
    
    # 4. データ分割
    train_data, val_data, test_data = splitter.split_periods()
    
    # 5. 分割データの保存
    train_file, val_file, test_file, metadata_file = splitter.save_split_data(
        train_data, val_data, test_data
    )
    
    # 6. 結果の可視化
    splitter.visualize_split_result()
    
    print("\n=== 処理完了 ===")
    print(f"\n次のステップ:")
    print(f"1. 学習: python initial_training.py --data {train_file}")
    print(f"2. 検証: python validate_prediction.py --data {val_file}")
    print(f"3. テスト: python validate_prediction.py --data {test_file}")


if __name__ == "__main__":
    main()