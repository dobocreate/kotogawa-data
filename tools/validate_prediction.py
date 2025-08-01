#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
予測モデルの検証・評価スクリプト
実データを使用して予測精度を評価
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import json
from realtime_prediction_model import RealtimePredictionModel

# 日本語フォント設定
plt.rcParams['font.sans-serif'] = ['Yu Gothic', 'Hiragino Sans', 'Meirio', 'DejaVu Sans']
plt.rcParams['font.size'] = 10
plt.rcParams['axes.unicode_minus'] = False

class PredictionValidator:
    def __init__(self, data_path, config_path=None):
        """
        初期化
        
        Parameters:
        -----------
        data_path : str
            検証用データのパス
        config_path : str
            モデル設定ファイルのパス
        """
        self.model = RealtimePredictionModel(config_path)
        self.data = None
        self.data_path = data_path
        self.validation_results = []
        
    def load_data(self):
        """データの読み込み"""
        print("=== 検証データ読み込み ===")
        self.data = pd.read_csv(self.data_path, encoding='utf-8')
        self.data['時刻'] = pd.to_datetime(self.data['時刻'])
        
        # 必要なカラムの確認
        required_cols = ['時刻', '水位_水位', 'ダム_全放流量']
        for col in required_cols:
            if col not in self.data.columns:
                raise ValueError(f"必要なカラム '{col}' が見つかりません")
        
        print(f"データ読み込み完了: {len(self.data)}行")
        
        # 放流量150m³/s以上のデータのみ抽出
        self.data = self.data[self.data['ダム_全放流量'] >= 150.0].reset_index(drop=True)
        print(f"放流量150m³/s以上のデータ: {len(self.data)}行")
    
    def validate_period(self, start_idx, period_hours=24):
        """
        指定期間での検証を実行
        
        Parameters:
        -----------
        start_idx : int
            開始インデックス
        period_hours : int
            検証期間（時間）
        """
        period_steps = period_hours * 6  # 10分刻み
        
        if start_idx + period_steps > len(self.data):
            return None
        
        results = {
            "start_time": self.data.iloc[start_idx]['時刻'],
            "predictions": [],
            "actual": [],
            "errors": [],
            "discharge": []
        }
        
        # 初期履歴の設定（2時間分）
        history_steps = 12
        if start_idx >= history_steps:
            for i in range(history_steps):
                idx = start_idx - history_steps + i
                self.model.update_history(
                    self.data.iloc[idx]['ダム_全放流量'],
                    self.data.iloc[idx]['水位_水位']
                )
        
        # 3時間ごとに予測を実行
        prediction_interval = 18  # 3時間 = 18ステップ
        
        for i in range(0, period_steps - prediction_interval, 6):  # 1時間ごと
            current_idx = start_idx + i
            
            # 現在の状態
            current_discharge = self.data.iloc[current_idx]['ダム_全放流量']
            current_water_level = self.data.iloc[current_idx]['水位_水位']
            
            # 将来の放流量（実績値）
            future_discharge = []
            for j in range(prediction_interval):
                if current_idx + j < len(self.data):
                    future_discharge.append(self.data.iloc[current_idx + j]['ダム_全放流量'])
            
            # 予測実行
            pred_result = self.model.predict(current_discharge, current_water_level, future_discharge)
            
            if pred_result["status"] == "success":
                # 30分、60分、90分、120分、180分後の予測を評価
                eval_points = [3, 6, 9, 12, 18]  # ステップ数
                
                for point in eval_points:
                    if point < len(pred_result["water_levels"]) and current_idx + point < len(self.data):
                        pred_level = pred_result["water_levels"][point - 1]
                        actual_level = self.data.iloc[current_idx + point]['水位_水位']
                        error = pred_level - actual_level
                        
                        results["predictions"].append({
                            "time": self.data.iloc[current_idx + point]['時刻'],
                            "lead_time": point * 10,
                            "predicted": pred_level,
                            "actual": actual_level,
                            "error": error,
                            "confidence": pred_result["confidence"][point - 1],
                            "state": pred_result["state"],
                            "discharge": self.data.iloc[current_idx + point]['ダム_全放流量']
                        })
                        
                        # オンライン学習
                        if point == 6:  # 60分後の予測で学習
                            self.model.update_online_learning(pred_level, actual_level)
            
            # 履歴更新
            for j in range(min(6, len(self.data) - current_idx)):
                self.model.update_history(
                    self.data.iloc[current_idx + j]['ダム_全放流量'],
                    self.data.iloc[current_idx + j]['水位_水位']
                )
        
        return results
    
    def analyze_results(self, results):
        """検証結果の分析"""
        if not results["predictions"]:
            return None
        
        df = pd.DataFrame(results["predictions"])
        
        analysis = {}
        
        # リードタイム別の精度
        for lead_time in [30, 60, 90, 120, 180]:
            subset = df[df["lead_time"] == lead_time]
            if len(subset) > 0:
                mae = np.mean(np.abs(subset["error"]))
                rmse = np.sqrt(np.mean(subset["error"] ** 2))
                within_5cm = (np.abs(subset["error"]) < 0.05).sum() / len(subset) * 100
                within_10cm = (np.abs(subset["error"]) < 0.10).sum() / len(subset) * 100
                
                analysis[f"{lead_time}min"] = {
                    "mae": mae,
                    "rmse": rmse,
                    "within_5cm": within_5cm,
                    "within_10cm": within_10cm,
                    "count": len(subset)
                }
        
        # 状態別の精度
        for state in df["state"].unique():
            subset = df[df["state"] == state]
            if len(subset) > 0:
                mae = np.mean(np.abs(subset["error"]))
                analysis[f"state_{state}"] = {
                    "mae": mae,
                    "count": len(subset)
                }
        
        # 放流量範囲別
        discharge_ranges = [(150, 300), (300, 500), (500, 1000)]
        for min_q, max_q in discharge_ranges:
            subset = df[(df["discharge"] >= min_q) & (df["discharge"] < max_q)]
            if len(subset) > 0:
                mae = np.mean(np.abs(subset["error"]))
                analysis[f"discharge_{min_q}-{max_q}"] = {
                    "mae": mae,
                    "count": len(subset)
                }
        
        return analysis
    
    def visualize_validation(self, results, output_path=None):
        """検証結果の可視化"""
        if not results["predictions"]:
            return
        
        df = pd.DataFrame(results["predictions"])
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'予測モデル検証結果 ({results["start_time"].strftime("%Y-%m-%d %H:%M")}～)', fontsize=14)
        
        # 1. 時系列比較（60分予測）
        ax1 = axes[0, 0]
        df_60min = df[df["lead_time"] == 60].sort_values("time")
        
        if len(df_60min) > 0:
            ax1.plot(df_60min["time"], df_60min["actual"], 'b-', label='実測値', linewidth=2)
            ax1.plot(df_60min["time"], df_60min["predicted"], 'r--', label='予測値', linewidth=2)
            ax1.fill_between(df_60min["time"], 
                           df_60min["predicted"] - 0.1, 
                           df_60min["predicted"] + 0.1,
                           alpha=0.3, color='red', label='±10cm範囲')
            
            ax1.set_xlabel('時刻')
            ax1.set_ylabel('水位 (m)')
            ax1.set_title('60分先予測の時系列比較')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # x軸の表示調整
            ax1.tick_params(axis='x', rotation=45)
        
        # 2. 誤差分布（リードタイム別）
        ax2 = axes[0, 1]
        lead_times = [30, 60, 90, 120, 180]
        errors_by_lead = []
        
        for lt in lead_times:
            subset = df[df["lead_time"] == lt]
            if len(subset) > 0:
                errors_by_lead.append(subset["error"].values * 100)  # cmに変換
            else:
                errors_by_lead.append([])
        
        bp = ax2.boxplot(errors_by_lead, labels=[f'{lt}分' for lt in lead_times], patch_artist=True)
        for patch in bp['boxes']:
            patch.set_facecolor('lightblue')
        
        ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax2.axhline(y=10, color='red', linestyle='--', linewidth=1, label='±10cm')
        ax2.axhline(y=-10, color='red', linestyle='--', linewidth=1)
        ax2.set_xlabel('予測リードタイム')
        ax2.set_ylabel('予測誤差 (cm)')
        ax2.set_title('リードタイム別の予測誤差分布')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        # 3. 散布図（予測 vs 実測）
        ax3 = axes[1, 0]
        
        # 全データ
        ax3.scatter(df["actual"], df["predicted"], alpha=0.5, s=20, c=df["lead_time"], cmap='viridis')
        
        # 理想線
        min_val = min(df["actual"].min(), df["predicted"].min())
        max_val = max(df["actual"].max(), df["predicted"].max())
        ax3.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=2, label='理想線')
        ax3.plot([min_val, max_val], [min_val - 0.1, max_val - 0.1], 'r--', linewidth=1, alpha=0.5)
        ax3.plot([min_val, max_val], [min_val + 0.1, max_val + 0.1], 'r--', linewidth=1, alpha=0.5)
        
        ax3.set_xlabel('実測水位 (m)')
        ax3.set_ylabel('予測水位 (m)')
        ax3.set_title('予測値 vs 実測値')
        ax3.grid(True, alpha=0.3)
        
        # カラーバー
        sm = plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(vmin=30, vmax=180))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax3)
        cbar.set_label('リードタイム (分)')
        
        # 4. 精度サマリー
        ax4 = axes[1, 1]
        ax4.axis('tight')
        ax4.axis('off')
        
        # 分析結果を取得
        analysis = self.analyze_results(results)
        
        # サマリーテーブル作成
        summary_data = []
        for lead_time in [30, 60, 90, 120, 180]:
            key = f"{lead_time}min"
            if key in analysis:
                summary_data.append([
                    f"{lead_time}分",
                    f"{analysis[key]['mae']*100:.1f}cm",
                    f"{analysis[key]['rmse']*100:.1f}cm",
                    f"{analysis[key]['within_10cm']:.1f}%",
                    analysis[key]['count']
                ])
        
        if summary_data:
            table = ax4.table(
                cellText=summary_data,
                colLabels=['リードタイム', 'MAE', 'RMSE', '10cm以内率', 'サンプル数'],
                cellLoc='center',
                loc='center'
            )
            table.auto_set_font_size(False)
            table.set_fontsize(10)
            table.scale(1.2, 1.5)
        
        ax4.set_title('予測精度サマリー', fontsize=12, pad=20)
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"検証結果を保存: {output_path}")
        
        plt.show()
        
        return analysis
    
    def run_full_validation(self, num_periods=10, period_hours=24):
        """
        複数期間での検証を実行
        
        Parameters:
        -----------
        num_periods : int
            検証期間数
        period_hours : int
            各期間の長さ（時間）
        """
        print("\n=== 予測モデル検証開始 ===")
        
        # データ読み込み
        self.load_data()
        
        # ランダムに期間を選択
        max_start = len(self.data) - period_hours * 6
        if max_start <= 0:
            print("データが不足しています")
            return
        
        np.random.seed(42)  # 再現性のため
        start_indices = np.random.choice(max_start, min(num_periods, max_start // (period_hours * 6)), replace=False)
        
        all_predictions = []
        
        for i, start_idx in enumerate(start_indices):
            print(f"\n検証期間 {i+1}/{len(start_indices)}: {self.data.iloc[start_idx]['時刻']}")
            
            # 検証実行
            results = self.validate_period(start_idx, period_hours)
            
            if results and results["predictions"]:
                all_predictions.extend(results["predictions"])
                
                # 最初の期間は詳細を可視化
                if i == 0:
                    output_path = f"validation_result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
                    self.visualize_validation(results, output_path)
        
        # 全体の統計
        if all_predictions:
            print("\n=== 全体統計 ===")
            df_all = pd.DataFrame(all_predictions)
            
            for lead_time in [30, 60, 90, 120, 180]:
                subset = df_all[df_all["lead_time"] == lead_time]
                if len(subset) > 0:
                    mae = np.mean(np.abs(subset["error"]))
                    within_10cm = (np.abs(subset["error"]) < 0.10).sum() / len(subset) * 100
                    print(f"\n{lead_time}分先予測:")
                    print(f"  MAE: {mae*100:.1f}cm")
                    print(f"  10cm以内率: {within_10cm:.1f}%")
                    print(f"  サンプル数: {len(subset)}")
            
            # 結果を保存
            output_file = f"tools/results/validation_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump({
                    "total_predictions": len(all_predictions),
                    "summary": self.analyze_results({"predictions": all_predictions})
                }, f, indent=2, ensure_ascii=False)
            print(f"\n検証結果サマリーを保存: {output_file}")


def main():
    """メイン処理"""
    # 検証実行
    validator = PredictionValidator(
        data_path="統合データ_水位ダム_20250730_030746.csv",
        config_path="core/configs/prediction_config.json"
    )
    
    # 10期間、各24時間で検証
    validator.run_full_validation(num_periods=10, period_hours=24)


if __name__ == "__main__":
    main()