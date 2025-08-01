#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
リアルタイム水位予測デモンストレーション（実データ使用版）
実際のCSVデータを使用して予測の動作を可視化
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from datetime import datetime, timedelta
from realtime_prediction_model import RealtimePredictionModel
import os
import warnings
warnings.filterwarnings('ignore')

# 日本語フォント設定
plt.rcParams['font.sans-serif'] = ['Yu Gothic', 'Hiragino Sans', 'Meirio', 'DejaVu Sans']
plt.rcParams['font.size'] = 10
plt.rcParams['axes.unicode_minus'] = False

class PredictionDemoWithRealData:
    def __init__(self, data_path="統合データ_水位ダム_20250730_142903.csv", config_path="core/configs/learned_config.json"):
        """
        初期化
        
        Parameters:
        -----------
        data_path : str
            使用するデータファイルのパス
        config_path : str
            モデル設定ファイルのパス
        """
        self.data_path = data_path
        self.model = RealtimePredictionModel(config_path)
        self.current_time = datetime.now()
        
        # データ読み込み
        self.load_data()
        
        # 結果保存用
        self.history_time = []
        self.history_discharge = []
        self.history_water_level = []
        self.predictions_history = []
    
    def load_data(self):
        """実データの読み込み"""
        print(f"データファイルを読み込み中: {self.data_path}")
        
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"データファイルが見つかりません: {self.data_path}")
        
        # データ読み込み
        self.df = pd.read_csv(self.data_path, encoding='utf-8')
        self.df['時刻'] = pd.to_datetime(self.df['時刻'])
        
        # 必要なカラムの確認
        required_cols = ['時刻', '水位_水位', 'ダム_全放流量']
        for col in required_cols:
            if col not in self.df.columns:
                raise ValueError(f"必要なカラム '{col}' が見つかりません")
        
        # 放流量150m³/s以上のデータのみ使用
        self.df_filtered = self.df[self.df['ダム_全放流量'] >= 150.0].reset_index(drop=True)
        
        print(f"読み込み完了: 全{len(self.df)}行中、{len(self.df_filtered)}行が対象")
        print(f"期間: {self.df_filtered['時刻'].min()} ～ {self.df_filtered['時刻'].max()}")
    
    def find_demo_periods(self):
        """デモに適した期間を自動検出"""
        periods = []
        
        # 変動が大きい期間を探す
        window_size = 36  # 6時間
        
        for i in range(0, len(self.df_filtered) - window_size, 18):  # 3時間ごと
            window = self.df_filtered.iloc[i:i+window_size]
            
            # 変動係数を計算
            cv_discharge = window['ダム_全放流量'].std() / window['ダム_全放流量'].mean()
            
            # 変動が大きい期間を選択
            if cv_discharge > 0.1:  # 10%以上の変動
                max_discharge = window['ダム_全放流量'].max()
                min_discharge = window['ダム_全放流量'].min()
                
                periods.append({
                    'start_idx': i,
                    'start_time': window.iloc[0]['時刻'],
                    'cv': cv_discharge,
                    'discharge_range': (min_discharge, max_discharge),
                    'type': 'variable'
                })
            elif cv_discharge < 0.05:  # 安定期間
                periods.append({
                    'start_idx': i,
                    'start_time': window.iloc[0]['時刻'],
                    'cv': cv_discharge,
                    'discharge_range': (window['ダム_全放流量'].min(), window['ダム_全放流量'].max()),
                    'type': 'stable'
                })
        
        # 変動期間と安定期間を1つずつ選択
        variable_periods = [p for p in periods if p['type'] == 'variable']
        stable_periods = [p for p in periods if p['type'] == 'stable']
        
        selected = []
        if variable_periods:
            selected.append(max(variable_periods, key=lambda x: x['cv']))
        if stable_periods:
            selected.append(min(stable_periods, key=lambda x: x['cv']))
        
        return selected
    
    def find_time_index(self, target_time=None, target_index=None, event_type=None):
        """
        指定された条件に最も近い時刻のインデックスを返す
        
        Parameters:
        -----------
        target_time : str or datetime
            目標時刻
        target_index : int
            直接指定するインデックス
        event_type : str
            イベントタイプ ('max_discharge', 'min_discharge', 'rapid_change')
        
        Returns:
        --------
        int : 該当するインデックス
        """
        if target_time:
            # 文字列をdatetimeに変換
            if isinstance(target_time, str):
                target_time = pd.to_datetime(target_time)
            
            # 最も近い時刻を探す
            time_diff = abs(self.df_filtered['時刻'] - target_time)
            closest_idx = time_diff.idxmin()
            
            # 完全一致しない場合は警告
            if time_diff.iloc[closest_idx] > timedelta(minutes=10):
                actual_time = self.df_filtered.iloc[closest_idx]['時刻']
                print(f"警告: 指定時刻 {target_time} に最も近いデータは {actual_time} です")
            
            return closest_idx
        
        elif target_index is not None:
            # 直接インデックスを使用
            if 0 <= target_index < len(self.df_filtered):
                return target_index
            else:
                raise ValueError(f"インデックス {target_index} は範囲外です (0-{len(self.df_filtered)-1})")
        
        elif event_type:
            # イベントタイプに基づいて選択
            if event_type == 'max_discharge':
                # 最大放流量の時点
                return self.df_filtered['ダム_全放流量'].idxmax()
            elif event_type == 'min_discharge':
                # 最小放流量の時点（ただし150以上）
                return self.df_filtered['ダム_全放流量'].idxmin()
            elif event_type == 'rapid_change':
                # 変化率が最大の時点を探す
                discharge_change = self.df_filtered['ダム_全放流量'].diff().abs()
                # 最初と最後の数点は除外
                discharge_change.iloc[:6] = 0
                discharge_change.iloc[-6:] = 0
                return discharge_change.idxmax()
            else:
                raise ValueError(f"不明なイベントタイプ: {event_type}")
        
        else:
            # デフォルト: データの中央付近を使用
            return len(self.df_filtered) // 2
    
    def run_static_demo_with_real_data(self, demo_period_idx=None, period_info=None):
        """実データを使用した静的デモ"""
        print("=== リアルタイム水位予測デモ（実データ使用） ===\n")
        
        # デモ期間の選択
        if demo_period_idx is None:
            # 適切な期間を自動選択
            demo_periods = self.find_demo_periods()
            if not demo_periods:
                print("適切なデモ期間が見つかりません")
                return
            
            # 最初の期間を使用
            demo_period = demo_periods[0]
            demo_point = demo_period['start_idx'] + 18  # 期間の中間点
        else:
            demo_point = demo_period_idx
            demo_period = period_info if period_info else {}
        
        # デモ時点の情報表示
        demo_time = self.df_filtered.iloc[demo_point]['時刻']
        print(f"デモ時点: {demo_time}")
        if demo_period:
            print(f"期間タイプ: {demo_period.get('type', '不明')}")
        
        # その時点までの履歴を設定（2時間分）
        history_size = min(12, demo_point)
        for i in range(demo_point - history_size, demo_point):
            if i >= 0:
                self.model.update_history(
                    self.df_filtered.iloc[i]['ダム_全放流量'],
                    self.df_filtered.iloc[i]['水位_水位']
                )
        
        # 現在の状態
        current_discharge = self.df_filtered.iloc[demo_point]['ダム_全放流量']
        current_water_level = self.df_filtered.iloc[demo_point]['水位_水位']
        
        print(f"現在の放流量: {current_discharge:.1f} m³/s")
        print(f"現在の水位: {current_water_level:.2f} m")
        
        # 将来の放流量（実データから取得）
        future_discharge = []
        for i in range(18):  # 3時間分
            idx = demo_point + i
            if idx < len(self.df_filtered):
                future_discharge.append(self.df_filtered.iloc[idx]['ダム_全放流量'])
            else:
                future_discharge.append(current_discharge)  # データがない場合は現在値を維持
        
        # 予測実行
        predictions = self.model.predict(current_discharge, current_water_level, future_discharge)
        
        if predictions["status"] == "success":
            # 結果表示
            print(self.model.get_prediction_summary(predictions))
            
            # 実際の水位と比較（可能な範囲で）
            print("\n=== 予測精度の検証 ===")
            for lead_time, idx in [(30, 3), (60, 6), (90, 9)]:
                future_idx = demo_point + idx
                if future_idx < len(self.df_filtered):
                    pred_level = predictions["water_levels"][idx - 1]
                    actual_level = self.df_filtered.iloc[future_idx]['水位_水位']
                    error = pred_level - actual_level
                    print(f"{lead_time}分後 - 予測: {pred_level:.2f}m, 実測: {actual_level:.2f}m, 誤差: {error*100:.1f}cm")
            
            # 可視化
            self.visualize_prediction_with_real_data(demo_point, predictions)
        else:
            print(f"予測エラー: {predictions.get('message', 'Unknown error')}")
    
    def visualize_prediction_with_real_data(self, current_idx, predictions):
        """実データでの予測結果の可視化"""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
        
        # 表示範囲の設定（現在時刻の前後6時間）
        start_idx = max(0, current_idx - 36)
        end_idx = min(len(self.df_filtered), current_idx + 36)
        
        # データの抽出
        display_data = self.df_filtered.iloc[start_idx:end_idx]
        current_time = self.df_filtered.iloc[current_idx]['時刻']
        
        # 1. 放流量
        # 過去のデータ（実線）
        past_discharge = display_data[display_data['時刻'] <= current_time]
        ax1.plot(past_discharge['時刻'], past_discharge['ダム_全放流量'], 
                'b-', linewidth=2, label='放流量（実績）')
        
        # 将来のデータ（破線）
        future_discharge_data = display_data[display_data['時刻'] > current_time]
        if len(future_discharge_data) > 0:
            ax1.plot(future_discharge_data['時刻'], future_discharge_data['ダム_全放流量'], 
                    'b--', linewidth=1, alpha=0.5, label='放流量（実測）')
        
        ax1.axvline(x=current_time, color='red', linestyle='--', alpha=0.7, label='予測実行時点')
        
        # 予測期間をハイライト
        pred_end_time = current_time + timedelta(hours=3)
        ax1.axvspan(current_time, pred_end_time, alpha=0.1, color='yellow', label='予測期間')
        
        ax1.set_ylabel('放流量 (m³/s)')
        ax1.set_title(f'リアルタイム水位予測デモ - {current_time.strftime("%Y-%m-%d %H:%M")}')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. 水位
        # 実績
        past_data = display_data[display_data['時刻'] <= current_time]
        ax2.plot(past_data['時刻'], past_data['水位_水位'], 
                'b-', linewidth=2, label='水位（実績）')
        
        # 実測値（予測期間）
        future_data = display_data[display_data['時刻'] > current_time]
        if len(future_data) > 0:
            ax2.plot(future_data['時刻'], future_data['水位_水位'], 
                    'b--', linewidth=1, alpha=0.5, label='水位（実測）')
        
        # 予測値
        pred_times = [current_time + timedelta(minutes=(i+1)*10) for i in range(len(predictions["water_levels"]))]
        pred_levels = [self.df_filtered.iloc[current_idx]['水位_水位']] + predictions["water_levels"]
        pred_times_plot = [current_time] + pred_times
        
        ax2.plot(pred_times_plot, pred_levels, 
                'r-', linewidth=2, label='予測値', marker='o', markersize=4)
        
        # 信頼区間
        upper_bound = [l + 0.1 for l in pred_levels[1:]]
        lower_bound = [l - 0.1 for l in pred_levels[1:]]
        ax2.fill_between(pred_times, lower_bound, upper_bound, 
                       alpha=0.3, color='red', label='±10cm範囲')
        
        # 重要な時点の精度表示
        for lead_time, idx in [(30, 3), (60, 6), (90, 9)]:
            future_idx = current_idx + idx
            if idx <= len(predictions["water_levels"]) and future_idx < len(self.df_filtered):
                pred_level = predictions["water_levels"][idx - 1]
                actual_level = self.df_filtered.iloc[future_idx]['水位_水位']
                error = pred_level - actual_level
                
                # 予測点にマーカー
                ax2.plot(pred_times[idx - 1], pred_level, 'ro', markersize=8)
                
                # 誤差を表示
                ax2.annotate(f'{lead_time}分後\n誤差:{error*100:.1f}cm', 
                           xy=(pred_times[idx - 1], pred_level),
                           xytext=(10, 20), textcoords='offset points',
                           bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7),
                           fontsize=8)
        
        ax2.set_xlabel('時刻')
        ax2.set_ylabel('水位 (m)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 状態表示（遅延時間情報を追加）
        current_discharge_value = self.df_filtered.iloc[current_idx]["ダム_全放流量"]
        discharge_range = self.model.get_discharge_range(current_discharge_value)
        delay_minutes = self.model.config["delay_params"][discharge_range]["base_delay"]
        
        state_text = f'システム状態: {predictions["state"]}\n'
        state_text += f'現在の放流量: {current_discharge_value:.1f} m³/s\n'
        state_text += f'現在の水位: {self.df_filtered.iloc[current_idx]["水位_水位"]:.2f} m\n'
        state_text += f'遅延時間: {delay_minutes}分 ({discharge_range})'
        ax2.text(0.02, 0.98, state_text, transform=ax2.transAxes,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                verticalalignment='top', fontsize=9)
        
        # x軸の表示調整
        fig.autofmt_xdate()
        
        plt.tight_layout()
        
        # 保存
        output_path = f"prediction_demo_real_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"\nデモ結果を保存: {output_path}")
        
        plt.show()
    
    def run_continuous_evaluation(self, start_idx=None, hours=24):
        """連続評価モード（実データで予測精度を評価）"""
        print(f"=== 連続評価モード（{hours}時間） ===\n")
        
        if start_idx is None:
            # ランダムに開始点を選択
            start_idx = np.random.randint(12, len(self.df_filtered) - hours * 6 - 18)
        
        steps = hours * 6  # 10分刻み
        results = []
        
        # 初期履歴の設定
        for i in range(max(0, start_idx - 12), start_idx):
            self.model.update_history(
                self.df_filtered.iloc[i]['ダム_全放流量'],
                self.df_filtered.iloc[i]['水位_水位']
            )
        
        # 評価ループ
        for step in range(0, steps - 18, 6):  # 1時間ごとに予測
            current_idx = start_idx + step
            
            if current_idx + 18 >= len(self.df_filtered):
                break
            
            # 現在の状態
            current_discharge = self.df_filtered.iloc[current_idx]['ダム_全放流量']
            current_water_level = self.df_filtered.iloc[current_idx]['水位_水位']
            current_time = self.df_filtered.iloc[current_idx]['時刻']
            
            # 将来の放流量
            future_discharge = []
            for i in range(18):
                idx = current_idx + i
                if idx < len(self.df_filtered):
                    future_discharge.append(self.df_filtered.iloc[idx]['ダム_全放流量'])
            
            # 予測実行
            predictions = self.model.predict(current_discharge, current_water_level, future_discharge)
            
            if predictions["status"] == "success":
                # 予測精度の計算
                errors = {}
                for lead_time, idx in [(30, 3), (60, 6), (90, 9)]:
                    future_idx = current_idx + idx
                    if idx <= len(predictions["water_levels"]) and future_idx < len(self.df_filtered):
                        pred_level = predictions["water_levels"][idx - 1]
                        actual_level = self.df_filtered.iloc[future_idx]['水位_水位']
                        errors[f'error_{lead_time}min'] = pred_level - actual_level
                
                results.append({
                    'time': current_time,
                    'discharge': current_discharge,
                    'water_level': current_water_level,
                    'state': predictions["state"],
                    **errors
                })
            
            # 履歴更新
            for i in range(min(6, len(self.df_filtered) - current_idx)):
                self.model.update_history(
                    self.df_filtered.iloc[current_idx + i]['ダム_全放流量'],
                    self.df_filtered.iloc[current_idx + i]['水位_水位']
                )
            
            # 進捗表示
            if step % 36 == 0:  # 6時間ごと
                print(f"進捗: {step/6:.0f}時間/{hours}時間")
        
        # 結果の集計と表示
        if results:
            df_results = pd.DataFrame(results)
            
            print("\n=== 評価結果サマリー ===")
            for lead_time in [30, 60, 90]:
                col = f'error_{lead_time}min'
                if col in df_results.columns:
                    mae = df_results[col].abs().mean() * 100
                    rmse = np.sqrt((df_results[col] ** 2).mean()) * 100
                    within_10cm = (df_results[col].abs() < 0.1).mean() * 100
                    
                    print(f"\n{lead_time}分予測:")
                    print(f"  MAE: {mae:.1f}cm")
                    print(f"  RMSE: {rmse:.1f}cm")
                    print(f"  10cm以内率: {within_10cm:.1f}%")
                    print(f"  サンプル数: {len(df_results[col].dropna())}")
            
            # 結果をCSVに保存
            output_file = f"evaluation_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            df_results.to_csv(output_file, index=False, encoding='utf-8')
            print(f"\n評価結果を保存: {output_file}")


def main():
    """メイン処理"""
    import argparse
    
    parser = argparse.ArgumentParser(description='実データを使用した水位予測デモ')
    parser.add_argument('--data', type=str, default='統合データ_水位ダム_20250730_142903.csv',
                       help='使用するデータファイル')
    parser.add_argument('--config', type=str, default='prediction_config.json',
                       help='モデル設定ファイル (prediction_config.json: 分析結果, learned_config.json: 学習結果)')

    parser.add_argument('--mode', type=str, default='demo',
                       choices=['demo', 'evaluate'],
                       help='実行モード: demo（デモ）またはevaluate（評価）')
    parser.add_argument('--hours', type=int, default=24,
                       help='評価モードでの評価時間')
    
    # 時刻指定オプション
    parser.add_argument('--time', type=str, default=None,
                       help='デモ実行時刻 (YYYY-MM-DD HH:MM:SS形式)')
    parser.add_argument('--index', type=int, default=None,
                       help='デモ実行時のデータインデックス')
    parser.add_argument('--event', type=str, default=None,
                       choices=['max_discharge', 'min_discharge', 'rapid_change'],
                       help='特定のイベント時点でデモを実行')
    
    args = parser.parse_args()
    
    # デモ実行
    demo = PredictionDemoWithRealData(args.data, args.config)
    
    if args.mode == 'demo':
        # 時刻指定がある場合
        if args.time or args.index is not None or args.event:
            try:
                # 指定された時刻のインデックスを取得
                target_idx = demo.find_time_index(
                    target_time=args.time,
                    target_index=args.index,
                    event_type=args.event
                )
                
                # その時点の情報を表示
                target_time = demo.df_filtered.iloc[target_idx]['時刻']
                target_discharge = demo.df_filtered.iloc[target_idx]['ダム_全放流量']
                target_level = demo.df_filtered.iloc[target_idx]['水位_水位']
                
                print(f"\n{'='*60}")
                print(f"指定時点でのデモ実行")
                print(f"時刻: {target_time}")
                print(f"放流量: {target_discharge:.1f} m³/s")
                print(f"水位: {target_level:.2f} m")
                print(f"{'='*60}")
                
                # 期間情報を作成
                period_info = {
                    'type': 'user_specified',
                    'start_idx': target_idx,
                    'start_time': target_time
                }
                
                demo.run_static_demo_with_real_data(target_idx, period_info)
                
            except Exception as e:
                print(f"エラー: {e}")
                return
        
        else:
            # 自動選択モード（従来の動作）
            periods = demo.find_demo_periods()
            for i, period in enumerate(periods[:1]):  # 最大1つの期間をデモ
                print(f"\n{'='*60}")
                print(f"デモ {i+1}: {period['type']}期間")
                print(f"{'='*60}")
                demo.run_static_demo_with_real_data(period['start_idx'] + 18, period)
    else:
        # 連続評価モード
        demo.run_continuous_evaluation(hours=args.hours)


if __name__ == "__main__":
    main()