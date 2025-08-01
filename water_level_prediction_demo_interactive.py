#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
水位予測モデル インタラクティブデモ
実計測データを使用して、任意の時刻での予測をテスト
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import pickle
import os
from water_level_prediction_model_v2 import WaterLevelPredictorV2
# from discharge_prediction_model import DischargePredictionModel  # v1は廃止
try:
    from discharge_prediction_model_v2 import DischargePredictionModelV2
except ImportError:
    DischargePredictionModelV2 = None

# 日本語フォント設定
plt.rcParams['font.sans-serif'] = ['Yu Gothic', 'Hiragino Sans', 'Meirio', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class InteractivePredictionDemo:
    """インタラクティブ予測デモ"""
    
    def __init__(self, data_file='統合データ_水位ダム_20250730_142903.csv'):
        """初期化"""
        self.data_file = data_file
        self.predictor = WaterLevelPredictorV2()
        self.discharge_predictor = None  # 後でv2モデルを設定
        self.data = None
        self.interesting_events = []
        
        # モデル読み込み（最新のモデルを探す）
        self.load_latest_model()
        
        # 放流量予測モデルの読み込み
        self.load_discharge_model()
        
        # データ読み込み
        self.load_data()
        
        # 興味深いイベントの検出
        self.find_interesting_events()
    
    def load_latest_model(self):
        """最新の訓練済みモデルを読み込み"""
        model_files = [f for f in os.listdir('.') if f.startswith('water_level_predictor_v2_') and f.endswith('.pkl')]
        
        if model_files:
            latest_model = sorted(model_files)[-1]
            self.predictor.load_model(latest_model)
            print(f"モデルを読み込みました: {latest_model}")
        else:
            print("訓練済みモデルが見つかりません。デフォルトパラメータを使用します。")
    
    def load_discharge_model(self):
        """放流量予測モデルを読み込み"""
        # 最新の放流量予測モデルを探す（v2を優先）
        discharge_model_v2_files = [f for f in os.listdir('.') if f.startswith('discharge_prediction_model_v2_') and f.endswith('.pkl')]
        discharge_model_v1_files = [f for f in os.listdir('.') if f.startswith('discharge_prediction_model_') and not f.startswith('discharge_prediction_model_v2_') and f.endswith('.pkl')]
        
        if discharge_model_v2_files and DischargePredictionModelV2:
            # v2モデルを優先して使用（作成時刻でソート）
            discharge_model_v2_files.sort(key=lambda x: os.path.getmtime(x))
            latest_model = discharge_model_v2_files[-1]
            self.discharge_predictor = DischargePredictionModelV2()
            self.discharge_predictor.load_model(latest_model)
            print(f"放流量予測モデルv2を読み込みました: {latest_model}")
            self.use_v2_model = True
        elif discharge_model_v1_files:
            # v1モデルを使用
            latest_model = sorted(discharge_model_v1_files)[-1]
            self.discharge_predictor.load_model(latest_model)
            print(f"放流量予測モデルを読み込みました: {latest_model}")
            self.use_v2_model = False
        else:
            print("放流量予測モデルが見つかりません。")
            self.use_v2_model = False
    
    def load_data(self):
        """データ読み込み"""
        print(f"\nデータを読み込み中: {self.data_file}")
        
        self.full_data = pd.read_csv(self.data_file, encoding='utf-8')
        self.full_data['時刻'] = pd.to_datetime(self.full_data['時刻'])
        
        # 水位予測用のデータ（互換性維持）
        self.data = self.full_data[['時刻', '水位_水位', 'ダム_全放流量']].copy()
        self.data.columns = ['時刻', '水位', '放流量']
        
        # フィルタリング（水位≥3m、放流量≥150m³/s）
        self.filtered_data = self.data[(self.data['水位'] >= 3.0) & 
                                       (self.data['放流量'] >= 150.0)].copy()
        
        print(f"データ期間: {self.data['時刻'].min()} ～ {self.data['時刻'].max()}")
        print(f"フィルタ後のデータ数: {len(self.filtered_data)}件")
    
    def find_interesting_events(self):
        """興味深いイベントを自動検出"""
        print("\n興味深いイベントを検出中...")
        
        # 1. 放流量の急増イベント
        discharge_diff = self.filtered_data['放流量'].diff()
        rapid_increase = self.filtered_data[discharge_diff > 100].copy()
        
        if len(rapid_increase) > 0:
            # 最大の増加イベントを選択
            max_increase_idx = discharge_diff.idxmax()
            self.interesting_events.append({
                'name': '放流量急増',
                'time': self.filtered_data.loc[max_increase_idx, '時刻'],
                'description': f"放流量が{discharge_diff[max_increase_idx]:.0f} m³/s増加"
            })
        
        # 2. 高水位イベント
        high_water = self.filtered_data[self.filtered_data['水位'] >= 5.0]
        if len(high_water) > 0:
            # 最高水位の時刻
            max_water_idx = self.filtered_data['水位'].idxmax()
            self.interesting_events.append({
                'name': '高水位',
                'time': self.filtered_data.loc[max_water_idx, '時刻'],
                'description': f"水位 {self.filtered_data.loc[max_water_idx, '水位']:.2f}m"
            })
        
        # 3. 放流量ピーク
        if len(self.filtered_data) > 0:
            max_discharge_idx = self.filtered_data['放流量'].idxmax()
            self.interesting_events.append({
                'name': '放流量ピーク',
                'time': self.filtered_data.loc[max_discharge_idx, '時刻'],
                'description': f"放流量 {self.filtered_data.loc[max_discharge_idx, '放流量']:.0f} m³/s"
            })
        
        # 4. 典型的な状況（中間的な水位・放流量）
        median_water = self.filtered_data['水位'].median()
        typical_data = self.filtered_data[
            (self.filtered_data['水位'] > median_water - 0.5) & 
            (self.filtered_data['水位'] < median_water + 0.5)
        ]
        
        if len(typical_data) > 0:
            typical_idx = typical_data.index[len(typical_data)//2]
            self.interesting_events.append({
                'name': '典型的状況',
                'time': self.filtered_data.loc[typical_idx, '時刻'],
                'description': f"水位 {self.filtered_data.loc[typical_idx, '水位']:.2f}m, " +
                             f"放流量 {self.filtered_data.loc[typical_idx, '放流量']:.0f} m³/s"
            })
        
        print(f"{len(self.interesting_events)}個のイベントを検出しました")
    
    def predict_water_level_with_discharge(self, current_time, current_water_level, 
                                          discharge_history_with_pred, prediction_hours=3):
        """予測された放流量を使用した水位予測"""
        # WaterLevelPredictorV2のpredict_water_levelメソッドを修正して使用
        # 放流量が予測値を含むdischarge_history_with_predを使用
        
        n_steps = int(prediction_hours * 6)
        predictions = []
        predicted_water_level = current_water_level
        
        for step in range(n_steps):
            pred_time = current_time + timedelta(minutes=(step + 1) * 10)
            water_level_change = 0
            
            # 遅延を考慮した放流量変化の影響を積算
            for delay_direction in ['increase', 'decrease']:
                delay_minutes = self.predictor.delay_params[delay_direction]
                influence_time = pred_time - timedelta(minutes=delay_minutes)
                
                if influence_time >= discharge_history_with_pred['時刻'].min():
                    time_before = influence_time - timedelta(minutes=30)
                    time_after = influence_time + timedelta(minutes=30)
                    
                    # 最も近い時刻のデータを取得
                    idx_before = discharge_history_with_pred['時刻'].searchsorted(time_before)
                    idx_after = discharge_history_with_pred['時刻'].searchsorted(time_after)
                    
                    if idx_before < len(discharge_history_with_pred) and idx_after < len(discharge_history_with_pred):
                        q_before = discharge_history_with_pred.iloc[idx_before]['放流量']
                        q_after = discharge_history_with_pred.iloc[idx_after]['放流量']
                        
                        dQ_dt = (q_after - q_before) / 60
                        
                        if dQ_dt > 0.5 and delay_direction == 'increase':
                            response_rate = self.predictor.get_response_rate(predicted_water_level, 'increase')
                            water_level_change += dQ_dt * response_rate * 10
                        elif dQ_dt < -0.5 and delay_direction == 'decrease':
                            response_rate = self.predictor.get_response_rate(predicted_water_level, 'decrease')
                            water_level_change += dQ_dt * response_rate * 10
            
            predicted_water_level += water_level_change
            
            # 予測区間の計算
            uncertainty = self.predictor.error_stats['rmse'] * np.sqrt(step + 1) / np.sqrt(n_steps)
            lower_bound = predicted_water_level - self.predictor.error_stats['confidence_factor'] * uncertainty
            upper_bound = predicted_water_level + self.predictor.error_stats['confidence_factor'] * uncertainty
            
            predictions.append({
                '時刻': pred_time,
                '予測水位': predicted_water_level,
                '予測下限': lower_bound,
                '予測上限': upper_bound,
                '予測ステップ': step + 1
            })
        
        return pd.DataFrame(predictions)
    
    def run_prediction(self, current_time_str=None, show_actual=True, use_discharge_model=True, rainfall_forecast_data=None):
        """予測を実行"""
        if current_time_str:
            # 文字列から時刻を解析
            try:
                current_time = pd.to_datetime(current_time_str)
            except Exception as e:
                print(f"時刻の解析に失敗しました: {current_time_str}")
                print(f"エラー: {e}")
                return None
        else:
            # デフォルトは最新データの1日前
            current_time = self.filtered_data['時刻'].max() - timedelta(days=1)
        
        # 現在時刻のデータを取得
        current_idx = self.data['時刻'].searchsorted(current_time)
        if current_idx >= len(self.data) or current_idx < 18:
            print(f"指定された時刻 {current_time} のデータが不十分です")
            return None
        
        current_data = self.data.iloc[current_idx]
        current_water_level = current_data['水位']
        
        # 放流量履歴（過去3時間）
        history_start = max(0, current_idx - 18)
        discharge_history = self.data.iloc[history_start:current_idx + 1][['時刻', '放流量']].copy()
        
        print(f"\n=== 予測実行 ===")
        print(f"現在時刻: {current_time}")
        print(f"現在の水位: {current_water_level:.2f} m")
        print(f"現在の放流量: {current_data['放流量']:.0f} m³/s")
        
        # 放流量予測（機械学習モデルを使用）
        predicted_discharge = None
        if use_discharge_model and hasattr(self, 'discharge_predictor') and (hasattr(self.discharge_predictor, 'models') or hasattr(self, 'use_v2_model')):
            try:
                # 放流量予測のための現在時刻のインデックスを取得
                full_idx = self.full_data['時刻'].searchsorted(current_time)
                
                # 降雨予測データの作成（1時間先まで利用可能）
                current_rainfall = self.full_data.iloc[full_idx]['ダム_60分雨量'] if 'ダム_60分雨量' in self.full_data.columns else 0
                
                # 降雨予測データの準備
                rainfall_forecast = None
                if rainfall_forecast_data is not None:
                    # 外部から降雨予測が提供された場合
                    rainfall_forecast = rainfall_forecast_data
                    print(f"降雨予測: 外部データを使用（{len(rainfall_forecast_data)}件）")
                else:
                    # デフォルト: 1時間先までの実際のデータを天気予報データとして使用
                    future_1h_mask = (self.full_data['時刻'] > current_time) & \
                                   (self.full_data['時刻'] <= current_time + pd.Timedelta(hours=1))
                    future_1h_data = self.full_data[future_1h_mask]
                    
                    if len(future_1h_data) > 0:
                        rainfall_forecast = pd.DataFrame({
                            '時刻': future_1h_data['時刻'],
                            '降雨強度': future_1h_data['ダム_60分雨量']
                        })
                        print(f"降雨予測: 1時間先までの実データを使用（天気予報APIを想定）")
                    else:
                        # データがない場合は現在値継続
                        forecast_times = pd.date_range(start=current_time + pd.Timedelta(minutes=10), 
                                                     end=current_time + pd.Timedelta(hours=1), 
                                                     freq='10min')
                        rainfall_forecast = pd.DataFrame({
                            '時刻': forecast_times,
                            '降雨強度': [current_rainfall] * len(forecast_times)
                        })
                        print(f"降雨予測: データ不足のため現在値継続（{current_rainfall:.1f}mm/h）")
                
                # 放流量予測を実行
                predicted_discharge = self.discharge_predictor.predict(
                    current_time, 
                    self.full_data.iloc[:full_idx+1],
                    prediction_hours=3,
                    rainfall_forecast=rainfall_forecast
                )
                
                print(f"放流量予測: 機械学習モデルを使用")
                
                # 予測された放流量を履歴に追加
                discharge_history_with_pred = discharge_history.copy()
                for _, pred in predicted_discharge.iterrows():
                    discharge_history_with_pred = pd.concat([
                        discharge_history_with_pred,
                        pd.DataFrame({
                            '時刻': [pred['時刻']],
                            '放流量': [pred['予測放流量']]
                        })
                    ], ignore_index=True)
                
                # 水位予測を実行（予測された放流量を使用）
                predictions = self.predict_water_level_with_discharge(
                    current_time, current_water_level, discharge_history_with_pred,
                    prediction_hours=3
                )
                
            except Exception as e:
                print(f"放流量予測でエラーが発生しました: {e}")
                print("現在値継続の仮定で予測を続行します")
                predicted_discharge = None
                # 従来の方法（放流量現在値継続）で予測
                predictions = self.predictor.predict_water_level(
                    current_time, current_water_level, discharge_history,
                    prediction_hours=3
                )
        else:
            # 従来の方法（放流量現在値継続）で予測
            print(f"放流量予測: 現在値継続の仮定")
            predictions = self.predictor.predict_water_level(
                current_time, current_water_level, discharge_history,
                prediction_hours=3
            )
        
        # 可視化（rainfall_forecastを渡す）
        self.visualize_prediction(current_time, current_water_level, 
                                discharge_history, predictions, 
                                predicted_discharge=predicted_discharge,
                                show_actual=show_actual,
                                rainfall_forecast=rainfall_forecast if 'rainfall_forecast' in locals() else None)
        
        # 予測精度の評価（実際のデータがある場合）
        if show_actual:
            self.evaluate_prediction(predictions)
        
        return predictions
    
    def visualize_prediction(self, current_time, current_water_level, 
                           discharge_history, predictions, predicted_discharge=None,
                           show_actual=True, rainfall_forecast=None):
        """予測結果の可視化"""
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(14, 14), sharex=True)
        
        # 表示期間
        display_start = current_time - timedelta(hours=6)
        display_end = predictions['時刻'].max()
        
        # 1. 水位グラフ
        # 過去の実データ
        past_mask = (self.data['時刻'] >= display_start) & (self.data['時刻'] <= current_time)
        ax1.plot(self.data.loc[past_mask, '時刻'], 
                self.data.loc[past_mask, '水位'],
                'ko-', linewidth=2, markersize=4, label='過去の水位')
        
        # 現在位置
        ax1.plot(current_time, current_water_level, 'ro', markersize=12, 
                label=f'現在 ({current_water_level:.2f}m)', zorder=5)
        
        # 予測
        ax1.plot(predictions['時刻'], predictions['予測水位'], 
                'b-', linewidth=3, label='予測水位')
        ax1.fill_between(predictions['時刻'], 
                        predictions['予測下限'], 
                        predictions['予測上限'],
                        alpha=0.3, color='blue', label='95%予測区間')
        
        # 実際の水位（あれば）
        if show_actual:
            future_mask = (self.data['時刻'] > current_time) & (self.data['時刻'] <= display_end)
            if future_mask.sum() > 0:
                ax1.plot(self.data.loc[future_mask, '時刻'], 
                        self.data.loc[future_mask, '水位'],
                        'g--', linewidth=2, label='実際の水位', alpha=0.7)
        
        # 危険水位
        ax1.axhline(y=5.5, color='red', linestyle='--', alpha=0.5, label='氾濫危険水位')
        
        # 現在時刻ライン
        ax1.axvline(x=current_time, color='gray', linestyle=':', alpha=0.5)
        
        ax1.set_ylabel('水位 (m)', fontsize=12)
        ax1.set_title(f'水位予測 - {current_time.strftime("%Y年%m月%d日 %H:%M")}時点', fontsize=14)
        ax1.legend(loc='best')
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(bottom=2.5)
        
        # 2. 放流量グラフ
        # 過去の放流量
        ax2.plot(self.data.loc[past_mask, '時刻'], 
                self.data.loc[past_mask, '放流量'],
                'go-', linewidth=2, markersize=4, label='過去の放流量')
        
        # 現在の放流量
        current_discharge = discharge_history.iloc[-1]['放流量']
        ax2.plot(current_time, current_discharge, 'ro', markersize=12, 
                label=f'現在 ({current_discharge:.0f} m³/s)', zorder=5)
        
        # 将来の放流量
        if predicted_discharge is not None:
            # 機械学習モデルの予測を表示
            ax2.plot(predicted_discharge['時刻'], predicted_discharge['予測放流量'],
                    'b-', linewidth=2, label='予測放流量（MLモデル）')
        else:
            # 現在値継続の仮定
            future_times = predictions['時刻']
            ax2.plot(future_times, [current_discharge] * len(future_times),
                    'g--', linewidth=2, alpha=0.5, label='放流量（現在値継続と仮定）')
        
        # 実際の放流量（あれば）
        if show_actual:
            if future_mask.sum() > 0:
                ax2.plot(self.data.loc[future_mask, '時刻'], 
                        self.data.loc[future_mask, '放流量'],
                        'k:', linewidth=2, label='実際の放流量', alpha=0.7)
        
        ax2.axvline(x=current_time, color='gray', linestyle=':', alpha=0.5)
        
        ax2.set_ylabel('放流量 (m³/s)', fontsize=12)
        ax2.legend(loc='best')
        ax2.grid(True, alpha=0.3)
        
        # 3. 降雨強度グラフ
        # 降雨強度（棒グラフ）
        if 'ダム_60分雨量' in self.full_data.columns:
            past_mask_full = (self.full_data['時刻'] >= display_start) & (self.full_data['時刻'] <= current_time)
            ax3.bar(self.full_data.loc[past_mask_full, '時刻'], 
                   self.full_data.loc[past_mask_full, 'ダム_60分雨量'],
                   width=0.007, color='blue', alpha=0.7, label='過去の降雨強度')
            
            # 現在の降雨強度
            current_idx_full = self.full_data['時刻'].searchsorted(current_time)
            if current_idx_full < len(self.full_data):
                current_rainfall = self.full_data.iloc[current_idx_full]['ダム_60分雨量']
                ax3.bar(current_time, current_rainfall, 
                       width=0.007, color='red', alpha=0.9, 
                       label=f'現在 ({current_rainfall:.1f} mm/h)')
            
            # 将来の降雨予測（1時間先まで）
            if rainfall_forecast is not None and len(rainfall_forecast) > 0:
                # 降雨予測を表示
                forecast_mask = (rainfall_forecast['時刻'] > current_time) & (rainfall_forecast['時刻'] <= display_end)
                ax3.bar(rainfall_forecast.loc[forecast_mask, '時刻'], 
                       rainfall_forecast.loc[forecast_mask, '降雨強度'],
                       width=0.007, color='orange', alpha=0.7, 
                       label='降雨予測（天気予報API想定）')
            
            # 実際の降雨強度（あれば）
            if show_actual:
                future_mask_full = (self.full_data['時刻'] > current_time) & (self.full_data['時刻'] <= display_end)
                if future_mask_full.sum() > 0:
                    ax3.bar(self.full_data.loc[future_mask_full, '時刻'], 
                           self.full_data.loc[future_mask_full, 'ダム_60分雨量'],
                           width=0.007, color='green', alpha=0.5, label='実際の降雨強度')
        
        ax3.axvline(x=current_time, color='gray', linestyle=':', alpha=0.5)
        ax3.set_ylabel('降雨強度 (mm/h)', fontsize=12)
        ax3.legend(loc='best')
        ax3.grid(True, alpha=0.3)
        ax3.set_ylim(bottom=0)
        
        # 4. 貯水位グラフ
        if 'ダム_貯水位' in self.full_data.columns:
            # 過去の貯水位
            past_mask_full = (self.full_data['時刻'] >= display_start) & (self.full_data['時刻'] <= current_time)
            ax4.plot(self.full_data.loc[past_mask_full, '時刻'], 
                    self.full_data.loc[past_mask_full, 'ダム_貯水位'],
                    'g-', linewidth=2, label='過去の貯水位')
            
            # 現在の貯水位
            current_idx_full = self.full_data['時刻'].searchsorted(current_time)
            if current_idx_full < len(self.full_data):
                current_reservoir_level = self.full_data.iloc[current_idx_full]['ダム_貯水位']
                ax4.plot(current_time, current_reservoir_level, 'ro', 
                        markersize=10, label=f'現在 ({current_reservoir_level:.2f}m)', zorder=5)
            
            # 洪水貯留準備水位と洪水時最高水位のライン
            ax4.axhline(y=38.0, color='orange', linestyle='--', alpha=0.7, 
                       label='洪水貯留準備水位 (38.0m)')
            ax4.axhline(y=39.2, color='red', linestyle='--', alpha=0.7, 
                       label='洪水時最高水位 (39.2m)')
            
            # 実際の貯水位（あれば）
            if show_actual:
                if future_mask_full.sum() > 0:
                    ax4.plot(self.full_data.loc[future_mask_full, '時刻'], 
                            self.full_data.loc[future_mask_full, 'ダム_貯水位'],
                            'g--', linewidth=2, alpha=0.7, label='実際の貯水位')
        
        ax4.axvline(x=current_time, color='gray', linestyle=':', alpha=0.5)
        ax4.set_xlabel('時刻', fontsize=12)
        ax4.set_ylabel('貯水位 (m)', fontsize=12)
        ax4.legend(loc='best')
        ax4.grid(True, alpha=0.3)
        ax4.set_ylim(bottom=30)  # 貯水位の最小値を設定
        
        # x軸のフォーマット
        import matplotlib.dates as mdates
        ax4.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d %H:%M'))
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        
        # 保存
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"prediction_demo_{timestamp}.png"
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"\n予測グラフを保存: {filename}")
        
        plt.show()
    
    def evaluate_prediction(self, predictions):
        """予測精度の評価（実データとの比較）"""
        print("\n=== 予測精度評価 ===")
        
        errors = []
        for _, pred in predictions.iterrows():
            # 実際の値を取得
            actual_idx = self.data['時刻'].searchsorted(pred['時刻'])
            if actual_idx < len(self.data):
                actual_water = self.data.iloc[actual_idx]['水位']
                error = pred['予測水位'] - actual_water
                
                hours = pred['予測ステップ'] / 6
                errors.append({
                    '予測時間': hours,
                    '予測値': pred['予測水位'],
                    '実際値': actual_water,
                    '誤差': error,
                    '絶対誤差': abs(error),
                    '区間内': pred['予測下限'] <= actual_water <= pred['予測上限']
                })
        
        if errors:
            error_df = pd.DataFrame(errors)
            
            # 時間別の精度
            for hours in [1, 2, 3]:
                hour_data = error_df[error_df['予測時間'] == hours]
                if len(hour_data) > 0:
                    mae = hour_data['絶対誤差'].mean()
                    rmse = np.sqrt((hour_data['誤差'] ** 2).mean())
                    coverage = hour_data['区間内'].mean() * 100
                    
                    print(f"{hours}時間先: MAE={mae:.3f}m, RMSE={rmse:.3f}m, " +
                          f"区間カバー率={coverage:.1f}%")
    
    def show_event_menu(self):
        """イベント選択メニューを表示"""
        print("\n=== 予測したい状況を選択してください ===")
        print("0: カスタム時刻を入力")
        
        for i, event in enumerate(self.interesting_events, 1):
            print(f"{i}: {event['name']} - {event['time'].strftime('%Y-%m-%d %H:%M')} " +
                  f"({event['description']})")
        
        print(f"{len(self.interesting_events) + 1}: 終了")
        
        return len(self.interesting_events) + 1
    
    def run_interactive(self):
        """インタラクティブモード実行"""
        print("\n" + "="*60)
        print("水位予測モデル インタラクティブデモ")
        print("="*60)
        
        while True:
            max_option = self.show_event_menu()
            
            try:
                choice = int(input("\n選択番号: "))
                
                if choice == max_option:
                    print("デモを終了します")
                    break
                elif choice == 0:
                    # カスタム時刻
                    time_str = input("予測開始時刻を入力 (例: 2019-07-03 12:00): ")
                    self.run_prediction(time_str, show_actual=True)
                elif 1 <= choice < max_option:
                    # 事前定義イベント
                    event = self.interesting_events[choice - 1]
                    print(f"\n選択: {event['name']}")
                    self.run_prediction(event['time'].strftime('%Y-%m-%d %H:%M:%S'), 
                                      show_actual=True)
                else:
                    print("無効な選択です")
                
            except ValueError:
                print("数値を入力してください")
            except Exception as e:
                print(f"エラーが発生しました: {e}")
            
            input("\nEnterキーを押して続行...")

def main():
    """メイン処理"""
    # データファイルの確認
    data_file = '統合データ_水位ダム_20250730_142903.csv'
    
    if not os.path.exists(data_file):
        print(f"データファイル {data_file} が見つかりません")
        print("analyze_water_level_delay.py を実行してデータを選択してください")
        return
    
    # デモ実行
    demo = InteractivePredictionDemo(data_file)
    
    # 自動デモモード（引数で指定可能）
    import sys
    if len(sys.argv) > 1:
        # コマンドライン引数で時刻指定
        demo.run_prediction(sys.argv[1], show_actual=True)
    else:
        # インタラクティブモード
        demo.run_interactive()

if __name__ == "__main__":
    main()