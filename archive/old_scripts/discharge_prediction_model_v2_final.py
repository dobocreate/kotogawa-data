#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
放流量予測モデル v2 (最終修正版)
- 動的遅延時間
- 状態識別機能
- 増加・減少の非対称性を考慮
- 1時間先までの降雨予測制限を正しく実装
- 現実的な状態判定ロジック
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from datetime import datetime, timedelta
import pickle
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# 日本語フォント設定
plt.rcParams['font.sans-serif'] = ['Yu Gothic', 'Hiragino Sans', 'Meirio', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class DischargePredictionModelV2:
    """改良版放流量予測モデル"""
    
    def __init__(self):
        """初期化"""
        # モデルパラメータ（分析結果に基づく）
        self.params = {
            # 遅延時間（分）
            'delay_onset': 0,        # 増加開始時の遅延
            'delay_increase': 120,   # 増加継続時の遅延
            'delay_decrease': 60,    # 減少開始時の遅延
            
            # 降雨強度閾値（mm/h）
            'rain_threshold_low': 10,    # 増加/減少の境界
            'rain_threshold_high': 20,   # 積極的増加の閾値
            
            # 状態判定閾値（m³/s per 30min）
            'state_threshold': 50,
            
            # 変化率（m³/s per 10min）- 分析結果に基づく
            'rate_increase_low': 27.1,    # 低降雨時の増加率
            'rate_decrease_low': 24.3,    # 低降雨時の減少率
            'rate_increase_high': 29.1,   # 中降雨時の増加率
            'rate_decrease_ratio': 0.89,  # 減少率/増加率の比率
        }
        
        # 機械学習モデル（補助的に使用）
        self.ml_models = {}
        self.scalers = {}
        
        # 特徴量名（簡素化）
        self.feature_names = [
            'current_discharge',      # 現在の放流量
            'discharge_state',        # 状態（-1:減少, 0:定常, 1:増加）
            'rainfall_current',       # 現在の降雨強度
            'rainfall_delayed',       # 遅延考慮後の降雨強度
            'rainfall_category',      # 降雨カテゴリ（0:低, 1:中, 2:高）
        ]
        
        # 予測ステップ
        self.prediction_steps = [1, 3, 6, 12, 18]  # 10分, 30分, 1時間, 2時間, 3時間
        
    def identify_discharge_state(self, discharge_history):
        """
        放流量の状態を識別
        
        Returns:
        --------
        state : int
            -1: 減少中, 0: 定常, 1: 増加中
        rate : float
            変化率（m³/s per 10min）
        """
        if len(discharge_history) < 4:  # 30分以上のデータが必要
            return 0, 0.0
        
        # 過去30分の変化量
        change_30min = discharge_history.iloc[-1] - discharge_history.iloc[-4]
        rate = change_30min / 3  # m³/s per 10min
        
        # 状態判定
        if change_30min > self.params['state_threshold']:
            state = 1  # 増加中
        elif change_30min < -self.params['state_threshold']:
            state = -1  # 減少中
        else:
            state = 0  # 定常
            
        return state, rate
    
    def get_dynamic_delay(self, state, rainfall_current, rainfall_history, 
                         effective_rainfall_60min_ago, effective_rainfall_120min_ago):
        """
        動的な遅延時間を取得（改良版）
        
        Parameters:
        -----------
        state : int
            現在の放流量状態
        rainfall_current : float
            現在の降雨強度
        rainfall_history : pd.Series
            降雨強度の履歴
        effective_rainfall_60min_ago : float
            60分前の実効降雨
        effective_rainfall_120min_ago : float
            120分前の実効降雨
        
        Returns:
        --------
        delay_minutes : int
            適用する遅延時間（分）
        """
        # 降雨の増加を検出（過去30分）
        if len(rainfall_history) >= 4:
            rain_change = rainfall_current - rainfall_history.iloc[-4]
            rain_increasing = rain_change > 10  # 10mm/h以上の増加
        else:
            rain_increasing = False
        
        # 状態と降雨に基づく遅延時間の決定
        if rain_increasing and (state <= 0):
            # 降雨が急増した場合は即座に反応
            return self.params['delay_onset']
        elif state == 1:
            # 増加継続中 → 通常の遅延
            return self.params['delay_increase']
        elif state == -1:
            # 減少中
            # 過去の降雨が高い場合は短い遅延、低い場合は長い遅延
            if effective_rainfall_60min_ago >= self.params['rain_threshold_low']:
                return self.params['delay_decrease']
            else:
                return self.params['delay_increase']
        else:
            # 定常状態
            # 過去の降雨状況に応じて遅延を決定
            if effective_rainfall_120min_ago >= self.params['rain_threshold_low']:
                return self.params['delay_increase']
            elif effective_rainfall_60min_ago >= self.params['rain_threshold_low']:
                return self.params['delay_decrease']
            else:
                # 過去も現在も低降雨
                return self.params['delay_decrease']
    
    def predict_discharge_change(self, state, rainfall_intensity, current_discharge, 
                               past_rainfall_avg):
        """
        放流量の変化を予測（改良版）
        
        Parameters:
        -----------
        state : int
            現在の状態
        rainfall_intensity : float
            実効降雨強度
        current_discharge : float
            現在の放流量
        past_rainfall_avg : float
            過去1時間の平均降雨
        
        Returns:
        --------
        change_rate : float
            予測変化率（m³/s per 10min）
        """
        # 降雨カテゴリの判定
        if rainfall_intensity >= self.params['rain_threshold_high']:
            rain_category = 'high'
        elif rainfall_intensity >= self.params['rain_threshold_low']:
            rain_category = 'medium'
        else:
            rain_category = 'low'
        
        # 基本変化率の決定（より現実的なロジック）
        if rain_category == 'low':
            if state == 1 and past_rainfall_avg < self.params['rain_threshold_low']:
                # 増加中だが過去も現在も低降雨 → 減速
                base_rate = -self.params['rate_decrease_low'] * 0.5
            elif state <= 0:
                # 減少中または定常で低降雨 → 減少
                base_rate = -self.params['rate_decrease_low']
            else:
                # 増加中だが低降雨 → 緩やかに減速
                base_rate = self.params['rate_increase_low'] * 0.2
        elif rain_category == 'medium':
            if state >= 0:
                # 増加中または定常で中降雨 → 増加
                base_rate = self.params['rate_increase_low']
            else:
                # 減少中で中降雨 → 減速を緩める
                base_rate = -self.params['rate_decrease_low'] * 0.3
        else:  # high
            # 高降雨では積極的に増加
            base_rate = self.params['rate_increase_high']
        
        # 現在の放流量による調整
        if current_discharge > 1000:
            rate_modifier = 0.7
        elif current_discharge > 800:
            rate_modifier = 0.85
        else:
            rate_modifier = 1.0
        
        return base_rate * rate_modifier
    
    def predict(self, current_time, historical_data, prediction_hours=3, 
                rainfall_forecast=None):
        """
        放流量予測（ルールベース＋機械学習のハイブリッド）
        
        Parameters:
        -----------
        current_time : datetime
            現在時刻
        historical_data : pd.DataFrame
            過去データ（時刻、降雨強度、放流量を含む）
        prediction_hours : float
            予測時間
        rainfall_forecast : pd.DataFrame, optional
            降雨予測データ（1時間先まで）
        
        Returns:
        --------
        predictions : pd.DataFrame
            予測結果
        """
        # データ準備
        df = historical_data.copy()
        df = df.sort_values('時刻')
        
        current_idx = len(df) - 1
        current_discharge = df.iloc[current_idx]['ダム_全放流量']
        current_rainfall = df.iloc[current_idx]['ダム_60分雨量']
        
        # 放流量履歴（過去1時間）
        discharge_history = df.iloc[-7:]['ダム_全放流量']
        rainfall_history = df.iloc[-7:]['ダム_60分雨量'].copy()
        
        # 過去の実効降雨を計算
        if len(df) >= 7:
            effective_rainfall_60min_ago = df.iloc[-7]['ダム_60分雨量']
        else:
            effective_rainfall_60min_ago = 0
            
        if len(df) >= 13:
            effective_rainfall_120min_ago = df.iloc[-13]['ダム_60分雨量']
        else:
            effective_rainfall_120min_ago = 0
        
        # 過去1時間の平均降雨
        past_rainfall_avg = rainfall_history.mean()
        
        # 現在の状態を識別
        current_state, current_rate = self.identify_discharge_state(discharge_history)
        
        # 予測結果の格納
        predictions = []
        
        # 各時間ステップで予測
        predicted_discharge = current_discharge
        predicted_state = current_state
        
        # 動的に更新される降雨履歴（予測用）
        dynamic_rainfall_history = rainfall_history.copy()
        
        for step in range(1, int(prediction_hours * 6) + 1):
            pred_time = current_time + timedelta(minutes=step * 10)
            
            # 降雨予測の取得（1時間先まで利用可能）
            if rainfall_forecast is not None and step <= 6:
                # 予測データから取得
                forecast_idx = rainfall_forecast['時刻'].searchsorted(pred_time)
                if forecast_idx < len(rainfall_forecast):
                    future_rainfall = rainfall_forecast.iloc[forecast_idx]['降雨強度']
                else:
                    future_rainfall = 0
            else:
                # 1時間以降または予測データがない場合は0
                future_rainfall = 0
            
            # 動的遅延時間の取得
            delay = self.get_dynamic_delay(
                predicted_state, 
                current_rainfall, 
                dynamic_rainfall_history,
                effective_rainfall_60min_ago,
                effective_rainfall_120min_ago
            )
            
            # 遅延を考慮した降雨強度の計算
            delay_steps = delay // 10
            
            if delay_steps == 0:
                # 即座に反応
                effective_rainfall = future_rainfall if step <= 6 else 0
            elif delay_steps <= 6:
                # 60分以内の遅延
                if step <= delay_steps:
                    # まだ過去のデータを参照
                    hist_idx = delay_steps - step
                    if hist_idx < len(dynamic_rainfall_history):
                        effective_rainfall = dynamic_rainfall_history.iloc[-(hist_idx + 1)]
                    else:
                        effective_rainfall = 0
                else:
                    # 予測期間のデータを参照
                    future_idx = step - delay_steps
                    if future_idx <= 6:
                        effective_rainfall = future_rainfall
                    else:
                        effective_rainfall = 0
            else:
                # 120分の遅延
                if step <= 12:
                    # まだ過去のデータを参照
                    hist_idx = 12 - step
                    if hist_idx < len(df):
                        effective_rainfall = df.iloc[-(hist_idx + 1)]['ダム_60分雨量']
                    else:
                        effective_rainfall = 0
                else:
                    # 予測期間のデータを参照（ただし1時間以内のみ）
                    effective_rainfall = 0
            
            # 放流量変化の予測
            change_rate = self.predict_discharge_change(
                predicted_state, 
                effective_rainfall, 
                predicted_discharge,
                past_rainfall_avg
            )
            
            # 放流量の更新
            predicted_discharge += change_rate
            predicted_discharge = max(150, predicted_discharge)  # 最小値制限
            
            # 状態の更新
            if change_rate > 5:
                predicted_state = 1
            elif change_rate < -5:
                predicted_state = -1
            else:
                predicted_state = 0
            
            # 結果の保存
            predictions.append({
                '時刻': pred_time,
                '予測放流量': predicted_discharge,
                '使用降雨強度': effective_rainfall,
                '適用遅延': delay,
                '状態': predicted_state,
                '変化率': change_rate
            })
            
            # 履歴の更新（1時間先まで）
            if step <= 6:
                dynamic_rainfall_history = pd.concat([
                    dynamic_rainfall_history[1:],
                    pd.Series([future_rainfall])
                ])
                # 過去の実効降雨も更新
                if step == 6:
                    effective_rainfall_120min_ago = effective_rainfall_60min_ago
                    effective_rainfall_60min_ago = dynamic_rainfall_history.iloc[0]
        
        return pd.DataFrame(predictions)
    
    def train_ml_component(self, training_data):
        """
        機械学習コンポーネントの訓練（オプション）
        ルールベースの予測を補正するために使用
        """
        print("機械学習コンポーネントの訓練（補助用）...")
        # 簡易実装のため省略
        pass
    
    def save_model(self, filepath):
        """モデルの保存"""
        model_data = {
            'params': self.params,
            'ml_models': self.ml_models,
            'scalers': self.scalers,
            'feature_names': self.feature_names
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        print(f"モデルを保存: {filepath}")
    
    def load_model(self, filepath):
        """モデルの読み込み"""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
            
        self.params = model_data['params']
        self.ml_models = model_data['ml_models']
        self.scalers = model_data['scalers']
        self.feature_names = model_data['feature_names']
        
        print(f"モデルを読み込み: {filepath}")

def demonstrate_model():
    """モデルのデモンストレーション"""
    print("=== 改良版放流量予測モデル デモ ===")
    
    # モデル初期化
    model = DischargePredictionModelV2()
    
    # テストデータの作成（2023年7月1日のケース）
    test_data = pd.DataFrame({
        '時刻': pd.date_range('2023-06-30 23:00', '2023-07-01 01:00', freq='10min'),
        'ダム_60分雨量': [1, 1, 1, 29, 40, 54, 54, 50, 45, 40, 35, 30, 25],
        'ダム_全放流量': [390, 385, 384, 398, 450, 550, 700, 838, 900, 950, 1000, 1050, 1078]
    })
    
    # 予測実行
    current_time = pd.to_datetime('2023-07-01 00:00')
    predictions = model.predict(current_time, test_data.iloc[:4], prediction_hours=2)
    
    print("\n予測結果:")
    print(predictions[['時刻', '予測放流量', '使用降雨強度', '適用遅延', '変化率']])
    
    # 可視化
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    
    # 放流量
    ax1.plot(test_data['時刻'], test_data['ダム_全放流量'], 'ko-', label='実績')
    ax1.plot(predictions['時刻'], predictions['予測放流量'], 'r--', label='予測', linewidth=2)
    ax1.axvline(x=current_time, color='gray', linestyle='--', alpha=0.5)
    ax1.set_ylabel('放流量 (m³/s)')
    ax1.set_title('改良版モデルによる放流量予測')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 降雨強度と遅延
    ax2.bar(test_data['時刻'], test_data['ダム_60分雨量'], width=0.007, 
            color='blue', alpha=0.7, label='降雨強度')
    
    # 適用遅延を表示
    ax2_twin = ax2.twinx()
    ax2_twin.plot(predictions['時刻'], predictions['適用遅延'], 
                  'g-', label='適用遅延', linewidth=2)
    ax2_twin.set_ylabel('遅延時間 (分)', color='green')
    ax2_twin.tick_params(axis='y', labelcolor='green')
    
    ax2.axvline(x=current_time, color='gray', linestyle='--', alpha=0.5)
    ax2.set_xlabel('時刻')
    ax2.set_ylabel('降雨強度 (mm/h)', color='blue')
    ax2.tick_params(axis='y', labelcolor='blue')
    ax2.legend(loc='upper left')
    ax2_twin.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f'discharge_model_v2_demo_{timestamp}.png'
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\nグラフを保存: {filename}")
    
    return model

if __name__ == "__main__":
    model = demonstrate_model()
    
    # モデルの保存
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    model.save_model(f'discharge_prediction_model_v2_{timestamp}.pkl')