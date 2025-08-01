#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
放流量予測モデル v2
- 動的遅延時間
- 状態識別機能
- 増加・減少の非対称性を考慮
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
            'state_threshold': 10,
            
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
        
    def identify_discharge_state(self, discharge_history, future_discharge_10min=None):
        """
        放流量の状態を識別（過去10分と10分先予測を考慮）
        
        Parameters:
        -----------
        discharge_history : pd.Series
            放流量の履歴
        future_discharge_10min : float, optional
            10分先の予測放流量
        
        Returns:
        --------
        state : int
            -1: 減少中, 0: 定常, 1: 増加中
        rate : float
            変化率（m³/s per 10min）
        """
        if len(discharge_history) < 2:  # 10分以上のデータが必要
            return 0, 0.0
        
        # 過去10分の変化量
        change_past_10min = discharge_history.iloc[-1] - discharge_history.iloc[-2]
        
        # 10分先予測がある場合は前後20分での変化を考慮
        if future_discharge_10min is not None:
            # 過去10分から10分先までの総変化
            total_change = future_discharge_10min - discharge_history.iloc[-2]
            # 平均変化率
            rate = total_change / 2  # m³/s per 10min
            
            # 状態判定は総変化量で行う（20分での変化）
            threshold_20min = self.params['state_threshold'] * 2 / 3  # 30分→20分への調整
            if total_change > threshold_20min:
                state = 1  # 増加中
            elif total_change < -threshold_20min:
                state = -1  # 減少中
            else:
                state = 0  # 定常
        else:
            # 10分先予測がない場合は過去10分のみで判定
            rate = change_past_10min
            
            # 10分での判定用しきい値（元の30分用の1/3）
            threshold_10min = self.params['state_threshold'] / 3
            if change_past_10min > threshold_10min:
                state = 1  # 増加中
            elif change_past_10min < -threshold_10min:
                state = -1  # 減少中
            else:
                state = 0  # 定常
            
        return state, rate
    
    def get_dynamic_delay(self, state, rainfall_current, rainfall_history, 
                         future_discharge_trend=None, future_rainfall_30min=None):
        """
        動的な遅延時間を取得（10分先の予測を考慮）
        
        Parameters:
        -----------
        state : int
            現在の放流量状態
        rainfall_current : float
            現在の降雨強度
        rainfall_history : pd.Series
            降雨強度の履歴
        future_discharge_trend : float, optional
            10分先の予測放流量変化（m³/s per 10min）
        future_rainfall_30min : float, optional
            10分先の降雨強度（mm/h）※変数名は互換性のため維持
        
        Returns:
        --------
        delay_minutes : int
            適用する遅延時間（分）
        """
        # 降雨の増加を検出（過去10分）
        if len(rainfall_history) >= 2:
            rain_change = rainfall_current - rainfall_history.iloc[-2]
            rain_increasing = rain_change > 5  # 10分で5mm/h以上の増加（30分で10mm/hの比率）
        else:
            rain_increasing = False
        
        # 将来の降雨傾向を考慮
        if future_rainfall_30min is not None:
            future_rain_high = future_rainfall_30min > self.params['rain_threshold_high']
            future_rain_increasing = future_rainfall_30min > rainfall_current + 5
        else:
            future_rain_high = False
            future_rain_increasing = False
        
        # 将来の放流量傾向を考慮（10分での判定用しきい値）
        if future_discharge_trend is not None:
            threshold_10min = self.params['state_threshold'] / 3  # 30分→10分への調整
            future_increasing = future_discharge_trend > threshold_10min
            future_decreasing = future_discharge_trend < -threshold_10min
        else:
            future_increasing = False
            future_decreasing = False
        
        # 状態と降雨に基づく遅延時間の決定
        if (rain_increasing or future_rain_increasing) and (state <= 0 or future_increasing):
            # 降雨が急増または今後急増予測 → 即座に反応
            return self.params['delay_onset']
        elif state == 1 and not future_decreasing and not (rainfall_current < 10 and (future_rainfall_30min is None or future_rainfall_30min < 10)):
            # 増加継続中で今後も減少しない → 通常の遅延
            return self.params['delay_increase']
        elif (rainfall_current < self.params['rain_threshold_low'] and state <= 0) or future_decreasing:
            # 低降雨かつ非増加状態、または将来減少予測 → 減少開始の遅延
            return self.params['delay_decrease']
        else:
            # その他 → 標準的な遅延
            return self.params['delay_increase']
    
    def predict_discharge_change(self, state, rainfall_intensity, current_discharge):
        """
        放流量の変化を予測
        
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
        
        # 基本変化率の決定
        if rain_category == 'low':
            if state <= 0:  # 減少中または定常
                # 低降雨では減少
                base_rate = -self.params['rate_decrease_low']
            else:  # 増加中
                # 増加中でも低降雨なら減少に転じる
                if rainfall_intensity < 5.0:  # 非常に低い降雨
                    base_rate = -self.params['rate_decrease_low'] * 0.5  # 緩やかに減少
                else:  # 5-10mm/h
                    base_rate = 0  # 現状維持
        elif rain_category == 'medium':
            if state >= 0:  # 増加中または定常
                base_rate = self.params['rate_increase_low']
            else:  # 減少中
                # 中降雨で減少中なら減速を緩める
                base_rate = -self.params['rate_decrease_low'] * 0.5
        else:  # high
            # 高降雨では積極的に増加
            base_rate = self.params['rate_increase_high']
        
        # 現在の放流量による調整（大規模放流時は変化を抑制）
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
        rainfall_history = df.iloc[-7:]['ダム_60分雨量']
        
        # 現在の状態を識別
        current_state, current_rate = self.identify_discharge_state(discharge_history)
        
        # 予測結果の格納
        predictions = []
        
        # 各時間ステップで予測
        predicted_discharge = current_discharge
        predicted_state = current_state
        last_forecast_value = 0  # 最後の予測値を保存
        
        for step in range(1, int(prediction_hours * 6) + 1):
            pred_time = current_time + timedelta(minutes=step * 10)
            
            # 降雨予測の取得（1時間先まで利用可能）
            if rainfall_forecast is not None and step <= 6:
                # 予測データから取得
                forecast_idx = rainfall_forecast['時刻'].searchsorted(pred_time)
                if forecast_idx < len(rainfall_forecast):
                    future_rainfall = rainfall_forecast.iloc[forecast_idx]['降雨強度']
                    last_forecast_value = future_rainfall  # 最後の予測値を更新
                else:
                    future_rainfall = last_forecast_value  # 予測がない場合は最後の値を使用
            elif step <= 6:
                # 予測データがない場合、1時間先までは現在値を使用
                future_rainfall = current_rainfall
                last_forecast_value = future_rainfall  # 最後の予測値を更新
            else:
                # 1時間以降は最後の予測値を継続
                future_rainfall = last_forecast_value
            
            # 10分先までの予測を考慮した遅延時間の計算
            # 最初のステップで10分先までの傾向を計算
            if step == 1:
                # 10分先の放流量変化の簡易予測
                temp_discharge = predicted_discharge
                temp_state = predicted_state
                
                # 10分先の降雨予測の取得
                if rainfall_forecast is not None:
                    look_ahead_time = current_time + timedelta(minutes=10)
                    look_ahead_idx = rainfall_forecast['時刻'].searchsorted(look_ahead_time)
                    if look_ahead_idx < len(rainfall_forecast):
                        future_rainfall_10min = rainfall_forecast.iloc[look_ahead_idx]['降雨強度']
                    else:
                        future_rainfall_10min = current_rainfall
                else:
                    future_rainfall_10min = current_rainfall
                
                # 10分先の簡易的な変化予測
                if future_rainfall_10min >= 20:
                    future_discharge_trend = self.params['rate_increase_high'] * 0.8
                elif future_rainfall_10min >= 10:
                    future_discharge_trend = self.params['rate_increase_low'] * 0.8
                else:
                    future_discharge_trend = -self.params['rate_decrease_low'] * 0.5
                
                # 10分先の降雨（30分平均の代わりに10分先の値を使用）
                future_rainfall_30min = future_rainfall_10min
            else:
                # 2ステップ目以降は前回の値を使用
                future_discharge_trend = None
                future_rainfall_30min = None
            
            # 動的遅延時間の取得（10分先の予測を考慮）
            delay = self.get_dynamic_delay(predicted_state, current_rainfall, rainfall_history,
                                         future_discharge_trend, future_rainfall_30min)
            
            # 遅延を考慮した降雨強度
            delay_steps = delay // 10
            
            # 遅延時間0分の特別処理
            if delay == 0:
                # 遅延なし＝現在または将来の降雨が即座に影響
                effective_rainfall = future_rainfall
            elif step > delay_steps:
                # 過去の降雨データを使用
                past_idx = step - delay_steps - 1
                if past_idx < len(rainfall_history):
                    effective_rainfall = rainfall_history.iloc[past_idx]
                else:
                    effective_rainfall = future_rainfall
            else:
                # まだ遅延時間内
                hist_idx = delay_steps - step
                if hist_idx < len(rainfall_history):
                    effective_rainfall = rainfall_history.iloc[-(hist_idx + 1)]
                else:
                    effective_rainfall = rainfall_history.iloc[0]
            
            # 放流量変化の予測
            change_rate = self.predict_discharge_change(
                predicted_state, effective_rainfall, predicted_discharge
            )
            
            # 放流量の更新
            predicted_discharge += change_rate
            predicted_discharge = max(150, predicted_discharge)  # 最小値制限
            
            # 状態の更新（改善版）
            # 降雨履歴による状態の強制変更
            recent_rainfall_avg = rainfall_history[-3:].mean() if len(rainfall_history) >= 3 else effective_rainfall
            
            if change_rate > 5:
                predicted_state = 1
            elif change_rate < -5:
                predicted_state = -1
            else:
                predicted_state = 0
            
            # 低降雨が継続している場合の状態修正（過去10分の降雨で判定）
            recent_rainfall_10min = rainfall_history.iloc[-1] if len(rainfall_history) >= 1 else effective_rainfall
            
            if predicted_state == 1 and recent_rainfall_10min < 5.0 and effective_rainfall < 5.0:
                # 増加状態でも低降雨が10分続いたら定常へ
                predicted_state = 0
            elif predicted_state == 0 and recent_rainfall_10min < 3.0 and effective_rainfall < 3.0:
                # 定常状態で非常に低い降雨が続いたら減少へ
                predicted_state = -1
            
            # 結果の保存
            predictions.append({
                '時刻': pred_time,
                '予測放流量': predicted_discharge,
                '使用降雨強度': effective_rainfall,
                '適用遅延': delay,
                '状態': predicted_state,
                '変化率': change_rate
            })
            
            # 履歴の更新（次のステップ用）
            # 1時間超でも最後の予測値を継続して使用
            rainfall_history = pd.concat([
                rainfall_history[1:], 
                pd.Series([future_rainfall])
            ])
        
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