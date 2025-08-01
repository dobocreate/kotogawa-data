#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
放流量予測モデル v3 - ハイブリッド版
- ルールベースと機械学習の組み合わせ
- 遅延時間の動的調整を機械学習で改善
- 高降雨継続時の適切な対応
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from datetime import datetime, timedelta
import pickle
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# 日本語フォント設定
plt.rcParams['font.sans-serif'] = ['Yu Gothic', 'Hiragino Sans', 'Meirio', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class DischargePredictionModelV3Hybrid:
    """ハイブリッド版放流量予測モデル"""
    
    def __init__(self):
        """初期化"""
        # 基本パラメータ（v2から継承）
        self.params = {
            # 遅延時間（分）
            'delay_onset': 0,        # 増加開始時の遅延
            'delay_increase': 120,   # 増加継続時の遅延
            'delay_decrease': 60,    # 減少開始時の遅延
            
            # 降雨強度閾値（mm/h）
            'rain_threshold_low': 10,    # 増加/減少の境界
            'rain_threshold_high': 20,   # 積極的増加の閾値
            
            # 状態判定閾値（m³/s per 10min）
            'state_threshold': 10,
            
            # 変化率（m³/s per 10min）
            'rate_increase_low': 27.1,    # 低降雨時の増加率
            'rate_decrease_low': 24.3,    # 低降雨時の減少率
            'rate_increase_high': 29.1,   # 中降雨時の増加率
            'rate_decrease_ratio': 0.89,  # 減少率/増加率の比率
        }
        
        # 機械学習モデル
        self.ml_models = {
            'delay_adjuster': None,      # 遅延時間調整モデル
            'rate_modifier': None,       # 変化率修正モデル
            'trend_predictor': None      # 短期トレンド予測モデル
        }
        
        self.scalers = {
            'delay_adjuster': StandardScaler(),
            'rate_modifier': StandardScaler(),
            'trend_predictor': StandardScaler()
        }
        
        # 特徴量名
        self.feature_names = {
            'delay_adjuster': [
                'current_discharge',      # 現在の放流量
                'discharge_state',        # 状態（-1:減少, 0:定常, 1:増加）
                'discharge_change_10min', # 過去10分の変化
                'discharge_change_30min', # 過去30分の変化
                'current_rainfall',       # 現在の降雨
                'rainfall_10min_future',  # 10分先の降雨
                'rainfall_30min_future',  # 30分先の降雨
                'rainfall_60min_future',  # 60分先の降雨
                'rainfall_trend',         # 降雨トレンド
                'max_rainfall_1h',        # 1時間先までの最大降雨
                'avg_rainfall_1h',        # 1時間先までの平均降雨
                'rule_based_delay'        # ルールベースの遅延時間
            ],
            'rate_modifier': [
                'current_discharge',
                'discharge_state',
                'discharge_change_10min',
                'effective_rainfall',     # 遅延考慮後の降雨
                'future_rainfall_avg',    # 将来降雨の平均
                'rainfall_category',      # 降雨カテゴリ（0:低, 1:中, 2:高）
                'continuous_high_rain',   # 高降雨継続時間
                'rule_based_rate'         # ルールベースの変化率
            ],
            'trend_predictor': [
                'current_discharge',
                'discharge_changes_1h',   # 過去1時間の変化パターン
                'current_rainfall',
                'rainfall_forecast_1h',   # 1時間先までの降雨予測
                'current_state',
                'predicted_delays',       # 予測遅延時間列
                'predicted_rates'         # 予測変化率列
            ]
        }
        
    def identify_discharge_state(self, discharge_history, future_discharge_10min=None):
        """
        放流量の状態を識別（v2と同じロジック）
        """
        if len(discharge_history) < 2:
            return 0, 0.0
        
        # 過去10分の変化量
        change_past_10min = discharge_history.iloc[-1] - discharge_history.iloc[-2]
        
        # 10分先予測がある場合は前後20分での変化を考慮
        if future_discharge_10min is not None:
            total_change = future_discharge_10min - discharge_history.iloc[-2]
            rate = total_change / 2  # m³/s per 10min
            
            threshold_20min = self.params['state_threshold'] * 2 / 3
            if total_change > threshold_20min:
                state = 1
            elif total_change < -threshold_20min:
                state = -1
            else:
                state = 0
        else:
            rate = change_past_10min
            threshold_10min = self.params['state_threshold'] / 3
            
            if change_past_10min > threshold_10min:
                state = 1
            elif change_past_10min < -threshold_10min:
                state = -1
            else:
                state = 0
            
        return state, rate
    
    def get_rule_based_delay(self, state, rainfall_current, rainfall_history, 
                            future_discharge_trend=None, future_rainfall_10min=None):
        """
        ルールベースの遅延時間（v2と同じ）
        """
        # 降雨の増加を検出（過去10分）
        if len(rainfall_history) >= 2:
            rain_change = rainfall_current - rainfall_history.iloc[-2]
            rain_increasing = rain_change > 5
        else:
            rain_increasing = False
        
        # 将来の降雨傾向を考慮
        if future_rainfall_10min is not None:
            future_rain_high = future_rainfall_10min > self.params['rain_threshold_high']
            future_rain_increasing = future_rainfall_10min > rainfall_current + 5
        else:
            future_rain_high = False
            future_rain_increasing = False
        
        # 将来の放流量傾向を考慮
        if future_discharge_trend is not None:
            threshold_10min = self.params['state_threshold'] / 3
            future_increasing = future_discharge_trend > threshold_10min
            future_decreasing = future_discharge_trend < -threshold_10min
        else:
            future_increasing = False
            future_decreasing = False
        
        # 状態と降雨に基づく遅延時間の決定
        if (rain_increasing or future_rain_increasing) and (state <= 0 or future_increasing):
            return self.params['delay_onset']
        elif state == 1 and not future_decreasing and not (rainfall_current < 10 and (future_rainfall_10min is None or future_rainfall_10min < 10)):
            return self.params['delay_increase']
        elif (rainfall_current < self.params['rain_threshold_low'] and state <= 0) or future_decreasing:
            return self.params['delay_decrease']
        else:
            return self.params['delay_increase']
    
    def get_ml_adjusted_delay(self, features):
        """
        機械学習による遅延時間の調整
        
        Returns:
        --------
        adjusted_delay : int
            調整後の遅延時間（分）
        """
        if self.ml_models['delay_adjuster'] is None:
            return features['rule_based_delay']
        
        # 特徴量の準備
        X = np.array([features[name] for name in self.feature_names['delay_adjuster']]).reshape(1, -1)
        X_scaled = self.scalers['delay_adjuster'].transform(X)
        
        # 予測（調整係数）
        adjustment_factor = self.ml_models['delay_adjuster'].predict(X_scaled)[0]
        
        # ルールベースの遅延時間を調整
        adjusted_delay = features['rule_based_delay'] * adjustment_factor
        
        # 制約を適用（0～120分の範囲内）
        adjusted_delay = np.clip(adjusted_delay, 0, 120)
        
        # 10分単位に丸める
        adjusted_delay = int(round(adjusted_delay / 10) * 10)
        
        return adjusted_delay
    
    def get_rule_based_rate(self, state, rainfall_intensity, current_discharge):
        """
        ルールベースの変化率（v2と同じ）
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
            if state <= 0:
                base_rate = -self.params['rate_decrease_low']
            else:
                if rainfall_intensity < 5.0:
                    base_rate = -self.params['rate_decrease_low'] * 0.5
                else:
                    base_rate = 0
        elif rain_category == 'medium':
            if state >= 0:
                base_rate = self.params['rate_increase_low']
            else:
                base_rate = -self.params['rate_decrease_low'] * 0.5
        else:  # high
            base_rate = self.params['rate_increase_high']
        
        # 現在の放流量による調整
        if current_discharge > 1000:
            rate_modifier = 0.7
        elif current_discharge > 800:
            rate_modifier = 0.85
        else:
            rate_modifier = 1.0
        
        return base_rate * rate_modifier
    
    def get_ml_adjusted_rate(self, features):
        """
        機械学習による変化率の調整
        """
        if self.ml_models['rate_modifier'] is None:
            return features['rule_based_rate']
        
        # 特徴量の準備
        X = np.array([features[name] for name in self.feature_names['rate_modifier']]).reshape(1, -1)
        X_scaled = self.scalers['rate_modifier'].transform(X)
        
        # 予測（修正係数）
        modification_factor = self.ml_models['rate_modifier'].predict(X_scaled)[0]
        
        # ルールベースの変化率を修正
        adjusted_rate = features['rule_based_rate'] * modification_factor
        
        # 制約を適用（物理的に妥当な範囲）
        max_rate = 40.0  # m³/s per 10min
        min_rate = -30.0
        adjusted_rate = np.clip(adjusted_rate, min_rate, max_rate)
        
        return adjusted_rate
    
    def predict(self, current_time, historical_data, prediction_hours=3, 
                rainfall_forecast=None):
        """
        ハイブリッド予測
        """
        # データ準備
        df = historical_data.copy()
        df = df.sort_values('時刻')
        
        current_idx = len(df) - 1
        current_discharge = df.iloc[current_idx]['ダム_全放流量']
        current_rainfall = df.iloc[current_idx]['ダム_60分雨量']
        
        # 履歴データ
        discharge_history = df.iloc[-7:]['ダム_全放流量']
        rainfall_history = df.iloc[-7:]['ダム_60分雨量']
        
        # 現在の状態を識別
        current_state, current_rate = self.identify_discharge_state(discharge_history)
        
        # 予測結果の格納
        predictions = []
        
        # 各時間ステップで予測
        predicted_discharge = current_discharge
        predicted_state = current_state
        last_forecast_value = 0
        continuous_high_rain = 0  # 高降雨継続カウンタ
        
        for step in range(1, int(prediction_hours * 6) + 1):
            pred_time = current_time + timedelta(minutes=step * 10)
            
            # 降雨予測の取得
            if rainfall_forecast is not None and step <= 6:
                forecast_idx = rainfall_forecast['時刻'].searchsorted(pred_time)
                if forecast_idx < len(rainfall_forecast):
                    future_rainfall = rainfall_forecast.iloc[forecast_idx]['降雨強度']
                    last_forecast_value = future_rainfall
                else:
                    future_rainfall = last_forecast_value
            elif step <= 6:
                future_rainfall = current_rainfall
                last_forecast_value = future_rainfall
            else:
                future_rainfall = last_forecast_value
            
            # 高降雨継続時間の更新
            if future_rainfall >= self.params['rain_threshold_high']:
                continuous_high_rain += 1
            else:
                continuous_high_rain = 0
            
            # 将来の降雨情報を収集（MLモデル用）
            future_rainfalls = []
            for i in range(1, 7):  # 60分先まで
                if rainfall_forecast is not None and step + i <= len(rainfall_forecast):
                    idx = step + i - 1
                    if idx < len(rainfall_forecast):
                        future_rainfalls.append(rainfall_forecast.iloc[idx]['降雨強度'])
                    else:
                        future_rainfalls.append(last_forecast_value)
                else:
                    future_rainfalls.append(future_rainfall)
            
            # 機械学習用の特徴量を準備
            delay_features = {
                'current_discharge': predicted_discharge,
                'discharge_state': predicted_state,
                'discharge_change_10min': discharge_history.iloc[-1] - discharge_history.iloc[-2] if len(discharge_history) >= 2 else 0,
                'discharge_change_30min': discharge_history.iloc[-1] - discharge_history.iloc[-4] if len(discharge_history) >= 4 else 0,
                'current_rainfall': current_rainfall,
                'rainfall_10min_future': future_rainfalls[0] if future_rainfalls else future_rainfall,
                'rainfall_30min_future': np.mean(future_rainfalls[:3]) if len(future_rainfalls) >= 3 else future_rainfall,
                'rainfall_60min_future': np.mean(future_rainfalls[:6]) if len(future_rainfalls) >= 6 else future_rainfall,
                'rainfall_trend': (future_rainfalls[0] - current_rainfall) if future_rainfalls else 0,
                'max_rainfall_1h': max(future_rainfalls) if future_rainfalls else future_rainfall,
                'avg_rainfall_1h': np.mean(future_rainfalls) if future_rainfalls else future_rainfall,
                'rule_based_delay': 0  # 後で設定
            }
            
            # ルールベースの遅延時間を取得
            if step == 1:
                # 初回は10分先予測を考慮
                future_rainfall_10min = future_rainfalls[0] if future_rainfalls else future_rainfall
                if future_rainfall_10min >= 20:
                    future_discharge_trend = self.params['rate_increase_high'] * 0.8
                elif future_rainfall_10min >= 10:
                    future_discharge_trend = self.params['rate_increase_low'] * 0.8
                else:
                    future_discharge_trend = -self.params['rate_decrease_low'] * 0.5
                
                rule_delay = self.get_rule_based_delay(predicted_state, current_rainfall, rainfall_history,
                                                       future_discharge_trend, future_rainfall_10min)
            else:
                rule_delay = self.get_rule_based_delay(predicted_state, current_rainfall, rainfall_history)
            
            delay_features['rule_based_delay'] = rule_delay
            
            # 機械学習による遅延時間の調整
            delay = self.get_ml_adjusted_delay(delay_features)
            
            # 遅延を考慮した降雨強度
            delay_steps = delay // 10
            
            if delay == 0:
                effective_rainfall = future_rainfall
            elif step > delay_steps:
                past_idx = step - delay_steps - 1
                if past_idx < len(rainfall_history):
                    effective_rainfall = rainfall_history.iloc[past_idx]
                else:
                    effective_rainfall = future_rainfall
            else:
                hist_idx = delay_steps - step
                if hist_idx < len(rainfall_history):
                    effective_rainfall = rainfall_history.iloc[-(hist_idx + 1)]
                else:
                    effective_rainfall = rainfall_history.iloc[0]
            
            # 変化率計算用の特徴量
            rate_features = {
                'current_discharge': predicted_discharge,
                'discharge_state': predicted_state,
                'discharge_change_10min': discharge_history.iloc[-1] - discharge_history.iloc[-2] if len(discharge_history) >= 2 else 0,
                'effective_rainfall': effective_rainfall,
                'future_rainfall_avg': np.mean(future_rainfalls[:3]) if future_rainfalls else future_rainfall,
                'rainfall_category': 2 if effective_rainfall >= 20 else (1 if effective_rainfall >= 10 else 0),
                'continuous_high_rain': continuous_high_rain,
                'rule_based_rate': 0  # 後で設定
            }
            
            # ルールベースの変化率を取得
            rule_rate = self.get_rule_based_rate(predicted_state, effective_rainfall, predicted_discharge)
            rate_features['rule_based_rate'] = rule_rate
            
            # 機械学習による変化率の調整
            change_rate = self.get_ml_adjusted_rate(rate_features)
            
            # 放流量の更新
            predicted_discharge += change_rate
            predicted_discharge = max(150, predicted_discharge)  # 最小値制限
            
            # 状態の更新
            recent_rainfall_10min = rainfall_history.iloc[-1] if len(rainfall_history) >= 1 else effective_rainfall
            
            if change_rate > 5:
                predicted_state = 1
            elif change_rate < -5:
                predicted_state = -1
            else:
                predicted_state = 0
            
            # 低降雨継続時の状態修正
            if predicted_state == 1 and recent_rainfall_10min < 5.0 and effective_rainfall < 5.0:
                predicted_state = 0
            elif predicted_state == 0 and recent_rainfall_10min < 3.0 and effective_rainfall < 3.0:
                predicted_state = -1
            
            # 結果の保存
            predictions.append({
                '時刻': pred_time,
                '予測放流量': predicted_discharge,
                '使用降雨強度': effective_rainfall,
                '適用遅延': delay,
                '状態': predicted_state,
                '変化率': change_rate,
                'ML調整係数_遅延': delay / rule_delay if rule_delay > 0 else 1.0,
                'ML調整係数_変化率': change_rate / rule_rate if abs(rule_rate) > 0.1 else 1.0
            })
            
            # 履歴の更新
            rainfall_history = pd.concat([
                rainfall_history[1:], 
                pd.Series([future_rainfall])
            ])
        
        return pd.DataFrame(predictions)
    
    def train_ml_components(self, training_data):
        """
        機械学習コンポーネントの訓練
        """
        print("=== 機械学習コンポーネントの訓練開始 ===")
        
        # データの準備
        df = training_data.copy()
        df = df.sort_values('時刻')
        
        # 訓練データの生成
        X_delay = []
        y_delay = []
        X_rate = []
        y_rate = []
        
        # 時系列でデータを処理
        for i in range(70, len(df) - 18):  # 7時間分の履歴と3時間先まで確認
            # 現在時点のデータ
            current_data = df.iloc[i]
            history_data = df.iloc[i-7:i+1]
            future_data = df.iloc[i:i+18]
            
            # 状態識別
            discharge_history = history_data['ダム_全放流量']
            state, _ = self.identify_discharge_state(discharge_history)
            
            # 実際の放流量変化から最適な遅延時間を推定
            actual_changes = future_data['ダム_全放流量'].diff()
            rainfall_changes = history_data['ダム_60分雨量'].diff()
            
            # 相関が最も高い遅延時間を見つける
            correlations = []
            for delay_steps in range(0, 13):  # 0～120分
                if delay_steps < len(actual_changes) and delay_steps < len(rainfall_changes):
                    # 両方の配列の長さを揃える
                    min_len = min(len(actual_changes) - delay_steps, len(rainfall_changes) - delay_steps)
                    if min_len > 2:
                        try:
                            corr = np.corrcoef(
                                actual_changes[delay_steps:delay_steps+min_len].values,
                                rainfall_changes[:min_len].values
                            )[0, 1]
                            if np.isnan(corr):
                                corr = 0
                            correlations.append(corr)
                        except:
                            correlations.append(0)
                    else:
                        correlations.append(0)
                else:
                    correlations.append(0)
            
            optimal_delay = np.argmax(np.abs(correlations)) * 10  # 絶対値で判断
            
            # 特徴量の生成（遅延時間予測用）
            delay_features = self._create_delay_features(current_data, history_data, future_data)
            rule_delay = self.get_rule_based_delay(
                state, 
                current_data['ダム_60分雨量'],
                history_data['ダム_60分雨量']
            )
            delay_features['rule_based_delay'] = rule_delay
            
            # 調整係数として学習
            if rule_delay > 0:
                delay_adjustment = optimal_delay / rule_delay
                delay_adjustment = np.clip(delay_adjustment, 0.2, 2.0)  # 極端な値を制限
                
                X_delay.append([delay_features[name] for name in self.feature_names['delay_adjuster']])
                y_delay.append(delay_adjustment)
            
            # 変化率予測用のデータ生成
            actual_rate = actual_changes.iloc[1]  # 10分後の実際の変化
            rate_features = self._create_rate_features(current_data, history_data, future_data, state)
            rule_rate = self.get_rule_based_rate(
                state,
                current_data['ダム_60分雨量'],
                current_data['ダム_全放流量']
            )
            rate_features['rule_based_rate'] = rule_rate
            
            # 調整係数として学習
            if abs(rule_rate) > 0.1:
                rate_adjustment = actual_rate / rule_rate
                rate_adjustment = np.clip(rate_adjustment, 0.5, 1.5)  # 極端な値を制限
                
                X_rate.append([rate_features[name] for name in self.feature_names['rate_modifier']])
                y_rate.append(rate_adjustment)
        
        # モデルの訓練
        if len(X_delay) > 100:
            print(f"\n遅延時間調整モデルの訓練: {len(X_delay)}サンプル")
            X_delay = np.array(X_delay)
            y_delay = np.array(y_delay)
            
            # スケーリング
            X_delay_scaled = self.scalers['delay_adjuster'].fit_transform(X_delay)
            
            # モデル訓練
            self.ml_models['delay_adjuster'] = GradientBoostingRegressor(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=42
            )
            self.ml_models['delay_adjuster'].fit(X_delay_scaled, y_delay)
            
            # 訓練精度
            train_pred = self.ml_models['delay_adjuster'].predict(X_delay_scaled)
            train_mae = mean_absolute_error(y_delay, train_pred)
            print(f"  訓練MAE: {train_mae:.3f}")
        
        if len(X_rate) > 100:
            print(f"\n変化率修正モデルの訓練: {len(X_rate)}サンプル")
            X_rate = np.array(X_rate)
            y_rate = np.array(y_rate)
            
            # スケーリング
            X_rate_scaled = self.scalers['rate_modifier'].fit_transform(X_rate)
            
            # モデル訓練
            self.ml_models['rate_modifier'] = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=42
            )
            self.ml_models['rate_modifier'].fit(X_rate_scaled, y_rate)
            
            # 訓練精度
            train_pred = self.ml_models['rate_modifier'].predict(X_rate_scaled)
            train_mae = mean_absolute_error(y_rate, train_pred)
            print(f"  訓練MAE: {train_mae:.3f}")
        
        print("\n=== 訓練完了 ===")
    
    def _create_delay_features(self, current_data, history_data, future_data):
        """遅延時間予測用の特徴量生成"""
        features = {
            'current_discharge': current_data['ダム_全放流量'],
            'discharge_state': 0,  # 後で設定
            'discharge_change_10min': history_data['ダム_全放流量'].iloc[-1] - history_data['ダム_全放流量'].iloc[-2],
            'discharge_change_30min': history_data['ダム_全放流量'].iloc[-1] - history_data['ダム_全放流量'].iloc[-4] if len(history_data) >= 4 else 0,
            'current_rainfall': current_data['ダム_60分雨量'],
            'rainfall_10min_future': future_data['ダム_60分雨量'].iloc[1] if len(future_data) > 1 else current_data['ダム_60分雨量'],
            'rainfall_30min_future': future_data['ダム_60分雨量'].iloc[:3].mean() if len(future_data) >= 3 else current_data['ダム_60分雨量'],
            'rainfall_60min_future': future_data['ダム_60分雨量'].iloc[:6].mean() if len(future_data) >= 6 else current_data['ダム_60分雨量'],
            'rainfall_trend': future_data['ダム_60分雨量'].iloc[1] - current_data['ダム_60分雨量'] if len(future_data) > 1 else 0,
            'max_rainfall_1h': future_data['ダム_60分雨量'].iloc[:6].max() if len(future_data) >= 6 else current_data['ダム_60分雨量'],
            'avg_rainfall_1h': future_data['ダム_60分雨量'].iloc[:6].mean() if len(future_data) >= 6 else current_data['ダム_60分雨量'],
        }
        
        # 状態を設定
        discharge_history = history_data['ダム_全放流量']
        state, _ = self.identify_discharge_state(discharge_history)
        features['discharge_state'] = state
        
        return features
    
    def _create_rate_features(self, current_data, history_data, future_data, state):
        """変化率予測用の特徴量生成"""
        # 高降雨継続時間の計算
        high_rain_count = 0
        for val in history_data['ダム_60分雨量'].values:
            if val >= self.params['rain_threshold_high']:
                high_rain_count += 1
            else:
                break
        
        features = {
            'current_discharge': current_data['ダム_全放流量'],
            'discharge_state': state,
            'discharge_change_10min': history_data['ダム_全放流量'].iloc[-1] - history_data['ダム_全放流量'].iloc[-2],
            'effective_rainfall': current_data['ダム_60分雨量'],  # 簡略化
            'future_rainfall_avg': future_data['ダム_60分雨量'].iloc[:3].mean() if len(future_data) >= 3 else current_data['ダム_60分雨量'],
            'rainfall_category': 2 if current_data['ダム_60分雨量'] >= 20 else (1 if current_data['ダム_60分雨量'] >= 10 else 0),
            'continuous_high_rain': high_rain_count,
        }
        
        return features
    
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

def train_hybrid_model():
    """ハイブリッドモデルの訓練"""
    print("=== ハイブリッドモデルの訓練 ===")
    
    # データ読み込み
    df = pd.read_csv('統合データ_水位ダム_20250730_142903.csv', encoding='utf-8')
    df['時刻'] = pd.to_datetime(df['時刻'])
    
    # 訓練期間のデータを使用（2023年6月～7月）
    train_mask = (df['時刻'] >= '2023-06-01') & (df['時刻'] <= '2023-07-31')
    train_data = df[train_mask].copy()
    
    print(f"訓練データ: {len(train_data)}件")
    
    # モデル初期化と訓練
    model = DischargePredictionModelV3Hybrid()
    model.train_ml_components(train_data)
    
    # モデルの保存
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    model.save_model(f'discharge_prediction_model_v3_hybrid_{timestamp}.pkl')
    
    return model

if __name__ == "__main__":
    # モデルの訓練
    model = train_hybrid_model()