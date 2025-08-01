#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
水位予測モデル v2
Figure 11, 12の分析結果を基にした3時間先までの水位予測
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from datetime import datetime, timedelta
import pickle
from scipy import interpolate
import warnings
warnings.filterwarnings('ignore')

# 日本語フォント設定
plt.rcParams['font.sans-serif'] = ['Yu Gothic', 'Hiragino Sans', 'Meirio', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class WaterLevelPredictorV2:
    """水位予測モデル v2"""
    
    def __init__(self):
        # モデルパラメータ（Figure 11, 12の分析結果）
        self.delay_params = {
            'increase': 40,  # 増加期の遅延時間（分）
            'decrease': 60   # 減少期の遅延時間（分）
        }
        
        # 水位レベル別応答率（dH/dt / dQ/dt）
        self.response_rates = {
            'increase': {
                '3.0-3.5': 0.00452,
                '3.5-4.0': 0.00380,
                '4.0-4.5': 0.00334,
                '4.5-5.0': 0.00565,
                '5.0+': 0.00551
            },
            'decrease': {
                '3.0-3.5': 0.00223,
                '3.5-4.0': 0.00250,
                '4.0-4.5': 0.00365,
                '4.5-5.0': 0.00468,
                '5.0+': 0.00325
            }
        }
        
        # 予測誤差統計（初期値）
        self.error_stats = {
            'rmse': 0.3,  # 初期RMSE推定値
            'confidence_factor': 1.96  # 95%信頼区間
        }
        
        # データ保存用
        self.training_data = None
        self.validation_results = []
        
    def get_water_level_category(self, water_level):
        """水位レベルのカテゴリを取得"""
        if water_level < 3.5:
            return '3.0-3.5'
        elif water_level < 4.0:
            return '3.5-4.0'
        elif water_level < 4.5:
            return '4.0-4.5'
        elif water_level < 5.0:
            return '4.5-5.0'
        else:
            return '5.0+'
    
    def get_response_rate(self, water_level, direction):
        """水位レベルと方向に応じた応答率を取得"""
        category = self.get_water_level_category(water_level)
        return self.response_rates[direction][category]
    
    def predict_water_level(self, current_time, current_water_level, discharge_history, 
                          prediction_hours=3, rainfall_forecast=None):
        """
        水位予測
        
        Parameters:
        -----------
        current_time : datetime
            現在時刻
        current_water_level : float
            現在の水位 (m)
        discharge_history : pd.DataFrame
            放流量履歴 (columns: ['時刻', '放流量'])
        prediction_hours : float
            予測時間（時間）
        rainfall_forecast : pd.DataFrame, optional
            降雨予測データ (columns: ['時刻', '降雨強度'])
        
        Returns:
        --------
        predictions : pd.DataFrame
            予測結果 (columns: ['時刻', '予測水位', '予測下限', '予測上限'])
        """
        
        # 予測ステップ数（10分間隔）
        n_steps = int(prediction_hours * 6)
        
        # 結果保存用
        predictions = []
        
        # 初期値
        predicted_water_level = current_water_level
        
        # 各ステップで予測
        for step in range(n_steps):
            # 予測時刻
            pred_time = current_time + timedelta(minutes=(step + 1) * 10)
            
            # 1. 放流量の影響を計算
            water_level_change = 0
            
            # 遅延を考慮した放流量変化の影響を積算
            for delay_direction in ['increase', 'decrease']:
                delay_minutes = self.delay_params[delay_direction]
                delay_steps = delay_minutes // 10
                
                # 影響を与える放流量の時刻
                influence_time = pred_time - timedelta(minutes=delay_minutes)
                
                # 放流量変化を取得
                if influence_time >= discharge_history['時刻'].min():
                    # 前後30分の放流量変化率を計算
                    time_before = influence_time - timedelta(minutes=30)
                    time_after = influence_time + timedelta(minutes=30)
                    
                    # 時刻に最も近いデータを取得
                    idx_before = discharge_history['時刻'].searchsorted(time_before)
                    idx_after = discharge_history['時刻'].searchsorted(time_after)
                    
                    if idx_before < len(discharge_history) and idx_after < len(discharge_history):
                        q_before = discharge_history.iloc[idx_before]['放流量']
                        q_after = discharge_history.iloc[idx_after]['放流量']
                        
                        # 変化率計算 (m³/s/min)
                        dQ_dt = (q_after - q_before) / 60
                        
                        # 変化の方向判定
                        if dQ_dt > 0.5 and delay_direction == 'increase':
                            # 増加期の応答
                            response_rate = self.get_response_rate(predicted_water_level, 'increase')
                            water_level_change += dQ_dt * response_rate * 10  # 10分間の変化
                        elif dQ_dt < -0.5 and delay_direction == 'decrease':
                            # 減少期の応答
                            response_rate = self.get_response_rate(predicted_water_level, 'decrease')
                            water_level_change += dQ_dt * response_rate * 10
            
            # 2. 降雨の影響（将来実装）
            if rainfall_forecast is not None:
                # TODO: 降雨による水位上昇を追加
                pass
            
            # 3. 水位更新
            predicted_water_level += water_level_change
            
            # 4. 予測区間の計算
            # 予測が先になるほど不確実性が増加
            uncertainty = self.error_stats['rmse'] * np.sqrt(step + 1) / np.sqrt(n_steps)
            lower_bound = predicted_water_level - self.error_stats['confidence_factor'] * uncertainty
            upper_bound = predicted_water_level + self.error_stats['confidence_factor'] * uncertainty
            
            # 結果保存
            predictions.append({
                '時刻': pred_time,
                '予測水位': predicted_water_level,
                '予測下限': lower_bound,
                '予測上限': upper_bound,
                '予測ステップ': step + 1
            })
            
            # 現在の放流量が継続すると仮定して履歴を更新
            current_discharge = discharge_history.iloc[-1]['放流量']
            discharge_history = pd.concat([
                discharge_history,
                pd.DataFrame({
                    '時刻': [pred_time],
                    '放流量': [current_discharge]
                })
            ], ignore_index=True)
        
        return pd.DataFrame(predictions)
    
    def train_with_historical_data(self, historical_data, cv_folds=5):
        """
        過去データを使用したモデル訓練（時系列クロスバリデーション）
        
        Parameters:
        -----------
        historical_data : pd.DataFrame
            過去データ (columns: ['時刻', '水位', '放流量'])
        cv_folds : int
            クロスバリデーションの分割数
        """
        print("=== モデル訓練開始 ===")
        
        # データ準備
        data = historical_data.copy()
        data = data.sort_values('時刻')
        
        # フィルタリング（水位≥3m、放流量≥150m³/s）
        data = data[(data['水位'] >= 3.0) & (data['放流量'] >= 150.0)]
        
        # 時系列クロスバリデーション
        fold_size = len(data) // (cv_folds + 1)
        all_errors = []
        
        for fold in range(cv_folds):
            print(f"\nFold {fold + 1}/{cv_folds}")
            
            # 訓練データとテストデータの分割
            train_end = (fold + 1) * fold_size
            test_start = train_end
            test_end = test_start + fold_size
            
            train_data = data.iloc[:train_end]
            test_data = data.iloc[test_start:test_end]
            
            # テストデータで予測と評価
            fold_errors = []
            
            # 1時間ごとに予測を実行
            for i in range(0, len(test_data) - 18, 6):  # 3時間先まで予測可能な位置
                # 現在の状態
                current_idx = test_start + i
                current_time = data.iloc[current_idx]['時刻']
                current_water_level = data.iloc[current_idx]['水位']
                
                # 放流量履歴（過去3時間）
                history_start = max(0, current_idx - 18)
                discharge_history = data.iloc[history_start:current_idx + 1][['時刻', '放流量']].copy()
                discharge_history.columns = ['時刻', '放流量']
                
                # 予測実行
                predictions = self.predict_water_level(
                    current_time, current_water_level, discharge_history,
                    prediction_hours=3
                )
                
                # 実際の値と比較
                for _, pred in predictions.iterrows():
                    # 実際の値を取得
                    actual_idx = data['時刻'].searchsorted(pred['時刻'])
                    if actual_idx < len(data):
                        actual_water_level = data.iloc[actual_idx]['水位']
                        error = pred['予測水位'] - actual_water_level
                        fold_errors.append({
                            '予測ステップ': pred['予測ステップ'],
                            '誤差': error,
                            '絶対誤差': abs(error),
                            '実際の水位': actual_water_level
                        })
            
            all_errors.extend(fold_errors)
        
        # 誤差統計の更新
        if all_errors:
            error_df = pd.DataFrame(all_errors)
            
            # ステップ別のRMSE計算
            step_rmse = {}
            for step in range(1, 19):  # 3時間 = 18ステップ
                step_errors = error_df[error_df['予測ステップ'] == step]['誤差'].values
                if len(step_errors) > 0:
                    step_rmse[step] = np.sqrt(np.mean(step_errors ** 2))
            
            # 全体のRMSE
            overall_rmse = np.sqrt(np.mean(error_df['絶対誤差'].values ** 2))
            self.error_stats['rmse'] = overall_rmse
            
            print(f"\n=== 訓練結果 ===")
            print(f"全体RMSE: {overall_rmse:.3f} m")
            print(f"1時間先RMSE: {step_rmse.get(6, 'N/A'):.3f} m")
            print(f"2時間先RMSE: {step_rmse.get(12, 'N/A'):.3f} m")
            print(f"3時間先RMSE: {step_rmse.get(18, 'N/A'):.3f} m")
            
            # 結果保存
            self.validation_results = error_df
            self.step_rmse = step_rmse
    
    def visualize_prediction(self, current_time, current_water_level, discharge_history, 
                           predictions, actual_data=None):
        """予測結果の可視化"""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
        
        # 1. 水位予測
        # 過去データ
        history_hours = 3
        history_start = current_time - timedelta(hours=history_hours)
        history_mask = discharge_history['時刻'] >= history_start
        
        # 実際の水位（あれば）
        if actual_data is not None:
            actual_mask = (actual_data['時刻'] >= history_start) & \
                         (actual_data['時刻'] <= predictions['時刻'].max())
            ax1.plot(actual_data.loc[actual_mask, '時刻'], 
                    actual_data.loc[actual_mask, '水位'],
                    'ko-', label='実際の水位', markersize=4)
        
        # 現在の水位
        ax1.plot(current_time, current_water_level, 'ro', markersize=10, label='現在')
        
        # 予測
        ax1.plot(predictions['時刻'], predictions['予測水位'], 
                'b-', linewidth=2, label='予測水位')
        ax1.fill_between(predictions['時刻'], 
                        predictions['予測下限'], 
                        predictions['予測上限'],
                        alpha=0.3, color='blue', label='95%予測区間')
        
        # 危険水位ライン
        ax1.axhline(y=5.5, color='red', linestyle='--', alpha=0.5, label='氾濫危険水位')
        
        ax1.set_ylabel('水位 (m)')
        ax1.set_title(f'水位予測 (現在: {current_time.strftime("%Y-%m-%d %H:%M")})')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. 放流量
        ax2.plot(discharge_history.loc[history_mask, '時刻'], 
                discharge_history.loc[history_mask, '放流量'],
                'g-', linewidth=2, label='放流量')
        
        # 将来の放流量（現在値継続）
        future_discharge = pd.DataFrame({
            '時刻': predictions['時刻'],
            '放流量': discharge_history.iloc[-1]['放流量']
        })
        ax2.plot(future_discharge['時刻'], future_discharge['放流量'],
                'g--', linewidth=2, alpha=0.5, label='放流量（仮定）')
        
        ax2.set_xlabel('時刻')
        ax2.set_ylabel('放流量 (m³/s)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 保存
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"water_level_prediction_{timestamp}.png"
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"予測グラフを保存: {filename}")
        
        plt.close()
        
        return filename
    
    def save_model(self, filepath):
        """モデルの保存"""
        model_data = {
            'delay_params': self.delay_params,
            'response_rates': self.response_rates,
            'error_stats': self.error_stats,
            'validation_results': self.validation_results,
            'step_rmse': getattr(self, 'step_rmse', None)
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        print(f"モデルを保存: {filepath}")
    
    def load_model(self, filepath):
        """モデルの読み込み"""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.delay_params = model_data['delay_params']
        self.response_rates = model_data['response_rates']
        self.error_stats = model_data['error_stats']
        self.validation_results = model_data.get('validation_results', [])
        self.step_rmse = model_data.get('step_rmse', None)
        
        print(f"モデルを読み込み: {filepath}")

def demonstrate_prediction():
    """予測デモ"""
    print("=== 水位予測モデル v2 デモ ===")
    
    # モデル初期化
    predictor = WaterLevelPredictorV2()
    
    # デモ用データ作成
    current_time = datetime.now()
    current_water_level = 4.2  # 現在の水位
    
    # 放流量履歴（過去3時間）
    times = [current_time - timedelta(minutes=i*10) for i in range(18, -1, -1)]
    discharges = [300 + 50 * np.sin(i/6) for i in range(19)]  # 変動する放流量
    
    discharge_history = pd.DataFrame({
        '時刻': times,
        '放流量': discharges
    })
    
    # 予測実行
    predictions = predictor.predict_water_level(
        current_time, current_water_level, discharge_history,
        prediction_hours=3
    )
    
    # 結果表示
    print("\n予測結果:")
    print(predictions[['時刻', '予測水位', '予測下限', '予測上限']].head(6))
    
    # 可視化
    predictor.visualize_prediction(
        current_time, current_water_level, discharge_history, predictions
    )
    
    return predictor, predictions

if __name__ == "__main__":
    demonstrate_prediction()