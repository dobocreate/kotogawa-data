#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
water_level_prediction_demo_interactive.pyに降雨予測機能を追加するパッチ
"""

import os
import shutil
from datetime import datetime

def create_updated_demo():
    """降雨予測を活用する更新版を作成"""
    
    # バックアップ作成
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    original_file = 'water_level_prediction_demo_interactive.py'
    backup_file = f'water_level_prediction_demo_interactive_backup_{timestamp}.py'
    
    # ファイルを読み込み
    with open(original_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 1. run_predictionメソッドに降雨予測パラメータを追加
    old_predict_sig = 'def run_prediction(self, current_time_str=None, show_actual=True, use_discharge_model=True):'
    new_predict_sig = 'def run_prediction(self, current_time_str=None, show_actual=True, use_discharge_model=True, rainfall_forecast_data=None):'
    content = content.replace(old_predict_sig, new_predict_sig)
    
    # 2. 降雨予測データの生成部分を改良
    old_rainfall_forecast = """                # 降雨予測データの作成（1時間先まで利用可能）
                # ここでは現在の降雨強度を1時間継続すると仮定
                current_rainfall = self.full_data.iloc[full_idx]['ダム_60分雨量'] if 'ダム_60分雨量' in self.full_data.columns else 0"""
    
    new_rainfall_forecast = """                # 降雨予測データの作成（1時間先まで利用可能）
                current_rainfall = self.full_data.iloc[full_idx]['ダム_60分雨量'] if 'ダム_60分雨量' in self.full_data.columns else 0
                
                # 降雨予測データの準備
                rainfall_forecast = None
                if rainfall_forecast_data is not None:
                    # 外部から降雨予測が提供された場合
                    rainfall_forecast = rainfall_forecast_data
                    print(f"降雨予測: 外部データを使用（{len(rainfall_forecast_data)}件）")
                else:
                    # デフォルト: 現在値から段階的に減少
                    import pandas as pd
                    forecast_times = pd.date_range(start=current_time + pd.Timedelta(minutes=10), 
                                                 end=current_time + pd.Timedelta(hours=1), 
                                                 freq='10min')
                    # 10分ごとに10%ずつ減少する予測
                    forecast_values = [current_rainfall * (0.9 ** i) for i in range(1, len(forecast_times) + 1)]
                    rainfall_forecast = pd.DataFrame({
                        '時刻': forecast_times,
                        '降雨強度': forecast_values
                    })
                    print(f"降雨予測: 段階的減少モデル（現在{current_rainfall:.1f}mm/h）")"""
    
    content = content.replace(old_rainfall_forecast, new_rainfall_forecast)
    
    # 3. discharge_predictor.predictにrainfall_forecastを渡す
    old_predict_call = """                # 放流量予測を実行
                predicted_discharge = self.discharge_predictor.predict(
                    current_time, 
                    self.full_data.iloc[:full_idx+1],
                    prediction_hours=3
                )"""
    
    new_predict_call = """                # 放流量予測を実行
                predicted_discharge = self.discharge_predictor.predict(
                    current_time, 
                    self.full_data.iloc[:full_idx+1],
                    prediction_hours=3,
                    rainfall_forecast=rainfall_forecast
                )"""
    
    content = content.replace(old_predict_call, new_predict_call)
    
    # 4. 可視化部分の降雨予測表示を改良
    old_future_rain_viz = """            # 将来の降雨予測（1時間先まで現在値継続と仮定）
            future_times_1h = pd.date_range(start=current_time + timedelta(minutes=10), 
                                           end=min(current_time + timedelta(hours=1), display_end), 
                                           freq='10min')
            if len(future_times_1h) > 0:
                ax3.bar(future_times_1h, [current_rainfall] * len(future_times_1h),
                       width=0.007, color='lightblue', alpha=0.5, 
                       label='降雨予測（1時間先まで）')"""
    
    new_future_rain_viz = """            # 将来の降雨予測（1時間先まで）
            if rainfall_forecast is not None and len(rainfall_forecast) > 0:
                # 実際の降雨予測を表示
                forecast_mask = (rainfall_forecast['時刻'] >= current_time) & (rainfall_forecast['時刻'] <= display_end)
                ax3.bar(rainfall_forecast.loc[forecast_mask, '時刻'], 
                       rainfall_forecast.loc[forecast_mask, '降雨強度'],
                       width=0.007, color='lightblue', alpha=0.5, 
                       label='降雨予測（1時間先）')
            else:
                # デフォルト表示
                future_times_1h = pd.date_range(start=current_time + timedelta(minutes=10), 
                                               end=min(current_time + timedelta(hours=1), display_end), 
                                               freq='10min')
                if len(future_times_1h) > 0:
                    ax3.bar(future_times_1h, [current_rainfall * (0.9 ** i) for i in range(1, len(future_times_1h) + 1)],
                           width=0.007, color='lightblue', alpha=0.5, 
                           label='降雨予測（段階的減少）')"""
    
    content = content.replace(old_future_rain_viz, new_future_rain_viz)
    
    # バックアップ保存
    shutil.copy(original_file, backup_file)
    print(f"バックアップを作成: {backup_file}")
    
    # 更新版を保存
    with open(original_file, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"更新完了: {original_file}")
    print("\n追加された機能:")
    print("1. rainfall_forecast_dataパラメータ（外部から降雨予測を提供可能）")
    print("2. デフォルトで段階的減少モデル（10分ごとに10%減少）")
    print("3. 降雨予測をdischarge_predictorに渡す")
    print("4. 降雨予測の可視化を改良")
    
    return True

if __name__ == "__main__":
    create_updated_demo()