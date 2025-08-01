#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
リアルタイム水位予測モデル
放流量150m³/s以上を対象に、3時間先までの水位を予測
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import json
from collections import deque

class RealtimePredictionModel:
    def __init__(self, config_path=None):
        """
        初期化
        
        Parameters:
        -----------
        config_path : str
            設定ファイルのパス
        """
        # デフォルト設定
        self.config = {
            "min_discharge": 150.0,  # 最小対象放流量
            "prediction_hours": 3,   # 予測時間（時間）
            "time_step": 10,        # 時間ステップ（分）
            "history_hours": 2,     # 履歴保持時間（時間）
            
            # 遅延時間パラメータ（分析結果より）
            "delay_params": {
                "150-300": {"base_delay": 25, "correlation": 0.824},
                "300-500": {"base_delay": 15, "correlation": 0.710},
                "500+": {"base_delay": 10, "correlation": 0.636}
            },
            
            # 応答率パラメータ（ΔH/ΔQ）
            "response_rates": {
                "increase": {
                    "150-300": 0.0045,
                    "300-500": 0.0040,
                    "500+": 0.0035
                },
                "decrease": {
                    "150-300": 0.0042,
                    "300-500": 0.0038,
                    "500+": 0.0033
                }
            },
            
            # ヒステリシス補正係数
            "hysteresis_correction": {
                "increase": 1.02,
                "decrease": 0.98
            },
            
            # 初期水位レベル補正
            "water_level_correction": {
                "low": 1.2,      # < 3.5m
                "medium": 1.0,   # 3.5-4.5m
                "high": 0.9      # > 4.5m
            },
            
            # 安定性判定閾値
            "stability_thresholds": {
                "stable": 0.05,
                "semi_variable": 0.15
            }
        }
        
        # 設定ファイルから読み込み
        if config_path:
            self.load_config(config_path)
        
        # データ履歴
        self.discharge_history = deque(maxlen=int(self.config["history_hours"] * 60 / self.config["time_step"]))
        self.water_level_history = deque(maxlen=int(self.config["history_hours"] * 60 / self.config["time_step"]))
        
        # オンライン学習パラメータ
        self.online_correction_factor = 1.0
        self.error_history = deque(maxlen=30)  # 過去30回分の誤差
    
    def load_config(self, config_path):
        """設定ファイルの読み込み"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                loaded_config = json.load(f)
                self.config.update(loaded_config)
            print(f"設定ファイルを読み込みました: {config_path}")
        except Exception as e:
            print(f"設定ファイルの読み込みエラー: {e}")
            print("デフォルト設定を使用します")
    
    def update_history(self, discharge, water_level):
        """データ履歴の更新"""
        self.discharge_history.append(discharge)
        self.water_level_history.append(water_level)
    
    def get_state(self):
        """現在の状態を判定（安定/準変動/変動）"""
        if len(self.discharge_history) < 12:  # 最低2時間分のデータが必要
            return "unknown"
        
        recent_discharge = list(self.discharge_history)
        mean_q = np.mean(recent_discharge)
        std_q = np.std(recent_discharge)
        
        if mean_q > 0:
            cv = std_q / mean_q
            if cv < self.config["stability_thresholds"]["stable"]:
                return "stable"
            elif cv < self.config["stability_thresholds"]["semi_variable"]:
                return "semi_variable"
            else:
                return "variable"
        return "unknown"
    
    def get_discharge_range(self, discharge):
        """放流量レンジを取得"""
        if discharge < 300:
            return "150-300"
        elif discharge < 500:
            return "300-500"
        elif discharge < 800:
            return "500-800" if "500-800" in self.config["delay_params"] else "500+"
        elif discharge < 1000:
            return "800-1000" if "800-1000" in self.config["delay_params"] else "500+"
        else:
            return "1000+" if "1000+" in self.config["delay_params"] else "500+"
    
    def get_water_level_category(self, water_level):
        """水位レベルカテゴリを取得"""
        if water_level < 3.5:
            return "low"
        elif water_level < 4.5:
            return "medium"
        else:
            return "high"
    
    def calculate_delay(self, discharge, direction):
        """動的遅延時間の計算"""
        discharge_range = self.get_discharge_range(discharge)
        base_delay = self.config["delay_params"][discharge_range]["base_delay"]
        
        # 変化方向による補正
        if direction == "increase":
            delay = base_delay * 0.9
        elif direction == "decrease":
            delay = base_delay * 1.1
        else:
            delay = base_delay
        
        return int(delay / self.config["time_step"])  # ステップ数に変換
    
    def get_response_rate(self, discharge, direction):
        """応答率の取得"""
        discharge_range = self.get_discharge_range(discharge)
        
        # response_ratesのキーにマッピング
        response_key = discharge_range
        if discharge_range in ["800-1000", "1000+"]:
            # 800-1000と1000+は800+にマッピング
            response_key = "800+" if "800+" in self.config["response_rates"]["increase"] else discharge_range
        elif discharge_range == "500-800":
            # 500-800がない場合は500+を使用
            if discharge_range not in self.config["response_rates"]["increase"] and "500+" in self.config["response_rates"]["increase"]:
                response_key = "500+"
        
        if direction in ["increase", "decrease"]:
            base_rate = self.config["response_rates"][direction].get(response_key, 0.003)  # デフォルト値
        else:
            # 安定状態の場合は平均値
            inc_rate = self.config["response_rates"]["increase"].get(response_key, 0.003)
            dec_rate = self.config["response_rates"]["decrease"].get(response_key, 0.003)
            base_rate = (inc_rate + dec_rate) / 2
        
        # オンライン学習による補正
        return base_rate * self.online_correction_factor
    
    def predict(self, current_discharge, current_water_level, future_discharge_plan=None):
        """
        水位予測
        
        Parameters:
        -----------
        current_discharge : float
            現在の放流量 (m³/s)
        current_water_level : float
            現在の水位 (m)
        future_discharge_plan : list of float, optional
            将来の放流量計画（3時間分、10分刻み）
        
        Returns:
        --------
        predictions : dict
            予測結果
        """
        # 最小放流量チェック
        if current_discharge < self.config["min_discharge"]:
            return {
                "status": "error",
                "message": f"放流量が{self.config['min_discharge']}m³/s未満です"
            }
        
        # 履歴更新
        self.update_history(current_discharge, current_water_level)
        
        # 状態判定
        state = self.get_state()
        
        # 予測ステップ数
        prediction_steps = int(self.config["prediction_hours"] * 60 / self.config["time_step"])
        
        # 将来の放流量計画がない場合は現在値を維持
        if future_discharge_plan is None:
            future_discharge_plan = [current_discharge] * prediction_steps
        
        # 予測実行
        predictions = {
            "status": "success",
            "state": state,
            "timestamp": datetime.now().isoformat(),
            "water_levels": [],
            "confidence": [],
            "time_labels": []
        }
        
        # 初期値
        predicted_levels = [current_water_level]
        
        for step in range(prediction_steps):
            # 時刻ラベル
            time_label = f"+{(step + 1) * self.config['time_step']}分"
            predictions["time_labels"].append(time_label)
            
            # 放流量の変化を計算
            if step < len(future_discharge_plan):
                next_discharge = future_discharge_plan[step]
            else:
                next_discharge = current_discharge
            
            if step > 0:
                prev_discharge = future_discharge_plan[step - 1]
            else:
                prev_discharge = current_discharge
            
            delta_q = next_discharge - prev_discharge
            
            # 変化方向
            if delta_q > 5:
                direction = "increase"
            elif delta_q < -5:
                direction = "decrease"
            else:
                direction = "stable"
            
            # 遅延時間を考慮
            delay_steps = self.calculate_delay(prev_discharge, direction)
            
            # 遅延を考慮した過去の放流量変化を取得
            if step >= delay_steps:
                delayed_discharge = future_discharge_plan[step - delay_steps]
                if step - delay_steps > 0:
                    prev_delayed_discharge = future_discharge_plan[step - delay_steps - 1]
                else:
                    prev_delayed_discharge = current_discharge
                effective_delta_q = delayed_discharge - prev_delayed_discharge
            else:
                # 履歴から取得
                if len(self.discharge_history) >= delay_steps:
                    delayed_discharge = self.discharge_history[-delay_steps]
                    if len(self.discharge_history) >= delay_steps + 1:
                        prev_delayed_discharge = self.discharge_history[-delay_steps - 1]
                    else:
                        prev_delayed_discharge = delayed_discharge
                    effective_delta_q = delayed_discharge - prev_delayed_discharge
                else:
                    effective_delta_q = 0
            
            # 応答率を取得
            response_rate = self.get_response_rate(next_discharge, direction)
            
            # 水位変化を計算
            delta_h = response_rate * effective_delta_q
            
            # 初期水位レベルによる補正
            water_level_category = self.get_water_level_category(predicted_levels[-1])
            level_correction = self.config["water_level_correction"][water_level_category]
            delta_h *= level_correction
            
            # ヒステリシス補正
            if direction in ["increase", "decrease"]:
                hysteresis_correction = self.config["hysteresis_correction"][direction]
                delta_h *= hysteresis_correction
            
            # 予測水位
            predicted_level = predicted_levels[-1] + delta_h
            predicted_levels.append(predicted_level)
            predictions["water_levels"].append(round(predicted_level, 3))
            
            # 信頼度計算（時間経過とともに低下）
            if step < 3:  # 30分以内
                confidence = 0.9
            elif step < 9:  # 90分以内
                confidence = 0.8 - (step - 3) * 0.02
            else:  # 90分以降
                confidence = 0.7 - (step - 9) * 0.01
            
            # 状態による信頼度調整
            if state == "stable":
                confidence *= 1.0
            elif state == "semi_variable":
                confidence *= 0.9
            else:
                confidence *= 0.8
            
            predictions["confidence"].append(round(confidence, 2))
        
        return predictions
    
    def update_online_learning(self, predicted_level, actual_level):
        """
        オンライン学習によるパラメータ更新
        
        Parameters:
        -----------
        predicted_level : float
            予測水位
        actual_level : float
            実測水位
        """
        error = actual_level - predicted_level
        self.error_history.append(error)
        
        # 誤差が5cm以上の場合、補正係数を調整
        if abs(error) > 0.05:
            if actual_level > 0:
                adjustment = 1 + 0.1 * error / actual_level
                self.online_correction_factor *= adjustment
                
                # 補正係数の範囲制限
                self.online_correction_factor = max(0.8, min(1.2, self.online_correction_factor))
        
        # 30回分の誤差が貯まったら統計を計算
        if len(self.error_history) == 30:
            mean_error = np.mean(self.error_history)
            std_error = np.std(self.error_history)
            
            # 系統的な誤差がある場合は補正
            if abs(mean_error) > 0.03:  # 3cm以上の平均誤差
                self.online_correction_factor *= (1 + mean_error * 0.5)
                self.online_correction_factor = max(0.8, min(1.2, self.online_correction_factor))
    
    def get_prediction_summary(self, predictions):
        """予測結果のサマリーを生成"""
        if predictions["status"] != "success":
            return predictions["message"]
        
        summary = f"""
予測結果サマリー
================
予測時刻: {predictions['timestamp']}
システム状態: {predictions['state']}
オンライン補正係数: {self.online_correction_factor:.3f}

予測水位:
  30分後: {predictions['water_levels'][2]:.2f}m (信頼度: {predictions['confidence'][2]:.0%})
  60分後: {predictions['water_levels'][5]:.2f}m (信頼度: {predictions['confidence'][5]:.0%})
  90分後: {predictions['water_levels'][8]:.2f}m (信頼度: {predictions['confidence'][8]:.0%})
  120分後: {predictions['water_levels'][11]:.2f}m (信頼度: {predictions['confidence'][11]:.0%})
  180分後: {predictions['water_levels'][17]:.2f}m (信頼度: {predictions['confidence'][17]:.0%})
"""
        return summary


def main():
    """デモンストレーション"""
    # モデル初期化
    model = RealtimePredictionModel()
    
    # サンプルデータで予測
    print("=== リアルタイム水位予測モデル デモ ===\n")
    
    # 現在の状態
    current_discharge = 250.0  # m³/s
    current_water_level = 3.8  # m
    
    # 将来の放流量計画（増加→維持→減少）
    future_plan = (
        [250] * 6 +      # 1時間維持
        [300] * 6 +      # 1時間で300に増加
        [250] * 6        # 1時間で250に減少
    )
    
    # 予測実行
    predictions = model.predict(current_discharge, current_water_level, future_plan)
    
    # 結果表示
    print(model.get_prediction_summary(predictions))


if __name__ == "__main__":
    main()