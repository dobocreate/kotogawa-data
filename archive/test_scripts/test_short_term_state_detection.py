#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
前後10分での状態判定ロジックのテスト
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def identify_discharge_state_short_term(discharge_past_10min, discharge_current, discharge_future_10min=None):
    """
    前後10分での放流量状態を識別
    
    Parameters:
    -----------
    discharge_past_10min : float
        10分前の放流量
    discharge_current : float
        現在の放流量
    discharge_future_10min : float, optional
        10分後の予測放流量（予測時に使用）
    
    Returns:
    --------
    state : int
        -1: 減少中, 0: 定常, 1: 増加中
    rate : float
        変化率（m³/s per 10min）
    trend : str
        変化の傾向（'accelerating', 'steady', 'decelerating'）
    """
    # 過去から現在への変化
    past_change = discharge_current - discharge_past_10min
    
    # 状態判定のしきい値（より敏感に）
    threshold = 5  # 10分で5 m³/s以上の変化
    
    # 基本的な状態判定
    if past_change > threshold:
        state = 1  # 増加中
    elif past_change < -threshold:
        state = -1  # 減少中
    else:
        state = 0  # 定常
    
    # 将来予測を含む場合のトレンド判定
    trend = 'steady'
    if discharge_future_10min is not None:
        future_change = discharge_future_10min - discharge_current
        
        # 加速/減速の判定
        if state == 1:  # 増加中
            if future_change > past_change * 1.2:
                trend = 'accelerating'
            elif future_change < past_change * 0.8:
                trend = 'decelerating'
        elif state == -1:  # 減少中
            if future_change < past_change * 1.2:  # より負の値
                trend = 'accelerating'
            elif future_change > past_change * 0.8:  # 減速
                trend = 'decelerating'
    
    return state, past_change, trend

def test_scenarios():
    """各種シナリオでのテスト"""
    print("=== 前後10分での状態判定テスト ===\n")
    
    scenarios = [
        # (10分前, 現在, 10分後, 説明)
        (400, 410, 420, "安定した増加"),
        (400, 410, 415, "増加が減速"),
        (400, 410, 430, "増加が加速"),
        (400, 405, 410, "緩やかな増加"),
        (400, 402, 404, "微増（定常判定）"),
        (400, 390, 380, "安定した減少"),
        (400, 390, 385, "減少が減速"),
        (400, 390, 370, "減少が加速"),
        (400, 395, 390, "緩やかな減少"),
        (400, 398, 396, "微減（定常判定）"),
        (400, 400, 400, "完全に定常"),
        (400, 405, 400, "増加から減少へ転換"),
        (400, 395, 400, "減少から増加へ転換"),
    ]
    
    for past, current, future, description in scenarios:
        state, rate, trend = identify_discharge_state_short_term(past, current, future)
        state_str = ['減少', '定常', '増加'][state + 1]
        
        print(f"{description}:")
        print(f"  {past} → {current} → {future} m³/s")
        print(f"  状態: {state_str}, 変化率: {rate:+.1f} m³/s/10min, トレンド: {trend}")
        print()

def get_dynamic_delay_short_term(state, trend, rainfall_current, rainfall_change_10min):
    """
    短期的な状態変化に基づく動的遅延時間
    
    Parameters:
    -----------
    state : int
        現在の状態（-1: 減少, 0: 定常, 1: 増加）
    trend : str
        変化トレンド（'accelerating', 'steady', 'decelerating'）
    rainfall_current : float
        現在の降雨強度
    rainfall_change_10min : float
        10分間の降雨変化量
    
    Returns:
    --------
    delay_minutes : int
        適用する遅延時間（分）
    """
    # 降雨の急増を検出（10分で5mm/h以上の増加）
    rain_surge = rainfall_change_10min > 5
    
    # パラメータ
    delay_onset = 0      # 増加開始時
    delay_increase = 120  # 増加継続時
    delay_decrease = 60   # 減少時
    delay_transition = 30 # 状態遷移時
    
    # 遅延時間の決定
    if rain_surge:
        # 降雨急増時は即座に反応
        return delay_onset
    elif state == 1:  # 増加中
        if trend == 'decelerating':
            # 増加が減速中 → 遷移の可能性
            return delay_transition
        else:
            # 安定した増加
            return delay_increase
    elif state == -1:  # 減少中
        if trend == 'decelerating' and rainfall_current > 10:
            # 減少が減速中で降雨あり → 遷移の可能性
            return delay_transition
        else:
            # 安定した減少
            return delay_decrease
    else:  # 定常
        if abs(rainfall_change_10min) > 3:
            # 降雨変化あり → 短めの遅延
            return delay_transition
        else:
            # 安定状態
            return delay_decrease
    
    return delay_decrease

def demonstrate_delay_logic():
    """遅延時間決定ロジックのデモ"""
    print("\n=== 動的遅延時間の決定例 ===\n")
    
    cases = [
        # (状態, トレンド, 現在降雨, 降雨変化, 説明)
        (1, 'steady', 30, 2, "安定した増加・安定降雨"),
        (1, 'decelerating', 15, -5, "増加減速中・降雨減少"),
        (0, 'steady', 5, 8, "定常状態・降雨急増"),
        (-1, 'decelerating', 12, 3, "減少減速中・降雨あり"),
        (-1, 'steady', 2, -1, "安定した減少・低降雨"),
    ]
    
    for state, trend, rain_current, rain_change, description in cases:
        delay = get_dynamic_delay_short_term(state, trend, rain_current, rain_change)
        state_str = ['減少', '定常', '増加'][state + 1]
        
        print(f"{description}:")
        print(f"  状態: {state_str}, トレンド: {trend}")
        print(f"  降雨: {rain_current} mm/h (変化: {rain_change:+.1f} mm/h/10min)")
        print(f"  → 遅延時間: {delay}分")
        print()

if __name__ == "__main__":
    test_scenarios()
    demonstrate_delay_logic()