#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
学習済み設定ファイルの完全性を確保するスクリプト
"""

import json
import os

def ensure_complete_config(config_path="core/configs/learned_config.json"):
    """
    設定ファイルに必要なすべてのパラメータが含まれているか確認し、
    不足している場合はデフォルト値を追加
    """
    # デフォルト値の定義
    default_config = {
        "min_discharge": 150.0,
        "prediction_hours": 3,
        "time_step": 10,
        "history_hours": 2,
        "delay_params": {
            "150-300": {"base_delay": 25, "correlation": 0.824},
            "300-500": {"base_delay": 15, "correlation": 0.710},
            "500+": {"base_delay": 10, "correlation": 0.636}
        },
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
        "hysteresis_correction": {
            "increase": 1.02,
            "decrease": 0.98
        },
        "water_level_correction": {
            "low": 1.2,
            "medium": 1.0,
            "high": 0.9
        },
        "stability_thresholds": {
            "stable": 0.05,
            "semi_variable": 0.15
        }
    }
    
    # 既存の設定を読み込み
    if os.path.exists(config_path):
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        print(f"既存の設定ファイルを読み込みました: {config_path}")
    else:
        config = {}
        print("設定ファイルが見つかりません。新規作成します。")
    
    # 修正が必要かどうかのフラグ
    modified = False
    
    # 基本パラメータの確認
    for key in ["min_discharge", "prediction_hours", "time_step", "history_hours"]:
        if key not in config:
            config[key] = default_config[key]
            modified = True
    
    # delay_paramsの確認
    if "delay_params" not in config:
        config["delay_params"] = {}
    
    for range_name in ["150-300", "300-500", "500+"]:
        if range_name not in config["delay_params"]:
            config["delay_params"][range_name] = default_config["delay_params"][range_name]
            print(f"delay_params['{range_name}']を追加しました（デフォルト値）")
            modified = True
    
    # response_ratesの確認
    if "response_rates" not in config:
        config["response_rates"] = {"increase": {}, "decrease": {}}
    
    for direction in ["increase", "decrease"]:
        if direction not in config["response_rates"]:
            config["response_rates"][direction] = {}
        
        for range_name in ["150-300", "300-500", "500+"]:
            if range_name not in config["response_rates"][direction]:
                config["response_rates"][direction][range_name] = default_config["response_rates"][direction][range_name]
                print(f"response_rates['{direction}']['{range_name}']を追加しました（デフォルト値）")
                modified = True
    
    # hysteresis_correctionの確認
    if "hysteresis_correction" not in config:
        config["hysteresis_correction"] = default_config["hysteresis_correction"]
        modified = True
    else:
        for key in ["increase", "decrease"]:
            if key not in config["hysteresis_correction"]:
                config["hysteresis_correction"][key] = default_config["hysteresis_correction"][key]
                modified = True
    
    # water_level_correctionの確認
    if "water_level_correction" not in config:
        config["water_level_correction"] = default_config["water_level_correction"]
        modified = True
    else:
        for level in ["low", "medium", "high"]:
            if level not in config["water_level_correction"]:
                config["water_level_correction"][level] = default_config["water_level_correction"][level]
                modified = True
    
    # stability_thresholdsの確認
    if "stability_thresholds" not in config:
        config["stability_thresholds"] = default_config["stability_thresholds"]
        modified = True
    
    # 修正された場合は保存
    if modified:
        # バックアップを作成
        if os.path.exists(config_path):
            backup_path = config_path.replace('.json', '_backup.json')
            with open(backup_path, 'w', encoding='utf-8') as f:
                with open(config_path, 'r', encoding='utf-8') as original:
                    f.write(original.read())
            print(f"元のファイルをバックアップしました: {backup_path}")
        
        # 修正した設定を保存
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=4, ensure_ascii=False)
        print(f"\n修正した設定を保存しました: {config_path}")
    else:
        print("\n設定ファイルは完全です。修正は不要でした。")
    
    return config


def main():
    """メイン処理"""
    print("=== 設定ファイルの完全性確認 ===\n")
    
    # learned_config.jsonを確認・修正
    config = ensure_complete_config("core/configs/learned_config.json")
    
    print("\n=== 最終的な設定内容 ===")
    print(json.dumps(config, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()