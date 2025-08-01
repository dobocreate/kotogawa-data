#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
特徴量重要度エラーの分析スクリプト
"""

import pickle
import numpy as np
import pandas as pd
import glob

def analyze_model_features():
    """保存されたモデルの特徴量を分析"""
    
    # 最新のモデルファイルを探す
    model_files = glob.glob("water_level_predictor_*.pkl")
    if not model_files:
        print("モデルファイルが見つかりません")
        return
    
    model_files.sort()
    model_path = model_files[-1]
    print(f"分析するモデル: {model_path}")
    
    # モデルを読み込む
    with open(model_path, 'rb') as f:
        model_data = pickle.load(f)
    
    models = model_data['model']
    feature_names = model_data['feature_names']
    
    print(f"\n全特徴量数: {len(feature_names)}")
    print(f"モデル数: {len(models)}")
    
    # 1. 各モデルの重要度の形状を確認
    print("\n=== 1. 各モデルの特徴量重要度の形状 ===")
    for i, model in enumerate(models):
        try:
            # gain タイプ
            importance_gain = model.feature_importance(importance_type='gain')
            # split タイプ
            importance_split = model.feature_importance(importance_type='split')
            
            print(f"\nモデル {(i+1)*10}分先:")
            print(f"  gain重要度の長さ: {len(importance_gain)}")
            print(f"  split重要度の長さ: {len(importance_split)}")
            print(f"  非ゼロ要素数(gain): {np.count_nonzero(importance_gain)}")
            print(f"  非ゼロ要素数(split): {np.count_nonzero(importance_split)}")
            
            # 使用された特徴量を確認
            if hasattr(model, 'feature_name'):
                used_features = model.feature_name()
                print(f"  使用特徴量数: {len(used_features)}")
            
        except Exception as e:
            print(f"  エラー: {e}")
    
    # 2. 実際に使用された特徴量の確認
    print("\n=== 2. 実際に使用された特徴量 ===")
    all_used_features = set()
    feature_usage_count = {feature: 0 for feature in feature_names}
    
    for i, model in enumerate(models):
        try:
            importance_split = model.feature_importance(importance_type='split')
            
            # 重要度が0より大きい特徴量をカウント
            for j, imp in enumerate(importance_split):
                if imp > 0:
                    if j < len(feature_names):
                        feature_usage_count[feature_names[j]] += 1
                        all_used_features.add(feature_names[j])
        except:
            pass
    
    print(f"\n全モデルで使用された特徴量数: {len(all_used_features)}")
    print(f"全く使用されなかった特徴量数: {len(feature_names) - len(all_used_features)}")
    
    # 3. 特徴量使用頻度のトップ10
    print("\n=== 3. 特徴量使用頻度トップ10 ===")
    usage_df = pd.DataFrame(list(feature_usage_count.items()), 
                           columns=['feature', 'usage_count'])
    usage_df = usage_df.sort_values('usage_count', ascending=False)
    
    for idx, row in usage_df.head(10).iterrows():
        print(f"{row['feature']:40s}: {row['usage_count']:2d} / {len(models)} モデル")
    
    # 4. 全く使用されなかった特徴量
    print("\n=== 4. 全く使用されなかった特徴量 ===")
    unused_features = usage_df[usage_df['usage_count'] == 0]['feature'].tolist()
    if unused_features:
        for feature in unused_features[:10]:  # 最初の10個だけ表示
            print(f"  - {feature}")
        if len(unused_features) > 10:
            print(f"  ... 他 {len(unused_features) - 10} 個")
    else:
        print("  なし（すべての特徴量が少なくとも1つのモデルで使用）")
    
    # 5. 重要度集計の修正方法の提案
    print("\n=== 5. 修正方法の提案 ===")
    
    # 方法1: splitタイプを使用
    print("\n方法1: split（分岐回数）タイプの重要度を使用")
    importance_sum_split = np.zeros(len(feature_names))
    valid_models_split = 0
    
    for model in models:
        try:
            importance = model.feature_importance(importance_type='split')
            if len(importance) == len(feature_names):
                importance_sum_split += importance
                valid_models_split += 1
        except:
            pass
    
    if valid_models_split > 0:
        importance_avg_split = importance_sum_split / valid_models_split
        print(f"  成功: {valid_models_split}/{len(models)} モデルで集計可能")
    
    # 方法2: 個別集計
    print("\n方法2: 各モデルの重要度を個別に保存")
    individual_importance = {}
    
    for i, model in enumerate(models):
        try:
            importance_gain = model.feature_importance(importance_type='gain')
            importance_split = model.feature_importance(importance_type='split')
            
            if len(importance_gain) > 0 or len(importance_split) > 0:
                individual_importance[f'{(i+1)*10}min'] = {
                    'gain': importance_gain if len(importance_gain) > 0 else None,
                    'split': importance_split if len(importance_split) > 0 else None
                }
        except:
            pass
    
    print(f"  成功: {len(individual_importance)}/{len(models)} モデルの重要度を取得")
    
    # 6. 推奨される修正方法
    print("\n=== 6. 推奨される修正方法 ===")
    print("1. importance_type='split' を使用（分岐回数ベース）")
    print("2. 各モデルの重要度を個別に処理")
    print("3. 空の重要度を持つモデルをスキップ")
    print("4. 時刻別の重要度変化を可視化")
    
    return {
        'feature_names': feature_names,
        'models': models,
        'usage_count': feature_usage_count,
        'individual_importance': individual_importance
    }


if __name__ == "__main__":
    analyze_model_features()