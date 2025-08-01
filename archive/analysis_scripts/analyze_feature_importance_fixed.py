#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
特徴量重要度の修正版分析関数
"""

import numpy as np
import pandas as pd


def analyze_feature_importance_fixed(predictor):
    """特徴量の重要度分析（修正版）"""
    if predictor.model_type != 'lightgbm' or not predictor.is_fitted:
        print("LightGBMモデルが学習されていません")
        return None
    
    print("\n=== 特徴量重要度分析（修正版） ===")
    
    # 1. 各モデルの特徴量重要度を個別に確認
    print("\n1. 各予測時刻での重要度取得状況:")
    model_importance_data = {}
    
    for i, model in enumerate(predictor.model):
        time_ahead = (i + 1) * 10
        try:
            # gainとsplitの両方を試す
            importance_gain = model.feature_importance(importance_type='gain')
            importance_split = model.feature_importance(importance_type='split')
            
            print(f"  {time_ahead}分先: gain={len(importance_gain)}要素, split={len(importance_split)}要素")
            
            model_importance_data[time_ahead] = {
                'gain': importance_gain,
                'split': importance_split,
                'gain_nonzero': np.count_nonzero(importance_gain),
                'split_nonzero': np.count_nonzero(importance_split)
            }
        except Exception as e:
            print(f"  {time_ahead}分先: エラー - {e}")
            model_importance_data[time_ahead] = None
    
    # 2. splitタイプでの集計（より安定）
    print("\n2. split（分岐回数）タイプでの重要度集計:")
    importance_sum_split = np.zeros(len(predictor.feature_names))
    valid_models_split = 0
    
    for i, model in enumerate(predictor.model):
        try:
            importance = model.feature_importance(importance_type='split')
            if len(importance) == len(predictor.feature_names):
                importance_sum_split += importance
                valid_models_split += 1
        except:
            continue
    
    if valid_models_split > 0:
        importance_avg_split = importance_sum_split / valid_models_split
        
        # 特徴量重要度のDataFrame作成
        feature_importance_split = pd.DataFrame({
            'feature': predictor.feature_names,
            'importance': importance_avg_split
        }).sort_values('importance', ascending=False)
        
        print(f"  集計成功: {valid_models_split}/{len(predictor.model)} モデル")
        print("\n  Top 10 重要な特徴量（split）:")
        for idx, row in feature_importance_split.head(10).iterrows():
            if row['importance'] > 0:
                print(f"  {row['feature']:40s}: {row['importance']:8.0f}")
    
    # 3. gainタイプでの集計（情報利得ベース）
    print("\n3. gain（情報利得）タイプでの重要度集計:")
    
    # 各モデルで個別に正規化してから集計
    normalized_importances = []
    valid_models_gain = 0
    
    for i, model in enumerate(predictor.model):
        try:
            importance = model.feature_importance(importance_type='gain')
            if len(importance) > 0 and np.sum(importance) > 0:
                # 各モデル内で正規化
                if len(importance) == len(predictor.feature_names):
                    normalized_imp = importance / np.sum(importance)
                    normalized_importances.append(normalized_imp)
                    valid_models_gain += 1
        except:
            continue
    
    if valid_models_gain > 0:
        # 正規化された重要度の平均
        importance_avg_gain = np.mean(normalized_importances, axis=0)
        
        feature_importance_gain = pd.DataFrame({
            'feature': predictor.feature_names,
            'importance': importance_avg_gain * 100  # パーセンテージ表示
        }).sort_values('importance', ascending=False)
        
        print(f"  集計成功: {valid_models_gain}/{len(predictor.model)} モデル")
        print("\n  Top 10 重要な特徴量（gain, %）:")
        for idx, row in feature_importance_gain.head(10).iterrows():
            if row['importance'] > 0:
                print(f"  {row['feature']:40s}: {row['importance']:6.2f}%")
    
    # 4. 時刻別の最重要特徴量
    print("\n4. 予測時刻別の最重要特徴量:")
    for i, model in enumerate(predictor.model[:6]):  # 最初の60分まで表示
        time_ahead = (i + 1) * 10
        try:
            importance = model.feature_importance(importance_type='gain')
            if len(importance) == len(predictor.feature_names) and np.sum(importance) > 0:
                top_idx = np.argmax(importance)
                top_feature = predictor.feature_names[top_idx]
                top_value = importance[top_idx] / np.sum(importance) * 100
                print(f"  {time_ahead}分先: {top_feature} ({top_value:.1f}%)")
        except:
            print(f"  {time_ahead}分先: 取得できません")
    
    # 5. カテゴリ別の重要度集計
    print("\n5. 特徴量カテゴリ別の重要度:")
    if valid_models_split > 0:
        category_importance = {
            '現在値': 0,
            '過去の放流量': 0,
            '過去の水位': 0,
            '統計量': 0,
            'その他': 0
        }
        
        for feature, importance in zip(predictor.feature_names, importance_avg_split):
            if 'current' in feature or '現在' in feature:
                category_importance['現在値'] += importance
            elif 'discharge_lag' in feature:
                category_importance['過去の放流量'] += importance
            elif 'water_level_lag' in feature:
                category_importance['過去の水位'] += importance
            elif 'mean' in feature or 'std' in feature or 'trend' in feature:
                category_importance['統計量'] += importance
            else:
                category_importance['その他'] += importance
        
        total_importance = sum(category_importance.values())
        if total_importance > 0:
            for category, imp in sorted(category_importance.items(), key=lambda x: x[1], reverse=True):
                percentage = imp / total_importance * 100
                print(f"  {category:15s}: {percentage:5.1f}%")
    
    return {
        'model_data': model_importance_data,
        'split_importance': feature_importance_split if valid_models_split > 0 else None,
        'gain_importance': feature_importance_gain if valid_models_gain > 0 else None
    }