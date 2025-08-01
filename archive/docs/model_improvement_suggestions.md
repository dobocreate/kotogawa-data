# 水位予測モデルの改善提案

## 現状の課題
- 10分後から約1.3mの予測誤差が発生
- 急激な水位上昇（4.5m→7.5m）を過小評価
- 予測が保守的で、変化を十分に捉えられていない

## 誤差の原因分析

### 1. モデルの学習データの偏り
- 極端な放流量変化（400→1200m³/s）のケースが少ない
- 急激な水位上昇のパターンが不足している可能性

### 2. 特徴量の問題
- 変化率の特徴量が十分に重視されていない
- 遅延時間の計算式が極端なケースに対応できていない

### 3. モデルアーキテクチャ
- LightGBMが保守的な予測を行う傾向
- 極端な値を予測しにくい

## 改善提案

### 1. データの前処理
```python
# 極端なケースの重み付けを増やす
def create_sample_weights(df):
    # 放流量の変化率を計算
    df['discharge_change_rate'] = df['ダム_全放流量'].pct_change().abs()
    
    # 変化率が大きいサンプルに高い重みを付ける
    weights = 1 + df['discharge_change_rate'] * 10
    return weights
```

### 2. 特徴量エンジニアリング
```python
# より動的な特徴量を追加
def add_dynamic_features(df):
    # 加速度的な変化を捉える特徴量
    df['discharge_acceleration'] = df['ダム_全放流量'].diff().diff()
    df['water_level_acceleration'] = df['水位_水位'].diff().diff()
    
    # 変化の持続性を表す特徴量
    df['consecutive_increase'] = (df['ダム_全放流量'].diff() > 0).cumsum()
    
    # 極端値フラグ
    df['extreme_discharge'] = df['ダム_全放流量'] > df['ダム_全放流量'].quantile(0.95)
    
    return df
```

### 3. モデルのハイパーパラメータ調整
```python
# より積極的な予測を行うパラメータ
lgb_params = {
    'objective': 'regression',
    'metric': 'rmse',
    'boosting_type': 'gbdt',
    'num_leaves': 63,  # 増やして複雑なパターンを学習
    'learning_rate': 0.05,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'min_child_samples': 10,  # 減らして極端なケースも学習
    'lambda_l1': 0.1,  # 正則化を弱める
    'lambda_l2': 0.1
}
```

### 4. アンサンブル手法
```python
# 保守的なモデルと積極的なモデルの組み合わせ
class EnsemblePredictor:
    def __init__(self):
        self.conservative_model = LightGBMModel(conservative_params)
        self.aggressive_model = LightGBMModel(aggressive_params)
        self.extreme_model = ExtremeGradientBoostingModel()
    
    def predict(self, X):
        # 状況に応じて重み付けを変える
        if self.detect_extreme_situation(X):
            weights = [0.2, 0.3, 0.5]  # 極端なモデルを重視
        else:
            weights = [0.5, 0.3, 0.2]  # 通常は保守的
        
        predictions = []
        for model, weight in zip(self.models, weights):
            predictions.append(model.predict(X) * weight)
        
        return sum(predictions)
```

### 5. 後処理による補正
```python
def adjust_predictions(predictions, features):
    # 変化率に基づく補正
    change_rate = features['discharge_change_rate']
    
    # 急激な変化の場合、予測を増幅
    if change_rate > 0.5:  # 50%以上の変化
        adjustment_factor = 1 + (change_rate - 0.5) * 0.5
        predictions = predictions * adjustment_factor
    
    return predictions
```

### 6. オンライン学習の強化
```python
def enhanced_online_update(self, X_new, y_new):
    # 予測誤差が大きいサンプルを重点的に学習
    predictions = self.predict(X_new)
    errors = abs(predictions - y_new)
    
    # 誤差が大きいサンプルに高い重みを付ける
    weights = 1 + errors / errors.mean()
    
    # 重み付き更新
    self.model.update(X_new, y_new, sample_weight=weights)
```

## 実装優先順位
1. **特徴量の追加**（実装が簡単で効果が期待できる）
2. **サンプル重み付け**（極端なケースの学習を強化）
3. **ハイパーパラメータ調整**（既存モデルの改善）
4. **後処理による補正**（即効性がある）
5. **アンサンブル手法**（より高度な改善）

## 検証方法
1. 極端な放流量変化を含む期間でのテスト
2. 交差検証での誤差分布の確認
3. 予測の過小評価率の測定
4. リアルタイムでの性能モニタリング