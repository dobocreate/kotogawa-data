# リアルタイム水位予測システム使用ガイド

## システム概要

本システムは、琴川の放流量データからリアルタイムで3時間先までの水位を予測します。機械学習とオンライン適応機能を組み合わせた高精度な予測を実現しています。

## 初期セットアップ

### 1. 初期学習の実行（推奨）

新しい環境でシステムを使用する場合、過去データを使用した初期学習を行うことを強く推奨します：

```bash
# 初期学習の実行
python initial_training.py
```

これにより：
- 過去データから最適なパラメータを自動学習
- `learned_config.json`として保存
- 学習結果の可視化

学習済み設定でモデルを初期化：
```python
model = RealtimePredictionModel("learned_config.json")
```

### 2. デフォルト設定の使用

初期学習を行わない場合は、分析済みのデフォルト設定を使用：
```python
model = RealtimePredictionModel("prediction_config.json")
```

## クイックスタート

### 1. 必要な環境

```bash
# Python 3.8以上
# 必要なライブラリのインストール
pip install -r requirements.txt
```

### 2. 基本的な使用方法

```python
from realtime_prediction_model import RealtimePredictionModel

# モデルの初期化
model = RealtimePredictionModel("prediction_config.json")

# 予測の実行
predictions = model.predict(
    current_discharge=250.0,     # 現在の放流量 (m³/s)
    current_water_level=3.8,     # 現在の水位 (m)
    future_discharge_plan=None   # 将来の放流計画（省略可）
)

# 結果の表示
print(model.get_prediction_summary(predictions))
```

## 詳細な使用方法

### 1. モデルの初期化と設定

```python
# デフォルト設定で初期化
model = RealtimePredictionModel()

# カスタム設定ファイルで初期化
model = RealtimePredictionModel("custom_config.json")
```

### 2. 履歴データの設定

予測精度向上のため、過去2時間分のデータを設定することを推奨：

```python
# 過去のデータを順次追加
for discharge, water_level in historical_data:
    model.update_history(discharge, water_level)
```

### 3. 将来の放流量計画の指定

```python
# 3時間分の放流量計画（10分刻みで18個）
future_discharge = [
    250, 250, 250,  # 30分維持
    300, 350, 400,  # 30分で増加
    400, 400, 400,  # 30分維持
    350, 300, 250,  # 30分で減少
    250, 250, 250,  # 30分維持
    250, 250, 250   # 30分維持
]

predictions = model.predict(
    current_discharge=250.0,
    current_water_level=3.8,
    future_discharge_plan=future_discharge
)
```

### 4. 予測結果の解釈

```python
if predictions["status"] == "success":
    # 状態情報
    state = predictions["state"]  # "stable", "semi_variable", "variable"
    
    # 各時点の予測水位（10分刻み）
    water_levels = predictions["water_levels"]  # 18個の予測値
    
    # 信頼度（0-1の値）
    confidence = predictions["confidence"]  # 各予測の信頼度
    
    # 主要時点の予測値
    h_30min = predictions["water_levels"][2]   # 30分後
    h_60min = predictions["water_levels"][5]   # 60分後
    h_90min = predictions["water_levels"][8]   # 90分後
    h_120min = predictions["water_levels"][11] # 120分後
    h_180min = predictions["water_levels"][17] # 180分後
```

### 5. オンライン学習

実測値が得られたら、モデルを更新して精度を向上：

```python
# 60分前の予測と実測値でモデルを更新
model.update_online_learning(
    predicted_level=predicted_60min,
    actual_level=actual_60min
)
```

## 実運用での実装例

### リアルタイム予測ループ

```python
import time
from datetime import datetime

def realtime_prediction_loop():
    model = RealtimePredictionModel("prediction_config.json")
    
    while True:
        try:
            # 現在のデータを取得（実際のデータソースから）
            current_data = get_current_data()  # 実装が必要
            future_plan = get_discharge_plan() # 実装が必要
            
            # 予測実行
            predictions = model.predict(
                current_discharge=current_data['discharge'],
                current_water_level=current_data['water_level'],
                future_discharge_plan=future_plan
            )
            
            if predictions["status"] == "success":
                # 予測結果を保存/送信
                save_predictions(predictions)  # 実装が必要
                
                # ログ出力
                print(f"[{datetime.now()}] 予測完了")
                print(f"  30分後: {predictions['water_levels'][2]:.2f}m")
                print(f"  60分後: {predictions['water_levels'][5]:.2f}m")
                print(f"  90分後: {predictions['water_levels'][8]:.2f}m")
            
            # 履歴更新
            model.update_history(
                current_data['discharge'],
                current_data['water_level']
            )
            
            # 10分待機
            time.sleep(600)
            
        except Exception as e:
            print(f"エラー発生: {e}")
            time.sleep(60)  # エラー時は1分後に再試行
```

### 過去データでの検証

```python
from validate_prediction import PredictionValidator

# 検証の実行
validator = PredictionValidator(
    data_path="historical_data.csv",
    config_path="prediction_config.json"
)

# 10期間、各24時間で検証
validator.run_full_validation(num_periods=10, period_hours=24)
```

### デモンストレーション

```python
from prediction_demo import PredictionDemo

# デモの実行
demo = PredictionDemo()

# 静的デモ（特定時点での予測を可視化）
demo.run_static_demo()

# リアルタイムシミュレーション
demo.run_realtime_simulation()
```

## 設定ファイルのカスタマイズ

`prediction_config.json`の主要パラメータ：

```json
{
    "min_discharge": 150.0,        // 最小対象放流量
    "prediction_hours": 3,         // 予測時間
    "time_step": 10,              // 時間ステップ（分）
    
    "delay_params": {
        "150-300": {
            "base_delay": 25,      // 基準遅延時間（分）
            "correlation": 0.824   // 相関係数
        }
    },
    
    "response_rates": {
        "increase": {
            "150-300": 0.0045     // 応答率 m/(m³/s)
        }
    }
}
```

## トラブルシューティング

### よくある問題

1. **放流量が150m³/s未満の場合**
   - エラーメッセージが返されます
   - このシステムは150m³/s以上で最適化されています

2. **履歴データが不足している場合**
   - 最低2時間分のデータが推奨されます
   - データが少ない場合、状態判定が "unknown" になります

3. **予測精度が低い場合**
   - オンライン学習機能を活用してください
   - 設定パラメータの調整を検討してください

### パフォーマンス最適化

- 予測は10分ごとに実行することを推奨
- 履歴データは2時間分で十分です
- オンライン学習は60分予測で実行するのが最適

## API リファレンス

### RealtimePredictionModel

**メソッド**:
- `__init__(config_path=None)`: モデルの初期化
- `predict(current_discharge, current_water_level, future_discharge_plan=None)`: 予測実行
- `update_history(discharge, water_level)`: 履歴更新
- `update_online_learning(predicted_level, actual_level)`: オンライン学習
- `get_state()`: 現在の状態を取得
- `get_prediction_summary(predictions)`: 予測結果のサマリー生成

**戻り値の形式**:
```python
{
    "status": "success",
    "state": "stable",
    "timestamp": "2024-01-30T12:00:00",
    "water_levels": [3.81, 3.82, ...],  # 18個の予測値
    "confidence": [0.9, 0.9, ...],      # 18個の信頼度
    "time_labels": ["+10分", "+20分", ...]
}
```