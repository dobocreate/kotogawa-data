# 放流量予測モデル統合サマリー

## 概要
`water_level_prediction_demo_interactive.py`を修正し、学習済みの放流量予測モデル（`discharge_prediction_model_20250801_051151.pkl`）を統合しました。これにより、従来の「放流量現在値継続」の仮定から、機械学習モデルによる動的な放流量予測へと改善されました。

## 主な変更点

### 1. 放流量予測モデルの導入
- `DischargePredictionModel`クラスをインポート
- 初期化時に放流量予測モデルをロード
- 120分遅延を考慮した降雨強度と貯水位から放流量を予測

### 2. 予測フローの改善
```python
# 従来：放流量を現在値で固定
放流量予測 = [現在値] * 予測時間数

# 改善後：機械学習モデルで予測
放流量予測 = discharge_predictor.predict(
    current_time, 
    historical_data,
    prediction_hours=3
)
```

### 3. 水位予測の高度化
- 予測された放流量を使用して水位変化を計算
- 増加期（40分遅延）と減少期（60分遅延）で異なる応答率を適用

### 4. 可視化の改善
- 放流量グラフに「予測放流量（MLモデル）」を青線で表示
- 従来の「現在値継続」仮定との比較が可能

## 実行方法

### 基本的な使用方法
```bash
# 仮想環境の有効化
source venv/bin/activate

# インタラクティブモードで実行
python water_level_prediction_demo_interactive.py

# 特定時刻を指定して実行
python water_level_prediction_demo_interactive.py "2023-07-03 12:00"
```

### 予測モードの切り替え
```python
# 機械学習モデルを使用（デフォルト）
demo.run_prediction(time_str, use_discharge_model=True)

# 従来の現在値継続モデルを使用
demo.run_prediction(time_str, use_discharge_model=False)
```

## 技術仕様

### 放流量予測モデル
- **アルゴリズム**: RandomForest
- **予測時間**: 最大3時間先（10分刻み）
- **主要特徴量**:
  - 120分前の降雨強度（最重要）
  - 現在の放流量
  - 貯水位と洪水レベルとの差
  - 過去2時間の降雨統計

### 降雨予測の扱い
- 現在の実装：現在の降雨強度が1時間継続すると仮定
- 拡張可能：外部の降雨予測データを`rainfall_forecast`パラメータで渡すことが可能

## 予測精度の向上
- 放流量予測により、水位予測の精度が向上
- 特に放流量が変化する局面での予測改善が期待される

## 今後の拡張可能性
1. 実際の降雨予測データの統合
2. 不確実性の定量化（予測区間の改善）
3. リアルタイムデータストリームへの対応
4. 複数シナリオの同時予測

## 注意事項
- モデルファイルが必要：
  - `water_level_predictor_v2_*.pkl`
  - `discharge_prediction_model_*.pkl`
- データファイルの形式に依存（統合データ_水位ダム_*.csv）
- 予測は過去データの傾向に基づくため、異常気象時は精度が低下する可能性