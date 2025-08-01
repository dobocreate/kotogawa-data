# Toolsフォルダ

このフォルダには、水位予測システムの補助ツールが含まれています。

## ファイル一覧

### データ準備・設定
- **ensure_complete_config.py** - 学習済み設定ファイルの完全性確保
- **filter_and_split_data.py** - データのフィルタリングと学習/検証/テスト分割

### 学習・検証
- **train_validate_with_split.py** - 初期学習と検証の一括実行
- **validate_prediction.py** - 予測モデルの検証・評価

### 本番環境
- **production_example.py** - リアルタイム予測システムの実装例

## 依存関係
これらのツールは`core/`フォルダ内のモジュールに依存しています：
- `core/realtime_prediction_model.py`
- `core/initial_training.py`
- `core/configs/learned_config.json`