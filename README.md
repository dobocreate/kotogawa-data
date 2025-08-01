# 琴川ダム水位・放流量予測システム

## プロジェクト概要
このプロジェクトは、琴川ダムの水位と放流量を予測するシステムです。降雨データを基に、機械学習とルールベースのハイブリッドアプローチで予測を行います。

## ディレクトリ構造

```
kotogawa-data/
├── models/                     # 学習済みモデル
│   ├── discharge/             # 放流量予測モデル
│   └── water_level/           # 水位予測モデル
├── core/                      # コアモジュール
├── tools/                     # ユーティリティツール
├── experimental/              # 実験的な機能
├── archive/                   # アーカイブファイル
│   ├── test_scripts/         # テストスクリプト
│   ├── analysis_scripts/     # 分析スクリプト
│   ├── debug_scripts/        # デバッグスクリプト
│   ├── old_models/           # 古いモデル
│   ├── csv_results/          # CSV結果ファイル
│   └── figures/              # 生成された図
└── docs/                      # ドキュメント

```

## 主要ファイル

### メインプログラム
- `water_level_prediction_demo_interactive.py` - インタラクティブな予測デモ
- `merge_water_and_dam_data.py` - 水位とダムデータの統合
- `train_water_level_model_auto.py` - 水位モデルの自動学習

### モデル定義
- `discharge_prediction_model_v2.py` - 放流量予測モデル（10分単位統一版）
- `water_level_prediction_model_v2.py` - 水位予測モデル

### 現在の安定版モデル
- 放流量予測: `discharge_prediction_model_v2_20250801_221633.pkl`
- 水位予測: `water_level_predictor_v2_20250731_143150.pkl`

## 使用方法

### 予測デモの実行
```bash
source venv/bin/activate
python water_level_prediction_demo_interactive.py "2023-07-01 00:00"
```

注：実行前に仮想環境がアクティブになっていることを確認してください。

### モデルの学習
```bash
source venv/bin/activate
python train_water_level_model_auto.py
```

## モデルの特徴

### 放流量予測モデル v2（10分単位統一版）
- 状態判定: 過去10分+10分先予測を考慮
- 遅延時間: 0/60/120分の動的選択
- しきい値: 10 m³/s per 30min
- 低降雨時の適切な減少処理

### 水位予測モデル v2
- Random Forestベース
- 特徴量: 降雨、放流量、過去水位、気象データ
- 10分間隔の逐次予測

## 開発状況
2025年8月1日時点で開発を一時休止。
ハイブリッドモデルの実験結果は `experimental/hybrid_model_backup/` に保存。

## データソース
- 統合データ: `統合データ_水位ダム_20250730_142903.csv`