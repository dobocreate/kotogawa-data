# リアルタイム水位予測システム統合ガイド

## 1. 既存システムへの統合

### 1.1 SCADA/DCSシステムとの統合

```python
# SCADA通信用のアダプター例
class SCADAAdapter:
    def __init__(self, scada_config):
        self.scada = initialize_scada_connection(scada_config)
        self.model = RealtimePredictionModel("learned_config.json")
    
    def get_realtime_data(self):
        # SCADAからリアルタイムデータを取得
        tags = self.scada.read_tags(['DISCHARGE_PV', 'WATER_LEVEL_PV'])
        return {
            'discharge': tags['DISCHARGE_PV'],
            'water_level': tags['WATER_LEVEL_PV']
        }
    
    def write_predictions(self, predictions):
        # 予測結果をSCADAに書き込み
        self.scada.write_tag('PRED_LEVEL_30MIN', predictions['water_levels'][2])
        self.scada.write_tag('PRED_LEVEL_60MIN', predictions['water_levels'][5])
        self.scada.write_tag('PRED_LEVEL_90MIN', predictions['water_levels'][8])
```

### 1.2 データベースとの統合

```python
import psycopg2
from contextlib import contextmanager

class DatabaseIntegration:
    def __init__(self, db_config):
        self.db_config = db_config
        self.model = RealtimePredictionModel("learned_config.json")
    
    @contextmanager
    def get_db_connection(self):
        conn = psycopg2.connect(**self.db_config)
        try:
            yield conn
        finally:
            conn.close()
    
    def get_latest_data(self):
        with self.get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT discharge, water_level 
                FROM sensor_data 
                ORDER BY timestamp DESC 
                LIMIT 1
            """)
            row = cursor.fetchone()
            return {'discharge': row[0], 'water_level': row[1]}
    
    def save_prediction(self, prediction_data):
        with self.get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO predictions 
                (timestamp, current_discharge, current_level, 
                 pred_30min, pred_60min, pred_90min, confidence, state)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            """, (
                prediction_data['timestamp'],
                prediction_data['current_discharge'],
                prediction_data['current_water_level'],
                prediction_data['pred_30min'],
                prediction_data['pred_60min'],
                prediction_data['pred_90min'],
                prediction_data['confidence_60min'],
                prediction_data['state']
            ))
            conn.commit()
```

## 2. Web APIとしての実装

```python
from flask import Flask, jsonify, request
from realtime_prediction_model import RealtimePredictionModel

app = Flask(__name__)
model = RealtimePredictionModel("learned_config.json")

@app.route('/predict', methods=['POST'])
def predict():
    """水位予測APIエンドポイント"""
    data = request.json
    
    # 入力検証
    required_fields = ['discharge', 'water_level']
    for field in required_fields:
        if field not in data:
            return jsonify({'error': f'Missing field: {field}'}), 400
    
    # 予測実行
    try:
        predictions = model.predict(
            current_discharge=data['discharge'],
            current_water_level=data['water_level'],
            future_discharge_plan=data.get('discharge_plan')
        )
        
        if predictions['status'] == 'success':
            return jsonify({
                'status': 'success',
                'predictions': {
                    '30min': predictions['water_levels'][2],
                    '60min': predictions['water_levels'][5],
                    '90min': predictions['water_levels'][8],
                    '120min': predictions['water_levels'][11],
                    '180min': predictions['water_levels'][17]
                },
                'confidence': predictions['confidence'][5],
                'state': predictions['state']
            })
        else:
            return jsonify({'error': predictions['message']}), 400
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    """ヘルスチェックエンドポイント"""
    return jsonify({'status': 'healthy', 'model': 'loaded'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
```

## 3. Docker化

```dockerfile
# Dockerfile
FROM python:3.8-slim

WORKDIR /app

# 依存関係のインストール
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# アプリケーションファイルのコピー
COPY realtime_prediction_model.py .
COPY learned_config.json .
COPY production_example.py .

# ポート公開（Web API使用時）
EXPOSE 5000

# 実行
CMD ["python", "production_example.py", "--mode", "continuous"]
```

```yaml
# docker-compose.yml
version: '3.8'

services:
  prediction-system:
    build: .
    environment:
      - TZ=Asia/Tokyo
    volumes:
      - ./data:/app/data
      - ./logs:/app/logs
    restart: unless-stopped
    command: python production_example.py --mode continuous --interval 10
```

## 4. 監視とメンテナンス

### 4.1 ログ監視

```python
# log_monitor.py
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

def analyze_prediction_accuracy():
    """予測精度の分析"""
    # ログファイルから予測と実測を読み込み
    df = pd.read_csv('prediction_log.csv')
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # 60分予測の精度を計算
    # （60分前の予測と現在の実測を比較）
    df['actual_60min_later'] = df['current_water_level'].shift(-6)
    df['error_60min'] = df['pred_60min'] - df['actual_60min_later']
    
    # 統計表示
    mae = df['error_60min'].abs().mean()
    within_10cm = (df['error_60min'].abs() < 0.1).mean() * 100
    
    print(f"60分予測の精度:")
    print(f"  MAE: {mae*100:.1f}cm")
    print(f"  10cm以内率: {within_10cm:.1f}%")
    
    # グラフ表示
    plt.figure(figsize=(12, 6))
    plt.plot(df['timestamp'], df['error_60min'] * 100, label='予測誤差')
    plt.axhline(y=10, color='r', linestyle='--', label='±10cm')
    plt.axhline(y=-10, color='r', linestyle='--')
    plt.xlabel('時刻')
    plt.ylabel('予測誤差 (cm)')
    plt.title('60分予測の精度推移')
    plt.legend()
    plt.grid(True)
    plt.show()
```

### 4.2 定期的な再学習

```bash
# 月次再学習スクリプト
#!/bin/bash

# 最新データで再学習
python initial_training.py

# 設定ファイルの完全性確認
python ensure_complete_config.py

# 新旧モデルの比較検証
python validate_prediction.py

# 結果が良好なら本番に反映
if [ $? -eq 0 ]; then
    cp learned_config.json learned_config_production.json
    echo "モデル更新完了"
else
    echo "モデル更新をスキップ"
fi
```

## 5. セキュリティ考慮事項

1. **アクセス制御**: APIエンドポイントには認証を実装
2. **入力検証**: すべての入力データを検証
3. **ログ管理**: 機密情報を含まないよう注意
4. **通信暗号化**: HTTPS/TLSを使用

## 6. パフォーマンス最適化

1. **予測のキャッシュ**: 同じ条件での再計算を避ける
2. **非同期処理**: 予測処理を非同期化
3. **バッチ処理**: 複数地点の予測をまとめて実行

これらの例を参考に、実際の環境に合わせて実装してください。