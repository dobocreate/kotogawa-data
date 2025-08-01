import pandas as pd
import numpy as np
from datetime import datetime
import os
import glob
from tkinter import Tk, filedialog, messagebox
import warnings
warnings.filterwarnings('ignore')

class WaterDamDataIntegrator:
    """水位データとダムデータを統合するクラス"""
    
    def __init__(self):
        self.water_data_list = []
        self.dam_data_list = []
        self.integrated_data = None
        
    def select_files(self, title="ファイルを選択"):
        """ファイル選択ダイアログを表示"""
        root = Tk()
        root.withdraw()
        
        files = filedialog.askopenfilenames(
            title=title,
            filetypes=[("CSVファイル", "*.csv"), ("すべてのファイル", "*.*")]
        )
        
        root.destroy()
        return list(files)
    
    def read_csv_with_encoding(self, filepath, file_type=None):
        """エンコーディングを考慮してCSVを読み込む"""
        # Shift-JISを優先的に試す（日本語ファイルの可能性が高いため）
        encodings = ['shift_jis', 'cp932', 'utf-8', 'iso-8859-1']
        
        for encoding in encodings:
            try:
                # ファイルの種類を判別
                # file_typeが指定されていればそれを使用
                if file_type == 'water':
                    # 水位ファイル: 6行目からデータ開始
                    df = pd.read_csv(filepath, encoding=encoding, skiprows=5)
                elif file_type == 'dam':
                    # ダムファイル: 7行目からデータ開始
                    df = pd.read_csv(filepath, encoding=encoding, skiprows=6)
                else:
                    # ファイル名から判別を試みる
                    is_water_file = '水位' in os.path.basename(filepath)
                    
                    if is_water_file:
                        # 水位ファイル: 6行目からデータ開始
                        df = pd.read_csv(filepath, encoding=encoding, skiprows=5)
                    else:
                        # ダムファイル: 7行目からデータ開始
                        df = pd.read_csv(filepath, encoding=encoding, skiprows=6)
                
                # 空の行を削除
                df = df.dropna(how='all')
                
                # カラム名の前後の空白を削除
                df.columns = df.columns.str.strip()
                
                # データの前後の空白も削除
                for col in df.columns:
                    if df[col].dtype == 'object':
                        df[col] = df[col].str.strip()
                
                # 数値カラムの変換を試みる（時刻カラム以外）
                for col in df.columns:
                    if '時刻' not in col and '日時' not in col and 'source_file' not in col:
                        try:
                            # 空文字列やNaNを処理
                            df[col] = df[col].replace(['', ' ', '　'], np.nan)
                            # 数値に変換（エラーは無視してそのまま）
                            df[col] = pd.to_numeric(df[col], errors='coerce')
                        except:
                            pass
                
                print(f"  - {os.path.basename(filepath)} を {encoding} で読み込み成功")
                print(f"    カラム: {list(df.columns)}")
                print(f"    データ行数: {len(df)}")
                
                return df
            except Exception as e:
                continue
        
        # すべてのエンコーディングで失敗した場合、詳細なエラー情報を表示
        print(f"  - エラー: {os.path.basename(filepath)} の読み込みに失敗")
        return None
    
    def load_data(self):
        """水位データとダムデータを読み込む"""
        print("=== データ読み込み開始 ===")
        
        # 水位データの選択と読み込み
        print("\n水位データファイルを選択してください（複数選択可）")
        print("注意: 水位データは観測時刻、水位などのカラムを含むファイルです")
        water_files = self.select_files("水位データファイルを選択")
        
        if not water_files:
            print("水位データファイルが選択されませんでした")
            return False
        
        print(f"\n選択された水位データファイル数: {len(water_files)}")
        for file in sorted(water_files):
            df = self.read_csv_with_encoding(file, file_type='water')
            if df is not None:
                df['source_file'] = os.path.basename(file)
                self.water_data_list.append(df)
        
        # ダムデータの選択と読み込み
        print("\nダムデータファイルを選択してください（複数選択可）")
        print("注意: ダムデータは観測時刻、雨量、貯水位、流入量、放流量などのカラムを含むファイルです")
        dam_files = self.select_files("ダムデータファイルを選択")
        
        if not dam_files:
            print("ダムデータファイルが選択されませんでした")
            return False
        
        print(f"\n選択されたダムデータファイル数: {len(dam_files)}")
        for file in sorted(dam_files):
            df = self.read_csv_with_encoding(file, file_type='dam')
            if df is not None:
                df['source_file'] = os.path.basename(file)
                self.dam_data_list.append(df)
        
        return True
    
    def analyze_data_structure(self):
        """データ構造を分析"""
        print("\n=== データ構造分析 ===")
        
        if self.water_data_list:
            print("\n【水位データ】")
            df = self.water_data_list[0]
            print(f"カラム数: {len(df.columns)}")
            print(f"カラム名: {list(df.columns)}")
            print(f"最初の5行:")
            print(df.head())
            print(f"データ型:")
            print(df.dtypes)
            
            # 時刻カラムの候補を表示
            time_columns = ['観測時刻', '時刻', '日時', 'datetime', 'timestamp', 'date', 'time']
            print("\n時刻カラムの候補:")
            for col in df.columns:
                if any(tc in col for tc in time_columns):
                    print(f"  - {col}")
        
        if self.dam_data_list:
            print("\n【ダムデータ】")
            df = self.dam_data_list[0]
            print(f"カラム数: {len(df.columns)}")
            print(f"カラム名: {list(df.columns)}")
            print(f"最初の5行:")
            print(df.head())
            print(f"データ型:")
            print(df.dtypes)
            
            # 時刻カラムの候補を表示
            time_columns = ['観測時刻', '時刻', '日時', 'datetime', 'timestamp', 'date', 'time']
            print("\n時刻カラムの候補:")
            for col in df.columns:
                if any(tc in col for tc in time_columns):
                    print(f"  - {col}")
    
    def integrate_data(self):
        """データを統合"""
        print("\n=== データ統合処理開始 ===")
        
        # 水位データの結合
        print("\n水位データを結合中...")
        water_df = pd.concat(self.water_data_list, ignore_index=True)
        print(f"結合後の水位データ: {len(water_df)}行")
        
        # ダムデータの結合
        print("\nダムデータを結合中...")
        dam_df = pd.concat(self.dam_data_list, ignore_index=True)
        print(f"結合後のダムデータ: {len(dam_df)}行")
        
        # 時刻カラムの特定と処理
        # 一般的な時刻カラム名のパターン（日本語も追加）
        time_columns = ['観測時刻', '時刻', '日時', 'datetime', 'timestamp', 'date', 'time']
        
        # 水位データの時刻カラムを特定
        water_time_col = None
        for col in water_df.columns:
            if any(tc in col for tc in time_columns):
                water_time_col = col
                break
        
        # ダムデータの時刻カラムを特定
        dam_time_col = None
        for col in dam_df.columns:
            if any(tc in col for tc in time_columns):
                dam_time_col = col
                break
        
        if not water_time_col or not dam_time_col:
            print("\nエラー: 時刻カラムが見つかりません")
            print(f"水位データの時刻カラム: {water_time_col}")
            print(f"ダムデータの時刻カラム: {dam_time_col}")
            
            # デバッグ情報を表示
            print("\n【水位データのカラム一覧】")
            print(list(water_df.columns))
            
            print("\n【ダムデータのカラム一覧】")
            print(list(dam_df.columns))
            
            # カラム名が完全一致しない場合の対処
            if not water_time_col:
                # 水位データの最初のカラムが時刻の可能性が高い
                first_col = water_df.columns[0]
                print(f"\n水位データの最初のカラム '{first_col}' を時刻カラムとして使用します")
                water_time_col = first_col
            
            if not dam_time_col:
                # ダムデータの最初のカラムが時刻の可能性が高い
                first_col = dam_df.columns[0]
                print(f"\nダムデータの最初のカラム '{first_col}' を時刻カラムとして使用します")
                dam_time_col = first_col
            
            if not water_time_col or not dam_time_col:
                return False
        
        print(f"\n時刻カラム - 水位: {water_time_col}, ダム: {dam_time_col}")
        
        # 時刻をdatetime型に変換（24:00を考慮）
        try:
            # 24:00を翌日の00:00に変換する関数
            def convert_datetime(dt_str):
                if isinstance(dt_str, str) and '24:00' in dt_str:
                    # 日付部分を抽出
                    date_part = dt_str.split(' ')[0]
                    # datetime型に変換して1日加算
                    base_date = pd.to_datetime(date_part)
                    return base_date + pd.Timedelta(days=1)
                else:
                    return pd.to_datetime(dt_str)
            
            # 各データフレームの時刻を変換
            print("時刻データを変換中...")
            water_df[water_time_col] = water_df[water_time_col].apply(convert_datetime)
            dam_df[dam_time_col] = dam_df[dam_time_col].apply(convert_datetime)
            
        except Exception as e:
            print(f"時刻変換エラー: {e}")
            # エラー詳細を表示
            print("問題のある時刻データの例:")
            problematic_water = water_df[water_df[water_time_col].astype(str).str.contains('24:00', na=False)]
            if not problematic_water.empty:
                print(f"水位データ: {problematic_water.head()}")
            problematic_dam = dam_df[dam_df[dam_time_col].astype(str).str.contains('24:00', na=False)]
            if not problematic_dam.empty:
                print(f"ダムデータ: {problematic_dam.head()}")
            return False
        
        # カラム名の調整（重複を避けるため）
        water_df = water_df.add_prefix('水位_')
        water_df = water_df.rename(columns={f'水位_{water_time_col}': '時刻'})
        
        dam_df = dam_df.add_prefix('ダム_')
        dam_df = dam_df.rename(columns={f'ダム_{dam_time_col}': '時刻'})
        
        # 時刻でソート
        water_df = water_df.sort_values('時刻')
        dam_df = dam_df.sort_values('時刻')
        
        # 重複の削除
        water_df = water_df.drop_duplicates(subset='時刻', keep='first')
        dam_df = dam_df.drop_duplicates(subset='時刻', keep='first')
        
        # データの結合方法を決定
        print("\n時刻間隔の分析中...")
        try:
            # 少なくとも2行のデータがあることを確認
            if len(water_df) >= 2:
                water_interval = (water_df['時刻'].iloc[1] - water_df['時刻'].iloc[0]).total_seconds() / 60
                print(f"水位データの時刻間隔: 約{water_interval:.0f}分")
            else:
                print("水位データの時刻間隔: 判定不能（データ不足）")
            
            if len(dam_df) >= 2:
                dam_interval = (dam_df['時刻'].iloc[1] - dam_df['時刻'].iloc[0]).total_seconds() / 60
                print(f"ダムデータの時刻間隔: 約{dam_interval:.0f}分")
            else:
                print("ダムデータの時刻間隔: 判定不能（データ不足）")
        except Exception as e:
            print(f"時刻間隔の分析中にエラー: {e}")
            print("デフォルトの設定で続行します")
        
        # 外部結合で統合（すべてのデータを保持）
        print("\nデータを結合中...")
        self.integrated_data = pd.merge(
            water_df, dam_df, 
            on='時刻', 
            how='outer'
        )
        
        # 時刻でソート
        self.integrated_data = self.integrated_data.sort_values('時刻')
        self.integrated_data = self.integrated_data.reset_index(drop=True)
        
        # 重複時刻のチェック
        duplicate_times = self.integrated_data['時刻'].duplicated().sum()
        if duplicate_times > 0:
            print(f"\n警告: {duplicate_times}個の重複時刻が検出されました")
            print("最初の重複を保持し、残りは削除します")
            self.integrated_data = self.integrated_data.drop_duplicates(subset='時刻', keep='first')
        
        # 欠損値を前の時刻のデータで補完
        print("\n欠損データを前時刻のデータで補完中...")
        
        # 数値カラムのみを対象に前方補完（forward fill）
        numeric_columns = self.integrated_data.select_dtypes(include=[np.number]).columns
        self.integrated_data[numeric_columns] = self.integrated_data[numeric_columns].fillna(method='ffill')
        
        # 補完後の欠損値の確認
        remaining_nulls = self.integrated_data[numeric_columns].isnull().sum().sum()
        if remaining_nulls > 0:
            print(f"警告: {remaining_nulls}個の欠損値が残っています（最初の行など）")
            # 最初の行の欠損値は後方補完（backward fill）で対応
            self.integrated_data[numeric_columns] = self.integrated_data[numeric_columns].fillna(method='bfill')
        
        print(f"\n統合完了: {len(self.integrated_data)}行")
        
        # 統計情報
        print("\n【統合データの統計情報】")
        print(f"期間: {self.integrated_data['時刻'].min()} ～ {self.integrated_data['時刻'].max()}")
        print(f"総日数: {(self.integrated_data['時刻'].max() - self.integrated_data['時刻'].min()).days + 1}日")
        print(f"カラム数: {len(self.integrated_data.columns)}")
        print(f"総データ行数: {len(self.integrated_data)}")
        
        # データの完全性チェック
        print("\n【データの完全性】")
        # 水位とダムの両方のデータがある行数
        both_data = self.integrated_data.dropna(subset=['水位_水位', 'ダム_全放流量'])
        print(f"水位・ダム両方のデータがある行数: {len(both_data)} ({len(both_data)/len(self.integrated_data)*100:.1f}%)")
        
        # カラム一覧
        print("\n【カラム一覧】")
        water_cols = [col for col in self.integrated_data.columns if col.startswith('水位_')]
        dam_cols = [col for col in self.integrated_data.columns if col.startswith('ダム_')]
        print(f"水位関連カラム ({len(water_cols)}個): {water_cols}")
        print(f"ダム関連カラム ({len(dam_cols)}個): {dam_cols}")
        
        # 欠損値の確認（主要カラムのみ）
        print("\n【主要カラムの欠損値情報（補完前）】")
        important_cols = ['水位_水位', 'ダム_60分雨量', 'ダム_貯水位', 'ダム_流入量', 'ダム_全放流量']
        for col in important_cols:
            if col in self.integrated_data.columns:
                missing = self.integrated_data[col].isnull().sum()
                percentage = (missing / len(self.integrated_data)) * 100
                print(f"  {col}: {missing}個 ({percentage:.1f}%)")
        
        # 補完後の統計
        print("\n【データ補完統計】")
        print("- 前方補完（forward fill）: 前の時刻のデータで欠損値を補完")
        print("- 後方補完（backward fill）: 最初の行の欠損値のみ次の時刻のデータで補完")
        print("- 補完後、すべての数値データの欠損値が0になりました")
        
        return True
    
    def save_integrated_data(self):
        """統合データを保存"""
        if self.integrated_data is None:
            print("保存するデータがありません")
            return
        
        # 保存ダイアログ
        root = Tk()
        root.withdraw()
        
        # デフォルトファイル名の生成
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        default_filename = f"統合データ_水位ダム_{timestamp}.csv"
        
        filepath = filedialog.asksaveasfilename(
            title="統合データの保存先を選択",
            defaultextension=".csv",
            initialfile=default_filename,
            filetypes=[("CSVファイル", "*.csv"), ("すべてのファイル", "*.*")]
        )
        
        root.destroy()
        
        if not filepath:
            print("保存がキャンセルされました")
            return
        
        # データの保存
        try:
            # 数値の精度を保持して保存
            self.integrated_data.to_csv(
                filepath, 
                index=False, 
                encoding='utf-8-sig',
                float_format='%.2f',  # 小数点以下2桁で保存
                date_format='%Y/%m/%d %H:%M'  # 日時形式を指定
            )
            print(f"\n統合データを保存しました: {filepath}")
            
            # 保存確認メッセージ
            root = Tk()
            root.withdraw()
            messagebox.showinfo("保存完了", f"統合データを保存しました:\n{filepath}")
            root.destroy()
            
        except Exception as e:
            print(f"保存エラー: {e}")
            root = Tk()
            root.withdraw()
            messagebox.showerror("保存エラー", f"データの保存に失敗しました:\n{e}")
            root.destroy()

def main():
    """メイン処理"""
    print("河川水位・ダムデータ統合プログラム")
    print("=" * 50)
    
    integrator = WaterDamDataIntegrator()
    
    # データの読み込み
    if not integrator.load_data():
        print("データの読み込みに失敗しました")
        return
    
    # データ構造の分析
    integrator.analyze_data_structure()
    
    # 統合前の確認情報を表示
    print("\n" + "=" * 50)
    print("【統合前の確認】")
    print(f"水位データ: 合計{sum(len(df) for df in integrator.water_data_list)}行")
    print(f"ダムデータ: 合計{sum(len(df) for df in integrator.dam_data_list)}行")
    print("両データは10分間隔で記録されており、時刻をキーとして結合されます")
    print("24:00表記は翌日00:00に自動変換されます")
    
    # ユーザー確認
    print("\n" + "=" * 50)
    response = input("データを統合しますか？ (y/n): ")
    if response.lower() != 'y':
        print("処理を中止しました")
        return
    
    # データの統合
    if not integrator.integrate_data():
        print("データの統合に失敗しました")
        return
    
    # 保存確認
    print("\n" + "=" * 50)
    response = input("統合データを保存しますか？ (y/n): ")
    if response.lower() == 'y':
        integrator.save_integrated_data()
    
    print("\n処理が完了しました")

if __name__ == "__main__":
    main()