# -*- coding: utf-8 -*-
"""
ウェアラブル + Outlookスケジュール 集中・疲労予測アプリ (Streamlit版)
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import lightgbm as lgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, roc_auc_score, log_loss
from sklearn.model_selection import TimeSeriesSplit
import google.generativeai as genai
import shap
import warnings
import math

# 日本語フォントの設定 (環境によってjapanize_matplotlibが必要な場合があります)
try:
    import japanize_matplotlib
except ImportError:
    pass

warnings.filterwarnings('ignore')

# --- Streamlit ページ設定 ---
st.set_page_config(page_title="集中・疲労予測システム", layout="wide")
st.title("🧠 ウェアラブル×スケジュール 集中予測システム")
st.markdown("""
ウェアラブルデバイスのデータと予定表データを組み合わせて、数時間後の集中スコアを予測し、推奨する働き方を提案します。
""")

# --- サイドバー (設定) ---
st.sidebar.header("⚙️ 設定")
api_key = st.sidebar.text_input("Gemini APIキー (省略時は固定ルールで出力)", type="password")

st.sidebar.subheader("分析パラメータ")
RESAMPLE_FREQ = st.sidebar.selectbox("分析単位", ['10T', '30T', '1H'], index=1)
PREDICT_AHEAD = st.sidebar.selectbox("予測先", ['10T', '30T', '1H'], index=1)
LOOKBACK_PERIOD = st.sidebar.selectbox("過去参照", ['1H', '2H', '3H'], index=1)
INTERPOLATE_LIMIT = st.sidebar.selectbox("補完限界", ['10T', '30T', '1H'], index=1)

st.sidebar.subheader("予測基準日時 (任意)")
TARGET_DATETIME_STR = st.sidebar.text_input("例）2025-12-18 16:00 (空欄で最新データ)")
TARGET_DATETIME = TARGET_DATETIME_STR if TARGET_DATETIME_STR.strip() != "" else None

freq_td = pd.Timedelta(RESAMPLE_FREQ)
ahead_td = pd.Timedelta(PREDICT_AHEAD)
lookback_td = pd.Timedelta(LOOKBACK_PERIOD)
interp_td = pd.Timedelta(INTERPOLATE_LIMIT)

ahead_steps = max(1, int(ahead_td / freq_td))
lookback_steps = max(2, int(lookback_td / freq_td))
interp_steps = max(1, int(interp_td / freq_td))

# --- 予測ターゲットの選択機能を追加 ---
TARGET_OPTIONS = {
    'CVRR_SCORE_NEW': '集中スコア',
    'RMSSD_SCORE_NEW': '疲労・回復スコア',
    'LP_SCORE_NEW': 'リラックススコア',
    'NEMUKE_SCORE_NEW': '低覚醒スコア(眠気)',
    'TP': 'TP(自律神経トータルパワー)',
    '集中判定': '集中判定(1=集中)',
    '疲労判定': '疲労判定(1=疲労)',
    '強い疲労判定': '強い疲労判定(1=強い)'
}
st.sidebar.subheader("🎯 予測ターゲット")
selected_target_name = st.sidebar.selectbox("予測する指標を選択", list(TARGET_OPTIONS.values()), index=0)
target_col = [k for k, v in TARGET_OPTIONS.items() if v == selected_target_name][0]

# --- 新規: 長期分析用フィルタ ---
st.sidebar.subheader("📅 長期分析フィルタ (特性インサイト用)")
dow_options = ["月", "火", "水", "木", "金", "土", "日"]
selected_dows = st.sidebar.multiselect("対象曜日", dow_options, default=dow_options)
time_range = st.sidebar.slider("対象時間帯", 0, 23, (9, 19))

# --- 特徴量名日本語化ヘルパー ---
def jp_feat_name(col_name: str) -> str:
    mapping = {
        'CVRR_SCORE_NEW': '集中スコア', 'SkinTemp': '皮膚温度', 'LP_SCORE_NEW': 'リラックススコア',
        'LFHF_SCORE_NEW': 'LF/HF(自律神経バランス)', 'TP': 'TP(自律神経トータルパワー)', 'NEMUKE_SCORE_NEW': '低覚醒スコア',
        'PR_SCORE_NEW': '脈拍', 'RMSSD_SCORE_NEW': '疲労・回復スコア', '1分間歩数': '歩数', 'accDeviation': '活動量(加速度)',
        'has_schedule': '予定の有無', 'is_meeting': '会議中かどうか', 'schedule_density_2h': '最近の予定の詰まり具合',
        'time_to_next_event_min': '次の予定までの時間', 'time_since_prev_event_min': '前の予定からの経過時間',
        'daily_schedule_hours': '1日の総予定時間', 'consecutive_schedules': '連続予定ブロック数',
        '今日からの累積歩数': '今日からの累積歩数', '今日からの累積会議時間_分': '今日からの累積会議時間(分)',
        '現在の集中継続時間_分': '現在の集中継続時間(分)', '現在の疲労継続時間_分': '現在の疲労継続時間(分)',
        '集中判定': '集中判定', '疲労判定': '疲労判定', '強い疲労判定': '強い疲労判定',
        '集中状態': '集中状態', '眠気状態': '眠気状態', '疲労状態': '疲労状態',
        '集中継続時間': '集中継続時間', '深い集中継続時間': '深い集中継続時間',
        '疲労状態継続時間': '疲労状態継続時間', '疲労継続時間': '疲労状態継続時間', '高心拍継続時間': '高心拍継続時間'
    }
    
    base_jp = col_name
    remainder = ""
    for k, v in mapping.items():
        if col_name.startswith(k):
            base_jp = v
            remainder = col_name[len(k):]
            break
            
    # サフィックス（_roll_meanなど）を分かりやすい言葉に変換
    if remainder == "":
        return base_jp
    elif remainder == "_roll_mean":
        return f"最近の「{base_jp}」の平均的な高さ"
    elif remainder == "_roll_slope":
        return f"最近の「{base_jp}」の急な変化(トレンド)"
    elif remainder == "_diff1":
        return f"前回からの「{base_jp}」の変動幅"
    elif remainder.startswith("_lag"):
        return f"少し前の「{base_jp}」の状態"
    elif remainder == "_is_missing":
        return f"「{base_jp}」が未計測であること"
    else:
        return f"{base_jp}{remainder}"

# --- ベース指標名（サフィックスなし）を取得するヘルパー ---
def get_base_feature_name(feat: str) -> str:
    mapping = {
        'CVRR_SCORE_NEW': '集中スコア', 'SkinTemp': '皮膚温度', 'LP_SCORE_NEW': 'リラックススコア',
        'LFHF_SCORE_NEW': 'LF/HF(自律神経バランス)', 'TP': 'TP(自律神経トータルパワー)', 'NEMUKE_SCORE_NEW': '低覚醒スコア',
        'PR_SCORE_NEW': '脈拍', 'RMSSD_SCORE_NEW': '疲労・回復スコア', '1分間歩数': '歩数', 'accDeviation': '活動量(加速度)',
        'has_schedule': '予定', 'is_meeting': '会議', 'schedule_density_2h': '予定の密度',
        'time_to_next_event_min': '次の予定までの時間', 'time_since_prev_event_min': '前の予定からの経過時間',
        'daily_schedule_hours': '1日の総予定時間', 'consecutive_schedules': '連続予定ブロック数',
        '今日からの累積歩数': '今日からの累積歩数', '今日からの累積会議時間_分': '今日からの累積会議時間',
        '現在の集中継続時間_分': '現在の集中継続時間', '現在の疲労継続時間_分': '現在の疲労継続時間',
        '集中判定': '集中判定', '疲労判定': '疲労判定', '強い疲労判定': '強い疲労判定',
        '集中状態': '集中状態', '眠気状態': '眠気状態', '疲労状態': '疲労状態',
        '集中継続時間': '集中継続時間', '深い集中継続時間': '深い集中継続時間',
        '疲労状態継続時間': '疲労状態継続時間', '疲労継続時間': '疲労状態継続時間', '高心拍継続時間': '高心拍継続時間'
    }
    for k, v in mapping.items():
        if feat.startswith(k):
            return v
    return feat

# --- 特徴量名と値から「増加」「低下」を加味した表現を生成するヘルパー ---
def get_factor_direction_text(feat: str, val: float, df_all: pd.DataFrame) -> str:
    mapping = {
        'CVRR_SCORE_NEW': '集中スコア', 'SkinTemp': '皮膚温度', 'LP_SCORE_NEW': 'リラックススコア',
        'LFHF_SCORE_NEW': 'LF/HF(自律神経バランス)', 'TP': 'TP(自律神経トータルパワー)', 'NEMUKE_SCORE_NEW': '低覚醒スコア',
        'PR_SCORE_NEW': '脈拍', 'RMSSD_SCORE_NEW': '疲労・回復スコア', '1分間歩数': '歩数', 'accDeviation': '活動量(加速度)',
        'has_schedule': '予定', 'is_meeting': '会議', 'schedule_density_2h': '予定の密度',
        'time_to_next_event_min': '次の予定までの時間', 'time_since_prev_event_min': '前の予定からの経過時間',
        'daily_schedule_hours': '1日の総予定時間', 'consecutive_schedules': '連続予定ブロック数',
        '集中判定': '集中判定', '疲労判定': '疲労判定', '強い疲労判定': '強い疲労判定',
        '集中状態': '集中状態', '眠気状態': '眠気状態', '疲労状態': '疲労状態',
        '集中継続時間': '集中継続時間', '深い集中継続時間': '深い集中継続時間',
        '疲労状態継続時間': '疲労状態継続時間', '疲労継続時間': '疲労状態継続時間', '高心拍継続時間': '高心拍継続時間'
    }
    
    base_jp = feat
    remainder = ""
    for k, v in mapping.items():
        if feat.startswith(k):
            base_jp = v
            remainder = feat[len(k):]
            break

    # カテゴリやフラグ系の処理
    if remainder == "_is_missing":
        return f"「{base_jp}」が未計測であること"
    elif feat in ["has_schedule", "is_meeting"]:
        if val > 0:
            return f"「{base_jp}」が入っていること"
        else:
            return f"「{base_jp}」が入っていないこと"
    elif feat in ["集中状態", "眠気状態", "疲労状態"]:
        return f"「{base_jp}」が「{val}」であること"
            
    # 増減をみるもの（トレンドや差分）
    if remainder in ["_roll_slope", "_diff1"]:
        if val > 0:
            direction = "の増加"
        elif val < 0:
            direction = "の低下"
        else:
            direction = "の変化なし"
            
        if remainder == "_roll_slope":
            return f"最近の「{base_jp}」のトレンド{direction}"
        elif remainder == "_diff1":
            return f"前回からの「{base_jp}」{direction}"
            
    # 指標そのものを見る場合（値の大きさ自体が影響するため、無理に「増加/低下」を付けない）
    else:
        if remainder == "_roll_mean":
            return f"最近の「{base_jp}」"
        elif remainder.startswith("_lag"):
            return f"少し前の「{base_jp}」"
        else:
            return f"「{base_jp}」"

# --- 分析メイン処理 ---
def run_analysis(df_ts, df_sched, use_gemini=False):
    # 1. データ前処理
    with st.spinner("データを集約・前処理しています..."):
        # CSVのtimestamp列からタイムゾーン文字列（ GMT+0900 など）を除去して日時に変換
        if 'timestamp' in df_ts.columns:
            df_ts['timestamp_clean'] = df_ts['timestamp'].astype(str).str.split(' GMT').str[0]
            df_ts['datetime'] = pd.to_datetime(df_ts['timestamp_clean'], errors='coerce')
            df_ts = df_ts.dropna(subset=['datetime'])
            df_ts.set_index('datetime', inplace=True)
            df_ts.drop(columns=['timestamp', 'timestamp_clean'], inplace=True, errors='ignore')

        base_agg_dict = {
            'SkinTemp': 'mean', 'CVRR_SCORE_NEW': 'mean', 'LP_SCORE_NEW': 'mean',
            'LFHF_SCORE_NEW': 'mean', 'TP': 'mean', 'NEMUKE_SCORE_NEW': 'mean',
            'PR_SCORE_NEW': 'mean', 'RMSSD_SCORE_NEW': 'mean', '1分間歩数': 'sum', 'accDeviation': 'mean',
            '集中判定': 'mean', '疲労判定': 'mean', '強い疲労判定': 'mean',
            '集中継続時間': 'mean', '深い集中継続時間': 'mean', '疲労状態継続時間': 'mean', '疲労継続時間': 'mean', '高心拍継続時間': 'mean'
        }
        
        cat_agg_dict = {
            '集中状態': 'last',
            '眠気状態': 'last',
            '疲労状態': 'last'
        }
        
        # データフレームに実際に存在する列のみを抽出して集約対象にする（存在しない場合はスキップ）
        agg_dict = {col: func for col, func in base_agg_dict.items() if col in df_ts.columns}
        for col, func in cat_agg_dict.items():
            if col in df_ts.columns:
                agg_dict[col] = func
        
        # 予測ターゲット変数が存在しない場合はエラーを出して終了
        if target_col not in agg_dict:
            st.error(f"エラー: 予測に必要な目的変数「{selected_target_name}（列名: {target_col}）」がデータに含まれていません。")
            return

        # 集約前に確実に数値型に変換しておく (数値データのみ)
        for col in agg_dict.keys():
            if col in base_agg_dict.keys():
                df_ts[col] = pd.to_numeric(df_ts[col], errors='coerce')

        df_resampled = df_ts.resample(RESAMPLE_FREQ).agg(agg_dict)

        if df_sched is not None:
            # 予定表データの処理
            df_sched = df_sched[df_sched['終日イベント'].astype(str).str.upper() != 'TRUE']
            df_sched['start_dt'] = pd.to_datetime(df_sched['開始日'].astype(str) + ' ' + df_sched['開始時刻'].astype(str), errors='coerce')
            df_sched['end_dt']   = pd.to_datetime(df_sched['終了日'].astype(str) + ' ' + df_sched['終了時刻'].astype(str), errors='coerce')
            df_sched = df_sched.dropna(subset=['start_dt', 'end_dt']).sort_values('start_dt')

            df_resampled['has_schedule'] = 0
            df_resampled['is_meeting'] = 0
            meeting_keywords = ['会議', '打合せ', '打ち合わせ', 'MTG', '面談', '商談', '来客', '訪問']

            for _, row in df_sched.iterrows():
                mask = (df_resampled.index < row['end_dt']) & ((df_resampled.index + freq_td) > row['start_dt'])
                df_resampled.loc[mask, 'has_schedule'] = 1
                subject = str(row.get('件名', ''))
                if any(kw in subject for kw in meeting_keywords):
                    df_resampled.loc[mask, 'is_meeting'] = 1

            s = df_resampled['has_schedule']
            df_resampled['consecutive_schedules'] = s.groupby((s != s.shift()).cumsum()).cumsum()

            # 日次特徴量
            df_resampled['date'] = df_resampled.index.date
            df_resampled = df_resampled.join(df_resampled.groupby('date')['has_schedule'].sum().rename('daily_schedule_hours'), on='date').fillna({'daily_schedule_hours': 0})
            df_resampled.drop(columns=['date'], inplace=True)

            # 次・前の予定までの時間 (Safe Lookup)
            event_starts = df_sched['start_dt'].to_numpy(dtype='datetime64[ns]')
            event_ends   = df_sched['end_dt'].to_numpy(dtype='datetime64[ns]')
            t = df_resampled.index.to_numpy(dtype='datetime64[ns]')

            next_start_idx = np.searchsorted(event_starts, t, side='left')
            has_next = next_start_idx < len(event_starts)
            next_idx_safe = np.clip(next_start_idx, 0, max(len(event_starts) - 1, 0))
            next_start = np.full(t.shape, np.datetime64('NaT'), dtype='datetime64[ns]')
            if len(event_starts) > 0: next_start[has_next] = event_starts[next_idx_safe[has_next]]

            prev_end_idx = np.searchsorted(event_ends, t, side='right') - 1
            has_prev = prev_end_idx >= 0
            prev_idx_safe = np.clip(prev_end_idx, 0, max(len(event_ends) - 1, 0))
            prev_end = np.full(t.shape, np.datetime64('NaT'), dtype='datetime64[ns]')
            if len(event_ends) > 0: prev_end[has_prev] = event_ends[prev_idx_safe[has_prev]]

            df_resampled['time_to_next_event_min'] = (next_start - t) / np.timedelta64(1, 'm')
            df_resampled['time_since_prev_event_min'] = (t - prev_end) / np.timedelta64(1, 'm')

            win_steps = max(1, int(pd.Timedelta('2H') / freq_td))
            df_resampled['schedule_density_2h'] = df_resampled['has_schedule'].rolling(win_steps, min_periods=1).mean()

    # 2. 特徴量エンジニアリング
    with st.spinner("特徴量を生成しています..."):
        df_features = df_resampled.copy()
        df_features['hour'] = df_features.index.hour.astype('category')
        df_features['dayofweek'] = df_features.index.dayofweek.astype('category')
        
        for c in ['集中状態', '眠気状態', '疲労状態']:
            if c in df_features.columns:
                df_features[c] = df_features[c].astype('category')
        
        numeric_cols = df_resampled.select_dtypes(include=[np.number]).columns.tolist()
        
        win = lookback_steps
        x = np.arange(win, dtype=float)
        x_mean = x.mean()
        x_var = ((x - x_mean) ** 2).sum()
        def rolling_slope(arr):
            y = arr.astype(float)
            if x_var == 0: return 0.0
            return ((x - x_mean) * (y - y.mean())).sum() / x_var

        for col in numeric_cols:
            df_features[f'{col}_is_missing'] = df_resampled[col].isna().astype(int)
            r = df_features[col].rolling(win, min_periods=win)
            df_features[f'{col}_roll_mean'] = r.mean()
            df_features[f'{col}_roll_slope'] = r.apply(rolling_slope, raw=True)
            df_features[f'{col}_diff1'] = df_features[col] - df_features[col].shift(1)

        # -- Step 2: 追加特徴量 (累積負荷, 継続時間) の計算 --
        df_features['date'] = df_features.index.date
        if '1分間歩数' in df_features.columns:
            df_features['今日からの累積歩数'] = df_features.groupby('date')['1分間歩数'].cumsum()
        if 'is_meeting' in df_features.columns:
            df_features['今日からの累積会議時間_分'] = df_features.groupby('date')['is_meeting'].cumsum() * (freq_td.total_seconds() / 60)
        
        def calc_duration_mins(series):
            group_id = (series != series.shift()).cumsum()
            return (series.groupby(group_id).cumcount() + 1) * (freq_td.total_seconds() / 60)
            
        if '集中判定' in df_features.columns:
            focus_mask = (df_features['集中判定'] >= 0.5).astype(int)
            df_features['現在の集中継続時間_分'] = calc_duration_mins(focus_mask) * focus_mask
            
        if '疲労判定' in df_features.columns:
            fatigue_mask = (df_features['疲労判定'] >= 0.5).astype(int)
            df_features['現在の疲労継続時間_分'] = calc_duration_mins(fatigue_mask) * fatigue_mask
            
        df_features.drop(columns=['date'], inplace=True)

        # Step 1: 分類モデルへのターゲット変数の生成
        target_threshold = 0.5 if target_col in ['集中判定', '疲労判定', '強い疲労判定'] else df_features[target_col].median()
        # 予測先のウィンドウ内での平均値が閾値以上かどうかを分類ターゲットにする（未来の絶対状態）
        df_features['target_ahead_class'] = (df_features[target_col].shift(-ahead_steps) >= target_threshold).astype(int)

    # 3. モデル学習
    with st.spinner("LightGBM分類モデルを学習しています..."):
        drop_cols = ['target_ahead_class']
        df_all = df_features.copy()
        split_idx = int(len(df_all) * 0.8)
        
        # 簡易欠損補完 (カテゴリ変数も含めて補完するよう修正)
        df_imp = df_all.copy()
        for col in df_imp.columns:
            if col not in drop_cols:
                df_imp[col] = df_imp[col].ffill(limit=interp_steps).bfill(limit=interp_steps)
        
        train_df = df_imp.iloc[:split_idx].dropna(subset=drop_cols + [target_col])
        test_df  = df_imp.iloc[split_idx:].dropna(subset=drop_cols + [target_col])

        X_train = train_df.drop(columns=drop_cols)
        y_train_class = train_df['target_ahead_class']
        X_test  = test_df.drop(columns=drop_cols)
        y_test_class = test_df['target_ahead_class']
        
        cat_cols = [c for c in X_train.columns if str(X_train[c].dtype) == 'category']
        
        model = lgb.LGBMClassifier(
            objective='binary', n_estimators=500, learning_rate=0.03, random_state=42
        )
        # 簡易的に全データで学習
        model.fit(X_train, y_train_class, categorical_feature=cat_cols if cat_cols else 'auto')
        
        preds_proba = model.predict_proba(X_test)[:, 1]
        
        try:
            auc_test = roc_auc_score(y_test_class, preds_proba)
            logloss_test = log_loss(y_test_class, preds_proba)
        except ValueError:
            auc_test = np.nan
            logloss_test = np.nan

    # === パーソナル特性インサイト (長期分析) を先に表示 ===
    st.header("👤 パーソナル特性インサイト (長期分析)")
    
    # 長期分析用フィルタの適用
    df_insight = df_imp.copy()
    selected_dow_indices = [dow_options.index(d) for d in selected_dows]
    df_insight = df_insight[df_insight.index.dayofweek.isin(selected_dow_indices)]
    df_insight = df_insight[(df_insight.index.hour >= time_range[0]) & (df_insight.index.hour <= time_range[1])]

    # 状態開始のフラグ作成 (インサイト全体で利用)
    if '集中判定' in df_insight.columns:
        df_insight['focus_start'] = (df_insight['集中判定'] >= 0.5) & (df_insight['集中判定'].shift(1) < 0.5)
    if '疲労判定' in df_insight.columns:
        df_insight['fatigue_start'] = (df_insight['疲労判定'] >= 0.5) & (df_insight['疲労判定'].shift(1) < 0.5)

    tab1, tab2, tab3 = st.tabs(["📝 マイ・スペック", "📊 時間帯・曜日特性", "💡 行動・予定インサイト"])
    
    with tab1:
        st.markdown(f"### 🎯 あなたの集中特性 (選択条件での集計)")
        if '集中判定' in df_ts.columns:
            # 継続時間をより正確に計算するため、元データを1分単位でリサンプリングして計算
            df_1min = df_ts[['集中判定']].resample('1T').mean()
            df_1min = df_1min.ffill(limit=5)
            
            # フィルタリングの適用
            selected_dow_indices = [dow_options.index(d) for d in selected_dows]
            df_1min = df_1min[df_1min.index.dayofweek.isin(selected_dow_indices)]
            df_1min = df_1min[(df_1min.index.hour >= time_range[0]) & (df_1min.index.hour <= time_range[1])]
            
            focus_mask = df_1min['集中判定'] >= 0.5
            focus_blocks = focus_mask.groupby((focus_mask != focus_mask.shift()).cumsum())
            focus_durations = focus_blocks.sum() 
            focus_durations = focus_durations[focus_durations > 0]
            
            if not focus_durations.empty:
                col_m1, col_m2, col_m3, col_m4 = st.columns(4)
                col_m1.metric("平均集中持続時間", f"{focus_durations.mean():.0f} 分")
                col_m2.metric("最長集中記録", f"{focus_durations.max():.0f} 分")
                
                # 総集中回数と1日あたりの平均集中回数
                total_focus_count = len(focus_durations)
                num_days = df_1min.index.normalize().nunique()
                daily_focus_count = total_focus_count / num_days if num_days > 0 else 0
                
                col_m3.metric("総集中回数", f"{total_focus_count} 回", help="選択期間内で集中状態に入った合計回数")
                col_m4.metric("1日あたりの平均集中回数", f"{daily_focus_count:.1f} 回/日")
                
                # 集中持続時間の分布グラフ (10分刻みに修正)
                st.markdown("#### 集中持続時間の分布")
                fig_dist, ax_dist = plt.subplots(figsize=(8, 4))
                max_duration = int(focus_durations.max())
                max_bin = math.ceil(max_duration / 10) * 10
                bins = np.arange(0, max_bin + 20, 10) 
                
                # エッジでビンを切る
                counts, edges, patches = ax_dist.hist(focus_durations, bins=bins, color='#4A90E2', edgecolor='white', alpha=0.8)
                
                # 表示用のラベル作成 (0-9, 10-19...)
                bin_centers = edges[:-1] + 5
                xtick_labels = [f"{int(edges[i])}-{int(edges[i+1])-1}" for i in range(len(edges)-1)]
                
                ax_dist.set_xticks(bin_centers)
                ax_dist.set_xticklabels(xtick_labels, rotation=45, ha='right', fontsize=9)
                
                ax_dist.set_xlabel("集中持続時間 (分)")
                ax_dist.set_ylabel("回数")
                ax_dist.set_title("集中持続時間のヒストグラム (10分刻み)")
                ax_dist.spines['top'].set_visible(False)
                ax_dist.spines['right'].set_visible(False)
                fig_dist.tight_layout()
                st.pyplot(fig_dist)
            else:
                st.write("指定された条件で十分な集中データがありません。")
        else:
            st.write("データに「集中判定」列が含まれていないため計算できません。")
                
    with tab2:
        st.markdown("### 📊 時間帯・曜日別の傾向 (集中・疲労)")
        col_h1, col_h2 = st.columns(2)
        
        def plot_heatmap(target_metric, title_prefix, cmap_name):
            if target_metric not in df_imp.columns:
                return None, None, None
                
            pivot_df = df_imp.pivot_table(
                values=target_metric, 
                index=df_imp.index.hour, 
                columns=df_imp.index.dayofweek, 
                aggfunc='mean'
            )
            
            start_hour_hm = time_range[0]
            end_hour_hm = time_range[1]
            num_hours_hm = end_hour_hm - start_hour_hm + 1
            
            fig_hm, ax_hm = plt.subplots(figsize=(6, 4))
            heatmap_data = np.full((num_hours_hm, 7), np.nan)
            
            for h in pivot_df.index:
                if start_hour_hm <= h <= end_hour_hm:
                    for d in pivot_df.columns:
                        if d in selected_dow_indices:
                            heatmap_data[int(h) - start_hour_hm, int(d)] = pivot_df.loc[h, d]
            
            im = ax_hm.imshow(heatmap_data, cmap=cmap_name, aspect='auto')
            
            ax_hm.set_xticks(np.arange(7))
            ax_hm.set_xticklabels(dow_options)
            
            yticks = []
            yticklabels = []
            for i, h in enumerate(range(start_hour_hm, end_hour_hm + 1)):
                yticks.append(i)
                yticklabels.append(str(h))
                
            ax_hm.set_yticks(yticks)
            ax_hm.set_yticklabels(yticklabels)
            
            ax_hm.set_xlabel("曜日")
            ax_hm.set_ylabel("時間帯 (時)")
            ax_hm.set_title(f"{title_prefix} ({start_hour_hm}時〜{end_hour_hm}時)")
            
            cbar = plt.colorbar(im, ax=ax_hm)
            cbar.set_label("確率")
            
            daytime_pivot = pivot_df.loc[start_hour_hm:end_hour_hm, selected_dow_indices]
            best_hour, best_dow_jp = None, None
            if not daytime_pivot.isna().all().all():
                best_hour, best_dow = daytime_pivot.stack().idxmax()
                best_hour = int(best_hour)
                best_dow = int(best_dow)
                best_dow_jp = dow_options[best_dow]
                
            return fig_hm, best_hour, best_dow_jp

        with col_h1:
            st.markdown("#### 🎯 集中しやすい時間帯")
            fig_focus, f_hour, f_dow = plot_heatmap('集中判定', "曜日・時間帯別の集中確率", 'Blues')
            if fig_focus:
                st.pyplot(fig_focus)
                if f_hour is not None:
                    st.write(f"※ 選択された条件では、**{f_dow}曜日の{f_hour}時台** に最も集中しやすい傾向があります。")
            else:
                st.write("「集中判定」データが不足しています。")

        with col_h2:
            st.markdown("#### 🔋 疲労しやすい時間帯")
            fig_fatigue, fat_hour, fat_dow = plot_heatmap('疲労判定', "曜日・時間帯別の疲労確率", 'Reds')
            if fig_fatigue:
                st.pyplot(fig_fatigue)
                if fat_hour is not None:
                    st.write(f"※ 選択された条件では、**{fat_dow}曜日の{fat_hour}時台** に最も疲労しやすい傾向があります。")
            else:
                st.write("「疲労判定」データが不足しています。")

    with tab3:
        st.markdown("### 📅 スケジュール・行動とパフォーマンスの関係")
        insight_texts = []
        
        # 0. 新規: 疲労のピークタイム
        if 'fatigue_start' in df_insight.columns:
            fatigue_starts = df_insight[df_insight['fatigue_start']]
            if not fatigue_starts.empty:
                peak_fatigue_hour = fatigue_starts.index.hour.value_counts().idxmax()
                insight_texts.append(f"- 🔋 **疲労のピークタイム**: あなたの疲労が最も始まりやすいのは **{peak_fatigue_hour}時台** です。この時間帯の前に小休憩を入れることをお勧めします。")
        
        # 1. 1時間の予定での疲労変化 / 次の予定の集中力
        if '疲労判定' in df_insight.columns and 'has_schedule' in df_insight.columns:
            sched_mask = df_insight['has_schedule'] >= 0.5
            sched_blocks = (sched_mask != sched_mask.shift()).cumsum()
            sched_df = df_insight[sched_mask]
            
            fatigue_diffs = []
            focus_scores_rested = []
            focus_scores_rushed = []
            
            for _, group in sched_df.groupby(sched_blocks):
                if len(group) > 1:
                    fatigue_start_val = group['疲労判定'].iloc[0]
                    fatigue_end_val = group['疲労判定'].iloc[-1]
                    duration_hours = len(group) * (freq_td.total_seconds() / 3600)
                    if duration_hours > 0:
                        fatigue_diffs.append((fatigue_end_val - fatigue_start_val) / duration_hours)
                    
                    if 'time_since_prev_event_min' in group.columns and '集中判定' in df_insight.columns:
                        rest_before = group['time_since_prev_event_min'].iloc[0]
                        avg_focus = group['集中判定'].mean()
                        if not np.isnan(rest_before):
                            if rest_before >= 30:
                                focus_scores_rested.append(avg_focus)
                            else:
                                focus_scores_rushed.append(avg_focus)
            
            if len(fatigue_diffs) > 0:
                avg_fatigue_diff = np.mean(fatigue_diffs) * 100
                if avg_fatigue_diff > 0:
                    insight_texts.append(f"- 🕒 **予定中の疲労蓄積**: 1時間の予定をこなすと、疲労の発生割合が平均して **{abs(avg_fatigue_diff):.1f} ポイント増加** します。")
                elif avg_fatigue_diff < 0:
                    insight_texts.append(f"- 🕒 **予定中のリフレッシュ**: 1時間の予定をこなすと、疲労の発生割合が平均して **{abs(avg_fatigue_diff):.1f} ポイント低下** します。予定がリフレッシュになっている可能性があります。")
            
            if len(focus_scores_rested) > 0 and len(focus_scores_rushed) > 0:
                diff_focus = (np.mean(focus_scores_rested) - np.mean(focus_scores_rushed)) * 100
                if diff_focus > 0:
                    insight_texts.append(f"- ☕ **事前の休憩効果**: 予定の前に30分以上の空き時間（休憩）があると、次の予定中の集中発生割合が **平均 {abs(diff_focus):.1f} ポイント高まります**。")
                elif diff_focus < 0:
                    insight_texts.append(f"- 🏃 **連続稼働の強さ**: 予定の前に空き時間がない（連続している）方が、次の予定中の集中発生割合が **平均 {abs(diff_focus):.1f} ポイント高まります**。勢いに乗ると集中できるタイプです。")
                    
        # 2. 予定の連続と回復時間 / 3. 予定後の休憩と回復時間
        if 'fatigue_start' in df_insight.columns and 'focus_start' in df_insight.columns:
            recovery_consecutive = [] 
            recovery_single = []      
            recovery_with_rest = []   
            recovery_no_rest = []     
            
            fatigue_times = df_insight[df_insight['fatigue_start']].index
            focus_times = df_insight[df_insight['focus_start']].index
            
            for fat_time in fatigue_times:
                future_focus = focus_times[focus_times > fat_time]
                if len(future_focus) > 0:
                    first_focus = future_focus[0]
                    if first_focus.date() == fat_time.date():
                        rec_time = (first_focus - fat_time).total_seconds() / 60
                        
                        if 'consecutive_schedules' in df_insight.columns:
                            cons_sched = df_insight.loc[fat_time, 'consecutive_schedules']
                            if cons_sched >= 2:
                                recovery_consecutive.append(rec_time)
                            else:
                                recovery_single.append(rec_time)
                                
                        if 'time_to_next_event_min' in df_insight.columns:
                            t2next = df_insight.loc[fat_time, 'time_to_next_event_min']
                            if not np.isnan(t2next) and t2next >= 30:
                                recovery_with_rest.append(rec_time)
                            else:
                                recovery_no_rest.append(rec_time)
            
            if len(recovery_consecutive) > 0 and len(recovery_single) > 0:
                delay = np.mean(recovery_consecutive) - np.mean(recovery_single)
                if delay > 0:
                    insight_texts.append(f"- 📅 **連続予定の負荷**: 予定が連続している状態での疲労は、単発の予定の疲労に比べて回復が **平均 {abs(delay):.0f} 分遅れます**。")
                elif delay < 0:
                    insight_texts.append(f"- 📅 **連続予定の耐性**: 予定が連続している状態での疲労は、単発の予定に比べて回復が **平均 {abs(delay):.0f} 分早まります**。")
                    
            if len(recovery_with_rest) > 0 and len(recovery_no_rest) > 0:
                speedup = np.mean(recovery_no_rest) - np.mean(recovery_with_rest)
                if speedup > 0:
                    insight_texts.append(f"- 🛋️ **事後の休憩効果**: 疲労状態になった後、次の予定まで30分以上の空き（休憩）があると、回復が **平均 {abs(speedup):.0f} 分早まります**。")
                elif speedup < 0:
                    insight_texts.append(f"- 🛋️ **短い間隔での回復**: 疲労状態になった後、次の予定まで30分以内の短い間隔の方が、回復が **平均 {abs(speedup):.0f} 分早まります**。")

        # 4. 行動特性 (どういった時に集中に入りやすいか)
        if '1分間歩数' in df_insight.columns and 'focus_start' in df_insight.columns:
            walk_before_focus = df_insight['1分間歩数'].shift(1)[df_insight['focus_start']].dropna()
            avg_walk_overall = df_insight['1分間歩数'].mean()
            
            if not walk_before_focus.empty and avg_walk_overall > 0:
                avg_walk_before = walk_before_focus.mean()
                if avg_walk_before > avg_walk_overall * 1.2:
                    insight_texts.append(f"- 🚶 **集中に入りやすい行動**: 集中が始まる直前は、普段より歩数（活動量）が約 {(avg_walk_before/avg_walk_overall):.1f}倍 多い傾向があります。少し歩くなど体を動かした後に集中モードに入りやすいタイプです。")
                elif avg_walk_before < avg_walk_overall * 0.8:
                    insight_texts.append(f"- 🧘 **集中に入りやすい行動**: 集中が始まる直前は、普段より歩数（活動量）が少ない傾向があります。静かな環境で落ち着いてから集中モードに入りやすいタイプです。")

        # 5. 新規: 疲労と回復パターンの追加分析 (アクティブレスト効果)
        if '1分間歩数' in df_insight.columns and 'fatigue_start' in df_insight.columns and 'focus_start' in df_insight.columns:
            active_recovery_times = []
            passive_recovery_times = []
            
            fatigue_times = df_insight[df_insight['fatigue_start']].index
            focus_times = df_insight[df_insight['focus_start']].index
            avg_walk_overall = df_insight['1分間歩数'].mean()
            
            for fat_time in fatigue_times:
                future_focus = focus_times[focus_times > fat_time]
                if len(future_focus) > 0:
                    first_focus = future_focus[0]
                    if first_focus.date() == fat_time.date():
                        rec_time = (first_focus - fat_time).total_seconds() / 60
                        
                        period_walk = df_insight.loc[fat_time:first_focus, '1分間歩数'].mean()
                        if pd.notna(period_walk):
                            if period_walk > avg_walk_overall:
                                active_recovery_times.append(rec_time)
                            else:
                                passive_recovery_times.append(rec_time)
                                
            if len(active_recovery_times) > 0 and len(passive_recovery_times) > 0:
                diff_rest = np.mean(passive_recovery_times) - np.mean(active_recovery_times)
                if diff_rest > 10: 
                    insight_texts.append(f"- 🏃 **アクティブレスト効果**: 疲労時に軽く体を動かす（歩数が平均より多い）と、じっとしている時より **平均 {abs(diff_rest):.0f} 分早く回復** します。")
                elif diff_rest < -10:
                    insight_texts.append(f"- 🛌 **パッシブレスト効果**: 疲労時に体を休める（歩数が平均より少ない）と、動いている時より **平均 {abs(diff_rest):.0f} 分早く回復** します。")

        if insight_texts:
            for text in insight_texts:
                st.write(text)
        else:
            st.write("予定データや疲労・集中データが不足しているため、十分なインサイトを算出できません。")

    # === リアルタイム予測を後ろに移動 ===
    st.header("⚡ リアルタイム予測 (Real-time Focus)")
    st.write(f"モデル精度 (AUC-ROC): **{auc_test:.3f}** / Log Loss: **{logloss_test:.3f}**")
    
    with st.expander("📊 テスト期間の予測確率推移を表示"):
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(test_df.index, y_test_class, label='実際の状態 (1=Yes, 0=No)', color='blue', alpha=0.6, marker='o', linestyle='None')
        ax.plot(test_df.index, preds_proba, label='LightGBM 予測確率', color='red', linestyle='-', alpha=0.8)
        ax.set_title(f"テスト期間の {selected_target_name} 予測確率の推移")
        ax.set_ylabel("確率 / 状態")
        ax.legend()
        st.pyplot(fig)
        
        # ズームグラフの追加
        if TARGET_DATETIME is not None:
            try:
                plot_date = pd.to_datetime(TARGET_DATETIME).date()
            except:
                plot_date = test_df.index[-1].date()
        else:
            plot_date = test_df.index[-1].date()
        
        if plot_date not in test_df.index.date:
            plot_date = test_df.index[-1].date()

        target_indices = test_df[test_df.index.date == plot_date].index
        if len(target_indices) > 0:
            fig_zoom, ax_zoom = plt.subplots(figsize=(10, 4))
            ax_zoom.plot(target_indices, y_test_class.loc[target_indices], label='実際の状態 (1=Yes, 0=No)', color='blue', marker='o', linestyle='None', alpha=0.6)
            
            preds_series = pd.Series(preds_proba, index=test_df.index)
            ax_zoom.plot(target_indices, preds_series.loc[target_indices], label='LightGBM 予測確率', color='red', linestyle='-', marker='x', alpha=0.8)
            
            ax_zoom.set_title(f"予測ズーム（{plot_date}）")
            ax_zoom.set_ylabel("確率 / 状態")
            ax_zoom.legend()
            st.pyplot(fig_zoom)

    # 4. 直近の予測とSHAP
    st.subheader("🔮 リアルタイム予測と要因分析")
    
    available_data_all = df_imp.drop(columns=drop_cols, errors='ignore')
    
    # --- 基準日時の適用 ---
    if TARGET_DATETIME is not None:
        try:
            target_dt = pd.to_datetime(TARGET_DATETIME)
            available_data = available_data_all[available_data_all.index <= target_dt]
            if len(available_data) == 0:
                st.warning("指定された基準日時以前のデータが存在しません。最新のデータを使用します。")
                available_data = available_data_all
        except Exception as e:
            st.warning(f"日時のパースに失敗しました（{e}）。最新のデータを使用します。")
    else:
        available_data = available_data_all

    # 抽出されたデータの最新行を取得
    target_data = available_data.iloc[-1:]
    current_time = target_data.index[0]
    current_val = float(target_data[target_col].values[0])
    current_state_bool = current_val >= target_threshold
    
    current_proba = model.predict_proba(target_data)[0, 1]
    predicted_state_bool = current_proba >= 0.5
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("基準日時", current_time.strftime('%Y-%m-%d %H:%M'))
    col2.metric(f"現在の {selected_target_name} 状態", "Yes" if current_state_bool else "No")
    col3.metric(f"{PREDICT_AHEAD}後の予測判定", "Yes" if predicted_state_bool else "No")
    col4.metric(f"発生確率", f"{current_proba * 100:.1f} %")
    
    st.caption(f"※ **予測判定と発生確率について**: {PREDICT_AHEAD}後にあなたが「{selected_target_name}」の状態になっている確率をAIが算出したものです。50%以上を「Yes」と判定しています。")

    with st.spinner("SHAPで要因を分析しています..."):
        explainer = shap.TreeExplainer(model)
        shap_values_latest = explainer(target_data)
        
        # 二値分類の場合のSHAP値の取り出し
        if len(shap_values_latest.shape) == 3:
            shap_vals = shap_values_latest[0, :, 1].values
            shap_base_obj = shap_values_latest[0, :, 1]
        else:
            shap_vals = shap_values_latest[0].values
            shap_base_obj = shap_values_latest[0]
        
        # 介入可能要因の抽出
        def is_actionable(col: str) -> bool:
            # 予測ターゲット自身に関する特徴量、および時刻・曜日などの直接制御できない要因を除外
            return not (target_col in col or col in ["hour", "dayofweek"])
            
        exp_df = pd.DataFrame({
            'Feature': target_data.columns,
            'Value': target_data.values[0],
            'SHAP': shap_vals
        })
        exp_df['AbsSHAP'] = exp_df['SHAP'].abs()
        exp_df_action = exp_df[exp_df['Feature'].apply(is_actionable)].sort_values('AbsSHAP', ascending=False)
        
        fig2, ax2 = plt.subplots(figsize=(8, 4))
        shap.plots.waterfall(shap_base_obj, show=False)
        st.pyplot(fig2)

        # 要因分析グラフへのコメント追加
        st.markdown("**【要因分析の解説】**")
        
        pos_factors = exp_df_action[exp_df_action['SHAP'] > 0]
        neg_factors = exp_df_action[exp_df_action['SHAP'] < 0]
        
        # ターゲット指標によって「確率上昇」が良いか悪いか分岐させる
        if target_col in ['NEMUKE_SCORE_NEW', '疲労判定', '強い疲労判定']:
            pos_effect_text = "確率上昇（悪化方向）"
            neg_effect_text = "確率低下（好転方向）"
            bar_desc = f"※グラフの赤いバーが{selected_target_name}の発生確率を押し上げる（悪化）要因、青いバーが押し下げる（好転）要因を示しています。"
        else:
            pos_effect_text = "確率上昇（好転方向）"
            neg_effect_text = "確率低下（悪化方向）"
            bar_desc = f"※グラフの赤いバーが{selected_target_name}の発生確率を押し上げる（好転）要因、青いバーが押し下げる（悪化）要因を示しています。"

        base_pos = None
        if not pos_factors.empty:
            top_pos = pos_factors.iloc[0]
            desc_pos = get_factor_direction_text(top_pos['Feature'], top_pos['Value'], available_data_all)
            base_pos = get_base_feature_name(top_pos['Feature'])
            st.write(f"- 📈 **確率を上げる要因**: **{desc_pos}** が{pos_effect_text}に働いています (影響度: {top_pos['SHAP']:+.2f})。")
            
        if not neg_factors.empty:
            top_neg = neg_factors.iloc[0]
            # プラス要因とベース指標が被った場合は、次点の要因を採用する（ユーザーの混乱を防ぐため）
            if base_pos is not None and get_base_feature_name(top_neg['Feature']) == base_pos and len(neg_factors) > 1:
                top_neg = neg_factors.iloc[1]
                
            desc_neg = get_factor_direction_text(top_neg['Feature'], top_neg['Value'], available_data_all)
            st.write(f"- 📉 **確率を下げる要因**: **{desc_neg}** が{neg_effect_text}に働いています (影響度: {top_neg['SHAP']:+.2f})。")
            
        st.caption(bar_desc)

    # 5. ロジックによる働き方判定
    schedule_density = float(target_data["schedule_density_2h"].values[0]) if "schedule_density_2h" in target_data else 0
    time_to_next = float(target_data["time_to_next_event_min"].values[0]) if "time_to_next_event_min" in target_data else np.nan
    is_meeting = float(target_data["is_meeting"].values[0]) if "is_meeting" in target_data else 0
    
    # ターゲット指標の増減を「好転(プラス)」「悪化(マイナス)」の共通軸に変換する
    if target_col in ['NEMUKE_SCORE_NEW', '疲労判定', '強い疲労判定']:
        state_trend_prob = 1.0 - current_proba  # 悪化確率が低い(好転)ほど高い値
    else:
        state_trend_prob = current_proba   # 発生確率が高い(好転)ほど高い値

    reasons = []
    if is_meeting > 0: reasons.append("現在会議中")
    if schedule_density >= 0.6: reasons.append("予定密度が高い")
    
    if state_trend_prob >= 0.6: 
        reasons.append(f"{selected_target_name}の好ましい確率が高い")
    elif state_trend_prob <= 0.4: 
        reasons.append(f"{selected_target_name}の好ましくない確率が高い")
    
    if is_meeting > 0:
        work_mode = "E: 段取り（会議モード）"
        advice = "論点を1枚に整理し、次アクションをToDo化しましょう。"
    elif state_trend_prob >= 0.6 and (np.isnan(time_to_next) or time_to_next >= 50) and schedule_density < 0.6:
        work_mode = "C: アウトプット（深）- 企画・戦略"
        advice = "状態が好転する確率が高く、まとまった時間もあります。設計・企画の骨格づくりなど、重い思考タスクを進めるのが最適です。"
    elif state_trend_prob <= 0.4 or schedule_density >= 0.6:
        work_mode = "D: アウトプット（軽）- 整理・レビュー"
        advice = "予定が細切れか、状態が悪化する確率が高いです。10〜20分で終わるToDo消化や、資料の整形・チェック作業に時間を当てましょう。"
    else:
        if (np.isnan(time_to_next) or time_to_next >= 30) and schedule_density < 0.6:
            work_mode = "A: インプット（重） または B: インプット（軽）"
            advice = "難しめ資料の読み込みや情報整理など、次の深い思考に向けたインプット作業に適しています。"
        else:
            work_mode = "E: 段取り"
            advice = "次の深い作業へスムーズに入れるよう、論点の列挙や優先順位付け、素材の洗い出しを行いましょう。"

    # 6. レポート出力
    st.subheader("📝 分析レポート (AIによる提案)")
    
    main_factor_desc = get_factor_direction_text(exp_df_action.iloc[0]['Feature'], exp_df_action.iloc[0]['Value'], available_data_all) if not exp_df_action.empty else "不明"

    prompt_context = f"""
    現在時刻: {current_time.strftime('%Y-%m-%d %H:%M')}
    現在の{selected_target_name}の状態: {'Yes' if current_state_bool else 'No'}
    {PREDICT_AHEAD}後の予測判定: {'Yes' if predicted_state_bool else 'No'} (発生確率: {current_proba * 100:.1f}%)
    直近の主要因: {main_factor_desc} (SHAP: {exp_df_action.iloc[0]['SHAP']:+.2f})
    判定された働き方: {work_mode}
    理由: {', '.join(reasons) if reasons else '特になし'}
    """
    
    if use_gemini and api_key:
        with st.spinner("Geminiがレポートを作成中..."):
            try:
                genai.configure(api_key=api_key)
                model_llm = genai.GenerativeModel('gemini-2.5-flash')
                prompt = f"以下のデータに基づき、ウェアラブルデータからの客観的な働き方アドバイスレポートを生成してください。\n\n{prompt_context}\n\n構成:\n1. 予測結果と主な要因\n2. 奨励する働き方の具体例"
                resp = model_llm.generate_content(prompt)
                st.write(resp.text)
            except Exception as e:
                st.error(f"Gemini APIエラー: {e}")
    else:
        st.info("💡 Gemini APIキーが未入力のため、ルールベースの詳細レポートを表示します。")
        st.markdown(f"#### 1. 近い将来（{PREDICT_AHEAD}後）の予測結果")
        st.write(f"基準日時（{current_time.strftime('%Y-%m-%d %H:%M')}）の {selected_target_name} は **{'Yes' if current_state_bool else 'No'}** の状態です。")
        st.write(f"{PREDICT_AHEAD}後は **{'Yes' if predicted_state_bool else 'No'}** （発生確率 **{current_proba * 100:.1f} %**）と予測されます。")
        
        st.write(f"この予測の主な要因として、**{main_factor_desc}** が影響しています。")

        st.markdown(f"#### 2. 奨励する働き方")
        st.write(f"現在の予測確率と予定状況（{', '.join(reasons) if reasons else '阻害要因なし'}）から、**「{work_mode}」**に取り組むことをお勧めします。")
        st.write(f"**💡 進め方のアドバイス**: {advice}")

# --- UI レイアウト ---
st.write("### データのアップロード")
col_file1, col_file2 = st.columns(2)
with col_file1:
    file_ts = st.file_uploader("1. 生体データ (CSV形式)", type=['csv'])
with col_file2:
    file_sched = st.file_uploader("2. 予定表データ (予定表.CSV) ※任意", type=['csv'])

if st.button("🚀 分析を実行する", type="primary"):
    if file_ts is not None:
        try:
            # 3行目（インデックス2）からデータラベルとして読み込む
            df_ts = pd.read_csv(file_ts, skiprows=2)
            df_sched = pd.read_csv(file_sched) if file_sched is not None else None
            
            # 分析処理の実行
            run_analysis(df_ts, df_sched, use_gemini=True if api_key else False)
            
        except Exception as e:
            st.error(f"エラーが発生しました: {e}")
    else:
        st.warning("⚠️ 生体データ (CSV形式) をアップロードしてください。")