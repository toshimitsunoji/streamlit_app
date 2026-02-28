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
from sklearn.tree import DecisionTreeRegressor, _tree
import google.generativeai as genai
import shap
import warnings
import math
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
import matplotlib as mpl
import matplotlib.font_manager as fm

# --- Streamlit ページ設定 ---
st.set_page_config(page_title="集中・疲労予測システム", layout="wide")

# 日本語フォントの設定 (アップロードされたOTFフォントを適用)
font_path = Path(__file__).parent / "assets" / "fonts" / "NotoSansCJKjp-Regular.otf"
if font_path.exists():
    fm.fontManager.addfont(str(font_path))
    prop = fm.FontProperties(fname=str(font_path))
    mpl.rcParams["font.family"] = prop.get_name()
else:
    st.warning("⚠️ NotoSansCJKjp-Regular.otf が見つかりません。GitHubで `assets/fonts/` フォルダ内にアップロードされているか確認してください。")

mpl.rcParams["axes.unicode_minus"] = False

warnings.filterwarnings('ignore')

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
TARGET_DATETIME_STR = st.sidebar.text_input("例）2026-01-01 16:00 (空欄で最新データ)")
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
    '集中判定': '集中判定(1=集中)',
    '疲労判定': '疲労判定(1=疲労)',
    '眠気判定': '眠気判定(1=眠気)',
    '強い眠気判定': '強い眠気判定(1=強い眠気)'
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
        '眠気判定': '眠気判定', '強い眠気判定': '強い眠気判定',
        '集中状態': '集中状態', '眠気状態': '眠気状態', '疲労状態': '疲労状態',
        '休憩判定': '休憩判定', '短時間歩行': '短時間歩行',
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
            
    if remainder == "": return base_jp
    elif remainder == "_roll_mean": return f"最近の「{base_jp}」の平均的な高さ"
    elif remainder == "_roll_slope": return f"最近の「{base_jp}」の急な変化(トレンド)"
    elif remainder == "_diff1": return f"前回からの「{base_jp}」の変動幅"
    elif remainder.startswith("_lag"): return f"少し前の「{base_jp}」の状態"
    elif remainder == "_is_missing": return f"「{base_jp}」が未計測であること"
    else: return f"{base_jp}{remainder}"

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
        '眠気判定': '眠気判定', '強い眠気判定': '強い眠気判定',
        '集中状態': '集中状態', '眠気状態': '眠気状態', '疲労状態': '疲労状態',
        '休憩判定': '休憩判定', '短時間歩行': '短時間歩行',
        '集中継続時間': '集中継続時間', '深い集中継続時間': '深い集中継続時間',
        '疲労状態継続時間': '疲労状態継続時間', '疲労継続時間': '疲労状態継続時間', '高心拍継続時間': '高心拍継続時間'
    }
    for k, v in mapping.items():
        if feat.startswith(k): return v
    return feat

def get_factor_direction_text(feat: str, val: float, df_all: pd.DataFrame) -> str:
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
        '眠気判定': '眠気判定', '強い眠気判定': '強い眠気判定',
        '集中状態': '集中状態', '眠気状態': '眠気状態', '疲労状態': '疲労状態',
        '休憩判定': '休憩判定', '短時間歩行': '短時間歩行',
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
            
    if "_is_missing" in feat: return f"「{base_jp}」が未計測であること"
    elif feat in ["has_schedule", "is_meeting"]: return f"「{base_jp}」が入っていること" if val > 0 else f"「{base_jp}」が入っていないこと"
    elif feat in ["休憩判定", "短時間歩行"]: return f"「{base_jp}」をしていること" if val > 0 else f"「{base_jp}」をしていないこと"
    elif feat in ["集中状態", "眠気状態", "疲労状態"]: return f"「{base_jp}」が「{val}」であること"
            
    if "_roll_slope" in feat or "_diff1" in feat:
        direction = "の増加" if val > 0 else "の低下" if val < 0 else "の変化なし"
        return f"最近の「{base_jp}」のトレンド{direction}" if "_roll_slope" in feat else f"前回からの「{base_jp}」{direction}"
    else:
        if "_roll_mean" in feat: return f"最近の「{base_jp}」"
        elif "_lag" in feat: return f"少し前の「{base_jp}」"
        else: return f"「{base_jp}」"

# --- 分析メイン処理 ---
def run_analysis(df_ts, df_sched, use_gemini=False):
    with st.spinner("データを集約・前処理しています..."):
        if 'timestamp' in df_ts.columns:
            df_ts['timestamp_clean'] = df_ts['timestamp'].astype(str).str.split(' GMT').str[0]
            df_ts['datetime'] = pd.to_datetime(df_ts['timestamp_clean'], errors='coerce')
            df_ts = df_ts.dropna(subset=['datetime'])
            df_ts.set_index('datetime', inplace=True)
            df_ts.drop(columns=['timestamp', 'timestamp_clean'], inplace=True, errors='ignore')
            df_ts = df_ts.sort_index()

        base_agg_dict = {
            'SkinTemp': 'mean', 'CVRR_SCORE_NEW': 'mean', 'LP_SCORE_NEW': 'mean',
            'LFHF_SCORE_NEW': 'mean', 'TP': 'mean', 'NEMUKE_SCORE_NEW': 'mean',
            'PR_SCORE_NEW': 'mean', 'RMSSD_SCORE_NEW': 'mean', '1分間歩数': 'sum', 'accDeviation': 'mean',
            '集中判定': 'mean', '疲労判定': 'mean', '強い疲労判定': 'mean', '眠気判定': 'mean', '強い眠気判定': 'mean',
            '休憩判定': 'mean', '短時間歩行': 'mean',
            '集中継続時間': 'mean', '深い集中継続時間': 'mean', '疲労状態継続時間': 'mean', '疲労継続時間': 'mean', '高心拍継続時間': 'mean'
        }
        
        cat_agg_dict = {'集中状態': 'last', '眠気状態': 'last', '疲労状態': 'last'}
        
        agg_dict = {col: func for col, func in base_agg_dict.items() if col in df_ts.columns}
        for col, func in cat_agg_dict.items():
            if col in df_ts.columns: agg_dict[col] = func
        
        if target_col not in agg_dict:
            st.error(f"エラー: 予測に必要な目的変数「{selected_target_name}」がデータに含まれていません。")
            return

        for col in agg_dict.keys():
            if col in base_agg_dict.keys():
                df_ts[col] = pd.to_numeric(df_ts[col], errors='coerce')

        df_resampled = df_ts.resample(RESAMPLE_FREQ).agg(agg_dict)

        if df_sched is not None:
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

            df_resampled['date'] = df_resampled.index.date
            df_resampled = df_resampled.join(df_resampled.groupby('date')['has_schedule'].sum().rename('daily_schedule_hours'), on='date').fillna({'daily_schedule_hours': 0})
            df_resampled.drop(columns=['date'], inplace=True)

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

    with st.spinner("特徴量を生成しています..."):
        df_features = df_resampled.copy()
        df_features['hour'] = df_features.index.hour.astype('category')
        df_features['dayofweek'] = df_features.index.dayofweek.astype('category')
        
        for c in ['集中状態', '眠気状態', '疲労状態']:
            if c in df_features.columns: df_features[c] = df_features[c].astype('category')
        
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

        target_threshold = 0.5 if target_col in ['集中判定', '疲労判定', '強い疲労判定', '眠気判定', '強い眠気判定'] else df_features[target_col].median()
        df_features['target_ahead_class'] = (df_features[target_col].shift(-ahead_steps) >= target_threshold).astype(int)

    with st.spinner("LightGBM分類モデルを学習しています..."):
        drop_cols = ['target_ahead_class']
        df_all = df_features.copy()
        split_idx = int(len(df_all) * 0.8)
        
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
        model = lgb.LGBMClassifier(objective='binary', n_estimators=500, learning_rate=0.03, random_state=42)
        model.fit(X_train, y_train_class, categorical_feature=cat_cols if cat_cols else 'auto')
        preds_proba = model.predict_proba(X_test)[:, 1]
        try:
            auc_test = roc_auc_score(y_test_class, preds_proba)
            logloss_test = log_loss(y_test_class, preds_proba)
        except ValueError:
            auc_test = np.nan
            logloss_test = np.nan


    # =========================================================================
    # パーソナル特性インサイト（長期・月次・日次）
    # =========================================================================
    st.header("👤 パーソナル特性インサイト")
    
    # 共通フィルタ適用
    df_insight = df_imp.copy()
    selected_dow_indices = [dow_options.index(d) for d in selected_dows]
    df_insight = df_insight[df_insight.index.dayofweek.isin(selected_dow_indices)]
    df_insight = df_insight[(df_insight.index.hour >= time_range[0]) & (df_insight.index.hour <= time_range[1])]

    if '集中判定' in df_insight.columns:
        df_insight['focus_start'] = (df_insight['集中判定'] >= 0.5) & (df_insight['集中判定'].shift(1) < 0.5)
    if '疲労判定' in df_insight.columns:
        df_insight['fatigue_start'] = (df_insight['疲労判定'] >= 0.5) & (df_insight['疲労判定'].shift(1) < 0.5)

    def get_peak_time(metric_col):
        if metric_col not in df_insight.columns: return "不明", "不明"
        pivot_df = df_insight.pivot_table(values=metric_col, index=df_insight.index.hour, columns=df_insight.index.dayofweek, aggfunc='mean')
        daytime_pivot = pivot_df.loc[time_range[0]:time_range[1], selected_dow_indices]
        if not daytime_pivot.isna().all().all():
            best_hour, best_dow = daytime_pivot.stack().idxmax()
            return dow_options[int(best_dow)], str(int(best_hour))
        return "不明", "不明"

    f_dow, f_hour = get_peak_time('集中判定')
    fat_dow, fat_hour = get_peak_time('疲労判定')

    avg_focus_duration_str = "不明"
    daily_focus_count_str = "不明"
    daily_total_focus_time_str = "不明"
    focus_durations = pd.Series(dtype=float)
    
    if '集中判定' in df_ts.columns:
        df_1min = df_ts[['集中判定']].resample('1T').mean().ffill(limit=5)
        df_1min = df_1min[df_1min.index.dayofweek.isin(selected_dow_indices)]
        df_1min = df_1min[(df_1min.index.hour >= time_range[0]) & (df_1min.index.hour <= time_range[1])]
        
        focus_mask = df_1min['集中判定'] >= 0.5
        focus_blocks = focus_mask.groupby((focus_mask != focus_mask.shift()).cumsum())
        focus_durations = focus_blocks.sum() 
        focus_durations = focus_durations[focus_durations > 0]
        
        if not focus_durations.empty:
            avg_focus_duration_str = f"{focus_durations.mean():.0f}"
            total_focus_count = len(focus_durations)
            num_days = df_1min.index.normalize().nunique()
            daily_focus_count_str = f"{(total_focus_count / num_days if num_days > 0 else 0):.1f}"
            total_focus_minutes = focus_mask.sum()
            daily_total_focus_time_str = f"{(total_focus_minutes / num_days if num_days > 0 else 0):.0f}"

    focus_actions = []
    if '1分間歩数' in df_insight.columns and 'focus_start' in df_insight.columns:
        walk_before_focus = df_insight['1分間歩数'].shift(1)[df_insight['focus_start']].dropna()
        avg_walk_overall = df_insight['1分間歩数'].mean()
        if not walk_before_focus.empty and avg_walk_overall > 0:
            avg_walk_before = walk_before_focus.mean()
            if avg_walk_before > avg_walk_overall * 1.2: focus_actions.append("事前に体を動かすこと（少し歩くなど）")
            elif avg_walk_before < avg_walk_overall * 0.8: focus_actions.append("事前に静かな環境で落ち着いて過ごすこと")

    if '短時間歩行' in df_insight.columns and 'focus_start' in df_insight.columns:
        walk_before = df_insight['短時間歩行'].shift(1)[df_insight['focus_start']].dropna()
        avg_overall = df_insight['短時間歩行'].mean()
        if not walk_before.empty and avg_overall > 0:
            if walk_before.mean() > avg_overall * 1.2: focus_actions.append("事前に短時間歩行（リフレッシュ）を取り入れること")

    if '休憩判定' in df_insight.columns and 'focus_start' in df_insight.columns:
        rest_before = df_insight['休憩判定'].shift(1)[df_insight['focus_start']].dropna()
        avg_overall = df_insight['休憩判定'].mean()
        if not rest_before.empty and avg_overall > 0:
            if rest_before.mean() > avg_overall * 1.2: focus_actions.append("事前にしっかり休憩をとること")

    if 'has_schedule' in df_insight.columns and '集中判定' in df_insight.columns:
        sched_mask = df_insight['has_schedule'] >= 0.5
        sched_blocks = (sched_mask != sched_mask.shift()).cumsum()
        sched_df = df_insight[sched_mask]
        focus_scores_rested, focus_scores_rushed = [], []
        for _, group in sched_df.groupby(sched_blocks):
            if len(group) > 1 and 'time_since_prev_event_min' in group.columns:
                rest_before = group['time_since_prev_event_min'].iloc[0]
                if not np.isnan(rest_before):
                    if rest_before >= 30: focus_scores_rested.append(group['集中判定'].mean())
                    else: focus_scores_rushed.append(group['集中判定'].mean())
        if focus_scores_rested and focus_scores_rushed:
            diff_focus = (np.mean(focus_scores_rested) - np.mean(focus_scores_rushed)) * 100
            if diff_focus > 0: focus_actions.append("予定の前に30分以上の空き時間（休憩）をとること")
            elif diff_focus < 0: focus_actions.append("予定と予定の間を空けずに連続して活動すること")

    focus_actions_str = "データ不足のため特定できません" if not focus_actions else "、".join(focus_actions)

    fatigue_actions = []
    if '疲労判定' in df_insight.columns and 'has_schedule' in df_insight.columns:
        sched_mask = df_insight['has_schedule'] >= 0.5
        sched_blocks = (sched_mask != sched_mask.shift()).cumsum()
        sched_df = df_insight[sched_mask]
        fatigue_diffs = []
        for _, group in sched_df.groupby(sched_blocks):
            if len(group) > 1:
                duration_hours = len(group) * (freq_td.total_seconds() / 3600)
                if duration_hours > 0:
                    fatigue_diffs.append((group['疲労判定'].iloc[-1] - group['疲労判定'].iloc[0]) / duration_hours)
        if fatigue_diffs and np.mean(fatigue_diffs) > 0:
            fatigue_actions.append("1時間以上の予定をこなすこと")

    if 'fatigue_start' in df_insight.columns and 'focus_start' in df_insight.columns:
        recovery_consecutive, recovery_single = [], []
        fatigue_times, focus_times = df_insight[df_insight['fatigue_start']].index, df_insight[df_insight['focus_start']].index
        for fat_time in fatigue_times:
            future_focus = focus_times[focus_times > fat_time]
            if len(future_focus) > 0 and future_focus[0].date() == fat_time.date():
                if 'consecutive_schedules' in df_insight.columns:
                    if df_insight.loc[fat_time, 'consecutive_schedules'] >= 2: recovery_consecutive.append(1)
                    else: recovery_single.append(1)
        if recovery_consecutive and recovery_single and (np.mean(recovery_consecutive) - np.mean(recovery_single)) > 0:
            fatigue_actions.append("予定を連続して入れること")

    fatigue_actions_str = "データ不足のため特定できません" if not fatigue_actions else "、".join(fatigue_actions)

    recovery_actions = []
    if 'fatigue_start' in df_insight.columns and 'focus_start' in df_insight.columns:
        fatigue_times, focus_times = df_insight[df_insight['fatigue_start']].index, df_insight[df_insight['focus_start']].index
        
        if '短時間歩行' in df_insight.columns:
            rec_with_walk, rec_no_walk = [], []
            for fat_time in fatigue_times:
                future_focus = focus_times[focus_times > fat_time]
                if len(future_focus) > 0 and future_focus[0].date() == fat_time.date():
                    first_focus = future_focus[0]
                    rec_time = (first_focus - fat_time).total_seconds() / 60
                    period_val = df_insight.loc[fat_time:first_focus, '短時間歩行'].mean()
                    if pd.notna(period_val):
                        if period_val > df_insight['短時間歩行'].mean(): rec_with_walk.append(rec_time)
                        else: rec_no_walk.append(rec_time)
            if rec_with_walk and rec_no_walk:
                diff = np.mean(rec_no_walk) - np.mean(rec_with_walk)
                if diff > 10: recovery_actions.append(f"短時間歩行（動的リフレッシュ）を行うこと（平均{abs(diff):.0f}分早く回復）")
                elif diff < -10: recovery_actions.append(f"歩き回らず静かに休むこと（平均{abs(diff):.0f}分早く回復）")
                
        if '休憩判定' in df_insight.columns:
            rec_with_rest, rec_no_rest = [], []
            for fat_time in fatigue_times:
                future_focus = focus_times[focus_times > fat_time]
                if len(future_focus) > 0 and future_focus[0].date() == fat_time.date():
                    first_focus = future_focus[0]
                    rec_time = (first_focus - fat_time).total_seconds() / 60
                    period_val = df_insight.loc[fat_time:first_focus, '休憩判定'].mean()
                    if pd.notna(period_val):
                        if period_val > df_insight['休憩判定'].mean(): rec_with_rest.append(rec_time)
                        else: rec_no_rest.append(rec_time)
            if rec_with_rest and rec_no_rest:
                diff = np.mean(rec_no_rest) - np.mean(rec_with_rest)
                if diff > 10: recovery_actions.append(f"意識的に休憩時間をとること（平均{abs(diff):.0f}分早く回復）")

    recovery_actions_str = "データ不足のため特定できません" if not recovery_actions else "、".join(recovery_actions)

    # --- 4つのタブを作成 ---
    tab1, tab2, tab3, tab4 = st.tabs(["📝 マイ・スペック", "📅 マンスリーインサイト", "☀️ デイリーインサイト", "📊 行動リターン分析"])
    
    with tab1:
        st.markdown("#### あなたの集中特性")
        st.markdown(f"　{f_dow}曜日の{f_hour}時台に最も集中しやすい傾向があります。<br>"
                    f"　平均集中持続時間は{avg_focus_duration_str}分です。<br>"
                    f"　1日の平均集中時間は{daily_total_focus_time_str}分です。<br>"
                    f"　1日に{daily_focus_count_str}回集中と緩和のリズムを繰り返しています。<br>"
                    f"　集中に入りやすい行動は{focus_actions_str}", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("#### あなたの疲労特性")
        st.markdown(f"　{fat_dow}曜日の{fat_hour}時台に最も疲労しやすい傾向があります。<br>"
                    f"　疲労しやすい行動は{fatigue_actions_str}<br>"
                    f"　疲労から早く回復する行動は{recovery_actions_str}", unsafe_allow_html=True)

        if not focus_durations.empty:
            st.markdown("<br>##### 集中持続時間の分布", unsafe_allow_html=True)
            max_duration = int(focus_durations.max())
            max_bin = math.ceil(max_duration / 10) * 10
            bins = np.arange(0, max_bin + 20, 10) 
            counts, edges = np.histogram(focus_durations, bins=bins)
            xtick_labels = [f"{int(edges[i])}-{int(edges[i+1])-1}" for i in range(len(edges)-1)]
            
            fig_dist = go.Figure(data=[go.Bar(
                x=xtick_labels, y=counts, marker_color='#4A90E2', opacity=0.8,
                hovertemplate="集中時間: %{x}分<br>回数: %{y}回<extra></extra>"
            )])
            fig_dist.update_layout(
                xaxis_title="集中持続時間 (分)", yaxis_title="回数", height=300,
                margin=dict(l=20, r=20, t=20, b=20), plot_bgcolor='rgba(0,0,0,0)', bargap=0.1
            )
            st.plotly_chart(fig_dist, use_container_width=True)

        st.markdown("##### 時間帯・曜日別の傾向 (ヒートマップ)")
        def plot_heatmap_plotly(target_metric, colorscale_name):
            if target_metric not in df_imp.columns: return None
            pivot_df = df_imp.pivot_table(values=target_metric, index=df_imp.index.hour, columns=df_imp.index.dayofweek, aggfunc='mean')
            heatmap_data = np.full((time_range[1] - time_range[0] + 1, 7), np.nan)
            for h in pivot_df.index:
                if time_range[0] <= h <= time_range[1]:
                    for d in pivot_df.columns:
                        if d in selected_dow_indices: heatmap_data[int(h) - time_range[0], int(d)] = pivot_df.loc[h, d]
            
            fig_hm = go.Figure(data=go.Heatmap(
                z=heatmap_data, x=dow_options, y=[f"{h}:00" for h in range(time_range[0], time_range[1] + 1)],
                colorscale=colorscale_name, hoverongaps=False, hovertemplate="曜日: %{x}<br>時間帯: %{y}<br>確率: %{z:.2f}<extra></extra>"
            ))
            fig_hm.update_layout(yaxis_autorange='reversed', height=350, margin=dict(l=20, r=20, t=20, b=20))
            return fig_hm

        col_h1, col_h2 = st.columns(2)
        with col_h1:
            st.markdown("**🎯 集中確率**")
            fig_focus = plot_heatmap_plotly('集中判定', 'Blues')
            if fig_focus: st.plotly_chart(fig_focus, use_container_width=True)
        with col_h2:
            st.markdown("**🔋 疲労確率**")
            fig_fatigue = plot_heatmap_plotly('疲労判定', 'Reds')
            if fig_fatigue: st.plotly_chart(fig_fatigue, use_container_width=True)

    with tab2:
        df_ts['year_month'] = df_ts.index.to_period('M').astype(str)
        available_months = sorted(df_ts['year_month'].unique().tolist(), reverse=True)
        
        if not available_months:
            st.write("分析可能な月のデータがありません。")
        else:
            selected_month = st.selectbox("分析対象とする年月を選択してください", available_months)
            df_month = df_ts[df_ts['year_month'] == selected_month]
            
            if '集中判定' in df_month.columns:
                # 1時間単位での集中回数（分数に相当）を集計
                df_m_1t = df_month[['集中判定']].resample('1T').mean()
                df_m_1t['集中判定_フラグ'] = (df_m_1t['集中判定'] >= 0.5).astype(int)
                
                df_m_hourly = df_m_1t.resample('1H').sum()
                df_m_hourly['day'] = df_m_hourly.index.day
                df_m_hourly['hour'] = df_m_hourly.index.hour
                df_m_hourly['dow'] = df_m_hourly.index.dayofweek
                
                # 設定された時間帯（time_range）のリストを作成
                target_hours = list(range(time_range[0], time_range[1] + 1))
                
                # グラフ: 曜日別・時間帯別
                col_m1, col_m2 = st.columns(2)
                
                with col_m1:
                    dow_sum = df_m_hourly.groupby('dow')['集中判定_フラグ'].sum().reindex(range(7), fill_value=0)
                    fig_dow = px.bar(x=dow_options, y=dow_sum.values, labels={'x': '曜日', 'y': '集中判定回数'}, title="曜日別 集中判定回数")
                    fig_dow.update_traces(marker_color='#4A90E2')
                    st.plotly_chart(fig_dow, use_container_width=True)
                    
                with col_m2:
                    # 時間帯を対象時間帯のみに絞り込む
                    hour_sum = df_m_hourly.groupby('hour')['集中判定_フラグ'].sum().reindex(target_hours, fill_value=0)
                    fig_hour = px.bar(x=[f"{h}:00" for h in target_hours], y=hour_sum.values, labels={'x': '時間帯', 'y': '集中判定回数'}, title="時間帯別 集中判定回数")
                    fig_hour.update_traces(marker_color='#4A90E2')
                    st.plotly_chart(fig_hour, use_container_width=True)
                
                # グラフ: 日×時間のヒートマップ
                st.markdown("#### 日付×時間帯の集中判定回数 (赤枠は予定あり)")
                hm_pivot = df_m_hourly.pivot_table(index='day', columns='hour', values='集中判定_フラグ', aggfunc='sum').fillna(0)
                
                # 欠けている日・時間を補完し、対象時間帯のみに絞り込む
                if len(df_month.index) > 0:
                    days_in_month = df_month.index[0].days_in_month
                else:
                    days_in_month = 31 # fallback
                all_days = list(range(1, days_in_month + 1))
                hm_pivot = hm_pivot.reindex(index=all_days, columns=target_hours, fill_value=0)
                
                fig_hm_month = go.Figure(data=go.Heatmap(
                    z=hm_pivot.values,
                    x=[f"{h}:00" for h in target_hours],
                    y=[f"{d}日" for d in all_days],
                    colorscale='Blues',
                    hovertemplate="日付: %{y}<br>時間帯: %{x}<br>集中回数: %{z}<extra></extra>"
                ))
                
                # 予定がある時間帯に赤枠（Shapes）を追加
                shapes = []
                if df_sched is not None and not df_sched.empty:
                    for d in all_days:
                        for h in target_hours:
                            try:
                                dt_start = pd.to_datetime(f"{selected_month}-{d:02d} {h:02d}:00:00")
                                dt_end = dt_start + pd.Timedelta('1H')
                                has_sched = ((df_sched['start_dt'] < dt_end) & (df_sched['end_dt'] > dt_start)).any()
                                if has_sched:
                                    # 横軸が絞り込まれたため、インデックスを計算し直す
                                    x_idx = h - time_range[0]
                                    shapes.append(dict(
                                        type="rect",
                                        x0=x_idx - 0.5, x1=x_idx + 0.5,
                                        y0=d - 1 - 0.5, y1=d - 1 + 0.5, # y0,y1 はインデックス(0始まり)で指定
                                        line=dict(color="red", width=2),
                                        fillcolor="rgba(0,0,0,0)"
                                    ))
                            except ValueError:
                                pass # 存在しない日付（うるう年など）はスキップ
                
                fig_hm_month.update_layout(
                    shapes=shapes,
                    yaxis_autorange='reversed',
                    height=600,
                    margin=dict(l=20, r=20, t=20, b=20)
                )
                st.plotly_chart(fig_hm_month, use_container_width=True)
                
                # コメントの生成
                best_dow_m = dow_options[dow_sum.idxmax()] if dow_sum.sum() > 0 else "不明"
                best_hour_m = hour_sum.idxmax() if hour_sum.sum() > 0 else "不明"
                
                st.info(f"**【{selected_month} のマンスリーインサイト】**\n\n"
                        f"- この月は **{best_dow_m}曜日** の集中判定回数が最も多くなっています。\n"
                        f"- 時間帯で見ると **{best_hour_m}時台** に集中する傾向が強かったです。\n"
                        f"- ヒートマップ上の赤枠は「予定（会議など）」が入っている時間帯を示しています。予定と集中の相関関係を視覚的に確認できます。")
            else:
                st.write("「集中判定」データが不足しているため表示できません。")

    with tab3:
        df_ts['date_str'] = df_ts.index.date.astype(str)
        available_days = sorted(df_ts['date_str'].unique().tolist(), reverse=True)
        
        if not available_days:
            st.write("分析可能な日のデータがありません。")
        else:
            selected_day = st.selectbox("分析対象とする年月日を選択してください", available_days)
            df_day = df_ts[df_ts['date_str'] == selected_day].copy()
            
            # 設定された時間帯（time_range）でデータをフィルタリング
            df_day = df_day[(df_day.index.hour >= time_range[0]) & (df_day.index.hour <= time_range[1])]
            
            score_col = 'CVRR_SCORE_NEW'
            graph_title_base = "集中と緩和"
            score_label = "CVRR SCORE (集中度合い)"
            state_high = "集中"
            state_low = "緩和（リラックス）"
            
            if target_col in ['疲労判定', '強い疲労判定']:
                score_col = 'RMSSD_SCORE_NEW'
                graph_title_base = "疲労と回復"
                score_label = "RMSSD SCORE (疲労・回復度合い)"
                state_high = "回復（リラックス）"
                state_low = "疲労（ストレス）"
            elif target_col in ['眠気判定', '強い眠気判定']:
                score_col = 'NEMUKE_SCORE_NEW'
                graph_title_base = "眠気と覚醒"
                score_label = "NEMUKE SCORE (眠気度合い)"
                state_high = "低覚醒（眠気）"
                state_low = "覚醒"
            
            if score_col in df_day.columns and not df_day.empty:
                st.markdown(f"#### モメンタルグラフ ({graph_title_base}の波)")
                
                base_val = 50.0 # 基準となる平均値
                
                fig_daily = go.Figure()
                
                # 基準線(50)を描画（ホバーはスキップ）
                fig_daily.add_trace(go.Scatter(
                    x=df_day.index, y=[base_val]*len(df_day),
                    mode='lines', line=dict(color='gray', width=1, dash='dash'),
                    name='基準(50)', hoverinfo='skip'
                ))
                
                # 上側（集中）の青い面
                y_upper = np.where(df_day[score_col] >= base_val, df_day[score_col], base_val)
                fig_daily.add_trace(go.Scatter(
                    x=df_day.index, y=y_upper,
                    fill='tonexty', fillcolor='rgba(54, 162, 235, 0.5)', # 青系
                    mode='lines', line=dict(color='rgba(0,0,0,0)', width=0),
                    showlegend=False, hoverinfo='skip'
                ))
                
                # 下側の面を描くために、ベースラインをもう一度引く
                fig_daily.add_trace(go.Scatter(
                    x=df_day.index, y=[base_val]*len(df_day),
                    mode='lines', line=dict(color='rgba(0,0,0,0)', width=0),
                    showlegend=False, hoverinfo='skip'
                ))
                
                # 下側（緩和）のオレンジ系の面
                y_lower = np.where(df_day[score_col] <= base_val, df_day[score_col], base_val)
                fig_daily.add_trace(go.Scatter(
                    x=df_day.index, y=y_lower,
                    fill='tonexty', fillcolor='rgba(255, 159, 64, 0.5)', # オレンジ系
                    mode='lines', line=dict(color='rgba(0,0,0,0)', width=0),
                    showlegend=False, hoverinfo='skip'
                ))
                
                # ホバー・表示用の実際の推移線（黒色）
                fig_daily.add_trace(go.Scatter(
                    x=df_day.index, 
                    y=df_day[score_col],
                    mode='lines',
                    line=dict(color='#333333', width=2),
                    name=score_col,
                    hovertemplate="時刻: %{x|%H:%M}<br>スコア: %{y:.1f}<extra></extra>"
                ))
                
                fig_daily.update_layout(
                    title=f"{selected_day} の{graph_title_base}の推移 ({time_range[0]}時〜{time_range[1]}時)",
                    xaxis_title="時刻",
                    yaxis_title=score_label,
                    height=400,
                    hovermode="x unified",
                    plot_bgcolor='rgba(0,0,0,0)',
                    margin=dict(l=20, r=20, t=40, b=20)
                )
                fig_daily.update_xaxes(showgrid=True, gridcolor='lightgray', showline=True, linewidth=1, linecolor='black')
                fig_daily.update_yaxes(showgrid=True, gridcolor='lightgray', showline=True, linewidth=1, linecolor='black')
                st.plotly_chart(fig_daily, use_container_width=True)
                
                # コメントの生成
                if not df_day[score_col].isna().all():
                    max_idx = df_day[score_col].idxmax()
                    max_val = df_day[score_col].max()
                    avg_val = df_day[score_col].mean()
                    
                    st.info(f"**【{selected_day} のデイリーインサイト】**\n\n"
                            f"- この日の設定時間帯（{time_range[0]}時〜{time_range[1]}時）におけるスコアのピークは **{max_idx.strftime('%H:%M')}頃** （スコア: {max_val:.1f}）でした。\n"
                            f"- 平均スコアは **{avg_val:.1f}** となっています。\n"
                            f"- グラフにおいて基準値(50)より上側の**青い面**が「{state_high}」している状態、下側の**オレンジの面**が「{state_low}」している状態を示しています。")
                else:
                    st.write("この日の有効なスコアデータがありません。")
            else:
                st.write(f"対象時間帯のデータがない、または「{score_col}」が含まれていないため、モメンタルグラフを表示できません。")

    with tab4:
        st.markdown("#### 行動リターン分析（重回帰分析）")
        st.markdown(f"過去のデータから、「現在」および「直前（{RESAMPLE_FREQ}前）」の休憩や短時間歩行といった行動が、パフォーマンスにどれだけのプラス/マイナス効果を与えているかを統計的に算出します。")
        
        reg_df = df_imp.copy()
        lag_steps = 1 # 直前（1ステップ前）の行動を評価するためにシフト
        
        action_cols = []
        if '休憩判定' in reg_df.columns: 
            action_cols.append('休憩判定')
            reg_df['休憩判定_前'] = reg_df['休憩判定'].shift(lag_steps)
            action_cols.append('休憩判定_前')
        if '短時間歩行' in reg_df.columns: 
            action_cols.append('短時間歩行')
            reg_df['短時間歩行_前'] = reg_df['短時間歩行'].shift(lag_steps)
            action_cols.append('短時間歩行_前')
        
        control_cols = []
        if 'is_meeting' in reg_df.columns: control_cols.append('is_meeting')
        if 'schedule_density_2h' in reg_df.columns: control_cols.append('schedule_density_2h')
        
        if not action_cols:
            st.write("分析に必要な行動データ（「休憩判定」や「短時間歩行」）が存在しません。")
        else:
            X_cols = action_cols + control_cols
            reg_df = reg_df.dropna(subset=X_cols + [target_col])
            
            if len(reg_df) > 10:
                X = reg_df[X_cols].astype(float)
                y = reg_df[target_col].astype(float)
                
                try:
                    import statsmodels.api as sm
                    # 定数項（切片）を追加してOLSモデルを学習
                    X_sm = sm.add_constant(X)
                    model_sm = sm.OLS(y, X_sm)
                    results = model_sm.fit()
                    
                    # 統計値の取得
                    nobs = int(results.nobs)
                    r2 = results.rsquared
                    r2_adj = results.rsquared_adj
                    
                    coef_dict = {}
                    pvalue_dict = {}
                    for col in action_cols:
                        if col in results.params:
                            coef_dict[col] = results.params[col]
                            pvalue_dict[col] = results.pvalues[col]
                    
                    # --- 統計サマリの表示 ---
                    st.markdown("##### 📈 統計サマリ")
                    col_s1, col_s2, col_s3 = st.columns(3)
                    col_s1.metric("サンプル数 (n)", f"{nobs} 件")
                    col_s2.metric("決定係数 (R²)", f"{r2:.3f}")
                    col_s3.metric("自由度調整済 R²", f"{r2_adj:.3f}")
                    
                    st.markdown("##### 📝 回帰係数とP値の詳細")
                    detail_data = []
                    for col in results.params.index:
                        if col == "const":
                            col_name = "定数項 (Intercept)"
                        elif '_前' in col:
                            base_name = jp_feat_name(col.replace('_前', ''))
                            col_name = f"直前の「{base_name}」"
                        else:
                            base_name = jp_feat_name(col)
                            if col in ['休憩判定', '短時間歩行']:
                                col_name = f"現在の「{base_name}」"
                            else:
                                col_name = f"「{base_name}」"
                            
                        pval = results.pvalues[col]
                        sig = "⭐ 有意" if pval < 0.05 else "ー"
                        
                        detail_data.append({
                            "変数名": col_name,
                            "係数 (効果量)": results.params[col],
                            "標準誤差": results.bse[col],
                            "t値": results.tvalues[col],
                            "P値": pval,
                            "有意判定": sig
                        })
                    
                    df_detail = pd.DataFrame(detail_data)
                    st.dataframe(df_detail.style.format({
                        "係数 (効果量)": "{:.4f}",
                        "標準誤差": "{:.4f}",
                        "t値": "{:.3f}",
                        "P値": "{:.4f}"
                    }), use_container_width=True)
                    st.caption("※ P値が0.05未満（5%水準）の場合、「統計的に有意（偶然ではなく実際に効果がある可能性が高い）」と判定されます。")

                except ImportError:
                    st.warning("詳細な統計情報を表示するためには `statsmodels` ライブラリが必要です。`requirements.txt` に `statsmodels` を追加してください。今回は `scikit-learn` による簡易分析を表示します。")
                    from sklearn.linear_model import LinearRegression
                    model_reg = LinearRegression()
                    model_reg.fit(X, y)
                    coef_dict = {col: coef for col, coef in zip(X_cols, model_reg.coef_) if col in action_cols}
                    pvalue_dict = {col: np.nan for col in action_cols}

                # グラフ描画
                action_names = []
                for col in coef_dict.keys():
                    if '_前' in col:
                        action_names.append(f"直前の「{jp_feat_name(col.replace('_前', ''))}」")
                    else:
                        action_names.append(f"現在の「{jp_feat_name(col)}」")

                coef_values = list(coef_dict.values())
                colors = ['#E24A4A' if c < 0 else '#4AE290' for c in coef_values]
                
                fig_roi = go.Figure(data=[go.Bar(
                    x=action_names, 
                    y=coef_values, 
                    marker_color=colors,
                    text=[f"{c*100:+.1f} pt" for c in coef_values],
                    textposition='auto',
                    hovertemplate="行動: %{x}<br>効果量: %{y:+.3f}<extra></extra>"
                )])
                
                target_label = jp_feat_name(target_col)
                fig_roi.update_layout(
                    title=f"各行動が「{target_label}」に与える純粋な効果量",
                    xaxis_title="行動",
                    yaxis_title="効果量 (係数)",
                    height=400,
                    margin=dict(l=20, r=20, t=40, b=20),
                    plot_bgcolor='rgba(0,0,0,0)'
                )
                fig_roi.update_yaxes(showgrid=True, gridcolor='lightgray', zeroline=True, zerolinecolor='black', zerolinewidth=1)
                st.plotly_chart(fig_roi, use_container_width=True)
                
                # インサイトの生成
                st.markdown("##### 💡 分析結果（行動の投資対効果）")
                for col, coef in coef_dict.items():
                    if '_前' in col:
                        action_desc = f"事前に「{jp_feat_name(col.replace('_前', ''))}」を行うこと"
                    else:
                        action_desc = f"現在「{jp_feat_name(col)}」を行うこと"

                    effect_pt = coef * 100
                    pval = pvalue_dict.get(col, np.nan)
                    
                    sig_note = ""
                    if not np.isnan(pval) and pval >= 0.05:
                        sig_note = " *(※P値が0.05以上のため、この効果は偶然の誤差の範囲である可能性があります)*"

                    if target_col in ['NEMUKE_SCORE_NEW', '疲労判定', '強い疲労判定', '眠気判定', '強い眠気判定']:
                        # 悪化系の指標の場合（マイナスが良い効果）
                        if coef < -0.01:
                            st.write(f"- 🟢 **{action_desc}**: 「{target_label}」の発生を **平均 {abs(effect_pt):.1f} ポイント抑える** 効果（リフレッシュ効果）が確認されました。{sig_note}")
                        elif coef > 0.01:
                            st.write(f"- 🔴 **{action_desc}**: 逆に「{target_label}」の発生を **平均 {abs(effect_pt):.1f} ポイント悪化** させてしまう傾向があります。タイミングの見直しが必要かもしれません。{sig_note}")
                        else:
                            st.write(f"- ⚪ **{action_desc}**: 「{target_label}」に対する直接的な増減効果はほとんど見られませんでした。")
                    else:
                        # 好転系の指標の場合（プラスが良い効果）
                        if coef > 0.01:
                            st.write(f"- 🟢 **{action_desc}**: 「{target_label}」の発生を **平均 {abs(effect_pt):.1f} ポイント高める** 効果（ブースト効果）が確認されました。積極的に取り入れましょう。{sig_note}")
                        elif coef < -0.01:
                            st.write(f"- 🔴 **{action_desc}**: 逆に「{target_label}」の発生を **平均 {abs(effect_pt):.1f} ポイント低下** させてしまう傾向があります。{sig_note}")
                        else:
                            st.write(f"- ⚪ **{action_desc}**: 「{target_label}」に対する直接的な増減効果はほとんど見られませんでした。")
                            
                st.caption("※この結果は「現在の予定の詰まり具合」や「会議中かどうか」といった他の条件（ノイズ）を統計的に除去し、行動そのものの純粋な効果を抽出したものです。")
                
                # --- 決定木分析によるマイルール抽出 ---
                st.markdown("---")
                st.markdown("##### 🌳 条件の組み合わせ分析（マイ・ルール抽出）")
                st.write("決定木アルゴリズムを用いて、複数の条件（予定の状況と行動）が組み合わさった時に、パフォーマンスがどう変化するかを分析します。")
                
                # ツリーモデルの学習 (分かりやすくするため深さを2に制限)
                from sklearn.tree import DecisionTreeRegressor, _tree, plot_tree
                tree_model = DecisionTreeRegressor(max_depth=2, min_samples_leaf=5, random_state=42)
                tree_model.fit(X, y)
                
                # 特徴量表示名と真偽値判定のリスト作成
                feature_display_names = []
                feature_is_bool = []
                for col in X_cols:
                    feature_is_bool.append(reg_df[col].dropna().nunique() <= 2)
                    if col == 'is_meeting':
                        feature_display_names.append("会議中")
                    elif col == 'schedule_density_2h':
                        feature_display_names.append("予定密度")
                    elif '_前' in col:
                        base = get_base_feature_name(col.replace('_前', ''))
                        feature_display_names.append(f"直前の{base}")
                    else:
                        base = get_base_feature_name(col)
                        if col in ['休憩判定', '短時間歩行']:
                            feature_display_names.append(f"現在の{base}")
                        else:
                            feature_display_names.append(jp_feat_name(col))

                def extract_rules(tree, feature_names, is_bool_list):
                    tree_ = tree.tree_
                    feature_name = [
                        feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
                        for i in tree_.feature
                    ]
                    rules = []
                    def recurse(node, current_rule):
                        if tree_.feature[node] != _tree.TREE_UNDEFINED:
                            name = feature_name[node]
                            threshold = tree_.threshold[node]
                            is_bool = is_bool_list[tree_.feature[node]]
                            
                            left_rule = current_rule.copy()
                            if is_bool:
                                left_rule.append(f"【{name}：なし】")
                            else:
                                left_rule.append(f"【{name}が低い (≦{threshold:.2f})】")
                            recurse(tree_.children_left[node], left_rule)
                            
                            right_rule = current_rule.copy()
                            if is_bool:
                                right_rule.append(f"【{name}：あり】")
                            else:
                                right_rule.append(f"【{name}が高い (>{threshold:.2f})】")
                            recurse(tree_.children_right[node], right_rule)
                        else:
                            val = tree_.value[node][0][0]
                            samples = tree_.n_node_samples[node]
                            rules.append((" ＋ ".join(current_rule), val, samples))
                    recurse(0, [])
                    return rules
                
                tree_rules = extract_rules(tree_model, feature_display_names, feature_is_bool)
                is_negative_target = target_col in ['NEMUKE_SCORE_NEW', '疲労判定', '強い疲労判定', '眠気判定', '強い眠気判定']
                
                # サンプル数が全体の5%以上のルールのみ抽出
                min_samples_threshold = max(3, int(len(reg_df) * 0.05))
                valid_rules = [r for r in tree_rules if r[2] >= min_samples_threshold]
                if not valid_rules:
                    valid_rules = tree_rules
                
                valid_rules.sort(key=lambda x: x[1], reverse=not is_negative_target)
                
                st.markdown(f"**🎯 あなたの「{target_label}」に関するベスト条件パターン**")
                
                if is_negative_target:
                    st.write(f"※スコアが**低い**（発生確率が低い）パターンをベスト条件として表示しています。")
                else:
                    st.write(f"※スコアが**高い**（発生確率が高い）パターンをベスト条件として表示しています。")

                if valid_rules:
                    rule_text, val, samples = valid_rules[0]
                    display_val = val * 100
                    st.markdown(f"🥇 **第1位** (データ数: {samples}件)")
                    st.markdown(f"　条件： {rule_text}")
                    st.markdown(f"　👉 予想スコア: **{display_val:.1f} pt**")
                else:
                    st.write("有効なルールが見つかりませんでした。")
                    
                # 樹形図の描画
                st.markdown("##### 🌿 決定木の樹形図")
                st.caption("※ 一番上のハコからスタートし、条件が「True（当てはまる）」なら左へ、「False（当てはまらない）」なら右へ進みます。色の濃さはスコアの高低を表します。")
                fig_tree, ax_tree = plt.subplots(figsize=(10, 6))
                plot_tree(tree_model, feature_names=feature_display_names, filled=True, rounded=True, ax=ax_tree, fontsize=12, precision=2)
                st.pyplot(fig_tree)
                
                # --- 分析データのダウンロードボタン追加 ---
                st.markdown("---")
                st.markdown("##### 📥 データダウンロード")
                st.write("この重回帰分析に実際に使用されたデータ（欠損値等を除去したクリーンなデータ）をCSVファイルとしてダウンロードできます。")
                
                # Excelで開いた際の文字化けを防ぐために utf-8-sig (BOM付きUTF-8) を使用
                csv_data = reg_df[X_cols + [target_col]].to_csv().encode('utf-8-sig')
                st.download_button(
                    label="📊 分析用データをダウンロード (.csv)",
                    data=csv_data,
                    file_name='regression_analysis_data.csv',
                    mime='text/csv',
                )

            else:
                st.write("有効なデータが少なすぎるため、統計分析を実行できません。")

    # =========================================================================
    # リアルタイム予測 (Real-time Focus)
    # =========================================================================
    st.header("⚡ リアルタイム予測 (Real-time Focus)")
    
    auc_eval = "算出不可"
    if not np.isnan(auc_test):
        if auc_test >= 0.8: auc_eval = "🟢 非常に良い"
        elif auc_test >= 0.7: auc_eval = "🔵 良い (実用レベル)"
        elif auc_test >= 0.6: auc_eval = "🟡 普通"
        else: auc_eval = "🔴 改善が必要"

    loss_eval = "算出不可"
    if not np.isnan(logloss_test):
        if logloss_test <= 0.4: loss_eval = "🟢 非常に良い"
        elif logloss_test <= 0.6: loss_eval = "🟡 普通"
        else: loss_eval = "🔴 改善が必要"

    col_m1, col_m2 = st.columns(2)
    col_m1.info(f"**モデル精度 (AUC-ROC)**: {auc_test:.3f} 👉 **{auc_eval}**\n\n*1.0に近いほど状態の判別が正確にできていることを示します（0.7以上が実用の目安）。*")
    col_m2.info(f"**予測の確信度 (Log Loss)**: {logloss_test:.3f} 👉 **{loss_eval}**\n\n*0.0に近いほどAIが「迷いなく」正解していることを示します（0.6以下が目安）。*")
    
    with st.expander("📊 テスト期間の予測確率推移を表示"):
        fig_ts_plot = go.Figure()
        fig_ts_plot.add_trace(go.Scatter(
            x=test_df.index, y=y_test_class, mode='markers', name='実際の状態 (1=Yes, 0=No)',
            marker=dict(color='blue', opacity=0.6, size=6), hovertemplate="日時: %{x}<br>状態: %{y}<extra></extra>"
        ))
        fig_ts_plot.add_trace(go.Scatter(
            x=test_df.index, y=preds_proba, mode='lines', name='LightGBM 予測確率',
            line=dict(color='red', width=2), opacity=0.8, hovertemplate="日時: %{x}<br>予測確率: %{y:.2f}<extra></extra>"
        ))
        fig_ts_plot.update_layout(title=f"テスト期間の {selected_target_name} 予測確率の推移", hovermode="x unified", legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
        st.plotly_chart(fig_ts_plot, use_container_width=True)

    st.subheader("🔮 リアルタイム予測と要因分析")
    available_data_all = df_imp.drop(columns=drop_cols, errors='ignore')
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

    target_data = available_data.iloc[-1:]
    current_time = target_data.index[0]
    current_val = float(target_data[target_col].values[0])
    current_state_bool = current_val >= target_threshold
    current_proba = model.predict_proba(target_data)[0, 1]
    predicted_state_bool = current_proba >= 0.5
    
    col_p1, col_p2, col_p3, col_p4 = st.columns(4)
    col_p1.metric("基準日時", current_time.strftime('%Y-%m-%d %H:%M'))
    col_p2.metric(f"現在の {selected_target_name} 状態", "Yes" if current_state_bool else "No")
    col_p3.metric(f"{PREDICT_AHEAD}後の予測判定", "Yes" if predicted_state_bool else "No")
    col_p4.metric(f"発生確率", f"{current_proba * 100:.1f} %")
    st.caption(f"※ **予測判定と発生確率について**: {PREDICT_AHEAD}後にあなたが「{selected_target_name}」の状態になっている確率をAIが算出したものです。50%以上を「Yes」と判定しています。")

    with st.spinner("SHAPで要因を分析しています..."):
        explainer = shap.TreeExplainer(model)
        shap_values_latest = explainer(target_data)
        if len(shap_values_latest.shape) == 3:
            shap_vals = shap_values_latest[0, :, 1].values
            shap_base_obj = shap_values_latest[0, :, 1]
        else:
            shap_vals = shap_values_latest[0].values
            shap_base_obj = shap_values_latest[0]
        
        def is_actionable(col: str) -> bool: return not (target_col in col or col in ["hour", "dayofweek"])
        exp_df = pd.DataFrame({'Feature': target_data.columns, 'Value': target_data.values[0], 'SHAP': shap_vals})
        exp_df['AbsSHAP'] = exp_df['SHAP'].abs()
        exp_df_action = exp_df[exp_df['Feature'].apply(is_actionable)].sort_values('AbsSHAP', ascending=False)
        
        fig2, ax2 = plt.subplots(figsize=(8, 4))
        shap.plots.waterfall(shap_base_obj, show=False)
        st.pyplot(fig2)

        st.markdown("**【要因分析の解説】**")
        st.caption("※ 上記のSHAPグラフは専用ライブラリのため静止画像で出力しています。一番長いバー（赤または青）が確率に最も影響を与えた要因です。")
        
        pos_factors = exp_df_action[exp_df_action['SHAP'] > 0]
        neg_factors = exp_df_action[exp_df_action['SHAP'] < 0]
        
        if target_col in ['NEMUKE_SCORE_NEW', '疲労判定', '強い疲労判定', '眠気判定', '強い眠気判定']:
            pos_effect_text, neg_effect_text = "確率上昇（悪化方向）", "確率低下（好転方向）"
            bar_desc = f"※グラフの赤いバーが{selected_target_name}の発生確率を押し上げる（悪化）要因、青いバーが押し下げる（好転）要因を示しています。"
        else:
            pos_effect_text, neg_effect_text = "確率上昇（好転方向）", "確率低下（悪化方向）"
            bar_desc = f"※グラフの赤いバーが{selected_target_name}の発生確率を押し上げる（好転）要因、青いバーが押し下げる（悪化）要因を示しています。"

        base_pos = None
        if not pos_factors.empty:
            top_pos = pos_factors.iloc[0]
            desc_pos = get_factor_direction_text(top_pos['Feature'], top_pos['Value'], available_data_all)
            base_pos = get_base_feature_name(top_pos['Feature'])
            st.write(f"- 📈 **確率を上げる要因**: **{desc_pos}** が{pos_effect_text}に働いています (影響度: {top_pos['SHAP']:+.2f})。")
            
        if not neg_factors.empty:
            top_neg = neg_factors.iloc[0]
            if base_pos is not None and get_base_feature_name(top_neg['Feature']) == base_pos and len(neg_factors) > 1:
                top_neg = neg_factors.iloc[1]
            desc_neg = get_factor_direction_text(top_neg['Feature'], top_neg['Value'], available_data_all)
            st.write(f"- 📉 **確率を下げる要因**: **{desc_neg}** が{neg_effect_text}に働いています (影響度: {top_neg['SHAP']:+.2f})。")
            
        st.caption(bar_desc)

    schedule_density = float(target_data["schedule_density_2h"].values[0]) if "schedule_density_2h" in target_data.columns else 0
    time_to_next = float(target_data["time_to_next_event_min"].values[0]) if "time_to_next_event_min" in target_data.columns else np.nan
    is_meeting = float(target_data["is_meeting"].values[0]) if "is_meeting" in target_data.columns else 0
    
    state_trend_prob = 1.0 - current_proba if target_col in ['NEMUKE_SCORE_NEW', '疲労判定', '強い疲労判定', '眠気判定', '強い眠気判定'] else current_proba

    reasons = []
    if is_meeting > 0: reasons.append("現在会議中")
    if schedule_density >= 0.6: reasons.append("予定密度が高い")
    if state_trend_prob >= 0.6: reasons.append(f"{selected_target_name}の好ましい確率が高い")
    elif state_trend_prob <= 0.4: reasons.append(f"{selected_target_name}の好ましくない確率が高い")
    
    if is_meeting > 0:
        work_mode, advice = "E: 段取り（会議モード）", "論点を1枚に整理し、次アクションをToDo化しましょう。"
    elif state_trend_prob >= 0.6 and (np.isnan(time_to_next) or time_to_next >= 50) and schedule_density < 0.6:
        work_mode, advice = "C: アウトプット（深）- 企画・戦略", "状態が好転する確率が高く、まとまった時間もあります。設計・企画の骨格づくりなど、重い思考タスクを進めるのが最適です。"
    elif state_trend_prob <= 0.4 or schedule_density >= 0.6:
        work_mode, advice = "D: アウトプット（軽）- 整理・レビュー", "予定が細切れか、状態が悪化する確率が高いです。10〜20分で終わるToDo消化や、資料の整形・チェック作業に時間を当てましょう。"
    else:
        if (np.isnan(time_to_next) or time_to_next >= 30) and schedule_density < 0.6:
            work_mode, advice = "A: インプット（重） または B: インプット（軽）", "難しめ資料の読み込みや情報整理など、次の深い思考に向けたインプット作業に適しています。"
        else:
            work_mode, advice = "E: 段取り", "次の深い作業へスムーズに入れるよう、論点の列挙や優先順位付け、素材の洗い出しを行いましょう。"

    st.subheader("📝 分析レポート (AIによる提案)")
    main_factor_desc = get_factor_direction_text(exp_df_action.iloc[0]['Feature'], exp_df_action.iloc[0]['Value'], available_data_all) if not exp_df_action.empty else "不明"
    prompt_context = f"現在時刻: {current_time.strftime('%Y-%m-%d %H:%M')}\n現在の{selected_target_name}の状態: {'Yes' if current_state_bool else 'No'}\n{PREDICT_AHEAD}後の予測判定: {'Yes' if predicted_state_bool else 'No'} (発生確率: {current_proba * 100:.1f}%)\n直近の主要因: {main_factor_desc} (SHAP: {exp_df_action.iloc[0]['SHAP']:+.2f})\n判定された働き方: {work_mode}\n理由: {', '.join(reasons) if reasons else '特になし'}"
    
    if use_gemini and api_key:
        with st.spinner("Geminiがレポートを作成中..."):
            try:
                genai.configure(api_key=api_key)
                model_llm = genai.GenerativeModel('gemini-2.5-flash')
                resp = model_llm.generate_content(f"以下のデータに基づき、客観的な働き方アドバイスレポートを生成してください。\n\n{prompt_context}\n\n構成:\n1. 予測結果と主な要因\n2. 奨励する働き方の具体例")
                st.write(resp.text)
            except Exception as e:
                st.error(f"Gemini APIエラー: {e}")
    else:
        st.info("💡 Gemini APIキーが未入力のため、ルールベースの詳細レポートを表示します。")
        st.markdown(f"#### 1. 近い将来（{PREDICT_AHEAD}後）の予測結果")
        st.write(f"基準日時（{current_time.strftime('%Y-%m-%d %H:%M')}）の {selected_target_name} は **{'Yes' if current_state_bool else 'No'}** の状態です。")
        st.write(f"{PREDICT_AHEAD}後は **{'Yes' if predicted_state_bool else 'No'}** （発生確率 **{current_proba * 100:.1f} %**）と予測されます。\nこの予測の主な要因として、**{main_factor_desc}** が影響しています。")
        st.markdown(f"#### 2. 奨励する働き方")
        st.write(f"現在の予測確率と予定状況（{', '.join(reasons) if reasons else '阻害要因なし'}）から、**「{work_mode}」**に取り組むことをお勧めします。\n**💡 進め方のアドバイス**: {advice}")

# --- UI レイアウト ---
st.write("### データのアップロード")
col_file1, col_file2 = st.columns(2)
with col_file1:
    file_ts = st.file_uploader("1. 生体データ (CSV形式)", type=['csv'])
with col_file2:
    file_sched = st.file_uploader("2. 予定表データ (予定表.CSV) ※任意", type=['csv'])

if st.button("🚀 分析を実行する", type="primary"):
    if file_ts is not None:
        # 分析実行フラグをセッションに保存（画面再描画で消えないようにする）
        st.session_state['run_analysis'] = True
    else:
        st.warning("⚠️ 生体データ (CSV形式) をアップロードしてください。")

# セッションにフラグがある場合のみ分析を実行・表示し続ける
if st.session_state.get('run_analysis', False) and file_ts is not None:
    try:
        # ドロップダウン変更時の再読み込みエラーを防ぐためにポインタを先頭に戻す
        file_ts.seek(0)
        df_ts = pd.read_csv(file_ts, skiprows=2)
        
        df_sched = None
        if file_sched is not None:
            file_sched.seek(0)
            df_sched = pd.read_csv(file_sched)
            
        run_analysis(df_ts, df_sched, use_gemini=True if api_key else False)
    except Exception as e:
        st.error(f"エラーが発生しました: {e}")
        st.session_state['run_analysis'] = False