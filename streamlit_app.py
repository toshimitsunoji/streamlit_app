# -*- coding: utf-8 -*-
"""
NeuroDesign　- 個人の深思考マネジメント -
（深思考成功確率予測エンジン版）
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import lightgbm as lgb
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, brier_score_loss
import scipy.signal as signal
import shap
import warnings
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import matplotlib as mpl
import matplotlib.font_manager as fm
import datetime
import math
import io

# --- Streamlit ページ設定 ---
st.set_page_config(page_title="NeuroDesign - 深思考成功確率予測 -", layout="wide", initial_sidebar_state="expanded")

# 日本語フォントの設定
font_path = Path(__file__).parent / "assets" / "fonts" / "NotoSansCJKjp-Regular.otf"
if font_path.exists():
    fm.fontManager.addfont(str(font_path))
    prop = fm.FontProperties(fname=str(font_path))
    mpl.rcParams["font.family"] = prop.get_name()

mpl.rcParams["axes.unicode_minus"] = False
warnings.filterwarnings('ignore')

# --- カスタムCSS ---
st.markdown("""
<style>
    .kpi-card { background-color: #ffffff; border-radius: 12px; padding: 24px; text-align: center; box-shadow: 0 4px 12px rgba(0,0,0,0.05); margin-bottom: 20px; border: 1px solid #f0f2f6; }
    .kpi-title { font-size: 1.1rem; color: #6c757d; margin-bottom: 8px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em; }
    .kpi-value-main { font-size: 3.5rem; color: #1e293b; font-weight: 800; line-height: 1.1; margin-bottom: 5px; }
    .kpi-unit { font-size: 1.2rem; color: #64748b; font-weight: 500; }
    .forecast-box { background-color: #ffffff; border: 1px solid #e2e8f0; border-radius: 8px; padding: 15px; text-align: center; margin-bottom: 15px; box-shadow: 0 2px 5px rgba(0,0,0,0.02); }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 🛑 A. 基礎レイヤー (1分粒度状態の生成)
# ==========================================
def compute_fatigue_features(df_1min):
    df = df_1min.copy()
    has_rmssd = 'RMSSD_SCORE_NEW' in df.columns
    has_tp = 'TP_SCORE_NEW' in df.columns
    
    if has_rmssd and has_tp:
        df['fatigue_score'] = 0.6 * df['RMSSD_SCORE_NEW'] + 0.4 * df['TP_SCORE_NEW']
    elif has_rmssd:
        df['fatigue_score'] = df['RMSSD_SCORE_NEW']
    elif has_tp:
        df['fatigue_score'] = df['TP_SCORE_NEW']
    else:
        df['fatigue_score'] = 50.0 
        
    df['fatigue_smooth'] = df['fatigue_score'].ewm(span=10, min_periods=1).mean()
    return df

def compute_low_arousal(df_1min, pr_col="PR_SCORE_NEW"):
    df = df_1min.copy()
    if pr_col not in df.columns:
        df['low_arousal'] = 0.0
        return df
        
    w5 = np.array([-2, -1, 0, 1, 2]) / 10.0
    slope = df[pr_col].rolling(5, min_periods=5).apply(lambda y: np.dot(w5, y), raw=True).fillna(0)
    eps = 0.02
    delta = np.maximum(0, -(slope + eps))
    alpha = 0.95
    k = 1.0
    
    low_arousal = np.zeros(len(df))
    dates = df.index.date
    
    for i in range(1, len(df)):
        if dates[i] != dates[i-1]:
            low_arousal[i] = 0
        else:
            low_arousal[i] = alpha * low_arousal[i-1] + k * delta.iloc[i]
            
    df['low_arousal'] = low_arousal
    return df

def add_1min_focus_wave(df_1min):
    df = df_1min.copy()
    focus_components = []
    if 'CVRR_SCORE_NEW' in df.columns: focus_components.append(df['CVRR_SCORE_NEW'])
    if 'RMSSD_SCORE_NEW' in df.columns: focus_components.append(100 - df['RMSSD_SCORE_NEW'])
    if 'LFHF_SCORE_NEW' in df.columns: focus_components.append(df['LFHF_SCORE_NEW'])
    
    if focus_components: df['focus_intensity'] = pd.concat(focus_components, axis=1).mean(axis=1)
    elif '集中判定' in df.columns: df['focus_intensity'] = df['集中判定'] * 100
    else: df['focus_intensity'] = 50.0
        
    df['focus_smooth'] = df['focus_intensity'].rolling(window=5, min_periods=1).mean()
    q70 = df['focus_smooth'].quantile(0.70) if not df['focus_smooth'].isna().all() else 50.0
    df['is_high_focus_wave'] = (df['focus_smooth'] >= q70).astype(int)
    return df

# ==========================================
# 🎯 B. 深思考成功確率予測エンジン
# ==========================================
def compute_base_success_prob(df_1min):
    """
    Step 1 & 2 & 4: 過去データから時間帯ごとの「ベース成功確率」を算出
    """
    df = df_1min.copy()
    df['date_hour'] = df.index.floor('H')
    
    # 個人基準値の算出
    focus_q70 = df['focus_intensity'].quantile(0.70) if not df['focus_intensity'].isna().all() else 50
    fatigue_median = df['fatigue_smooth'].median() if not df['fatigue_smooth'].isna().all() else 50
    la_threshold = df['low_arousal'].quantile(0.66) if not df['low_arousal'].isna().all() else 0
    
    # 連続高集中5分以上の判定
    focus_series = df['is_high_focus_wave']
    focus_streak = focus_series.groupby((focus_series != focus_series.shift()).cumsum()).transform('size') * focus_series
    df['has_5m_focus'] = (focus_streak >= 5).astype(int)
    
    records = []
    for name, group in df.groupby('date_hour'):
        hour = name.hour
        if not (9 <= hour <= 18): continue
        
        # 試行条件: その1時間内でFOCUS_SCOREが上位30%に達した瞬間があるか
        max_focus = group['focus_intensity'].max()
        is_trial = 1 if max_focus >= focus_q70 else 0
        
        if is_trial:
            # 成功条件
            cond_A = group['has_5m_focus'].max() > 0
            cond_B = group['fatigue_smooth'].mean() <= fatigue_median
            cond_C = group['low_arousal'].mean() <= la_threshold
            is_success = 1 if (cond_A and cond_B and cond_C) else 0
        else:
            is_success = 0
            
        records.append({'hour': hour, 'is_trial': is_trial, 'is_success': is_success})
        
    df_records = pd.DataFrame(records)
    
    if df_records.empty or df_records['is_trial'].sum() == 0:
        base_prob = pd.Series(0.5, index=np.arange(9, 19))
    else:
        hourly_stats = df_records.groupby('hour').agg({'is_trial': 'sum', 'is_success': 'sum'})
        base_prob = hourly_stats['is_success'] / hourly_stats['is_trial'].replace(0, np.nan)
        global_mean = df_records['is_success'].sum() / df_records['is_trial'].sum()
        base_prob = base_prob.reindex(np.arange(9, 19)).fillna(global_mean)
        
    # 平滑化 (前後1時間の移動平均)
    smooth_prob = base_prob.rolling(window=3, min_periods=1, center=True).mean()
    
    return smooth_prob, fatigue_median

def forecast_hourly_success_prob(df_1min, base_prob, fatigue_median, target_dt):
    """
    Step 3: 当日のリアルタイム値を用いて、各時間の成功確率を動的に補正
    """
    past_df = df_1min[df_1min.index <= target_dt]
    # 直近1時間(60T)のコンディションで補正
    recent_df = past_df.last('60T') if not past_df.empty else pd.DataFrame()
    
    fatigue_now = recent_df['fatigue_smooth'].mean() if not recent_df.empty else fatigue_median
    la_now = recent_df['low_arousal'].mean() if not recent_df.empty else 0
    
    alpha = 0.25 # 疲労ペナルティ係数
    beta = 0.25  # 眠気ペナルティ係数
    
    # 疲労補正: 当日平均 - 個人中央値
    fatigue_dev = (fatigue_now - fatigue_median) / 50.0 
    fatigue_dev = max(0.0, min(1.0, fatigue_dev)) # 悪化時のみ0〜1の範囲でペナルティ適用
    
    # 低覚醒補正: 正規化
    la_max = df_1min['low_arousal'].max()
    if pd.isna(la_max) or la_max == 0: la_max = 1.0
    la_risk = max(0.0, min(1.0, la_now / la_max))
    
    forecasts = {}
    for h in range(9, 19):
        bp = base_prob.get(h, 0.5)
        adj_prob = bp * (1 - alpha * fatigue_dev) * (1 - beta * la_risk)
        forecasts[h] = max(0.0, min(1.0, adj_prob))
        
    return forecasts

def get_today_best_hour(forecasts, df_sched_raw, target_dt):
    """
    UI連動: 未来の予定が空いている時間帯から、最大確率の枠を「今日の勝負時間」として抽出
    """
    best_hour = None
    best_prob = -1.0
    today_date = target_dt.date()
    
    for h, prob in forecasts.items():
        h_start = pd.Timestamp(datetime.datetime.combine(today_date, datetime.time(h, 0)))
        h_end = h_start + pd.Timedelta(hours=1)
        
        if h_end <= target_dt:
            continue # 過去の時間は除外
            
        has_conflict = False
        if df_sched_raw is not None and not df_sched_raw.empty:
            conflicts = df_sched_raw[(df_sched_raw['end_dt'] > h_start) & (df_sched_raw['start_dt'] < h_end)]
            if not conflicts.empty:
                has_conflict = True
                
        if not has_conflict and prob > best_prob:
            best_prob = prob
            best_hour = h
            
    # 全て予定で埋まっている場合は、純粋な予測最高値を提示
    if best_hour is None:
        future_fs = {k: v for k, v in forecasts.items() if pd.Timestamp(datetime.datetime.combine(today_date, datetime.time(k, 0))) + pd.Timedelta(hours=1) > target_dt}
        if future_fs:
            best_hour = max(future_fs, key=future_fs.get)
            best_prob = future_fs[best_hour]
            
    return best_hour, best_prob

def get_prob_color(prob):
    p = prob * 100
    if p >= 60: return "#10b981" # 緑 (好条件)
    elif p >= 40: return "#f59e0b" # 黄 (注意)
    else: return "#ef4444" # 赤 (悪条件)

# --- サイドバーUI ---
with st.sidebar:
    st.header("⚙️ データ入力")
    file_ts = st.file_uploader("1. 生体データ (CSV)", type=['csv'])
    file_sched = st.file_uploader("2. 予定表データ (CSV) ※任意", type=['csv'])
    
    with st.expander("🛠 設定"):
        TARGET_DATETIME_STR = st.text_input("予測基準日時 (空欄で最新)")
        
    st.markdown("---")
    run_btn = st.button("🚀 思考予報を更新", type="primary", use_container_width=True)

TARGET_DATETIME = TARGET_DATETIME_STR if TARGET_DATETIME_STR.strip() != "" else None

# === メイン処理パイプライン ===
if file_ts is not None:
    with st.spinner("深思考成功確率を計算中..."):
        # 1. データロード
        df_ts_raw = pd.read_csv(io.BytesIO(file_ts.getvalue()), skiprows=2)
        df_ts_raw['timestamp_clean'] = df_ts_raw['timestamp'].astype(str).str.split(' GMT').str[0]
        df_ts_raw['datetime'] = pd.to_datetime(df_ts_raw['timestamp_clean'], errors='coerce')
        df_ts_raw = df_ts_raw.dropna(subset=['datetime']).set_index('datetime').sort_index()

        df_sched_raw = None
        if file_sched:
            df_sched_raw = pd.read_csv(io.BytesIO(file_sched.getvalue()))
            df_sched_raw = df_sched_raw[df_sched_raw['終日イベント'].astype(str).str.upper() != 'TRUE']
            df_sched_raw['start_dt'] = pd.to_datetime(df_sched_raw['開始日'].astype(str) + ' ' + df_sched_raw['開始時刻'].astype(str), errors='coerce')
            df_sched_raw['end_dt']   = pd.to_datetime(df_sched_raw['終了日'].astype(str) + ' ' + df_sched_raw['終了時刻'].astype(str), errors='coerce')
            df_sched_raw = df_sched_raw.dropna(subset=['start_dt', 'end_dt']).sort_values('start_dt')
            
        # 2. 1分粒度データ構築
        num_cols = df_ts_raw.select_dtypes(include=[np.number]).columns
        df_1min = df_ts_raw[num_cols].resample('1T').mean().ffill(limit=5)
        
        df_1min = compute_fatigue_features(df_1min)
        df_1min = compute_low_arousal(df_1min, pr_col='PR_SCORE_NEW' if 'PR_SCORE_NEW' in df_1min.columns else None)
        df_1min = add_1min_focus_wave(df_1min)
        
        # 基準時刻 (t_now)
        if TARGET_DATETIME:
            try: t_now = pd.to_datetime(TARGET_DATETIME)
            except: t_now = df_1min.index[-1]
        else:
            t_now = df_1min.index[-1]
            
        # 3. 確率予測エンジンの実行
        base_prob, fatigue_median = compute_base_success_prob(df_1min)
        
        # 今日の予報
        today_forecasts = forecast_hourly_success_prob(df_1min, base_prob, fatigue_median, t_now)
        best_hour, best_prob = get_today_best_hour(today_forecasts, df_sched_raw, t_now)
        
        # 昨日比算出用
        yesterday_dt = t_now - pd.Timedelta(days=1)
        yesterday_forecasts = forecast_hourly_success_prob(df_1min, base_prob, fatigue_median, yesterday_dt)
        
        # KPI算出
        today_max = max(today_forecasts.values()) if today_forecasts else 0
        today_avg = sum(today_forecasts.values()) / len(today_forecasts) if today_forecasts else 0
        yesterday_avg = sum(yesterday_forecasts.values()) / len(yesterday_forecasts) if yesterday_forecasts else 0
        diff_avg = today_avg - yesterday_avg

    # ==========================================
    # UI 描画
    # ==========================================
    col_h1, col_h2 = st.columns([3, 1])
    with col_h1:
        st.markdown(f"<p style='color: gray; margin-top: 15px;'>最終予測時刻: {t_now.strftime('%Y/%m/%d %H:%M')}</p>", unsafe_allow_html=True)
    with col_h2:
        if st.button("🔄 最新状態で確率を再計算", use_container_width=True):
            st.rerun()

    # --- 習慣化強化 KPI パネル ---
    st.markdown("### 📊 本日のポテンシャル")
    col_k1, col_k2, col_k3 = st.columns(3)
    with col_k1:
        st.metric("今日の最高成功確率", f"{today_max*100:.0f}%")
    with col_k2:
        st.metric("今日の平均成功確率", f"{today_avg*100:.0f}%", f"{diff_avg*100:+.0f}% (昨日比)")

    # --- 今日の勝負時間 (1枠提示) ---
    st.markdown("<br>", unsafe_allow_html=True)
    if best_hour is not None:
        color = get_prob_color(best_prob)
        st.markdown(f"""
        <div style="background-color: #f8fafc; border-left: 6px solid {color}; padding: 20px; border-radius: 8px; margin-bottom: 30px;">
            <div style="font-size: 1.1rem; color: #64748b; font-weight: bold; margin-bottom: 5px;">👑 今日の勝負時間 (Deep Work)</div>
            <div style="font-size: 3rem; font-weight: 800; color: #1e293b;">
                🔥 {best_hour}:00 – {best_hour+1}:00 
                <span style="font-size: 2.2rem; color: {color}; margin-left: 20px;">{best_prob*100:.0f}%</span>
            </div>
            <div style="font-size: 1rem; color: #475569; margin-top: 10px;">
                リアルタイムの疲労・覚醒状況と予定の空きを考慮し、本日最も成功確率の高い時間を算出しました。
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="background-color: #fffbeb; border-left: 6px solid #f59e0b; padding: 20px; border-radius: 8px; margin-bottom: 30px;">
            <div style="font-size: 1.8rem; font-weight: 700; color: #92400e; margin-bottom: 10px;">
                ⚠ 本日は深思考に適した空き枠がありません
            </div>
            <div style="font-size: 1rem; color: #92400e;">
                戦略的な余白（休憩）として過ごすか、軽思考タスクを中心に配置してください。
            </div>
        </div>
        """, unsafe_allow_html=True)

    # --- 思考の天気予報 (1時間単位リスト) ---
    st.markdown("### 🌤 思考の天気予報 (時間帯別・成功確率)")
    st.write("各時間帯に深思考を実施した場合の予測成功確率です。絶対値のみを示しています。")
    
    cols = st.columns(5)
    for i, h in enumerate(range(9, 19)):
        p = today_forecasts.get(h, 0)
        color = get_prob_color(p)
        with cols[i % 5]:
            st.markdown(f"""
            <div class="forecast-box" style="border-top: 4px solid {color};">
                <div style="font-size: 1.2rem; font-weight: bold; color: #475569;">{h}:00</div>
                <div style="font-size: 2rem; font-weight: 900; color: {color}; margin-top: 5px;">{p*100:.0f}%</div>
            </div>
            """, unsafe_allow_html=True)