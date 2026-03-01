# -*- coding: utf-8 -*-
"""
ウェアラブル + Outlookスケジュール 集中・疲労予測アプリ (V2: アクション特化型 + インサイト充実版)
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import lightgbm as lgb
from sklearn.metrics import roc_auc_score, log_loss
from sklearn.tree import DecisionTreeRegressor, _tree
import google.generativeai as genai
import shap
import warnings
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
import matplotlib as mpl
import matplotlib.font_manager as fm
import datetime
import math

# --- Streamlit ページ設定 ---
st.set_page_config(page_title="Focus Battery | 集中予測", layout="wide", initial_sidebar_state="expanded")

# 日本語フォントの設定
font_path = Path(__file__).parent / "assets" / "fonts" / "NotoSansCJKjp-Regular.otf"
if font_path.exists():
    fm.fontManager.addfont(str(font_path))
    prop = fm.FontProperties(fname=str(font_path))
    mpl.rcParams["font.family"] = prop.get_name()

mpl.rcParams["axes.unicode_minus"] = False
warnings.filterwarnings('ignore')

# --- カスタムCSS (UIの洗練) ---
st.markdown("""
<style>
    .metric-container { background-color: #f8f9fa; border-radius: 10px; padding: 20px; text-align: center; box-shadow: 0 4px 6px rgba(0,0,0,0.05); margin-bottom: 20px; }
    .metric-title { font-size: 1.2rem; color: #6c757d; margin-bottom: 5px; font-weight: 600; }
    .metric-value { font-size: 3.5rem; color: #2b2b2b; font-weight: 800; line-height: 1.2; }
    .metric-sub { font-size: 1rem; color: #28a745; font-weight: bold; }
    .metric-sub.negative { color: #dc3545; }
    .window-box { background-color: #e3f2fd; border-left: 5px solid #1976d2; padding: 20px; border-radius: 5px; margin-bottom: 20px; }
    .window-time { font-size: 2rem; color: #1976d2; font-weight: 800; }
</style>
""", unsafe_allow_html=True)

# --- 特徴量名日本語化ヘルパー ---
def jp_feat_name(col_name: str) -> str:
    mapping = {'CVRR_SCORE_NEW': '集中スコア', '1分間歩数': '歩数', 'is_meeting': '会議', 'schedule_density_2h': '予定密度', '休憩判定': '休憩', '短時間歩行': '短時間歩行'}
    for k, v in mapping.items():
        if k in col_name: return col_name.replace(k, v)
    return col_name

def get_base_feature_name(feat: str) -> str:
    mapping = {'CVRR_SCORE_NEW': '集中スコア', '1分間歩数': '歩数', 'is_meeting': '会議', 'schedule_density_2h': '予定密度', '休憩判定': '休憩', '短時間歩行': '短時間歩行'}
    for k, v in mapping.items():
        if feat.startswith(k): return v
    return feat

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

# --- サイドバー (設定・データ入力) ---
with st.sidebar:
    st.header("⚙️ データ入力")
    file_ts = st.file_uploader("1. 生体データ (CSV)", type=['csv'])
    file_sched = st.file_uploader("2. 予定表データ (CSV)", type=['csv'])
    
    with st.expander("🛠 詳細設定 (分析フィルタ等)"):
        api_key = st.text_input("Gemini APIキー", type="password")
        RESAMPLE_FREQ = st.selectbox("分析単位", ['10T', '30T', '1H'], index=1)
        PREDICT_AHEAD = st.selectbox("予測先", ['10T', '30T', '1H'], index=1)
        TARGET_DATETIME_STR = st.text_input("予測基準日時 (空欄で最新)")
        target_col = '集中判定'
        
        st.markdown("**📅 分析対象フィルタ**")
        dow_options = ["月", "火", "水", "木", "金", "土", "日"]
        selected_dows = st.multiselect("対象曜日", dow_options, default=dow_options[0:5])
        time_range = st.slider("対象時間帯", 0, 23, (9, 19))
        
    st.markdown("---")
    st.button("🚀 今日のコンパスを更新", type="primary", use_container_width=True, key="run_btn")

selected_dow_indices = [dow_options.index(d) for d in selected_dows]
freq_td = pd.Timedelta(RESAMPLE_FREQ)
ahead_steps = max(1, int(pd.Timedelta(PREDICT_AHEAD) / freq_td))
TARGET_DATETIME = TARGET_DATETIME_STR if TARGET_DATETIME_STR.strip() != "" else None

# --- 分析エンジン ---
@st.cache_data(show_spinner=False)
def load_and_preprocess(file_ts_bytes, file_sched_bytes):
    import io
    df_ts = pd.read_csv(io.BytesIO(file_ts_bytes), skiprows=2)
    df_ts['timestamp_clean'] = df_ts['timestamp'].astype(str).str.split(' GMT').str[0]
    df_ts['datetime'] = pd.to_datetime(df_ts['timestamp_clean'], errors='coerce')
    df_ts = df_ts.dropna(subset=['datetime']).set_index('datetime').sort_index()

    df_sched = None
    if file_sched_bytes:
        df_sched = pd.read_csv(io.BytesIO(file_sched_bytes))
        df_sched = df_sched[df_sched['終日イベント'].astype(str).str.upper() != 'TRUE']
        df_sched['start_dt'] = pd.to_datetime(df_sched['開始日'].astype(str) + ' ' + df_sched['開始時刻'].astype(str), errors='coerce')
        df_sched['end_dt']   = pd.to_datetime(df_sched['終了日'].astype(str) + ' ' + df_sched['終了時刻'].astype(str), errors='coerce')
        df_sched = df_sched.dropna(subset=['start_dt', 'end_dt']).sort_values('start_dt')

    return df_ts, df_sched

def run_ml_pipeline(df_ts, df_sched):
    num_cols = df_ts.select_dtypes(include=[np.number]).columns
    df_resampled = df_ts[num_cols].resample(RESAMPLE_FREQ).mean()
    if '1分間歩数' in df_ts.columns:
        df_resampled['1分間歩数'] = df_ts['1分間歩数'].resample(RESAMPLE_FREQ).sum()

    if df_sched is not None:
        df_resampled['has_schedule'] = 0
        df_resampled['is_meeting'] = 0
        meeting_keywords = ['会議', '打合せ', 'MTG', '面談']
        for _, row in df_sched.iterrows():
            mask = (df_resampled.index < row['end_dt']) & ((df_resampled.index + freq_td) > row['start_dt'])
            df_resampled.loc[mask, 'has_schedule'] = 1
            if any(kw in str(row.get('件名', '')) for kw in meeting_keywords):
                df_resampled.loc[mask, 'is_meeting'] = 1
        win_steps = max(1, int(pd.Timedelta('2H') / freq_td))
        df_resampled['schedule_density_2h'] = df_resampled['has_schedule'].rolling(win_steps, min_periods=1).mean()

    df_features = df_resampled.copy()
    if '集中判定' in df_features.columns:
        focus_mask = (df_features['集中判定'] >= 0.5).astype(int)
        group_id = (focus_mask != focus_mask.shift()).cumsum()
        df_features['現在の集中継続時間_分'] = (focus_mask.groupby(group_id).cumcount() + 1) * (freq_td.total_seconds() / 60) * focus_mask
    
    if '休憩判定' in df_features.columns: df_features['休憩判定_前'] = df_features['休憩判定'].shift(1)
    if '短時間歩行' in df_features.columns: df_features['短時間歩行_前'] = df_features['短時間歩行'].shift(1)

    df_features['target_ahead'] = (df_features[target_col].shift(-ahead_steps) >= 0.5).astype(int)
    
    drop_cols = ['target_ahead']
    df_imp = df_features.ffill(limit=2).bfill(limit=2)
    train_df = df_imp.dropna(subset=drop_cols + [target_col])
    
    X = train_df.drop(columns=drop_cols)
    y = train_df['target_ahead']
    
    model = lgb.LGBMClassifier(objective='binary', n_estimators=100, learning_rate=0.05, random_state=42, verbose=-1)
    model.fit(X, y)
    
    return model, df_imp, X.columns, df_ts, df_sched

# === メイン処理 ===
if st.session_state.get('run_btn') or (file_ts is not None):
    if file_ts is None:
        st.info("👈 サイドバーから「1. 生体データ」をアップロードして分析を開始してください。")
        st.stop()
        
    with st.spinner("AIがコンディションを解析中..."):
        df_ts_raw, df_sched_raw = load_and_preprocess(file_ts.getvalue(), file_sched.getvalue() if file_sched else None)
        model, df_imp, feature_cols, df_ts_min, df_sched = run_ml_pipeline(df_ts_raw, df_sched_raw)
        
        # 基準日時の決定
        if TARGET_DATETIME is not None:
            try:
                current_time = pd.to_datetime(TARGET_DATETIME)
                target_data_all = df_imp[df_imp.index <= current_time]
                target_data = target_data_all.iloc[-1:] if not target_data_all.empty else df_imp.iloc[-1:]
            except:
                target_data = df_imp.iloc[-1:]
        else:
            target_data = df_imp.iloc[-1:]
        current_time = target_data.index[0]
            
        current_proba = model.predict_proba(target_data[feature_cols])[0, 1]
        
        # ==========================================
        # 🔋 Focus Battery ロジック (単調減少＆レンジ表現に修正)
        # ==========================================
        if '集中判定' in df_ts_min.columns:
            daily_focus = df_ts_min['集中判定'].resample('D').apply(lambda x: (x >= 0.5).sum())
            daily_focus = daily_focus[daily_focus > 0] # 計測がない日は除外
            if not daily_focus.empty:
                avg_focus_mins = daily_focus.mean() # 平均値をベースに
                focus_p80 = daily_focus.quantile(0.80) # 上振れ（80%タイル・好調時）
            else:
                avg_focus_mins, focus_p80 = 120, 180
        else:
            avg_focus_mins, focus_p80 = 120, 180
            
        today_str = current_time.strftime('%Y-%m-%d')
        if '集中判定' in df_ts_min.columns and today_str in df_ts_min.index.strftime('%Y-%m-%d'):
            today_data = df_ts_min[df_ts_min.index.date == current_time.date()]
            consumed_mins = (today_data.loc[:current_time, '集中判定'] >= 0.5).sum()
        else:
            consumed_mins = 0
            
        # 確率による変動を外し、純粋に「全体のポテンシャル - 消化済」で計算
        rem_avg = max(0, int(avg_focus_mins - consumed_mins))
        rem_p80 = max(0, int(focus_p80 - consumed_mins))
        
        # ==========================================
        # 🕒 Deep Work Window ロジック
        # ==========================================
        window_text = "本日は終了モードです"
        window_desc = "しっかり休んで明日に備えましょう。"
        
        if current_time.hour < 19:
            start_search = current_time
            end_search = current_time.replace(hour=20, minute=0, second=0)
            free_blocks = []
            curr_block_start = start_search
            
            if df_sched is not None and not df_sched.empty:
                today_sched = df_sched[(df_sched['start_dt'] >= start_search) & (df_sched['start_dt'] < end_search)].sort_values('start_dt')
                for _, row in today_sched.iterrows():
                    if row['start_dt'] > curr_block_start:
                        duration = (row['start_dt'] - curr_block_start).total_seconds() / 60
                        if duration >= 60: free_blocks.append((curr_block_start, row['start_dt'], duration))
                    curr_block_start = max(curr_block_start, row['end_dt'])
                if curr_block_start < end_search:
                    duration = (end_search - curr_block_start).total_seconds() / 60
                    if duration >= 60: free_blocks.append((curr_block_start, end_search, duration))
            else:
                next_hour = current_time.replace(minute=0, second=0) + pd.Timedelta('1H')
                if next_hour < end_search: free_blocks.append((next_hour, next_hour + pd.Timedelta('90T'), 90))
                    
            if free_blocks:
                scored_blocks = []
                for b_start, b_end, duration in free_blocks:
                    sim_data = target_data[feature_cols].copy()
                    if 'hour' in sim_data.columns: sim_data['hour'] = b_start.hour
                    if 'is_meeting' in sim_data.columns: sim_data['is_meeting'] = 0
                    if 'has_schedule' in sim_data.columns: sim_data['has_schedule'] = 0
                    
                    block_proba = model.predict_proba(sim_data)[0, 1]
                    scored_blocks.append((b_start, b_end, duration, block_proba))
                
                best_block = sorted(scored_blocks, key=lambda x: x[3], reverse=True)[0]
                w_start = best_block[0]
                w_end = w_start + pd.Timedelta(minutes=min(90, best_block[2]))
                window_text = f"{w_start.strftime('%H:%M')} – {w_end.strftime('%H:%M')}"
                window_desc = f"AIが本日最も集中しやすい（予測確率 {best_block[3]*100:.1f}%）と判断した空き時間です。"

        fatigue_risk = False
        if '疲労判定' in target_data.columns and target_data['疲労判定'].values[0] >= 0.5: fatigue_risk = True
        elif 'schedule_density_2h' in target_data.columns and target_data['schedule_density_2h'].values[0] >= 0.6: fatigue_risk = True

    # --- UI 描画開始 ---
    st.markdown(f"<p style='text-align: right; color: gray;'>更新日時: {current_time.strftime('%Y/%m/%d %H:%M')}</p>", unsafe_allow_html=True)
    
    tab_today, tab_weekly, tab_spec = st.tabs(["🧭 Today's Compass (今日の行動)", "📊 Weekly Report (振り返り)", "👤 My Spec (あなたの特性)"])

    # ==========================================
    # Tab 1: Today's Compass
    # ==========================================
    with tab_today:
        st.markdown("## 10秒で決める、今日の最適解")
        
        col1, col2 = st.columns([1.2, 1])
        with col1:
            st.markdown(f"""
            <div class="metric-container">
                <div class="metric-title">🔋 本日の高品質集中</div>
                <div style="font-size: 1rem; color: #555; margin-bottom: 5px;">
                    本日ここまで: <strong>{consumed_mins} 分</strong> 消化済
                </div>
                <div class="metric-value"><span style="font-size: 1.5rem; color: #6c757d;">残り</span> {rem_avg} <span style="font-size: 2rem;">〜</span> {rem_p80} <span style="font-size: 1.5rem;">分</span></div>
                <div style="font-size: 0.95rem; color: #6c757d; margin-top: 12px; font-weight: 500;">
                    ※ 平均値({int(avg_focus_mins)}分) 〜 好調時({int(focus_p80)}分) の予測レンジ
                </div>
            </div>
            """, unsafe_allow_html=True)
            
        with col2:
            st.markdown(f"""
            <div class="window-box">
                <div class="metric-title">🕒 今日の勝負枠 (Deep Work Window)</div>
                <div class="window-time">{window_text}</div>
                <div style="color: #555; margin-top: 10px;">👉 {window_desc}</div>
            </div>
            """, unsafe_allow_html=True)
            
        if fatigue_risk:
            st.error("⚠️ **疲労アラート**: 現在疲労が蓄積しているか、予定が過密です。10分程度の完全な休息をとるか、重要な意思決定を後回しにすることを推奨します。")
        else:
            st.success("✨ **コンディション良好**: 現在、集中を阻害する大きなノイズはありません。")

        st.markdown("---")
        st.markdown("### 🔮 アクション・シミュレーター (事前予測)")
        st.write("「今からどう行動を変えれば、どれくらいパフォーマンスが回復するか？」をAIが事前計算しました。")
        
        def simulate_battery_gain(mod_dict):
            sim_data = target_data[feature_cols].copy()
            for k, v in mod_dict.items():
                if k in sim_data.columns: sim_data[k] = v
            sim_proba = model.predict_proba(sim_data)[0, 1]
            prob_diff = sim_proba - current_proba
            gain = int(prob_diff * avg_focus_mins * 1.5)
            return gain

        sim_walk = simulate_battery_gain({'短時間歩行': 1.0, '短時間歩行_前': 1.0, '1分間歩数': 1000})
        sim_rest = simulate_battery_gain({'休憩判定': 1.0, '休憩判定_前': 1.0, 'time_since_prev_event_min': 30})
        sim_skip = simulate_battery_gain({'is_meeting': 0.0, 'has_schedule': 0.0, 'schedule_density_2h': max(0, target_data['schedule_density_2h'].values[0] - 0.25)})

        sim_col1, sim_col2, sim_col3 = st.columns(3)
        
        with sim_col1:
            st.info(f"**🚶 今から15分歩く**\n\n予測: バッテリー **{'+' + str(sim_walk) if sim_walk > 0 else sim_walk} 分**")
        with sim_col2:
            st.info(f"**☕ 予定の前に休憩をとる**\n\n予測: バッテリー **{'+' + str(sim_rest) if sim_rest > 0 else sim_rest} 分**")
        with sim_col3:
            st.info(f"**🚫 直近の会議を1つ減らす**\n\n予測: バッテリー **{'+' + str(sim_skip) if sim_skip > 0 else sim_skip} 分**")

    # ==========================================
    # Tab 2: Weekly Report
    # ==========================================
    with tab_weekly:
        st.markdown("## 週末の振り返りと分析 (Weekly Report)")
        
        # --- マイルールの文章化 ---
        st.markdown("#### 💡 AIが見つけた「あなた専用の集中ルール」")
        action_cols = [c for c in ['休憩判定', '短時間歩行', 'is_meeting', 'schedule_density_2h'] if c in df_imp.columns]
        if len(action_cols) > 0 and len(df_imp) > 10:
            reg_df = df_imp.dropna(subset=action_cols + [target_col])
            X_rule = reg_df[action_cols]
            y_rule = reg_df[target_col]
            
            tree_model = DecisionTreeRegressor(max_depth=2, min_samples_leaf=10, random_state=42)
            tree_model.fit(X_rule, y_rule)
            
            feat_names = []
            is_bool = []
            for col in action_cols:
                is_bool.append(reg_df[col].dropna().nunique() <= 2)
                if col == 'is_meeting': feat_names.append("会議中")
                elif col == 'schedule_density_2h': feat_names.append("予定密度")
                else: feat_names.append(jp_feat_name(col))
                
            tree_rules = extract_rules(tree_model, feat_names, is_bool)
            valid_rules = [r for r in tree_rules if r[2] >= 5]
            if not valid_rules: valid_rules = tree_rules
            valid_rules.sort(key=lambda x: x[1], reverse=True)
            
            # 良いパターンと危険なパターンの両方を抽出
            positive_rule = None
            overwork_rule = None
            
            for rule_text, val, samples in valid_rules:
                display_prob = val * 100
                conditions = rule_text.split(" ＋ ")
                cond_texts = [c.replace("【", "").replace("】", "") for c in conditions]
                
                has_positive_action = any(
                    ("休憩" in c and ("あり" in c or "高い" in c)) or
                    ("歩行" in c and ("あり" in c or "高い" in c))
                    for c in cond_texts
                )
                is_overwork = any(
                    ("予定密度" in c and "高い" in c) or
                    ("会議" in c and "あり" in c)
                    for c in cond_texts
                ) and not has_positive_action
                
                if has_positive_action and not positive_rule:
                    positive_rule = (cond_texts, display_prob, samples)
                if is_overwork and not overwork_rule:
                    overwork_rule = (cond_texts, display_prob, samples)
                    
                if positive_rule and overwork_rule:
                    break
                    
            if positive_rule:
                cond_joined = " かつ ".join(positive_rule[0])
                st.info(f"💡 **リフレッシュで集中を高める黄金パターン**\n\n**「{cond_joined}」** の状況が整ったとき、あなたが集中状態に入る確率は **{positive_rule[1]:.1f} %** まで高まります。\n\n*(過去の実績: {positive_rule[2]}件のデータより算出)*\n\n👉 **コーチからのアドバイス:**\n素晴らしい傾向です！意図的なリフレッシュ行動（休憩や歩行）が、確実なパフォーマンス向上に繋がっています。引き続きこのパターンを意識しましょう。")
                
            if overwork_rule:
                cond_joined = " かつ ".join(overwork_rule[0])
                st.warning(f"💡 **追い込み型の集中パターン（燃え尽き注意）**\n\n**「{cond_joined}」** のように、予定が詰まっていてリフレッシュがない切羽詰まった状況で、集中確率が **{overwork_rule[1]:.1f} %** まで高まる傾向があります。\n\n*(過去の実績: {overwork_rule[2]}件のデータより算出)*\n\n👉 **コーチからのアドバイス:**\n締め切り効果等でスコアは一時的に高まっていますが、この状態を続けると急激な疲労（バッテリー切れ）を招きます。意識的に予定に隙間を作り、短い歩行や休憩を挟むように行動を変えてみましょう。")
                
            if not positive_rule and not overwork_rule and valid_rules:
                rule_text, val, samples = valid_rules[0]
                display_prob = val * 100
                conditions = rule_text.split(" ＋ ")
                cond_texts = [c.replace("【", "").replace("】", "") for c in conditions]
                cond_joined = " かつ ".join(cond_texts)
                st.info(f"💡 **あなた専用の「集中モード」発動条件**\n\n**「{cond_joined}」** の状況が整ったとき、あなたが集中状態に入る確率は **{display_prob:.1f} %** まで高まります。\n\n*(過去の実績: {samples}件のデータより算出)*")

        st.markdown("---")
        st.markdown("#### 📅 今週の推移")
        
        week_start = (current_time - pd.to_timedelta(current_time.dayofweek, unit='d')).date()
        week_data_raw = df_ts_min[df_ts_min.index.date >= week_start].copy()
        week_data = week_data_raw[week_data_raw.index.dayofweek.isin(selected_dow_indices)]
        week_data = week_data[(week_data.index.hour >= time_range[0]) & (week_data.index.hour <= time_range[1])]
        
        if not week_data.empty and '集中判定' in week_data.columns:
            df_w_1t = week_data[['集中判定']].resample('1T').mean()
            df_w_1t['集中フラグ'] = (df_w_1t['集中判定'] >= 0.5).astype(int)
            df_w_hourly = df_w_1t.resample('1H').sum()
            df_w_hourly['hour'] = df_w_hourly.index.hour
            df_w_hourly['dow'] = df_w_hourly.index.dayofweek
            
            # --- 曜日別・時間帯別グラフ ---
            col_w1, col_w2 = st.columns(2)
            with col_w1:
                dow_sum = df_w_hourly.groupby('dow')['集中フラグ'].sum().reindex(selected_dow_indices, fill_value=0)
                fig_dow = px.bar(x=[dow_options[i] for i in selected_dow_indices], y=dow_sum.values, labels={'x': '曜日', 'y': '集中時間 (分)'}, title="曜日別の集中時間")
                fig_dow.update_traces(marker_color='#1976d2')
                st.plotly_chart(fig_dow, use_container_width=True)
                
            with col_w2:
                target_hours_list = list(range(time_range[0], time_range[1] + 1))
                hour_sum = df_w_hourly.groupby('hour')['集中フラグ'].sum().reindex(target_hours_list, fill_value=0)
                fig_hour = px.bar(x=[f"{h}:00" for h in target_hours_list], y=hour_sum.values, labels={'x': '時間帯', 'y': '集中時間 (分)'}, title="時間帯別の集中時間")
                fig_hour.update_traces(marker_color='#1976d2')
                st.plotly_chart(fig_hour, use_container_width=True)

            # --- ウィークリー・モメンタルグラフ ---
            st.markdown("##### 🌊 日別のモメンタルグラフ (CVRRの波)")
            st.caption("※ 上下の面がバランス良く見えるよう、基準値(グレー点線)は「今週の平均値」に合わせて自動調整されています。極端に低い値はグラフ下部で省略して表示しています。")
            week_dates = [(week_start + datetime.timedelta(days=i)) for i in range(7)]
            target_dates = [d for d in week_dates if d.weekday() in selected_dow_indices]
            
            base_val = week_data['CVRR_SCORE_NEW'].mean() if 'CVRR_SCORE_NEW' in week_data.columns else 50.0
            if pd.isna(base_val): base_val = 50.0
            
            if not week_data.empty and 'CVRR_SCORE_NEW' in week_data.columns:
                week_max = week_data['CVRR_SCORE_NEW'].max()
                amp = week_max - base_val
                if amp < 10: amp = 10
                y_max_global = base_val + (amp * 1.2)
                y_min_global = base_val - (amp * 1.5)
            else:
                y_max_global, y_min_global = 100, 0
            
            for i in range(0, len(target_dates), 2):
                cols = st.columns(2)
                for j in range(2):
                    if i + j < len(target_dates):
                        t_date = target_dates[i+j]
                        dow_str = dow_options[t_date.weekday()]
                        with cols[j]:
                            df_day = df_ts_min[df_ts_min.index.date == t_date].copy()
                            df_day = df_day[(df_day.index.hour >= time_range[0]) & (df_day.index.hour <= time_range[1])]
                            if 'CVRR_SCORE_NEW' in df_day.columns and not df_day.empty and not df_day['CVRR_SCORE_NEW'].isna().all():
                                fig_d = go.Figure()
                                fig_d.add_trace(go.Scatter(x=df_day.index, y=[base_val]*len(df_day), mode='lines', line=dict(color='gray', width=1, dash='dash'), hoverinfo='skip'))
                                y_up = np.where(df_day['CVRR_SCORE_NEW'] >= base_val, df_day['CVRR_SCORE_NEW'], base_val)
                                fig_d.add_trace(go.Scatter(x=df_day.index, y=y_up, fill='tonexty', fillcolor='rgba(54, 162, 235, 0.5)', mode='lines', line=dict(width=0), hoverinfo='skip'))
                                fig_d.add_trace(go.Scatter(x=df_day.index, y=[base_val]*len(df_day), mode='lines', line=dict(width=0), hoverinfo='skip'))
                                y_down = np.where(df_day['CVRR_SCORE_NEW'] <= base_val, df_day['CVRR_SCORE_NEW'], base_val)
                                fig_d.add_trace(go.Scatter(x=df_day.index, y=y_down, fill='tonexty', fillcolor='rgba(255, 159, 64, 0.5)', mode='lines', line=dict(width=0), hoverinfo='skip'))
                                fig_d.add_trace(go.Scatter(x=df_day.index, y=df_day['CVRR_SCORE_NEW'], mode='lines', line=dict(color='#333333', width=2), hovertemplate="%{x|%H:%M}<br>ｽｺｱ: %{y:.1f}<extra></extra>"))
                                fig_d.update_layout(title=f"{t_date.strftime('%m/%d')} ({dow_str})", height=250, hovermode="x unified", plot_bgcolor='rgba(0,0,0,0)', margin=dict(l=20, r=20, t=30, b=20), showlegend=False)
                                fig_d.update_xaxes(showgrid=True, gridcolor='lightgray')
                                fig_d.update_yaxes(showgrid=True, gridcolor='lightgray', title="CVRR", range=[y_min_global, y_max_global])
                                st.plotly_chart(fig_d, use_container_width=True)
                            else:
                                st.markdown(f"**{t_date.strftime('%m/%d')} ({dow_str})**")
                                st.info("データなし")

        # --- Gemini AIレポート ---
        if api_key:
            st.markdown("---")
            st.markdown("#### 🤖 専属AIコーチからのフィードバック")
            with st.spinner("レポートを生成中..."):
                try:
                    genai.configure(api_key=api_key)
                    model_llm = genai.GenerativeModel('gemini-2.5-flash')
                    prompt = f"""
                    あなたはプロの生産性コーチです。以下のデータに基づき、ユーザーの今週の働き方を振り返り、来週に向けた「改善アクション」を3つ提案してください。
                    ・ユーザーの平均集中バッテリー残量基準: {int(avg_focus_mins)}分
                    ・最近の集中スコア平均: {week_data['CVRR_SCORE_NEW'].mean() if not week_data.empty and 'CVRR_SCORE_NEW' in week_data.columns else '不明'}
                    ・現在のコンディション: {'疲労リスクあり' if fatigue_risk else '良好'}
                    出力形式:
                    1. 今週の総評（1〜2行）
                    2. 来週すぐできる改善アクション（箇条書きで3つ、具体的に）
                    """
                    resp = model_llm.generate_content(prompt)
                    st.success(resp.text)
                except Exception as e:
                    st.error(f"Gemini APIエラー: {e}")

    # ==========================================
    # Tab 3: My Spec
    # ==========================================
    with tab_spec:
        st.markdown("## 👤 あなたのパーソナル特性 (My Spec)")
        st.write("過去のすべてのデータから、あなた固有の集中と疲労のパターンを抽出した「あなたの取扱説明書」です。")
        
        df_insight = df_imp.copy()
        df_insight = df_insight[df_insight.index.dayofweek.isin(selected_dow_indices)]
        df_insight = df_insight[(df_insight.index.hour >= time_range[0]) & (df_insight.index.hour <= time_range[1])]

        target_hours_list = list(range(time_range[0], time_range[1] + 1))

        # --- 全期間の特性データ算出 (タイプ診断用) ---
        focus_type_name = "データ不足"
        focus_type_desc = "特徴を判定するためのデータが足りません。"
        hour_avg = pd.Series(dtype=float)
        dow_avg = pd.Series(dtype=float)

        if '集中判定' in df_insight.columns:
            df_ins_1t = df_insight[['集中判定']].resample('1T').mean().ffill(limit=5)
            df_ins_1t['集中フラグ'] = (df_ins_1t['集中判定'] >= 0.5).astype(int)
            
            df_ins_hourly = df_ins_1t.resample('1H').sum()
            df_ins_hourly['date'] = df_ins_hourly.index.date
            df_ins_hourly['hour'] = df_ins_hourly.index.hour
            df_ins_hourly['dow'] = df_ins_hourly.index.dayofweek
            
            df_ins_hourly = df_ins_hourly[df_ins_hourly['dow'].isin(selected_dow_indices)]
            df_ins_hourly = df_ins_hourly[(df_ins_hourly['hour'] >= time_range[0]) & (df_ins_hourly['hour'] <= time_range[1])]
            
            total_days = df_ins_hourly['date'].nunique()
            if total_days > 0:
                hour_total = df_ins_hourly.groupby('hour')['集中フラグ'].sum()
                hour_avg = (hour_total / total_days).reindex(target_hours_list, fill_value=0)
                
                dow_total = df_ins_hourly.groupby('dow')['集中フラグ'].sum()
                days_per_dow = df_ins_hourly.groupby('dow')['date'].nunique()
                dow_avg = (dow_total / days_per_dow).reindex(selected_dow_indices, fill_value=0)
                
                am_hours = [h for h in target_hours_list if h < 12]
                pm1_hours = [h for h in target_hours_list if 12 <= h < 16]
                pm2_hours = [h for h in target_hours_list if 16 <= h]
                
                am_avg = hour_avg.loc[am_hours].mean() if am_hours else 0
                pm1_avg = hour_avg.loc[pm1_hours].mean() if pm1_hours else 0
                pm2_avg = hour_avg.loc[pm2_hours].mean() if pm2_hours else 0
                
                max_period = max(am_avg, pm1_avg, pm2_avg)
                if max_period > 0:
                    if max_period == am_avg:
                        focus_type_name = "🌅 午前集中型 (Morning Sprinter)"
                        focus_type_desc = "午前中に最も高いパフォーマンスを発揮します。重いタスクは昼までに片付けるのがベストです。"
                    elif max_period == pm1_avg:
                        focus_type_name = "☀️ 午後スタート型 (Afternoon Engine)"
                        focus_type_desc = "昼食後から夕方にかけてエンジンがかかるタイプです。午後に勝負タスクを配置しましょう。"
                    else:
                        focus_type_name = "🌆 夕方追い込み型 (Evening Closer)"
                        focus_type_desc = "夕方以降に集中力が高まるタイプです。終業前の追い込みが得意ですが、オーバーワークに注意が必要です。"
                        
                    mean_val = hour_avg.mean()
                    cv = hour_avg.std() / mean_val if mean_val > 0 else 0
                    if cv > 0.4:
                        focus_type_name += " / 🌊 波型スプリンター"
                        focus_type_desc += " 集中する時間としない時間のメリハリが非常に強いため、波に乗れる時間を逃さないことが重要です。"
                    else:
                        focus_type_name += " / 🐢 安定持続型"
                        focus_type_desc += " 1日を通して安定して集中を保つことができます。こまめな休憩でスタミナを維持しましょう。"

        if '集中判定' in df_insight.columns: df_insight['focus_start'] = (df_insight['集中判定'] >= 0.5) & (df_insight['集中判定'].shift(1) < 0.5)
        if '疲労判定' in df_insight.columns: df_insight['fatigue_start'] = (df_insight['疲労判定'] >= 0.5) & (df_insight['疲労判定'].shift(1) < 0.5)

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

        avg_focus_duration_str, daily_focus_count_str, daily_total_focus_time_str = "不明", "不明", "不明"
        
        if '集中判定' in df_ts_min.columns:
            df_1min = df_ts_min[['集中判定']].resample('1T').mean().ffill(limit=5)
            df_1min = df_1min[df_1min.index.dayofweek.isin(selected_dow_indices)]
            df_1min = df_1min[(df_1min.index.hour >= time_range[0]) & (df_1min.index.hour <= time_range[1])]
            
            focus_mask = df_1min['集中判定'] >= 0.5
            focus_blocks = focus_mask.groupby((focus_mask != focus_mask.shift()).cumsum())
            focus_durations = focus_blocks.sum()[focus_blocks.sum() > 0]
            
            if not focus_durations.empty:
                avg_focus_duration_str = f"{focus_durations.mean():.0f}"
                num_days = df_1min.index.normalize().nunique()
                daily_focus_count_str = f"{(len(focus_durations) / num_days if num_days > 0 else 0):.1f}"
                daily_total_focus_time_str = f"{(focus_mask.sum() / num_days if num_days > 0 else 0):.0f}"

        focus_actions = []
        if '1分間歩数' in df_insight.columns and 'focus_start' in df_insight.columns:
            walk_before = df_insight['1分間歩数'].shift(1)[df_insight['focus_start']].dropna()
            avg_walk = df_insight['1分間歩数'].mean()
            if not walk_before.empty and avg_walk > 0:
                if walk_before.mean() > avg_walk * 1.2: focus_actions.append("事前に体を動かすこと（少し歩くなど）")
                elif walk_before.mean() < avg_walk * 0.8: focus_actions.append("事前に静かな環境で落ち着いて過ごすこと")

        if '休憩判定' in df_insight.columns and 'focus_start' in df_insight.columns:
            rest_before = df_insight['休憩判定'].shift(1)[df_insight['focus_start']].dropna()
            if not rest_before.empty and df_insight['休憩判定'].mean() > 0:
                if rest_before.mean() > df_insight['休憩判定'].mean() * 1.2: focus_actions.append("事前にしっかり休憩をとること")

        focus_actions_str = "データ不足のため特定できません" if not focus_actions else "、".join(focus_actions)

        fatigue_actions = []
        if '疲労判定' in df_insight.columns and 'has_schedule' in df_insight.columns:
            sched_mask = df_insight['has_schedule'] >= 0.5
            sched_blocks = (sched_mask != sched_mask.shift()).cumsum()
            fatigue_diffs = []
            for _, group in df_insight[sched_mask].groupby(sched_blocks):
                if len(group) > 1:
                    dh = len(group) * (freq_td.total_seconds() / 3600)
                    if dh > 0: fatigue_diffs.append((group['疲労判定'].iloc[-1] - group['疲労判定'].iloc[0]) / dh)
            if fatigue_diffs and np.mean(fatigue_diffs) > 0: fatigue_actions.append("1時間以上の予定をこなすこと")

        if 'fatigue_start' in df_insight.columns and 'focus_start' in df_insight.columns:
            fat_times, foc_times = df_insight[df_insight['fatigue_start']].index, df_insight[df_insight['focus_start']].index
            rec_c, rec_s = [], []
            for ft in fat_times:
                ff = foc_times[foc_times > ft]
                if len(ff) > 0 and ff[0].date() == ft.date() and 'consecutive_schedules' in df_insight.columns:
                    if df_insight.loc[ft, 'consecutive_schedules'] >= 2: rec_c.append(1)
                    else: rec_s.append(1)
            if rec_c and rec_s and (np.mean(rec_c) - np.mean(rec_s)) > 0: fatigue_actions.append("予定を連続して入れること")

        fatigue_actions_str = "データ不足のため特定できません" if not fatigue_actions else "、".join(fatigue_actions)

        st.markdown(f"""
        <div style="background-color: #f8f9fa; padding: 20px; border-radius: 10px; border-left: 5px solid #28a745; margin-bottom: 20px;">
            <h4 style="margin-top: 0; color: #333;">🎯 あなたの集中特性</h4>
            <ul style="font-size: 1.1rem; color: #555;">
                <li><strong>集中タイプ： <span style="color:#28a745;">{focus_type_name}</span></strong><br>
                    <span style="font-size: 0.95rem; color: #777;">{focus_type_desc}</span></li>
                <li style="margin-top: 10px;"><strong>{f_dow}曜日の{f_hour}時台</strong> に最も集中しやすい傾向があります。</li>
                <li>平均集中持続時間は <strong>{avg_focus_duration_str}分</strong> です。</li>
                <li>1日の平均集中時間は <strong>{daily_total_focus_time_str}分</strong> です。</li>
                <li>1日に平均 <strong>{daily_focus_count_str}回</strong> の集中サイクルを繰り返しています。</li>
                <li>集中に入りやすい行動： <strong>{focus_actions_str}</strong></li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div style="background-color: #f8f9fa; padding: 20px; border-radius: 10px; border-left: 5px solid #dc3545; margin-bottom: 20px;">
            <h4 style="margin-top: 0; color: #333;">🔋 あなたの疲労特性</h4>
            <ul style="font-size: 1.1rem; color: #555;">
                <li><strong>{fat_dow}曜日の{fat_hour}時台</strong> に最も疲労しやすい傾向があります。</li>
                <li>疲労しやすい行動： <strong>{fatigue_actions_str}</strong></li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        # --- 全期間の棒グラフ ---
        st.markdown("---")
        st.markdown("#### 📊 全期間の集中傾向 (曜日・時間帯別)")
        
        if not hour_avg.empty and not dow_avg.empty:
            col_s1, col_s2 = st.columns(2)
            with col_s1:
                fig_dow_all = px.bar(x=[dow_options[i] for i in selected_dow_indices], y=dow_avg.values, labels={'x': '曜日', 'y': '1日平均 集中時間 (分)'}, title="曜日別の平均集中時間")
                fig_dow_all.update_traces(marker_color='#28a745')
                st.plotly_chart(fig_dow_all, use_container_width=True)
            with col_s2:
                fig_hour_all = px.bar(x=[f"{h}:00" for h in target_hours_list], y=hour_avg.values, labels={'x': '時間帯', 'y': '1日平均 集中時間 (分)'}, title="時間帯別の平均集中時間")
                fig_hour_all.update_traces(marker_color='#28a745')
                st.plotly_chart(fig_hour_all, use_container_width=True)
        else:
            st.info("データを表示するための十分な記録がありません。")

        # --- ヒートマップ (全期間) ---
        st.markdown("##### 📍 曜日×時間帯 ヒートマップ (全期間)")
        col_hm1, col_hm2 = st.columns(2)
        def plot_overall_hm(metric_col, colorscale, title):
            if metric_col not in df_insight.columns: return None
            df_h = df_insight[[metric_col]].resample('1H').mean()
            df_h['hour'] = df_h.index.hour
            df_h['dow'] = df_h.index.dayofweek
            pivot = df_h.pivot_table(values=metric_col, index='hour', columns='dow', aggfunc='mean')
            heatmap_data = np.full((len(target_hours_list), len(selected_dow_indices)), np.nan)
            for i, h in enumerate(target_hours_list):
                for j, d in enumerate(selected_dow_indices):
                    if h in pivot.index and d in pivot.columns:
                        heatmap_data[i, j] = pivot.loc[h, d]
            fig = go.Figure(data=go.Heatmap(z=heatmap_data, x=[dow_options[d] for d in selected_dow_indices], y=[f"{h}:00" for h in target_hours_list], colorscale=colorscale, hoverongaps=False))
            fig.update_layout(title=title, yaxis_autorange='reversed', height=350, margin=dict(l=20, r=20, t=40, b=20))
            return fig
        
        with col_hm1:
            fig_hm_focus = plot_overall_hm('集中判定', 'Blues', "集中確率 (青いほど高い)")
            if fig_hm_focus: st.plotly_chart(fig_hm_focus, use_container_width=True)
        with col_hm2:
            if '疲労判定' in df_insight.columns:
                fig_hm_fat = plot_overall_hm('疲労判定', 'Reds', "疲労確率 (赤いほど高い)")
                if fig_hm_fat: st.plotly_chart(fig_hm_fat, use_container_width=True)