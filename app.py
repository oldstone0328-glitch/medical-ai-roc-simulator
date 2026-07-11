from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import auc, roc_curve


st.set_page_config(
    page_title="의료 AI 판정 기준 실험",
    layout="wide",
    initial_sidebar_state="collapsed",
)

PATIENT_COUNT = 200
CANCER_COUNT = 20
RANDOM_SEED = 20250712
ACCENT_COLOR = "#315f8c"


@dataclass(frozen=True)
class Metrics:
    tp: int
    fp: int
    tn: int
    fn: int
    accuracy: float
    sensitivity: float
    specificity: float
    fpr: float


def safe_divide(numerator: int, denominator: int) -> float:
    """분모가 0이어도 앱이 중단되지 않게 한다."""
    return numerator / denominator if denominator else 0.0


@st.cache_data(show_spinner=False)
def generate_virtual_data(seed: int = RANDOM_SEED) -> tuple[np.ndarray, np.ndarray]:
    """암 20명, 정상 180명의 고정된 가상 예측값을 만든다."""
    rng = np.random.default_rng(seed)
    actual = np.zeros(PATIENT_COUNT, dtype=int)
    actual[rng.choice(PATIENT_COUNT, size=CANCER_COUNT, replace=False)] = 1

    probability = np.empty(PATIENT_COUNT, dtype=float)
    probability[actual == 1] = rng.beta(4.6, 2.8, size=CANCER_COUNT)
    probability[actual == 0] = rng.beta(
        2.5, 5.0, size=PATIENT_COUNT - CANCER_COUNT
    )
    return actual, np.clip(probability, 0.0, 1.0)


def calculate_metrics(
    actual: np.ndarray, probability: np.ndarray, threshold: float
) -> Metrics:
    predicted = probability >= threshold
    tp = int(np.sum((actual == 1) & predicted))
    fp = int(np.sum((actual == 0) & predicted))
    tn = int(np.sum((actual == 0) & ~predicted))
    fn = int(np.sum((actual == 1) & ~predicted))

    return Metrics(
        tp=tp,
        fp=fp,
        tn=tn,
        fn=fn,
        accuracy=safe_divide(tp + tn, len(actual)),
        sensitivity=safe_divide(tp, tp + fn),
        specificity=safe_divide(tn, tn + fp),
        fpr=safe_divide(fp, fp + tn),
    )


def calculate_roc(
    actual: np.ndarray, probability: np.ndarray
) -> tuple[np.ndarray, np.ndarray, float]:
    if np.unique(actual).size < 2:
        return np.array([0.0, 1.0]), np.array([0.0, 1.0]), 0.5
    fpr_values, tpr_values, _ = roc_curve(actual, probability)
    return fpr_values, tpr_values, float(auc(fpr_values, tpr_values))


def create_roc_figure(
    fpr_values: np.ndarray,
    tpr_values: np.ndarray,
    auc_value: float,
    threshold: float,
    metrics: Metrics,
) -> go.Figure:
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            line=dict(color="#999999", width=1.5, dash="dash"),
            hoverinfo="skip",
            showlegend=False,
        )
    )
    figure.add_trace(
        go.Scatter(
            x=fpr_values,
            y=tpr_values,
            mode="lines",
            line=dict(color=ACCENT_COLOR, width=3),
            hoverinfo="skip",
            showlegend=False,
        )
    )
    figure.add_trace(
        go.Scatter(
            x=[metrics.fpr],
            y=[metrics.sensitivity],
            mode="markers",
            marker=dict(
                size=16,
                color=ACCENT_COLOR,
                line=dict(color="white", width=2),
            ),
            customdata=[[threshold, metrics.sensitivity, metrics.fpr]],
            hovertemplate=(
                "판정 기준: %{customdata[0]:.0%}<br>"
                "민감도: %{customdata[1]:.1%}<br>"
                "위양성률: %{customdata[2]:.1%}<extra></extra>"
            ),
            showlegend=False,
        )
    )
    figure.update_layout(
        title=dict(
            text=f"ROC Curve <span style='font-size:13px;color:#666'>AUC {auc_value:.3f}</span>",
            x=0.01,
            xanchor="left",
            font=dict(size=18, color="#222222"),
        ),
        height=270,
        margin=dict(l=48, r=12, t=34, b=38),
        paper_bgcolor="white",
        plot_bgcolor="white",
        xaxis=dict(
            title="위양성률 (1-특이도)",
            range=[0, 1],
            tickformat=".0%",
            title_font=dict(size=12),
            tickfont=dict(size=11),
            gridcolor="#e8e8e8",
            zeroline=False,
        ),
        yaxis=dict(
            title="민감도",
            range=[0, 1],
            tickformat=".0%",
            title_font=dict(size=12),
            tickfont=dict(size=11),
            gridcolor="#e8e8e8",
            zeroline=False,
        ),
        hovermode="closest",
    )
    return figure


def show_result(label: str, value: str, main: bool = False, suffix: str = "") -> None:
    css_class = "result-value main" if main else "result-value"
    st.markdown(
        f"""
        <div class="result">
          <div class="result-label">{label}</div>
          <div class="{css_class}">{value} <span class="result-suffix">{suffix}</span></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


st.markdown(
    """
    <style>
      html, body, [class*="css"] {font-family:Pretendard,"Noto Sans KR",Arial,sans-serif;}
      .block-container {max-width: 1200px; padding-top: 3.1rem; padding-left:2.6rem; padding-right:2.6rem; padding-bottom: 0;}
      .block-container > div > [data-testid="stVerticalBlock"] {gap:.24rem;}
      header[data-testid="stHeader"] {height: 2rem; background: white;}
      #MainMenu, footer {visibility: hidden;}
      h1 {font-size: 1.78rem !important; letter-spacing: -.025em; margin: 0 0 .02rem !important; color:#111;}
      .threshold-head {display:block; padding-top:0; margin-bottom:-.2rem;}
      .threshold-label {display:block; font-size:.92rem; font-weight:650; color:#333;}
      .threshold-value {display:block; font-size:1.72rem; line-height:1.05; font-weight:750; color:#111;}
      div[data-testid="stSlider"] {padding-top:0; padding-bottom:0; color:#111;}
      div[data-testid="stSlider"] [data-testid="stSliderThumbValue"] {display:none;}
      .direction {display:flex; justify-content:space-between; color:#777; font-size:.76rem; margin-top:-.5rem; margin-bottom:.08rem;}
      div[data-testid="stHorizontalBlock"] {gap:1.05rem;}
      .result {padding:.12rem 0 .04rem; min-height:49px; color:#111;}
      .result.accuracy {padding-top:.2rem; min-height:104px;}
      .result-label {font-size:.88rem; font-weight:600; color:#444;}
      .result-value {font-size:1.48rem; font-weight:750; line-height:1.08; margin:.02rem 0; color:#111;}
      .result-value.main {font-size:2.75rem; margin-top:.12rem;}
      .result-suffix {font-size:.7rem; font-weight:500; color:#777;}
      .commentary {padding:.06rem 0; margin:.02rem 0 .18rem; font-size:.82rem; font-weight:500; color:#444;}
      div[data-testid="stPlotlyChart"] {margin-top:.12rem;}
      @media (max-width: 800px) {.block-container{padding-top:2.8rem;padding-left:1.4rem;padding-right:1.4rem}.result{min-height:46px}}
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("정확도 99%인데도 암을 놓칠 수 있을까?")

threshold_placeholder = st.empty()
threshold_percent = st.slider(
    "판정 기준",
    min_value=0,
    max_value=100,
    value=50,
    step=1,
    format="%d%%",
    label_visibility="collapsed",
)
threshold_placeholder.markdown(
    f"""
    <div class="threshold-head">
      <span class="threshold-label">판정 기준</span>
      <span class="threshold-value">{threshold_percent}%</span>
    </div>
    """,
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="direction"><span>암 더 많이 찾음</span><span>잘못된 암 판정 줄어듦</span></div>',
    unsafe_allow_html=True,
)

threshold = threshold_percent / 100
actual, probability = generate_virtual_data()
metrics = calculate_metrics(actual, probability, threshold)
fpr_values, tpr_values, auc_value = calculate_roc(actual, probability)

result_area = st.columns([1.15, 2.35])
with result_area[0]:
    st.markdown(
        f"""
        <div class="result accuracy">
          <div class="result-label">정확도</div>
          <div class="result-value main">{metrics.accuracy:.1%}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with result_area[1]:
    upper_results = st.columns([1, 1.15])
    with upper_results[0]:
        show_result("민감도", f"{metrics.sensitivity:.1%}")
    with upper_results[1]:
        show_result("특이도", f"{metrics.specificity:.1%}")
    lower_results = st.columns([.85, 1.3])
    with lower_results[0]:
        show_result("놓친 암", f"{metrics.fn}명", suffix="(FN)")
    with lower_results[1]:
        show_result("잘못된 암 판정", f"{metrics.fp}명", suffix="(FP)")

if threshold_percent <= 35:
    interpretation = "암 발견 ↑&nbsp;&nbsp; 잘못된 암 판정 ↑"
elif threshold_percent >= 65:
    interpretation = "잘못된 암 판정 ↓&nbsp;&nbsp; 놓친 암 ↑"
else:
    interpretation = "민감도와 특이도 균형 구간"
st.markdown(f'<div class="commentary">{interpretation}</div>', unsafe_allow_html=True)

st.plotly_chart(
    create_roc_figure(fpr_values, tpr_values, auc_value, threshold, metrics),
    use_container_width=True,
    config={"displayModeBar": False, "responsive": True},
)
