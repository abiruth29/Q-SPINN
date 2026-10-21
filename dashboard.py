import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import time
from collections import deque

st.set_page_config(page_title="Q-SPINN Latent-Flow Monitor", layout="wide", page_icon="⚛️")

st.title("⚛️ Q-SPINN Latent-Flow Monitor (research prototype)")
st.markdown("Live display of the latent-flow diagnostic ω = λ_max(S). **Not a validated crash predictor**: the alert score below is an uncalibrated sigmoid of ω, and until the forward early-warning study (docs/RUNBOOK_TEAMMATE.md, item 10) shows ω beats a rolling-volatility baseline, treat it as a diagnostic only.")

# --- Sidebar Controls ---
with st.sidebar:
    st.header("⚙️ Monitor Controls")
    if st.button("Reset Dashboard History", use_container_width=True):
        st.session_state.history.clear()
        st.session_state.last_omega = -0.3
    
    st.markdown("---")
    dynamic_threshold = st.slider("Display threshold (ω, uncalibrated)", min_value=-1.0, max_value=2.0, value=0.60, step=0.05)

from pathlib import Path
import json

# --- Live feed (simulated data only when explicitly enabled) ---
import os
ALLOW_MOCK = os.environ.get('QSPINN_DASH_MOCK') == '1'
if 'history' not in st.session_state:
    st.session_state.history = deque(maxlen=60)
    
def get_latest_inference():
    log_file = Path("live_inference.jsonl")
    if log_file.exists():
        try:
            # Read the last line of the live inference stream
            with open(log_file, "r") as f:
                lines = f.readlines()
                if lines:
                    return json.loads(lines[-1])
        except Exception:
            pass
            
    if not ALLOW_MOCK:
        return None
    # Simulated feed - ONLY when QSPINN_DASH_MOCK=1; the page shows a SIMULATED banner
    if 'last_omega' not in st.session_state:
        st.session_state.last_omega = -0.3
        
    pull = (-0.3 - st.session_state.last_omega) * 0.15
    st.session_state.last_omega += pull + np.random.normal(loc=0.0, scale=0.04)
    if np.random.random() < 0.03 and st.session_state.last_omega < -0.1:
        st.session_state.last_omega += np.random.uniform(0.3, 0.6)
    if st.session_state.last_omega > 0.2 and np.random.random() < 0.3:
        st.session_state.last_omega = -0.4

    st.session_state.last_omega = np.clip(st.session_state.last_omega, -0.5, 0.5)
    omega = st.session_state.last_omega
    
    btc = 64000 + np.random.normal(0, 50)
    eth = 3500 + np.random.normal(0, 10)
    sol = 150 + np.random.normal(0, 2)
    
    return {"time": time.strftime("%H:%M:%S"), "omega": omega, "BTC": btc, "ETH": eth, "SOL": sol}

# --- Single Iteration Render ---
data = get_latest_inference()
if data is None:
    st.error('No live inference feed (live_inference.jsonl not found). Start pipeline.py, or set QSPINN_DASH_MOCK=1 for a clearly labelled simulation.')
    st.stop()
if ALLOW_MOCK and not data.get('mode'):
    data['mode'] = 'SIMULATED DATA - not model output'

if data.get('omega') is None:
    st.info("ω unavailable: the gate has not routed any window to the heavy expert yet (gated mode). "
            f"Latest cheap-path latency: {data.get('latency_ms', float('nan')):.1f} ms.")
    st.stop()
if data.get('omega_status') == 'cached':
    st.caption(f"ω is CACHED from {data.get('omega_age_ticks', '?')} ticks ago (heavy path skipped on the latest ticks).")
elif data.get('omega_status') == 'fresh':
    st.caption("ω is fresh (heavy path ran on this tick)." + ("" if data.get('omega_mode') != 'always'
               else " Always-on mode: no compute-saving claim."))

# Uncalibrated alert score (sigmoid of omega) - NOT a probability of a crash
crash_prob = 1 / (1 + np.exp(-20 * (data['omega'] - dynamic_threshold)))
data['crash_prob'] = crash_prob

if data.get('mode'):
    st.warning(f"Mode: {data['mode']}")
st.session_state.history.append(data)
df = pd.DataFrame(st.session_state.history)

# Top Metrics
c1, c2, c3, c4 = st.columns(4)
c1.metric(label="BTC/USDT", value=f"${data['BTC']:,.2f}")
c2.metric(label="ETH/USDT", value=f"${data['ETH']:,.2f}")
c3.metric(label="SOL/USDT", value=f"${data['SOL']:,.2f}")

status_color = "🟠 ω above display threshold (diagnostic only)" if data['omega'] > dynamic_threshold else "⚪ ω below display threshold (diagnostic only)"
c4.metric(label="System Status", value=status_color)

st.markdown("---")

# Main Charts
chart_col1, chart_col2 = st.columns([1, 2])

with chart_col1:
    # Numerical Abscissa Gauge
    fig_gauge = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = data['omega'],
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Numerical Abscissa ω(A_t)"},
        gauge = {
            'axis': {'range': [-0.5, 1.5]},
            'bar': {'color': "black"},
            'steps': [
                {'range': [-0.5, dynamic_threshold], 'color': "rgba(144, 238, 144, 0.5)"}, # lightgreen
                {'range': [dynamic_threshold, 1.5], 'color': "rgba(250, 128, 114, 0.5)"}   # salmon
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': dynamic_threshold
            }
        }
    ))
    fig_gauge.update_layout(height=350, margin=dict(l=20, r=20, t=50, b=20))
    st.plotly_chart(fig_gauge, use_container_width=True, key="gauge")
    
with chart_col2:
    # Time Series Chart
    fig_line = go.Figure()
    # Gradient fill for danger zone
    fig_line.add_trace(go.Scatter(
        x=df['time'], 
        y=df['crash_prob'], 
        mode='lines', 
        name='Alert score (uncalibrated)', 
        line=dict(color='red', width=3),
        fill='tozeroy',
        fillcolor='rgba(255, 0, 0, 0.1)',
        yaxis="y1"
    ))
    if 'omega_fresh_mean' in df.columns:
        fig_line.add_trace(go.Scatter(
            x=df['time'], 
            y=df['omega_fresh_mean'], 
            mode='lines', 
            name='Mean of recent fresh ω (sample-based, not a time integral)', 
            line=dict(color='orange', width=2, dash='dot'),
            yaxis="y2"
        ))
    if 'd_J' in df.columns:
        fig_line.add_trace(go.Scatter(
            x=df['time'], 
            y=df['d_J'], 
            mode='lines', 
            name='Phase Divergence (d_J)', 
            line=dict(color='blue', width=2, dash='dot'),
            yaxis="y2"
        ))
        
    fig_line.add_hline(y=0.5, line_dash="dash", line_color="gray", annotation_text=f"Critical Threshold (ω={dynamic_threshold:.2f})")
    fig_line.update_layout(
        title="Real-Time Diagnostic Metrics", 
        yaxis=dict(title="Alert score (not a probability)", range=[0, 1]),
        yaxis2=dict(title="Diagnostic Scale", overlaying="y", side="right"),
        height=350,
        margin=dict(l=20, r=20, t=50, b=20),
        xaxis_title="Time"
    )
    st.plotly_chart(fig_line, use_container_width=True, key="line")

time.sleep(1.0)
st.rerun()
