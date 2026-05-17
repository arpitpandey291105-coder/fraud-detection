import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go
import streamlit_authenticator as stauth
import yaml
from yaml.loader import SafeLoader

st.set_page_config(
    page_title="FraudShield AI",
    page_icon="🛡️",
    layout="wide"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
* { font-family: 'Inter', sans-serif; }
.stApp { background: #0a0a0f; }
.hero { text-align: center; padding: 50px 20px 30px; }
.hero-badge {
    display: inline-block;
    background: rgba(120,40,200,0.2);
    border: 1px solid rgba(120,40,200,0.5);
    color: #a78bfa;
    padding: 6px 20px;
    border-radius: 50px;
    font-size: 13px;
    font-weight: 500;
    letter-spacing: 2px;
    text-transform: uppercase;
    margin-bottom: 20px;
}
.hero-title {
    font-size: 64px;
    font-weight: 700;
    background: linear-gradient(135deg, #ffffff 0%, #a78bfa 50%, #06b6d4 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    line-height: 1.1;
    margin: 10px 0;
}
.hero-subtitle { color: #6b7280; font-size: 18px; font-weight: 300; margin-top: 15px; }
.stat-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 16px;
    margin: 30px 0;
}
.stat-card {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 16px;
    padding: 24px;
    text-align: center;
    position: relative;
    overflow: hidden;
}
.stat-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, #7c3aed, #06b6d4);
}
.stat-number {
    font-size: 32px;
    font-weight: 700;
    background: linear-gradient(135deg, #a78bfa, #06b6d4);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.stat-label { color: #6b7280; font-size: 13px; margin-top: 5px; text-transform: uppercase; letter-spacing: 1px; }
.section-title { font-size: 20px; font-weight: 600; color: #e5e7eb; margin-bottom: 20px; }
.result-fraud {
    background: linear-gradient(135deg, rgba(239,68,68,0.15), rgba(239,68,68,0.05));
    border: 1px solid rgba(239,68,68,0.4);
    border-radius: 20px;
    padding: 30px;
    text-align: center;
}
.result-fraud-title { font-size: 28px; font-weight: 700; color: #ef4444; margin-bottom: 10px; }
.result-legit {
    background: linear-gradient(135deg, rgba(16,185,129,0.15), rgba(16,185,129,0.05));
    border: 1px solid rgba(16,185,129,0.4);
    border-radius: 20px;
    padding: 30px;
    text-align: center;
}
.result-legit-title { font-size: 28px; font-weight: 700; color: #10b981; margin-bottom: 10px; }
.conf-bar-bg {
    background: rgba(255,255,255,0.05);
    border-radius: 50px;
    height: 10px;
    margin: 8px 0 20px;
    overflow: hidden;
}
.conf-bar-fill-legit {
    height: 100%;
    border-radius: 50px;
    background: linear-gradient(90deg, #10b981, #06b6d4);
}
.conf-bar-fill-fraud {
    height: 100%;
    border-radius: 50px;
    background: linear-gradient(90deg, #ef4444, #f97316);
}
.conf-label { display: flex; justify-content: space-between; color: #9ca3af; font-size: 13px; }
.summary-grid { display: grid; grid-template-columns: repeat(2, 1fr); gap: 12px; margin-top: 20px; }
.summary-item {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 12px;
    padding: 16px;
}
.summary-item-label { color: #6b7280; font-size: 12px; text-transform: uppercase; letter-spacing: 1px; }
.summary-item-value { color: #e5e7eb; font-size: 20px; font-weight: 600; margin-top: 4px; }
.placeholder {
    border: 1px dashed rgba(255,255,255,0.1);
    border-radius: 20px;
    padding: 60px 30px;
    text-align: center;
    color: #4b5563;
}
.stButton > button {
    width: 100%;
    background: linear-gradient(135deg, #7c3aed, #06b6d4);
    color: white;
    font-size: 16px;
    font-weight: 600;
    padding: 16px;
    border-radius: 12px;
    border: none;
    margin-top: 24px;
    box-shadow: 0 4px 20px rgba(124,58,237,0.4);
}
.stNumberInput input {
    background: #1a1a2e !important;
    border: 1px solid rgba(167,139,250,0.3) !important;
    border-radius: 10px !important;
    color: #ffffff !important;
    -webkit-text-fill-color: #ffffff !important;
    font-size: 16px !important;
}
input[type="number"] {
    background: #1a1a2e !important;
    color: #ffffff !important;
    -webkit-text-fill-color: #ffffff !important;
}
.stExpander {
    background: rgba(255,255,255,0.02) !important;
    border: 1px solid rgba(255,255,255,0.08) !important;
    border-radius: 12px !important;
}
label { color: #9ca3af !important; }
.divider { border: none; border-top: 1px solid rgba(255,255,255,0.06); margin: 30px 0; }
.footer { text-align: center; color: #374151; font-size: 13px; padding: 30px; }
</style>
""", unsafe_allow_html=True)

# ============================================================
# LOGIN SYSTEM
# ============================================================
with open('config.yaml') as file:
    config = yaml.load(file, Loader=SafeLoader)

authenticator = stauth.Authenticate(
    config['credentials'],
    config['cookie']['name'],
    config['cookie']['key'],
    config['cookie']['expiry_days']
)

# ✅ Login form render karo (yahi fix hai)
authenticator.login()

# ✅ Case 1: Wrong credentials
if st.session_state['authentication_status'] == False:
    st.error("❌ Username or Password is incorrect!")
    st.markdown("---")
    st.markdown("### 📝 New User? Register Here")
    try:
        result = authenticator.register_user()
        if result:
            st.success("✅ Registration Successful! Please Login now.")
            with open('config.yaml', 'w') as file:
                yaml.dump(config, file, default_flow_style=False)
    except Exception as e:
        st.error(f"Registration Error: {e}")
    st.stop()

# ✅ Case 2: Not logged in yet (None state)
elif st.session_state['authentication_status'] is None:
    st.markdown("""
    <div style='text-align:center; padding: 20px;'>
        <h1 style='color:#a78bfa'>🛡️ FraudShield AI</h1>
        <p style='color:#6b7280'>Enter your credentials above to continue</p>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("### 📝 New User? Register Here")
    try:
        result = authenticator.register_user()
        if result:
            st.success("✅ Registration Successful! Please Login now.")
            with open('config.yaml', 'w') as file:
                yaml.dump(config, file, default_flow_style=False)
    except Exception as e:
        st.error(f"Registration Error: {e}")
    st.stop()

# ✅ Case 3: Successfully logged in — poora app
elif st.session_state['authentication_status']:

    # Load Model
    model = pickle.load(open('model/fraud_model.pkl', 'rb'))

    # Session State
    if 'history' not in st.session_state:
        st.session_state.history = []

    # Logout Button in Sidebar
    authenticator.logout("🚪 Logout", "sidebar")

    # Hero Section
    st.markdown(f"""
    <div class="hero">
        <div class="hero-badge">🛡️ AI Powered Security</div>
        <div class="hero-title">FraudShield AI</div>
        <div class="hero-subtitle">Welcome back, <b>{st.session_state['name']}</b>! 👋</div>
    </div>
    """, unsafe_allow_html=True)

    # Stats
    st.markdown("""
    <div class="stat-grid">
        <div class="stat-card">
            <div class="stat-number">99.99%</div>
            <div class="stat-label">🎯 Accuracy</div>
        </div>
        <div class="stat-card">
            <div class="stat-number">0.9999</div>
            <div class="stat-label">📊 ROC-AUC Score</div>
        </div>
        <div class="stat-card">
            <div class="stat-number">284K+</div>
            <div class="stat-label">📁 Transactions Trained</div>
        </div>
        <div class="stat-card">
            <div class="stat-number">100</div>
            <div class="stat-label">🌲 Decision Trees</div>
        </div>
    </div>
    <hr class="divider">
    """, unsafe_allow_html=True)

    # Tabs
    tab1, tab2, tab3 = st.tabs(["🔍 Single Check", "📂 CSV Bulk Upload", "📊 Dashboard"])

    # ===== TAB 1: SINGLE CHECK =====
    with tab1:
        col_left, col_right = st.columns([1, 1], gap="large")

        with col_left:
            st.markdown('<div class="section-title">💰 Transaction Details</div>', unsafe_allow_html=True)
            amount = st.number_input("Transaction Amount ($)", min_value=0.0, max_value=100000.0, value=100.0)
            time = st.number_input("Time (seconds since first transaction)", min_value=0.0, value=50000.0)

            with st.expander("⚙️ Advanced Features — V1 to V28 (Optional)"):
                v_features = []
                c1, c2 = st.columns(2)
                for i in range(1, 29):
                    if i % 2 != 0:
                        with c1:
                            v = st.number_input(f"V{i}", value=0.0, key=f"v{i}")
                    else:
                        with c2:
                            v = st.number_input(f"V{i}", value=0.0, key=f"v{i}")
                    v_features.append(v)

            analyze = st.button("🚀 Analyze Transaction")

        with col_right:
            st.markdown('<div class="section-title">🔍 Detection Result</div>', unsafe_allow_html=True)

            if analyze:
                with st.spinner("Analyzing..."):
                    input_data = [time] + v_features + [amount]
                    input_df = pd.DataFrame([input_data], columns=[
                        'Time','V1','V2','V3','V4','V5','V6','V7','V8','V9','V10',
                        'V11','V12','V13','V14','V15','V16','V17','V18','V19','V20',
                        'V21','V22','V23','V24','V25','V26','V27','V28','Amount'
                    ])
                    sc = StandardScaler()
                    input_df['Amount'] = sc.fit_transform(input_df[['Amount']])
                    input_df['Time'] = sc.fit_transform(input_df[['Time']])
                    prediction = model.predict(input_df)[0]
                    probability = model.predict_proba(input_df)[0]

                legit_pct = round(probability[0] * 100, 2)
                fraud_pct = round(probability[1] * 100, 2)
                risk_score = round(fraud_pct)

                if prediction == 1:
                    st.markdown("""
                    <div class="result-fraud">
                        <div class="result-fraud-title">🚨 FRAUD DETECTED</div>
                        <p style="color:#9ca3af">This transaction has been flagged as suspicious</p>
                    </div>""", unsafe_allow_html=True)
                else:
                    st.markdown("""
                    <div class="result-legit">
                        <div class="result-legit-title">✅ LEGITIMATE</div>
                        <p style="color:#9ca3af">This transaction appears to be safe</p>
                    </div>""", unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)

                # Gauge
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=risk_score,
                    title={'text': "Risk Score", 'font': {'color': 'white', 'size': 16}},
                    gauge={
                        'axis': {'range': [0, 100], 'tickcolor': 'white'},
                        'bar': {'color': "#ef4444" if risk_score > 50 else "#10b981"},
                        'bgcolor': "rgba(255,255,255,0.05)",
                        'steps': [
                            {'range': [0, 40], 'color': 'rgba(16,185,129,0.2)'},
                            {'range': [40, 70], 'color': 'rgba(245,158,11,0.2)'},
                            {'range': [70, 100], 'color': 'rgba(239,68,68,0.2)'}
                        ],
                    },
                    number={'font': {'color': 'white', 'size': 40}}
                ))
                fig.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    height=250,
                    margin=dict(t=30, b=0, l=20, r=20),
                    font={'color': 'white'}
                )
                st.plotly_chart(fig, use_container_width=True)

                st.markdown(f"""
                <div class="conf-label"><span>✅ Legitimate</span><span>{legit_pct}%</span></div>
                <div class="conf-bar-bg"><div class="conf-bar-fill-legit" style="width:{legit_pct}%"></div></div>
                <div class="conf-label"><span>🚨 Fraud</span><span>{fraud_pct}%</span></div>
                <div class="conf-bar-bg"><div class="conf-bar-fill-fraud" style="width:{fraud_pct}%"></div></div>
                """, unsafe_allow_html=True)

                st.markdown(f"""
                <div class="summary-grid">
                    <div class="summary-item">
                        <div class="summary-item-label">💰 Amount</div>
                        <div class="summary-item-value">${amount:.2f}</div>
                    </div>
                    <div class="summary-item">
                        <div class="summary-item-label">⏱️ Time</div>
                        <div class="summary-item-value">{time:.0f}s</div>
                    </div>
                    <div class="summary-item">
                        <div class="summary-item-label">🎯 Verdict</div>
                        <div class="summary-item-value" style="color:{'#ef4444' if prediction==1 else '#10b981'}">
                            {'FRAUD' if prediction==1 else 'LEGIT'}
                        </div>
                    </div>
                    <div class="summary-item">
                        <div class="summary-item-label">📊 Risk Score</div>
                        <div class="summary-item-value" style="color:{'#ef4444' if risk_score>50 else '#10b981'}">
                            {risk_score}/100
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                st.session_state.history.append({
                    'User': st.session_state['name'],
                    'Amount': f"${amount:.2f}",
                    'Time': f"{time:.0f}s",
                    'Risk Score': f"{risk_score}/100",
                    'Result': '🚨 FRAUD' if prediction == 1 else '✅ LEGIT',
                    'Confidence': f"{max(probability)*100:.2f}%"
                })

            else:
                st.markdown("""
                <div class="placeholder">
                    <div style="font-size:48px">🛡️</div>
                    <p style="font-size:18px; color:#6b7280">Ready to analyze</p>
                    <p style="font-size:14px; color:#4b5563">
                        Enter details and click<br>
                        <b style="color:#7c3aed">Analyze Transaction</b>
                    </p>
                </div>""", unsafe_allow_html=True)

    # ===== TAB 2: CSV UPLOAD =====
    with tab2:
        st.markdown('<div class="section-title">📂 Bulk Transaction Check</div>', unsafe_allow_html=True)
        st.info("Upload a CSV file with transaction data to check multiple transactions at once.")

        uploaded_file = st.file_uploader("Upload CSV File", type=['csv'])

        if uploaded_file:
            df_upload = pd.read_csv(uploaded_file)
            st.write("Preview:", df_upload.head())

            if st.button("🔍 Analyze All Transactions"):
                with st.spinner("Analyzing all transactions..."):
                    try:
                        sc = StandardScaler()
                        df_upload['Amount'] = sc.fit_transform(df_upload[['Amount']])
                        df_upload['Time'] = sc.fit_transform(df_upload[['Time']])

                        features = ['Time','V1','V2','V3','V4','V5','V6','V7','V8','V9','V10',
                                    'V11','V12','V13','V14','V15','V16','V17','V18','V19','V20',
                                    'V21','V22','V23','V24','V25','V26','V27','V28','Amount']

                        X = df_upload[features]
                        predictions = model.predict(X)
                        probabilities = model.predict_proba(X)

                        df_upload['Prediction'] = ['🚨 FRAUD' if p == 1 else '✅ LEGIT' for p in predictions]
                        df_upload['Fraud %'] = [f"{p[1]*100:.2f}%" for p in probabilities]
                        df_upload['Risk Score'] = [round(p[1]*100) for p in probabilities]

                        total = len(df_upload)
                        fraud_count = sum(predictions)
                        legit_count = total - fraud_count

                        c1, c2, c3 = st.columns(3)
                        c1.metric("Total", total)
                        c2.metric("🚨 Fraud", fraud_count)
                        c3.metric("✅ Legit", legit_count)

                        st.dataframe(
                            df_upload[['Amount', 'Time', 'Prediction', 'Fraud %', 'Risk Score']],
                            use_container_width=True
                        )

                        csv = df_upload.to_csv(index=False)
                        st.download_button("⬇️ Download Results", csv, "fraud_results.csv", "text/csv")

                    except Exception as e:
                        st.error(f"Error: {e}")
        else:
            st.markdown("""
            <div class="placeholder">
                <div style="font-size:48px">📂</div>
                <p style="font-size:18px; color:#6b7280">Upload a CSV file to begin</p>
                <p style="font-size:13px; color:#4b5563">File must have: Time, V1-V28, Amount columns</p>
            </div>""", unsafe_allow_html=True)

    # ===== TAB 3: DASHBOARD =====
    with tab3:
        st.markdown('<div class="section-title">📊 Transaction Dashboard</div>', unsafe_allow_html=True)

        if len(st.session_state.history) > 0:
            hist_df = pd.DataFrame(st.session_state.history)
            fraud_count = hist_df['Result'].str.contains('FRAUD').sum()
            legit_count = hist_df['Result'].str.contains('LEGIT').sum()

            c1, c2, c3 = st.columns(3)
            c1.metric("Total Checked", len(hist_df))
            c2.metric("🚨 Fraud", fraud_count)
            c3.metric("✅ Legitimate", legit_count)

            fig_pie = go.Figure(go.Pie(
                labels=['Legitimate', 'Fraud'],
                values=[legit_count, fraud_count],
                hole=0.5,
                marker_colors=['#10b981', '#ef4444']
            ))
            fig_pie.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                font={'color': 'white'},
                title={'text': 'Fraud vs Legitimate', 'font': {'color': 'white'}},
                height=350
            )
            st.plotly_chart(fig_pie, use_container_width=True)

            st.markdown("### 📋 Transaction History")
            st.dataframe(hist_df, use_container_width=True)

            if st.button("🗑️ Clear History"):
                st.session_state.history = []
                st.rerun()
        else:
            st.markdown("""
            <div class="placeholder">
                <div style="font-size:48px">📊</div>
                <p style="font-size:18px; color:#6b7280">No transactions yet</p>
                <p style="font-size:14px; color:#4b5563">
                    Go to <b style="color:#7c3aed">Single Check</b> and analyze transactions!
                </p>
            </div>""", unsafe_allow_html=True)

    # Footer
    st.markdown("""
    <hr class="divider">
    <div class="footer">
        Built with using Python • Random Forest • Streamlit | FraudShield AI v3.0
    </div>
    """, unsafe_allow_html=True)