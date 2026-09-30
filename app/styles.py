"""Global CSS for the FraudGuard Streamlit app."""

APP_CSS = """
/* ---------- Base ---------- */
.main .block-container { padding-top: 1.5rem; padding-bottom: 3rem; max-width: 1400px; }
h1, h2, h3, h4 { color: #E6EDF3; letter-spacing: -0.01em; }
p, li, span, label { color: #C9D1D9; }

/* ---------- Hero ---------- */
.fg-hero {
    background: linear-gradient(135deg, #1f4068 0%, #162447 45%, #0f1626 100%);
    border: 1px solid #24344d;
    border-radius: 14px;
    padding: 22px 26px;
    margin-bottom: 18px;
    box-shadow: 0 6px 24px rgba(0,0,0,0.35);
}
.fg-hero h1 {
    margin: 0 0 6px 0;
    font-size: 1.65rem;
    color: #ffffff;
}
.fg-hero p {
    margin: 0;
    color: #b8c7dc;
    font-size: 0.95rem;
    line-height: 1.5;
}
.fg-hero .fg-pill {
    display: inline-block;
    background: #c45c26;
    color: white;
    padding: 2px 10px;
    border-radius: 999px;
    font-size: 0.72rem;
    font-weight: 600;
    margin-left: 8px;
    vertical-align: middle;
}

/* ---------- Metric cards ---------- */
.fg-card {
    background: linear-gradient(160deg, #161B22 0%, #1a2230 100%);
    border: 1px solid #24344d;
    border-left: 4px solid #4F8BF9;
    border-radius: 12px;
    padding: 14px 16px;
    transition: transform 0.15s ease, box-shadow 0.15s ease;
    height: 100%;
}
.fg-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 22px rgba(79,139,249,0.18);
}
.fg-card .label {
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #8b98a9;
    margin-bottom: 4px;
}
.fg-card .value {
    font-size: 1.55rem;
    font-weight: 700;
    color: #ffffff;
    line-height: 1.2;
}
.fg-card .hint {
    font-size: 0.72rem;
    color: #6e7b8c;
    margin-top: 4px;
}
.fg-card.danger { border-left-color: #c45c26; }
.fg-card.danger .value { color: #ffb38a; }
.fg-card.success { border-left-color: #2ea043; }
.fg-card.success .value { color: #7ee787; }

/* ---------- Risk badge ---------- */
.fg-badge {
    display: inline-block;
    padding: 6px 16px;
    border-radius: 999px;
    font-weight: 700;
    font-size: 0.95rem;
    letter-spacing: 0.04em;
    text-transform: uppercase;
    border: 1px solid transparent;
}
.fg-badge.low    { background: rgba(46,160,67,0.15); color: #7ee787; border-color: #2ea043; }
.fg-badge.medium { background: rgba(210,153,34,0.15); color: #f0c674; border-color: #d29922; }
.fg-badge.high   { background: rgba(196,92,38,0.2);  color: #ffb38a; border-color: #c45c26; }
.fg-badge.critical {
    background: rgba(220,38,38,0.22); color: #ff8a8a; border-color: #dc2626;
    animation: pulse 1.6s infinite;
}
@keyframes pulse {
    0%,100% { box-shadow: 0 0 0 0 rgba(220,38,38,0.45); }
    50%     { box-shadow: 0 0 0 10px rgba(220,38,38,0); }
}

/* ---------- Risk panel ---------- */
.risk-panel {
    background: linear-gradient(160deg, #161B22 0%, #101720 100%);
    border: 1px solid #24344d;
    border-radius: 14px;
    padding: 18px 20px;
    margin: 10px 0 18px 0;
}

/* ---------- Sidebar ---------- */
section[data-testid="stSidebar"] {
    background: #0b0f16;
    border-right: 1px solid #1e2836;
}
section[data-testid="stSidebar"] .stRadio label {
    padding: 6px 8px;
    border-radius: 8px;
    transition: background 0.15s ease;
}
section[data-testid="stSidebar"] .stRadio label:hover {
    background: #16202e;
}

/* ---------- Buttons ---------- */
.stButton > button {
    border-radius: 10px;
    border: 1px solid #24344d;
    transition: all 0.15s ease;
}
.stButton > button:hover {
    border-color: #4F8BF9;
    transform: translateY(-1px);
}
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #4F8BF9 0%, #2d6fd6 100%);
    border: none;
    font-weight: 600;
}
.stButton > button[kind="primary"]:hover {
    box-shadow: 0 6px 18px rgba(79,139,249,0.4);
}

/* ---------- Misc ---------- */
.fg-divider { height: 1px; background: #1e2836; margin: 18px 0; border: 0; }
.fg-muted { color: #8b98a9; font-size: 0.85rem; }
"""