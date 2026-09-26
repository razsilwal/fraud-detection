"""FraudGuard visual language — restrained finance/risk styling."""

APP_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&family=IBM+Plex+Mono:wght@500&display=swap');

html, body, [class*="css"]  {
  font-family: "IBM Plex Sans", sans-serif;
}

.stApp {
  background: #f4f1ea;
  color: #1b2429;
}

section[data-testid="stSidebar"] {
  background: #1b2a32;
}
section[data-testid="stSidebar"] * {
  color: #e8eef2 !important;
}
section[data-testid="stSidebar"] .stRadio label {
  padding: 0.35rem 0;
}

h1, h2, h3 {
  font-family: "IBM Plex Sans", sans-serif;
  letter-spacing: -0.02em;
  color: #132026;
}

.hero {
  background: linear-gradient(135deg, #132026 0%, #1f3b47 60%, #2c5364 100%);
  color: #f4f1ea;
  padding: 1.6rem 1.8rem;
  border-radius: 14px;
  margin-bottom: 1.2rem;
}
.hero h1 {
  color: #f4f1ea;
  margin: 0 0 0.35rem 0;
  font-size: 2rem;
}
.hero p {
  margin: 0;
  color: #c9d6dc;
  max-width: 52rem;
}

.metric-card {
  background: #fffdf8;
  border: 1px solid #ddd4c4;
  border-radius: 12px;
  padding: 0.95rem 1rem 0.8rem 1rem;
  box-shadow: 0 1px 0 rgba(19,32,38,0.04);
}
.metric-card .label {
  font-size: 0.78rem;
  text-transform: uppercase;
  letter-spacing: 0.08em;
  color: #5c6b73;
  margin-bottom: 0.2rem;
}
.metric-card .value {
  font-family: "IBM Plex Mono", monospace;
  font-size: 1.55rem;
  font-weight: 600;
  color: #132026;
}

.badge {
  display: inline-block;
  padding: 0.28rem 0.7rem;
  border-radius: 999px;
  font-weight: 600;
  letter-spacing: 0.04em;
  font-size: 0.85rem;
}
.badge-low { background: #d7efe3; color: #145c38; }
.badge-medium { background: #f7e3c2; color: #7a4b00; }
.badge-high { background: #f6cfcb; color: #8a1f18; }

.risk-panel {
  background: #fffdf8;
  border-radius: 14px;
  border: 1px solid #ddd4c4;
  padding: 1.2rem 1.3rem;
}
.footnote {
  color: #5c6b73;
  font-size: 0.9rem;
}
hr { border-color: #ddd4c4; }
</style>
"""
