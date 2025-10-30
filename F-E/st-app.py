import streamlit as st
import requests
import os
from PIL import Image
from io import BytesIO

# ------------------ PAGE CONFIG ------------------
st.set_page_config(
    page_title="FroQ Product Search",
    page_icon="📦",
    layout="centered",
)

# ------------------ CUSTOM CSS ------------------
st.markdown("""
    <style>
    body {
        background-color: #f4f5f7;
        font-family: 'Inter', sans-serif;
    }
    .main-title {
        text-align: center;
        font-size: 2.2em;
        font-weight: 700;
        color: #1e1e1e;
        margin-bottom: 0.4em;
    }
    .subtitle {
        text-align: center;
        font-size: 1.1em;
        color: #666;
        margin-bottom: 2em;
    }
    .stTextInput > div > div > input {
        border-radius: 12px;
        border: 1px solid #ccc;
        padding: 12px;
    }
    .result-card {
        background-color: #fff;
        border-radius: 12px;
        padding: 0.85em 1em;
        margin: 0.75em 0;
        border: 1px solid #eee;
    }
    .result-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 0.75rem;
        margin-bottom: 0.35rem;
    }
    .result-title {
        font-size: 1.02rem;
        font-weight: 600;
        color: #1f2937;
        margin: 0;
    }
    .score-badge {
        display: inline-block;
        background: #f1f5f9;
        color: #0f172a;
        border: 1px solid #e5e7eb;
        padding: 0.15rem 0.5rem;
        border-radius: 999px;
        font-size: 0.85rem;
        font-weight: 600;
        min-width: 2.5rem;
        text-align: center;
    }
    .result-meta {
        color: #6b7280;
        font-size: 0.85rem;
        margin-bottom: 0.35rem;
    }
    .result-preview {
        color: #374151;
        font-size: 0.95rem;
        line-height: 1.45;
    }
    </style>
""", unsafe_allow_html=True)

# ------------------ PAGE HEADER ------------------
st.markdown("<div class='main-title'>📦 FroQ Product Search Tool</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Search adapted package files from the local AI database</div>",
            unsafe_allow_html=True)

# ------------------ SEARCH INPUT ------------------
search_query = st.text_input("🔍 Enter your product name, EAN, or keyword:", "",
                             placeholder="e.g., Avocado stukjes, 8718907496346, Pancakes")

k_value = st.slider("Number of Results (k):", min_value=1, max_value=10, value=5)

search_button = st.button("Search")

"""
API integration expects FastAPI response shape:
{
  "query": str,
  "took_ms": float,
  "results": [{
     "id": str, "score": float, "source": str|null,
     "table_name": str|null, "row_id": str|null,
     "position": int, "preview": str
  }]
}
"""

# ------------------ PROCESS SEARCH ------------------
if search_button and search_query.strip():
    with st.spinner("🔎 Searching, please wait..."):
        try:
            base_url = os.environ.get("API_BASE_URL", "http://localhost:8000")
            response = requests.get(
                f"{base_url}/api/search",
                params={"q": search_query, "k": k_value},
                timeout=60,
            )

            if response.status_code == 200:
                payload = response.json()
                results = payload.get("results", []) if isinstance(payload, dict) else []
                took_ms = payload.get("took_ms") if isinstance(payload, dict) else None

                if not results:
                    st.warning("No results found for your query.")
                else:
                    if took_ms is not None:
                        st.caption(f"{len(results)} result(s) • {took_ms:.1f} ms")
                    else:
                        st.caption(f"{len(results)} result(s)")

                    for item in results:
                        src = item.get("source") or item.get("table_name") or "Result"
                        file_name = os.path.basename(src) if isinstance(src, str) else src
                        score = item.get("score")
                        desc = item.get("preview", "")

                        st.markdown(f"""
                            <div class="result-card">
                                <div class="result-header">
                                    <h4 class="result-title">📄 {file_name}</h4>
                                    {('<span class="score-badge">' + str(round(score, 2)) + '</span>') if score is not None else ''}
                                </div>
                                <div class="result-meta">{src}</div>
                                <div class="result-preview">{desc[:400]}{'...' if len(desc) > 400 else ''}</div>
                            </div>
                        """, unsafe_allow_html=True)

            else:
                st.error(f"❌ Error {response.status_code}: Could not reach the API.")
        except Exception as e:
            st.error(f"⚠️ Failed to connect to API: {e}")

elif search_button:
    st.warning("Please enter a valid search query before pressing Search.")

# ------------------ FOOTER ------------------
st.markdown("---")
st.markdown("<p style='text-align:center; color:#888;'>© 2025 FroQ AI Intern</p>", unsafe_allow_html=True)
 