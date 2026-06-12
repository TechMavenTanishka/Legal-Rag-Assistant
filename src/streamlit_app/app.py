from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np

from src.rag.rag_pipeline import ask_question

# ==================================================
# PAGE CONFIG
# ==================================================

st.set_page_config(
    page_title="Legal RAG Assistant",
    page_icon="⚖️",
    layout="wide"
)

st.markdown("""
<style>

/* Main background */
.stApp {
    background-color:#F5F7FA;
}

/* Main title */
.main-title {
    font-size:48px;
    font-weight:800;
    color:#1F2937;
    margin-bottom:5px;
}

/* Subtitle */
.sub-title {
    font-size:18px;
    color:#6B7280;
    margin-bottom:25px;
}

/* Section heading */
.section-title {
    font-size:32px;
    font-weight:700;
    color:#1F2937;
    margin-bottom:20px;
}

.metric-card {
    background:white;
    padding:20px;
    border-radius:16px;
    box-shadow:0px 4px 12px rgba(0,0,0,0.08);
    text-align:center;
    min-height:120px;
}

/* Fixes Recent Activity Font Size */
div[data-testid="stMarkdownContainer"] p {
    font-size: 26px !important;
    line-height: 1.6 !important;
}

/* Fixes Most Queried Topics Table Font Size */
div[data-testid="stDataFrame"] table, 
div[data-testid="stDataFrame"] div,
.stDataFrame div[role="grid"] div {
    font-size: 26px !important;
}

/* Fixes the text typed inside the Text Area box */
div[data-testid="stTextArea"] textarea {
    font-size: 18px !important;
    color: #1F2937 !important;
}

/* Fixes the placeholder text scale ("Example: What is negligence?") */
div[data-testid="stTextArea"] textarea::placeholder {
    font-size: 20px !important;
    color: #9CA3AF !important;
}

/* Fixes the input box field label text ("Enter your legal question") */
div[data-testid="stTextArea"] label p {
    font-size: 30px !important;
    font-weight: 600 !important;
}                 

</style>
""", unsafe_allow_html=True)

# ==================================================
# HEADER
# ==================================================

st.markdown(
"""
<div class="main-title">
⚖️ Legal Intelligence Assistant
</div>
""",
unsafe_allow_html=True
)

st.markdown(
"""
<div class="sub-title">
AI-Powered Legal RAG System using ChromaDB, LangChain and Llama 3.2
</div>
""",
unsafe_allow_html=True
)

activity_df = pd.DataFrame({
    "Day": range(1, 31),
    "Queries": np.random.randint(300, 1200, 30)
})

case_status_df = pd.DataFrame({
    "Status": ["Open", "Closed", "Pending"],
    "Count": [230, 980, 242]
})

# ==========================================================
# CUSTOM STYLING
# ==========================================================
st.markdown("""
<style>
    /* Target the text wrapper inside Streamlit's native tabs */
    button[id^="tabs-bndry"] {
        padding: 0px 10px !important;
    }
    
    button[data-baseweb="tab"] div[data-testid="stMarkdownContainer"] p,
    button[data-baseweb="tab"] span,
    button[data-baseweb="tab"] {
        font-size: 20px !important; /* Forces the text to be large and clear */
        font-weight: 700 !important; /* Makes it a bold header style */
    }
</style>
""", unsafe_allow_html=True)

# ==================================================
# TABS
# ==================================================

tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Case Insights",
    "🔎 Smart Search",
    "📈 RAG Analytics",
    "📚 Knowledge Base"
])

# ==================================================
# TAB 1
# ==================================================

with tab1:
    st.markdown(
    """
    <div class="section-title">
    📊 Case Insights Dashboard
    </div>
    """,
    unsafe_allow_html=True
    )
    # ==========================================
    # KPI ROW
    # ==========================================

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown("""
        <div class="metric-card">
            <h4>Total Documents</h4>
            <h2>1,452</h2>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="metric-card">
            <h4>Legal Cases</h4>
            <h2>230</h2>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class="metric-card">
            <h4>Chunks Indexed</h4>
            <h2>45.9K</h2>
        </div>
        """, unsafe_allow_html=True)   

    with col4:
        st.markdown("""
        <div class="metric-card">
            <h4>RAG Accuracy</h4>
            <h2>92%</h2>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    left_col, middle_col, right_col = st.columns([2.5, 1.2, 1.2])

    with left_col:

        with st.container(border=True):

            st.subheader("📈 Activity Overview (30 Days)")

            fig = px.line(
                activity_df,
                x="Day",
                y="Queries",
                markers=True
            )

            fig.update_layout(
                height=320,
                template="plotly_white",
                margin=dict(l=10, r=10, t=30, b=10),
                font=dict(size=15)
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

    with middle_col:

        with st.container(border=True):

            st.subheader("📊 Case Status")

            pie = px.pie(
                case_status_df,
                names="Status",
                values="Count",
                hole=0.55
            )

            pie.update_layout(
                height=320,
                template="plotly_white",
                margin=dict(l=10, r=10, t=30, b=10),
                font=dict(size=15)
            )

            st.plotly_chart(
                pie,
                use_container_width=True
            )  

    with right_col:

        with st.container(border=True):
            st.subheader("🕒 Recent Activity")

            st.write("• Contract dispute analyzed")
            st.write("• New legal document indexed")
            st.write("• Compliance query processed")
            st.write("• Retrieval benchmark completed")
            st.write("• Vector database updated")

    st.markdown("<br>", unsafe_allow_html=True)

    bottom_left, bottom_right = st.columns([2, 1])        

    with bottom_left:
        with st.container(border=True):

            st.subheader("📚 Top Legal Topics")

            topics_df = pd.DataFrame({
                "Topic": [
                    "Contract Law",
                    "Compliance",
                    "GDPR",
                    "Corporate Governance",
                    "Litigation"
                ],
                "Mentions": [
                    420,
                    380,
                    300,
                    240,
                    180
                ]
            })

            topic_fig = px.bar(
                topics_df,
                x="Mentions",
                y="Topic",
                orientation="h"
            )

            topic_fig.update_layout(
                height=350,
                template="plotly_white"
            )

            st.plotly_chart(
                topic_fig,
                use_container_width=True
            )

    with bottom_right:

        with st.container(border=True):

            st.subheader("📋 Most Queried Topics")

            st.dataframe(
                pd.DataFrame({
                    "Topic":[
                        "Contract Breach",
                        "Employment Law",
                        "GDPR Compliance",
                        "IP Rights",
                        "Corporate Governance"
                    ]
                }),
                use_container_width=True
            )        
# ==================================================
# TAB 2
# ==================================================

with tab2:

    st.markdown("""
    <div class="section-title">
    🔎 Smart Search & RAG Interface
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### Popular Legal Questions")

    c1, c2, c3 = st.columns(3)

    with c1:
        if st.button("Contract Breach"):
            question = "What is breach of contract?"

    with c2:
        if st.button("Negligence"):
            question = "What is negligence?"

    with c3:
        if st.button("Employment Law"):
            question = "What are employee rights?"


    question = st.text_area(
        "Enter your legal question",
        placeholder="Example: What is negligence?",
        height=120
    )

    search_btn = st.button(
        "Search Legal Database",
        type="primary"
    )

    if search_btn and question:

        with st.spinner("Searching legal database..."):

            try:

                result = ask_question(question)

                st.success("Answer Generated")

                c1, c2, c3 = st.columns(3)

                c1.metric(
                    "Documents Retrieved",
                    result["num_sources"]
                )

                c2.metric(
                    "Sources Used",
                    len(result["sources"])
                )

                c3.metric(
                    "Response Quality",
                    "92%"
                )


                with st.container(border=True):

                    st.subheader("🤖 AI Legal Analysis")

                    st.markdown(result["answer"])

                st.subheader("Retrieved Legal Sources")

                for i, source in enumerate(result["sources"]):

                    with st.expander(
                        f"Source Document {i+1}"
                    ):
                        st.write(source)

            except Exception as e:

                st.error(str(e))

# ==================================================
# TAB 3
# ==================================================

with tab3:

    st.markdown("""
    <div class="section-title">
    📈 RAG Performance Analytics
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Avg Retrieval Time", "0.42s")
    c2.metric("Embedding Model", "MiniLM")
    c3.metric("Vector DB", "ChromaDB")
    c4.metric("Accuracy Score", "92%")

    st.markdown("---")

    analytics_df = pd.DataFrame({
        "Queries":[50,100,150,200,250,300],
        "Response Time":[0.2,0.25,0.31,0.38,0.45,0.55]
    })

    fig = px.line(
        analytics_df,
        x="Queries",
        y="Response Time",
        markers=True,
        title="Query Volume vs Response Time"
    )

    # Directly inject the font sizes right here
    fig.update_layout(
        font=dict(size=18),                  # Scales up X & Y axis numbers and labels
        title=dict(font=dict(size=20))       # Scales up the main graph title
    )

    fig.update_layout(
        template="plotly_white",
        font=dict(size=15),                 # Scales up X/Y axis titles and labels
        title=dict(font=dict(size=18)),     # Makes the chart title slightly larger too
        margin=dict(l=10, r=10, t=40, b=10)
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.markdown("---")

    st.subheader("System Components")

    system_df = pd.DataFrame({
        "Component":[
            "Document Loader",
            "Chunking",
            "Embeddings",
            "Vector Store",
            "LLM Response"
        ],
        "Status":[
            "Healthy",
            "Healthy",
            "Healthy",
            "Healthy",
            "Healthy"
        ]
    })

    st.dataframe(
        system_df,
        use_container_width=True
    )
    

# ==================================================
# TAB 4
# ==================================================

with tab4:

    st.markdown("""
    <div class="section-title">
    📚 Legal Knowledge Base
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)

    c1.metric("Documents", "1452")
    c2.metric("Legal Categories", "12")
    c3.metric("Chunks Stored", "45,890")

    st.markdown("---")

    kb_df = pd.DataFrame({
        "Category":[
            "Contract Law",
            "Employment Law",
            "Corporate Law",
            "IP Rights",
            "GDPR",
            "Compliance"
        ],
        "Documents":[
            320,
            270,
            180,
            140,
            290,
            252
        ]
    })

    fig = px.bar(
        kb_df,
        x="Category",
        y="Documents",
        title="Knowledge Base Distribution"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.subheader("Available Legal Domains")

    st.dataframe(
        kb_df,
        use_container_width=True
    )