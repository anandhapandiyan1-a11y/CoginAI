import io
import os
import time
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


# Custom Core & Agent Modules
from agents.business_agent import BusinessAgent
from core.parser import DataParser



st.set_page_config(
    page_title="CoginAI - Autonomous Business Intelligence System",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .main {
        background-color: #0f172a;
        color: #f8fafc;
    }
    .main-title {
        font-size: 2.4rem;
        font-weight: 800;
        background: linear-gradient(90deg, #38bdf8 0%, #818cf8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        color: #94a3b8;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    .insight-box {
        background-color: #f1f5f9;
        color: #0f172a;
        border-left: 5px solid #0284c7;
        padding: 15px 20px;
        border-radius: 6px;
        margin-top: 10px;
        margin-bottom: 20px;
        font-size: 0.95rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    </style>
""",
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False)
def load_and_cache_data(file_bytes, filename):
  buffer = io.BytesIO(file_bytes)
  parser = DataParser(buffer)
  parser.filename = filename
  return parser.load_data()



st.markdown(
    '<div class="main-title">CoginAI: Autonomous Data & BI System</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="sub-title">End-to-End Automated Data Cleaning, Missing Value Audit & Strategic AI Insights</div>',
    unsafe_allow_html=True,
)

with st.sidebar:
  st.image("https://img.icons8.com/fluency/96/brain-insight.png", width=80)
  st.header("⚙️ Control Panel")
  uploaded_file = st.file_uploader(
      "📁 Upload Dataset (CSV, Excel, JSON)",
      type=["csv", "xlsx", "xls", "json"],
  )

  st.divider()
  st.markdown("### 📌 System Status")
  st.success("CoginAI Core Engine: Ready")

  if os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"):
    st.success("LLM Business Agent: Connected (Auth Key)")
  else:
    st.warning("LLM Business Agent: API Key Missing in .env")

  st.markdown("---")
  default_model = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")
  model_input = st.text_input(
      "LLM Model (override)",
      value=default_model,
      help="Set provider model name (e.g. gemini-3.8-flash).",
  )




if uploaded_file is not None:
  st.subheader("📁 1. Dataset Overview & Missing Value Audit")

  try:
    file_bytes = uploaded_file.getvalue()
    df = load_and_cache_data(file_bytes, uploaded_file.name)
  except Exception as e:
    st.error(f"Error loading file: {e}")
    st.stop()

  total_rows = len(df)
  total_cols = len(df.columns)
  missing_df = df.isnull().sum()
  total_missing = missing_df.sum()

  col1, col2, col3, col4 = st.columns(4)
  with col1:
    st.metric(label="Total Rows", value=f"{total_rows:,}")
  with col2:
    st.metric(label="Total Columns", value=total_cols)
  with col3:
    st.metric(
        label="Missing Values",
        value=int(total_missing),
        delta="100% Clean" if total_missing == 0 else "Action Required",
        delta_color="normal" if total_missing == 0 else "inverse",
    )
  with col4:
    st.metric(
        label="Memory Footprint",
        value=f"{df.memory_usage(deep=True).sum() / 1024**2:.2f} MB",
    )

  missing_summary = pd.DataFrame(
      {
          "Column Name": df.columns,
          "Data Type": df.dtypes.astype(str),
          "Missing Count": missing_df.values,
          "Missing Percentage (%)": (
              missing_df.values / total_rows * 100
          ).round(2),
      }
  ).reset_index(drop=True)

  st.dataframe(missing_summary, use_container_width=True)

  if total_missing > 0:
    st.markdown(
        """
        <div class="insight-box">
            <b>🛠️ How to Handle Missing Values (Imputation Strategy):</b><br>
            • <b>Numeric Columns:</b> Fill using median <code>df['col'].fillna(df['col'].median(), inplace=True)</code>.<br>
            • <b>Categorical Columns:</b> Fill using the mode or label as 'Unknown'.<br>
            • <b>Drop Threshold:</b> If >50% data is missing, drop the column.
        </div>
        """,
        unsafe_allow_html=True,
    )
  else:
    st.markdown(
        """
        <div class="insight-box">
            <b>✅ Quality Check Passed:</b> Zero missing values found in this dataset. Ready for visual analytics.
        </div>
        """,
        unsafe_allow_html=True,
    )

  st.divider()

  st.subheader("📈 2. Visual Analytics & Category Comparisons")

  if (
      "transaction_qty" in df.columns
      and "unit_price" in df.columns
      and "total_revenue" not in df.columns
  ):
    df["total_revenue"] = df["transaction_qty"] * df["unit_price"]

  col_chart1, col_chart2 = st.columns(2)

  with col_chart1:
    if "product_category" in df.columns and "total_revenue" in df.columns:
      cat_df = (
          df.groupby("product_category")["total_revenue"]
          .sum()
          .reset_index()
          .sort_values(by="total_revenue", ascending=False)
      )

      fig1 = px.bar(
          cat_df,
          x="product_category",
          y="total_revenue",
          title="<b>Revenue Contribution by Category</b>",
          color="total_revenue",
          color_continuous_scale="Blues",
          labels={
              "product_category": "Product Category",
              "total_revenue": "Revenue ($)",
          },
      )
      fig1.update_layout(
          template="plotly_dark",
          paper_bgcolor="rgba(0,0,0,0)",
          plot_bgcolor="rgba(0,0,0,0)",
          xaxis_tickangle=-45,
          margin=dict(b=80),
      )
      st.plotly_chart(fig1, use_container_width=True)

      top_cat = cat_df.iloc[0]["product_category"]
      st.markdown(
          f"""
            <div class="insight-box">
                <b>📌 Bundle Advisory:</b> Pair <b>'{top_cat}'</b> with slower categories to cross-sell and increase ticket size.
            </div>
            """,
          unsafe_allow_html=True,
      )

  with col_chart2:
    if "product_detail" in df.columns and "transaction_qty" in df.columns:
      top_prod = (
          df.groupby("product_detail")["transaction_qty"]
          .sum()
          .reset_index()
          .sort_values(by="transaction_qty", ascending=False)
          .head(8)
      )

      fig2 = px.bar(
          top_prod,
          x="transaction_qty",
          y="product_detail",
          orientation="h",
          title="<b>Top 8 Products by Volume</b>",
          color="transaction_qty",
          color_continuous_scale="Viridis",
          labels={
              "transaction_qty": "Quantity Sold",
              "product_detail": "Product Name",
          },
      )
      fig2.update_layout(
          template="plotly_dark",
          paper_bgcolor="rgba(0,0,0,0)",
          plot_bgcolor="rgba(0,0,0,0)",
          yaxis=dict(autorange="reversed"),
          margin=dict(l=120),
      )
      st.plotly_chart(fig2, use_container_width=True)

      top_p_name = top_prod.iloc[0]["product_detail"]
      st.markdown(
          f"""
            <div class="insight-box">
                <b>📌 Inventory Tip:</b> <b>'{top_p_name}'</b> drives high volume. Use it as an anchor item in your bundles.
            </div>
            """,
          unsafe_allow_html=True,
      )

  st.divider()

  # Instantiate the agent with the selected model input
  agent = BusinessAgent(model_name=model_input)

  st.subheader("💡 3. CoginAI Executive Strategic Analysis")

  if agent.model_error:
    st.warning(
        "LLM initialization issue detected. Check your API key environment"
        " variable."
    )
    st.code(agent.model_error)
  else:
    generate_btn = st.button("Generate Executive Analysis")

    if generate_btn:
      with st.spinner("🧠 AI Agent is generating executive recommendations..."):
        try:
          agent_summary = agent.generate_strategic_analysis(df)
          st.markdown(
              f"""
                    <div style="background-color: #f8fafc; color: #0f172a; padding: 28px; border-radius: 12px; border: 1px solid #e2e8f0; box-shadow: 0 4px 10px rgba(2,6,23,0.06);">
                        {agent_summary}
                    </div>
                    """,
              unsafe_allow_html=True,
          )
        except Exception as e:
          st.error(f"Unable to invoke AI Agent. Error details: {e}")
    else:
      st.info(
          "Click 'Generate Executive Analysis' to run the strategic AI agent"
          " with the selected model."
      )

  st.divider()

  st.subheader("💬 4. Ask CoginAI Agent")
  st.caption(
      "Ask questions about product bundling, stock risk reduction, or revenue"
      " growth."
  )

  if "messages" not in st.session_state:
    st.session_state.messages = []

  for message in st.session_state.messages:
    with st.chat_message(message["role"]):
      st.markdown(message["content"])

  if user_query := st.chat_input(
      "e.g., Which two products should we bundle together?"
  ):
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
      st.markdown(user_query)

    with st.chat_message("assistant"):
      with st.spinner("CoginAI Agent is thinking..."):
        chat_response = agent.answer_user_query(df, user_query)
        st.markdown(chat_response)
        st.session_state.messages.append(
            {"role": "assistant", "content": chat_response}
        )

else:
  st.info("👈 Please upload a dataset in the sidebar to launch the system.")