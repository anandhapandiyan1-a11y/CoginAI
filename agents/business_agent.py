import os
from dotenv import load_dotenv
import ollama
from google import genai

load_dotenv()


class BusinessAgent:

  def __init__(self, model_name="gemini-3.8-flash", local_model="llama3"):
    self.model_name = model_name
    self.local_model = local_model
    self.client = None
    self.model_error = None

    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if api_key:
      try:
        self.client = genai.Client(api_key=api_key)
      except Exception as e:
        self.model_error = str(e)
    else:
      self.model_error = "API Key Missing"

  def _build_dataset_context(self, df):
    num_rows = len(df)
    num_cols = len(df.columns)
    cols = list(df.columns)
    sample_data = df.head(5).to_dict(orient="records")

    aggregates = {}
    if "total_revenue" in df.columns:
      aggregates["Total Revenue"] = round(df["total_revenue"].sum(), 2)
    if "transaction_qty" in df.columns:
      aggregates["Total Volume Sold"] = int(df["transaction_qty"].sum())

    context = f"""
        - Dataset Rows: {num_rows:,}
        - Dataset Columns ({num_cols}): {cols}
        - Key Metrics/Aggregates: {aggregates}
        - Sample Data Snapshot: {sample_data}
        """
    return context

  def generate_strategic_analysis(self, df):
    context = self._build_dataset_context(df)
    prompt = f"""
        You are CoginAI, an Elite Enterprise Business Intelligence & Senior Strategy Agent.
        Analyze the following business dataset context and provide an executive-level strategic report.

        ### Dataset Context:
        {context}

        ### Instructions:
        Format your response cleanly in Markdown using structured sections:
        1. 🚀 Executive Growth & Performance Trends
        2. ⚠️ Operational Bottlenecks & Strategic Risks
        3. 🎯 Top 3 Actionable Strategic Recommendations
        Keep the tone executive, crisp, and direct.
        """

    if self.client and not self.model_error:
      try:
        response = self.client.models.generate_content(
            model=self.model_name, contents=prompt
        )
        return response.text
      except Exception as e:
        error_str = str(e)
        if "429" in error_str or "RESOURCE_EXHAUSTED" in error_str:
          pass
        else:
          return f"⚠️ **Gemini Error**: {error_str}"

    try:
      response = ollama.chat(
          model=self.local_model,
          messages=[{"role": "user", "content": prompt}],
      )
      return (
          "⚠️ *Cloud API Quota Exceeded. Automatically switched to Local Ollama"
          f" ({self.local_model}):*\n\n" + response["message"]["content"]
      )
    except Exception as local_err:
      return self._get_fallback_analysis(
          note=f"⚠️ Both Cloud and Local LLM failed. Error: {str(local_err)}"
      )

  def answer_user_query(self, df, user_query):
    context = self._build_dataset_context(df)
    prompt = f"""
        You are CoginAI, an AI Business Intelligence Analyst.
        Answer the user's specific follow-up question based on the provided dataset context.

        ### Dataset Context:
        {context}

        ### User Question:
        "{user_query}"
        """

    if self.client and not self.model_error:
      try:
        response = self.client.models.generate_content(
            model=self.model_name, contents=prompt
        )
        return response.text
      except Exception:
        pass  

    try:
      response = ollama.chat(
          model=self.local_model,
          messages=[{"role": "user", "content": prompt}],
      )
      return (
          "🤖 *(Local Ollama Response)*\n\n" + response["message"]["content"]
      )
    except Exception as e:
      return f"❌ Error responding via Local LLM: {str(e)}"

  def _get_fallback_analysis(self, note=""):
    prefix = f"{note}\n\n" if note else ""
    return (
        prefix
        + """# Executive Business Intelligence & Strategy Report
### 1. 🚀 Executive Growth & Performance Trends
* High-velocity micro-transactions anchor top-line revenue during morning transit windows.
### 2. ⚠️ Operational Bottlenecks & Strategic Risks
* Sub-optimal basket penetration and checkout counter friction during peak rush.
### 3. 🎯 Top 3 Actionable Strategic Recommendations
1. Morning Express Bundling to lift AOV.
2. Velocity-Based Dynamic Par Levels for inventory.
3. Digital Subscriptions & Daypart Diversification.
"""
    )