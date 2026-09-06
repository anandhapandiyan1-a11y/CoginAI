import os
from dotenv import load_dotenv
import pandas as pd
from google import genai

load_dotenv()


class BusinessAnalyzer:
    def __init__(self, df: pd.DataFrame):
        self.df = df

    def get_kpi_summary(self) -> dict:
        """Calculate high-level dataset metrics and KPIs."""
        total_rows = len(self.df)
        total_cols = len(self.df.columns)
        missing_count = int(self.df.isnull().sum().sum())
        
        # Calculate revenue if numeric columns match
        total_revenue = 0.0
        if 'transaction_qty' in self.df.columns and 'unit_price' in self.df.columns:
            total_revenue = float((self.df['transaction_qty'] * self.df['unit_price']).sum())
        elif 'revenue' in self.df.columns:
            total_revenue = float(self.df['revenue'].sum())

        return {
            "total_rows": total_rows,
            "total_cols": total_cols,
            "missing_values": missing_count,
            "total_revenue": round(total_revenue, 2)
        }

    def get_top_categories(self, top_n: int = 5) -> pd.DataFrame:
        """Extract top product categories based on sales or volume."""
        if 'product_category' in self.df.columns and 'transaction_qty' in self.df.columns:
            return (
                self.df.groupby('product_category')['transaction_qty']
                .sum()
                .reset_index()
                .sort_values(by='transaction_qty', ascending=False)
                .head(top_n)
            )
        return pd.DataFrame()



def analyze_with_gemini(df: pd.DataFrame, api_key: str = None) -> str:
    """Pass dataset summary to Gemini LLM for strategic business recommendations."""
    key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")

    if not key:
        return (
            "⚠️ **API Key Missing**: Please set your `GEMINI_API_KEY` or `GOOGLE_API_KEY` "
            "in environment variables or `.env` file to generate strategic business recommendations."
        )

    try:
        client = genai.Client(api_key=key)

        data_sample = df.head(5).to_dict(orient='records')
        cols = list(df.columns)
        num_rows = len(df)

        prompt = f"""
        You are CoginAI, an Autonomous Senior Data Scientist and Business Strategist.
        Analyze the following dataset overview and provide high-value, actionable strategic recommendations.

        ### Dataset Metadata:
        - Total Rows: {num_rows}
        - Columns Available: {cols}
        - Sample Records: {data_sample}

        ### Instructions:
        Provide a crisp executive summary in clean Markdown format with the following exact sections:
        1. **Executive Summary & Key Trends**: High-level observations on product/sales velocity.
        2. **Business Risks & Operational Anomalies**: Inventory, pricing, or supply chain bottlenecks.
        3. **Top 3 Actionable Strategic Recommendations**: Revenue growth and margin optimization tactics.
        """

        response = client.models.generate_content(
            model="gemini-1.5-flash",
            contents=prompt,
        )
        return response.text

    except Exception as e:
        return f"❌ **Error generating AI analysis**: {str(e)}"