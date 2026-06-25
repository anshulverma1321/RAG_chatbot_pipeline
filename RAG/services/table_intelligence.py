import logging
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI

logger = logging.getLogger(__name__)

# ===========================================================================
# 1. Pydantic Schemas for Table Categories
# ===========================================================================

class SimpleTableKnowledge(BaseModel):
    title: Optional[str] = Field(description="The title of the table, if present or inferred.")
    headers: List[str] = Field(description="List of column headers.")
    rows: List[List[str]] = Field(description="List of data rows, where each row is a list of strings matching column headers.")
    summary: str = Field(description="A concise summary of what this simple table represents.")

class ComparisonItem(BaseModel):
    entity: str = Field(description="The entity being compared (e.g. 'YOLOv8', 'Product X').")
    features: Dict[str, str] = Field(description="Key-value comparison points (attribute_name -> value).")

class ComparisonTableKnowledge(BaseModel):
    title: Optional[str] = Field(description="Title of the comparison table.")
    entities_compared: List[str] = Field(description="List of entities compared.")
    attributes_compared: List[str] = Field(description="List of features or attributes compared.")
    comparisons: List[ComparisonItem] = Field(description="Detailed comparison breakdown for each entity.")
    key_differences: List[str] = Field(description="List of key differences or highlights noted from the comparison.")
    summary: str = Field(description="A brief summary of the comparison matrix.")

class FinancialMetric(BaseModel):
    category: str = Field(description="The financial category or line item (e.g. 'Revenue', 'Operating Income').")
    values: Dict[str, str] = Field(description="Metric values indexed by reporting period (e.g. {'2022': '150M', '2023': '180M'}).")

class FinancialTableKnowledge(BaseModel):
    title: Optional[str] = Field(description="Title of the financial sheet.")
    reporting_period: Optional[str] = Field(description="Applicable periods (e.g. 'FY 2024', 'Q1-Q3').")
    currency: Optional[str] = Field(description="Reporting currency or scale (e.g. 'USD', 'Millions of EUR').")
    financial_metrics: List[FinancialMetric] = Field(description="List of financial line items and values.")
    key_financial_takeaways: List[str] = Field(description="Major trends, spikes, margins, or takeaways.")
    summary: str = Field(description="A summary explanation of the financial table.")

class StatMetricValue(BaseModel):
    variable_or_group: str = Field(description="The independent variable, experimental group, or subset analyzed.")
    metric_values: Dict[str, str] = Field(description="Key-value metric scores (e.g. {'Precision': '0.94', 'Recall': '0.91'}).")

class StatisticalTableKnowledge(BaseModel):
    title: Optional[str] = Field(description="Title of the statistical table.")
    variables: List[str] = Field(description="List of variables or configurations evaluated.")
    metrics: List[str] = Field(description="List of metrics reported.")
    data_summary: List[StatMetricValue] = Field(description="Structured values for each variable/group.")
    statistical_conclusions: List[str] = Field(description="Core conclusions, significance results, or takeaways.")
    summary: str = Field(description="Summary of the statistical findings.")

class TimeSeriesData(BaseModel):
    metric_name: str = Field(description="Name of the parameter tracked over time.")
    values_over_time: List[str] = Field(description="Values matching the order of the timestamps list.")

class TimeSeriesTableKnowledge(BaseModel):
    title: Optional[str] = Field(description="Title of the timeseries table.")
    time_interval: str = Field(description="The time interval, e.g. 'monthly', 'quarterly', 'yearly'.")
    timestamps: List[str] = Field(description="Ordered list of historical points/timestamps.")
    series_data: List[TimeSeriesData] = Field(description="Structured parameters and values corresponding to timestamps.")
    trends_observed: List[str] = Field(description="List of growth, decline, or seasonality trends observed.")
    summary: str = Field(description="Summary of chronological trend progression.")

class UnknownTableKnowledge(BaseModel):
    raw_content_description: str = Field(description="Detailed text description of the table layout and contents.")
    key_columns_detected: List[str] = Field(description="List of major header names or keys identified.")
    summary: str = Field(description="A basic summary of the table contents.")


# ===========================================================================
# 2. Specialized Extractor Services
# ===========================================================================

def extract_simple_table(table_markdown: str) -> dict:
    """Extracts simple structured table data from Markdown."""
    logger.info("Running extract_simple_table...")
    try:
        chat = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite")
        structured_chat = chat.with_structured_output(SimpleTableKnowledge)
        prompt = (
            "Analyze the following simple table. Extract the title, headers, rows of data, "
            "and provide a clear summary of what this table represents.\n\n"
            f"Table Markdown:\n{table_markdown}"
        )
        res: SimpleTableKnowledge = structured_chat.invoke(prompt)

        rich_text = (
            f"# Simple Table Data Extraction\n\n"
            f"- **Title**: {res.title or 'N/A'}\n"
            f"- **Summary**: {res.summary}\n\n"
            f"## Structure:\n"
            f"- **Headers**: {', '.join(res.headers)}\n"
            f"- **Row Count**: {len(res.rows)} rows extracted\n"
        )

        return {
            "image_type": "TABLE_SIMPLE",
            "title": res.title,
            "headers": res.headers,
            "rows": res.rows,
            "summary": res.summary,
            "rich_text_representation": rich_text
        }
    except Exception as e:
        logger.error("extract_simple_table failed: %s", e, exc_info=True)
        raise

def extract_comparison_table(table_markdown: str) -> dict:
    """Extracts comparison structured table data from Markdown."""
    logger.info("Running extract_comparison_table...")
    try:
        chat = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite")
        structured_chat = chat.with_structured_output(ComparisonTableKnowledge)
        prompt = (
            "Analyze the following comparison table. Identify the compared entities, features/attributes compared, "
            "parse features for each entity, list key differences, and summarize the comparison matrix.\n\n"
            f"Table Markdown:\n{table_markdown}"
        )
        res: ComparisonTableKnowledge = structured_chat.invoke(prompt)

        comparisons_formatted = []
        for item in res.comparisons:
            comparisons_formatted.append({
                "entity": item.entity,
                "features": item.features
            })

        rich_text = (
            f"# Comparison Table Analysis\n\n"
            f"- **Title**: {res.title or 'N/A'}\n"
            f"- **Summary**: {res.summary}\n\n"
            f"## Entities Compared:\n"
        )
        for ent in res.entities_compared:
            rich_text += f"- {ent}\n"

        rich_text += "\n## Key Differences & Highlights:\n"
        for diff in res.key_differences:
            rich_text += f"- {diff}\n"

        return {
            "image_type": "TABLE_COMPARISON",
            "title": res.title,
            "entities_compared": res.entities_compared,
            "attributes_compared": res.attributes_compared,
            "comparisons": comparisons_formatted,
            "key_differences": res.key_differences,
            "summary": res.summary,
            "rich_text_representation": rich_text
        }
    except Exception as e:
        logger.error("extract_comparison_table failed: %s", e, exc_info=True)
        raise

def extract_financial_table(table_markdown: str) -> dict:
    """Extracts financial structured table data from Markdown."""
    logger.info("Running extract_financial_table...")
    try:
        chat = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite")
        structured_chat = chat.with_structured_output(FinancialTableKnowledge)
        prompt = (
            "Analyze the following financial table. Extract the title, reporting periods, currency, "
            "financial metrics, key financial takeaways, and a summary of the financial performance.\n\n"
            f"Table Markdown:\n{table_markdown}"
        )
        res: FinancialTableKnowledge = structured_chat.invoke(prompt)

        metrics_formatted = []
        for m in res.financial_metrics:
            metrics_formatted.append({
                "category": m.category,
                "values": m.values
            })

        rich_text = (
            f"# Financial Data Analysis\n\n"
            f"- **Title**: {res.title or 'N/A'}\n"
            f"- **Reporting Period**: {res.reporting_period or 'N/A'}\n"
            f"- **Currency/Scale**: {res.currency or 'N/A'}\n"
            f"- **Summary**: {res.summary}\n\n"
            f"## Key Takeaways:\n"
        )
        for takeaway in res.key_financial_takeaways:
            rich_text += f"- {takeaway}\n"

        return {
            "image_type": "TABLE_FINANCIAL",
            "title": res.title,
            "reporting_period": res.reporting_period,
            "currency": res.currency,
            "financial_metrics": metrics_formatted,
            "key_financial_takeaways": res.key_financial_takeaways,
            "summary": res.summary,
            "rich_text_representation": rich_text
        }
    except Exception as e:
        logger.error("extract_financial_table failed: %s", e, exc_info=True)
        raise

def extract_statistical_table(table_markdown: str) -> dict:
    """Extracts statistical structured table data from Markdown."""
    logger.info("Running extract_statistical_table...")
    try:
        chat = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite")
        structured_chat = chat.with_structured_output(StatisticalTableKnowledge)
        prompt = (
            "Analyze the following statistical table. Identify variables, metrics, statistical values for each group, "
            "conclusions, and a summary of the findings.\n\n"
            f"Table Markdown:\n{table_markdown}"
        )
        res: StatisticalTableKnowledge = structured_chat.invoke(prompt)

        summary_formatted = []
        for s in res.data_summary:
            summary_formatted.append({
                "variable_or_group": s.variable_or_group,
                "metric_values": s.metric_values
            })

        rich_text = (
            f"# Statistical Findings Report\n\n"
            f"- **Title**: {res.title or 'N/A'}\n"
            f"- **Summary**: {res.summary}\n\n"
            f"## Core Statistical Conclusions:\n"
        )
        for conc in res.statistical_conclusions:
            rich_text += f"- {conc}\n"

        return {
            "image_type": "TABLE_STATISTICAL",
            "title": res.title,
            "variables": res.variables,
            "metrics": res.metrics,
            "data_summary": summary_formatted,
            "statistical_conclusions": res.statistical_conclusions,
            "summary": res.summary,
            "rich_text_representation": rich_text
        }
    except Exception as e:
        logger.error("extract_statistical_table failed: %s", e, exc_info=True)
        raise

def extract_timeseries_table(table_markdown: str) -> dict:
    """Extracts timeseries structured table data from Markdown."""
    logger.info("Running extract_timeseries_table...")
    try:
        chat = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite")
        structured_chat = chat.with_structured_output(TimeSeriesTableKnowledge)
        prompt = (
            "Analyze the following timeseries table. Identify the time interval, ordered timestamps, "
            "series metrics over time, trends observed, and provide a summary of the progression.\n\n"
            f"Table Markdown:\n{table_markdown}"
        )
        res: TimeSeriesTableKnowledge = structured_chat.invoke(prompt)

        series_formatted = []
        for s in res.series_data:
            series_formatted.append({
                "metric_name": s.metric_name,
                "values_over_time": s.values_over_time
            })

        rich_text = (
            f"# Chronological Trend Analysis\n\n"
            f"- **Title**: {res.title or 'N/A'}\n"
            f"- **Time Interval**: {res.time_interval}\n"
            f"- **Summary**: {res.summary}\n\n"
            f"## Observed Trends:\n"
        )
        for trend in res.trends_observed:
            rich_text += f"- {trend}\n"

        return {
            "image_type": "TABLE_TIMESERIES",
            "title": res.title,
            "time_interval": res.time_interval,
            "timestamps": res.timestamps,
            "series_data": series_formatted,
            "trends_observed": res.trends_observed,
            "summary": res.summary,
            "rich_text_representation": rich_text
        }
    except Exception as e:
        logger.error("extract_timeseries_table failed: %s", e, exc_info=True)
        raise

def extract_unknown_table(table_markdown: str) -> dict:
    """Extracts raw details as a fallback for unclassified tables."""
    logger.info("Running extract_unknown_table...")
    try:
        chat = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite")
        structured_chat = chat.with_structured_output(UnknownTableKnowledge)
        prompt = (
            "Analyze this irregular or complex table. Describe its contents, list key columns detected, "
            "and provide a basic summary of the table.\n\n"
            f"Table Markdown:\n{table_markdown}"
        )
        res: UnknownTableKnowledge = structured_chat.invoke(prompt)

        rich_text = (
            f"# Unclassified Table Data Description\n\n"
            f"- **Summary**: {res.summary}\n"
            f"- **Detected Columns**: {', '.join(res.key_columns_detected)}\n\n"
            f"## Raw Description:\n"
            f"{res.raw_content_description}\n"
        )

        return {
            "image_type": "TABLE_UNKNOWN",
            "raw_content_description": res.raw_content_description,
            "key_columns_detected": res.key_columns_detected,
            "summary": res.summary,
            "rich_text_representation": rich_text
        }
    except Exception as e:
        logger.error("extract_unknown_table failed: %s", e, exc_info=True)
        raise
