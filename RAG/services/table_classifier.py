import logging
from enum import Enum
from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Table Type Labels
# ---------------------------------------------------------------------------

class TableType(str, Enum):
    """Canonical label set for classifying tabular document elements."""
    TABLE_SIMPLE      = "TABLE_SIMPLE"       # Standard grid of data with basic rows/columns
    TABLE_COMPARISON  = "TABLE_COMPARISON"   # Comparing features, products, options, or models
    TABLE_FINANCIAL   = "TABLE_FINANCIAL"    # Balance sheets, income statements, revenue, pricing tables
    TABLE_STATISTICAL = "TABLE_STATISTICAL"  # Experimental results, metrics, distributions, confusion matrices
    TABLE_TIMESERIES  = "TABLE_TIMESERIES"   # Historical progression over standard intervals (days, years, etc.)
    TABLE_UNKNOWN     = "TABLE_UNKNOWN"      # Ambiguous, unstructured, or low-confidence layouts


# ---------------------------------------------------------------------------
# Pydantic Schema for Structured Gemini Output
# ---------------------------------------------------------------------------

class TableClassification(BaseModel):
    """Structured classification result returned by the Gemini model."""
    table_type: TableType = Field(
        description=(
            "The primary category of the table. Must be one of: "
            "TABLE_SIMPLE, TABLE_COMPARISON, TABLE_FINANCIAL, TABLE_STATISTICAL, TABLE_TIMESERIES, TABLE_UNKNOWN. "
            "Choose TABLE_SIMPLE for basic data grids with a flat schema. "
            "Choose TABLE_COMPARISON for matrices comparing multiple entities (e.g. models, systems) across specific attributes. "
            "Choose TABLE_FINANCIAL for tables representing monetary balances, costs, revenues, budgets, or pricing. "
            "Choose TABLE_STATISTICAL for confusion matrices, model training scores (precision, recall), and experimental test statistics. "
            "Choose TABLE_TIMESERIES for lists of metrics or values indexed chronologically. "
            "Choose TABLE_UNKNOWN only when the content is extremely cluttered or doesn't fit standard schemas."
        )
    )
    confidence: float = Field(
        description="A confidence score between 0.0 and 1.0 representing classification certainty.",
        ge=0.0,
        le=1.0
    )
    reason: str = Field(
        description="A concise explanation citing structural or textual cues observed in the table to justify the selection."
    )


# ---------------------------------------------------------------------------
# Public Classifier Function
# ---------------------------------------------------------------------------

def classify_table(table_markdown: str) -> dict:
    """
    Classifies a table represented in Markdown format using Gemini 3.1 Flash Lite.

    Args:
        table_markdown: The markdown string representation of the table.

    Returns:
        A dict containing keys:
            - "table_type" (str): One of the TableType values.
            - "confidence" (float): Score between 0.0 and 1.0.
            - "reason" (str): Narrative reasoning.
    """
    logger.info(
        "classify_table called | content_length=%d characters",
        len(table_markdown) if table_markdown else 0
    )

    if not table_markdown or not table_markdown.strip():
        return {
            "table_type": TableType.TABLE_UNKNOWN.value,
            "confidence": 1.0,
            "reason": "Table content is empty."
        }

    try:
        chat = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite")
        structured_chat = chat.with_structured_output(TableClassification)

        prompt = (
            "You are a professional data structure analyst. Your task is to classify the provided "
            "Markdown table into exactly one of the following categories:\n\n"
            "  • TABLE_SIMPLE      – Flat row-and-column data with no specialized analytical structure.\n"
            "  • TABLE_COMPARISON  – Compares items (features, products, systems, models) across several parameters.\n"
            "  • TABLE_FINANCIAL   – Contains balance sheets, incomes, cost metrics, pricing schedules, or currency calculations.\n"
            "  • TABLE_STATISTICAL – Lists experimental outcomes, standard deviations, precision, recall, confusion matrices, or score outputs.\n"
            "  • TABLE_TIMESERIES  – Lists parameters tracked over dates, months, years, or timeline intervals.\n"
            "  • TABLE_UNKNOWN     – Extremely irregular, sparse, or unidentifiable table elements.\n\n"
            f"Markdown Table:\n{table_markdown}\n"
        )

        result: TableClassification = structured_chat.invoke(prompt)
        logger.info(
            "Gemini table classification complete | type=%s | confidence=%.2f",
            result.table_type.value,
            result.confidence
        )
        return {
            "table_type": result.table_type.value,
            "confidence": round(result.confidence, 4),
            "reason": result.reason.strip()
        }
    except Exception as e:
        logger.error("classify_table failed: %s", e, exc_info=True)
        return {
            "table_type": TableType.TABLE_UNKNOWN.value,
            "confidence": 0.0,
            "reason": f"Classification exception: {str(e)}"
        }
