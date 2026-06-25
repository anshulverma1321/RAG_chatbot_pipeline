import base64
import logging
from typing import List
from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage

logger = logging.getLogger(__name__)

class ChartDataPoint(BaseModel):
    label: str = Field(description="The category label or time point (e.g. '2021', 'Q3', 'Product A').")
    value: str = Field(description="The numeric data value corresponding to the label.")

class ChartKnowledge(BaseModel):
    chart_type: str = Field(description="The type of chart or graph, e.g., 'bar_chart', 'line_graph', 'pie_chart', 'area_chart', 'histogram', or 'other'.")
    title: str = Field(description="The title of the chart or graph as indicated in the image.")
    x_axis: str = Field(description="The label or description of the horizontal x-axis / categories.")
    y_axis: str = Field(description="The label or description of the vertical y-axis / values.")
    data_points: List[ChartDataPoint] = Field(description="List of all extracted categories/coordinates and their values from the chart/graph.")
    trends: List[str] = Field(description="List of key trends observed in the chart data.")
    insights: List[str] = Field(description="List of core insights, peaks, dips, or overall takeaways.")


def extract_chart_knowledge(image_bytes: bytes, mime_type: str) -> dict:
    """Uses Gemini Vision to extract precise data coordinates and insights from charts and line graphs."""
    try:
        chat = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite")
        structured_chat = chat.with_structured_output(ChartKnowledge)
        
        image_base64 = base64.b64encode(image_bytes).decode("utf-8")
        image_url = f"data:{mime_type};base64,{image_base64}"
        
        prompt = (
            "Analyze the provided chart or line graph image and extract all retrievable statistical knowledge.\n"
            "Your output must contain:\n"
            "1. The type of chart (e.g. 'bar_chart', 'line_graph', etc.).\n"
            "2. The title of the chart or graph.\n"
            "3. Horizontal/X-axis title or category classification.\n"
            "4. Vertical/Y-axis title or value classification.\n"
            "5. A complete and exact list of data points with their labels and values as shown. Estimate values as accurately as possible from the axis lines.\n"
            "6. Key trends observed in the chart data.\n"
            "7. Core insights, observations, peaks, dips, or overall takeaways."
        )
        
        message = HumanMessage(
            content=[
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": image_url}}
            ]
        )
        
        res = structured_chat.invoke([message])
        
        # Format the rich text representation for vector indexing
        points_str = "\n".join([f"- **{dp.label}**: {dp.value}" for dp in res.data_points])
        trends_str = "\n".join([f"{i+1}. {trend}" for i, trend in enumerate(res.trends)])
        insights_str = "\n".join([f"{i+1}. {insight}" for i, insight in enumerate(res.insights)])
        
        rich_text = (
            f"# Chart Knowledge Extraction\n\n"
            f"- **Title**: {res.title}\n"
            f"- **Chart Type**: {res.chart_type}\n"
            f"- **X-Axis (Categories)**: {res.x_axis}\n"
            f"- **Y-Axis (Values)**: {res.y_axis}\n\n"
            f"## Extracted Data Points:\n"
            f"{points_str if points_str else '[No data points found]'}\n\n"
        )
        if res.trends:
            rich_text += f"## Key Trends:\n{trends_str}\n\n"
        if res.insights:
            rich_text += f"## Core Insights:\n{insights_str}\n"
        
        return {
            "image_type": "chart",
            "chart_type": res.chart_type,
            "title": res.title,
            "x_axis": res.x_axis,
            "y_axis": res.y_axis,
            "data_points": [dp.model_dump() for dp in res.data_points],
            "trends": res.trends,
            "insights": res.insights,
            "rich_text_representation": rich_text
        }
    except Exception as e:
        logger.error(f"Error in extract_chart_knowledge: {e}")
        raise RuntimeError(f"Chart knowledge extraction failed: {str(e)}")
