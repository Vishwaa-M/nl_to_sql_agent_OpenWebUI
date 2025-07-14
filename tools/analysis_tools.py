"""
This module provides high-level tools for data analysis and summarization.
It also includes a robust, programmatic tool for creating Plotly visualizations
directly from a structured plan.

REFACTORING V8:
- Added support for scatter, heatmap, box, and histogram chart types in `create_plotly_figure` to align with the updated VISUALIZATION_PLANNING_PROMPT_TEMPLATE.
- Updated `ChartComponent` Pydantic model to include `z_axis` for heatmap charts.
- Maintained robust handling of Decimal types and data validation.
"""

import logging
import json
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from decimal import Decimal
from typing import Dict, Any, List, Optional

from langchain_core.tools import tool
from pydantic import BaseModel, Field, ValidationError
from pandas.api.types import is_numeric_dtype

from core.llm import get_llm_client
from prompts.system_prompts import (
    SUMMARIZATION_PROMPT_TEMPLATE,
    VISUALIZATION_PLANNING_PROMPT_TEMPLATE,
)

# Configure a dedicated logger for these tools
logger = logging.getLogger(__name__)

# --- Pydantic Models for Robust Validation ---
class ChartComponent(BaseModel):
    """Defines the structure for a single chart to be generated."""
    chart_type: str = Field(..., description="The type of chart, e.g., 'bar', 'line', 'pie', 'scatter', 'heatmap', 'box', 'histogram', or 'kpi'.")
    title: str = Field(..., description="A descriptive title for the chart.")
    x_axis: Optional[str] = Field(None, description="The column for the x-axis. Not used for 'kpi' or 'histogram' charts.")
    y_axis: Optional[str] = Field(None, description="The column for the y-axis. Not used for 'kpi' charts.")
    z_axis: Optional[str] = Field(None, description="The column for the z-axis, used only for 'heatmap' charts.")
    value_column: Optional[str] = Field(None, description="The column containing the single value for 'kpi' charts.")
    explanation: str = Field(..., description="A brief explanation of why this chart is appropriate for the data.")

class VisualizationPlan(BaseModel):
    """Defines the overall plan for the dashboard, which can contain multiple charts."""
    charts: List[ChartComponent] = Field(..., description="A list of chart components to be included in the visualization.")

# --- JSON Serialization Helper for Decimal Types ---
def json_converter(o: Any) -> Any:
    """A custom converter to handle special data types like Decimal."""
    if isinstance(o, Decimal):
        return float(o)
    raise TypeError(f"Object of type {o.__class__.__name__} is not JSON serializable")

# --- Analysis and Summarization Tools ---

@tool
async def summarize_results(question: str, query_result: List[Dict[str, Any]]) -> str:
    """
    Analyzes raw data from a query and generates a concise, natural language summary
    that directly answers the original user question.
    """
    logger.info(f"Tool 'summarize_results' invoked for question: '{question[:100]}...'")
    
    if not query_result:
        logger.warning("No data provided to summarize_results tool.")
        return "The query returned no data. There is nothing to summarize."
        
    try:
        llm_client = get_llm_client()
        prompt = SUMMARIZATION_PROMPT_TEMPLATE.format(
            question=question,
            query_result=json.dumps(query_result, default=json_converter)
        )
        messages = [{"role": "user", "content": prompt}]
        
        summary = await llm_client.invoke_chat_completion(messages)
        if not summary:
            raise ValueError("LLM failed to generate a summary.")
            
        logger.info("Successfully generated summary.")
        return summary

    except Exception as e:
        logger.error(f"An unexpected error occurred in summarize_results: {e}", exc_info=True)
        return f"Error: Failed to generate summary. Details: {e}"

@tool
async def plan_visualizations(question: str, data: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Analyzes the user's question and dataset to create a structured JSON plan
    for one or more appropriate visualizations. Handles Decimal types correctly.
    """
    logger.info(f"Tool 'plan_visualizations' invoked for question: '{question[:100]}...'")

    if not data:
        logger.warning("No data provided to plan_visualizations tool.")
        return {"error": "No data available to create a visualization plan."}

    try:
        data_preview = {
            "columns": list(data[0].keys()) if data else [],
            "preview_rows": data[:3]
        }

        llm_client = get_llm_client()
        
        prompt = VISUALIZATION_PLANNING_PROMPT_TEMPLATE.format(
            question=question,
            data_preview=json.dumps(data_preview, indent=2, default=json_converter)
        )
        messages = [{"role": "user", "content": prompt}]
        
        response_str = await llm_client.invoke_chat_completion(messages, json_mode=True)
        if not response_str:
            raise ValueError("LLM returned no response for visualization plan.")

        validated_plan = VisualizationPlan.model_validate_json(response_str)
        logger.info(f"Successfully generated and validated visualization plan with {len(validated_plan.charts)} chart(s).")
        
        return {"plan": validated_plan.model_dump()}

    except ValidationError as e:
        error_message = f"LLM returned an invalid JSON structure for the visualization plan. Details: {e}"
        logger.error(error_message, exc_info=True)
        return {"error": error_message, "plan": None}
    except Exception as e:
        error_message = f"An unexpected error occurred in plan_visualizations: {e}"
        logger.error(error_message, exc_info=True)
        return {"error": error_message, "plan": None}

# --- Programmatic Visualization Tool ---

@tool
async def create_plotly_figure(chart_plan: Dict[str, Any], data: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Generates a single Plotly Figure object directly and programmatically
    based on a specific chart plan. Supports bar, pie, line, kpi, scatter, heatmap, box, and histogram charts.
    """
    logger.info(f"Tool 'create_plotly_figure' invoked for chart: '{chart_plan.get('title')}'")
    fig = None
    
    try:
        df = pd.DataFrame(data)
        chart_type = chart_plan.get("chart_type")
        title = chart_plan.get("title")

        if chart_type == "kpi":
            value_column_name = chart_plan.get("value_column")
            if not value_column_name or value_column_name not in df.columns:
                raise ValueError(f"Invalid value_column '{value_column_name}' for KPI chart.")

            df[value_column_name] = pd.to_numeric(df[value_column_name], errors='coerce')
            df.dropna(subset=[value_column_name], inplace=True)
            
            if df.empty:
                raise ValueError(f"No valid numeric data found in column '{value_column_name}' to create a KPI.")

            if len(df) > 1:
                numeric_value = df[value_column_name].sum()
                logger.info(f"Multiple rows found for KPI. Aggregating '{value_column_name}' to a sum of {numeric_value}.")
            else:
                numeric_value = df[value_column_name].iloc[0]

            fig = go.Figure(go.Indicator(
                mode="number",
                value=numeric_value,
                title={"text": title}
            ))
            fig.update_layout(height=250)

        elif chart_type == "bar":
            y_axis = chart_plan.get("y_axis")
            if y_axis and y_axis in df.columns:
                df[y_axis] = pd.to_numeric(df[y_axis], errors='coerce')
            
            x_axis = chart_plan.get("x_axis")
            if not all([x_axis, y_axis, x_axis in df.columns, y_axis in df.columns]):
                raise ValueError(f"Invalid x_axis or y_axis for bar chart.")
            fig = px.bar(df, x=x_axis, y=y_axis, title=title)
        
        elif chart_type == "pie":
            values_col = chart_plan.get("y_axis")
            if values_col and values_col in df.columns:
                df[values_col] = pd.to_numeric(df[values_col], errors='coerce')

            names_col = chart_plan.get("x_axis") 
            if not all([names_col, values_col, names_col in df.columns, values_col in df.columns]):
                raise ValueError(f"Invalid names_col or values_col for pie chart.")
            fig = px.pie(df, names=names_col, values=values_col, title=title)

        elif chart_type == "line":
            y_axis = chart_plan.get("y_axis")
            if y_axis and y_axis in df.columns:
                df[y_axis] = pd.to_numeric(df[y_axis], errors='coerce')
                
            x_axis = chart_plan.get("x_axis")
            if not all([x_axis, y_axis, x_axis in df.columns, y_axis in df.columns]):
                raise ValueError(f"Invalid x_axis or y_axis for line chart.")
            fig = px.line(df, x=x_axis, y=y_axis, title=title)

        elif chart_type == "scatter":
            x_axis = chart_plan.get("x_axis")
            y_axis = chart_plan.get("y_axis")
            if not all([x_axis, y_axis, x_axis in df.columns, y_axis in df.columns]):
                raise ValueError(f"Invalid x_axis or y_axis for scatter chart.")
            df[y_axis] = pd.to_numeric(df[y_axis], errors='coerce')
            df[x_axis] = pd.to_numeric(df[x_axis], errors='coerce')
            df.dropna(subset=[x_axis, y_axis], inplace=True)
            if df.empty:
                raise ValueError(f"No valid numeric data for scatter chart in columns '{x_axis}' and '{y_axis}'.")
            fig = px.scatter(df, x=x_axis, y=y_axis, title=title)

        elif chart_type == "heatmap":
            x_axis = chart_plan.get("x_axis")
            y_axis = chart_plan.get("y_axis")
            z_axis = chart_plan.get("z_axis")
            if not all([x_axis, y_axis, z_axis, x_axis in df.columns, y_axis in df.columns, z_axis in df.columns]):
                raise ValueError(f"Invalid x_axis, y_axis, or z_axis for heatmap.")
            df[z_axis] = pd.to_numeric(df[z_axis], errors='coerce')
            df.dropna(subset=[x_axis, y_axis, z_axis], inplace=True)
            if df.empty:
                raise ValueError(f"No valid data for heatmap in columns '{x_axis}', '{y_axis}', '{z_axis}'.")
            fig = px.density_heatmap(df, x=x_axis, y=y_axis, z=z_axis, title=title)

        elif chart_type == "box":
            y_axis = chart_plan.get("y_axis")
            if not y_axis or y_axis not in df.columns:
                raise ValueError(f"Invalid y_axis for box chart.")
            df[y_axis] = pd.to_numeric(df[y_axis], errors='coerce')
            df.dropna(subset=[y_axis], inplace=True)
            if df.empty:
                raise ValueError(f"No valid numeric data for box chart in column '{y_axis}'.")
            x_axis = chart_plan.get("x_axis")
            fig = px.box(df, x=x_axis if x_axis and x_axis in df.columns else None, y=y_axis, title=title)

        elif chart_type == "histogram":
            x_axis = chart_plan.get("x_axis")
            if not x_axis or x_axis not in df.columns:
                raise ValueError(f"Invalid x_axis for histogram.")
            df[x_axis] = pd.to_numeric(df[x_axis], errors='coerce')
            df.dropna(subset=[x_axis], inplace=True)
            if df.empty:
                raise ValueError(f"No valid numeric data for histogram in column '{x_axis}'.")
            fig = px.histogram(df, x=x_axis, title=title)

        else:
            logger.warning(f"Unsupported chart type: '{chart_type}'. Skipping.")
            return {"figure": None, "error": f"Unsupported chart type: {chart_type}"}

        logger.info(f"Successfully created Plotly figure for '{title}'.")
        return {"figure": fig, "error": None}

    except Exception as e:
        error_message = f"Failed to create Plotly figure for '{chart_plan.get('title')}': {e}"
        logger.error(error_message, exc_info=True)
        return {"figure": None, "error": error_message}