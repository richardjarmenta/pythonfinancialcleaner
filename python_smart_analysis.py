"""
Smart Analysis API - FastAPI service for analyzing business data with GPT
IMPROVED VERSION - Fixes critical prompt engineering and data analysis flaws
"""

import os
import re
import json
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator
import openai
from openai import OpenAI
import uvicorn
import logging
from datetime import datetime, timedelta
import sqlite3
import tempfile

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Smart Analysis API",
    description="Analyze tabular business data using natural language questions",
    version="2.0.0"
)

# Add CORS middleware for web integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Pydantic models for request and response validation
class AnalysisRequest(BaseModel):
    """Request model for the smart analysis endpoint"""
    question: str = Field(..., description="Natural language business question")
    rows: List[Dict[str, Any]] = Field(..., description="List of data rows from Supabase")
    file_title: Optional[str] = Field(None, description="Optional metadata about the data source")
    schema: Optional[List[str]] = Field(None, description="Optional column headers for reference")
    
    @field_validator('question')
    @classmethod
    def question_must_not_be_empty(cls, v):
        if not v or not v.strip():
            raise ValueError('Question cannot be empty')
        return v.strip()
    
    @field_validator('rows')
    @classmethod
    def rows_must_not_be_empty(cls, v):
        if not v:
            raise ValueError('Rows cannot be empty')
        return v

class AnalysisResponse(BaseModel):
    """Response model for the analysis results"""
    answer: str = Field(..., description="Direct answer to the question")
    explanation: str = Field(..., description="Brief reasoning behind the answer")
    table: Optional[str] = Field(None, description="Optional summary table or data")
    sql_query: Optional[str] = Field(None, description="SQL query used for analysis")
    confidence_score: float = Field(..., description="Confidence in analysis (0.0-1.0)")
    analysis_type: str = Field(..., description="Type of analysis performed")
    suggested_followup: Optional[str] = Field(None, description="Suggested follow-up questions")
    raw_insights: List[str] = Field(default_factory=list, description="Key insights for agent reasoning")
    metadata: Dict[str, Any] = Field(..., description="Additional information about the analysis")

def clean_and_normalize_data(rows: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Clean and normalize the dataset for analysis.
    
    Args:
        rows: List of dictionaries representing table rows
        
    Returns:
        Cleaned pandas DataFrame
    """
    try:
        # Convert to DataFrame
        df = pd.DataFrame(rows)
        
        if df.empty:
            raise ValueError("No data to process")
        
        logger.info(f"Processing {len(df)} rows with {len(df.columns)} columns")
        
        # Clean each column
        for column in df.columns:
            # Skip if column is all NaN
            if df[column].isna().all():
                continue
                
            # Convert to string for processing
            df[column] = df[column].astype(str)
            
            # Handle common null representations
            df[column] = df[column].replace(['nan', 'None', 'N/A', 'n/a', 'NULL', ''], pd.NA)
            
            # Try to clean currency values (e.g., "$1,250.00" -> 1250.0)
            currency_pattern = r'^\$?([\d,]+\.?\d*)$'
            if df[column].str.match(currency_pattern, na=False).any():
                logger.info(f"Cleaning currency values in column: {column}")
                df[column] = df[column].str.replace('$', '', regex=False)
                df[column] = df[column].str.replace(',', '', regex=False)
                df[column] = pd.to_numeric(df[column], errors='ignore')
            
            # Try to clean percentage values (e.g., "43%" -> 0.43)
            elif df[column].str.match(r'^\d+\.?\d*%$', na=False).any():
                logger.info(f"Cleaning percentage values in column: {column}")
                df[column] = df[column].str.replace('%', '', regex=False)
                df[column] = pd.to_numeric(df[column], errors='ignore') / 100
            
            # Try to convert numeric strings to numbers
            elif df[column].str.match(r'^\d+\.?\d*$', na=False).any():
                df[column] = pd.to_numeric(df[column], errors='ignore')
            
            # Try to detect and convert datetime columns
            elif df[column].str.match(r'\d{4}-\d{2}-\d{2}', na=False).any():
                try:
                    df[column] = pd.to_datetime(df[column], errors='ignore')
                    logger.info(f"Converted column {column} to datetime")
                except:
                    pass
        
        logger.info("Data cleaning completed successfully")
        return df
        
    except Exception as e:
        logger.error(f"Error cleaning data: {str(e)}")
        raise ValueError(f"Failed to clean data: {str(e)}")

def detect_analysis_requirements(question: str, df: pd.DataFrame) -> Dict[str, Any]:
    """
    Analyze the question and data to determine analysis requirements and confidence.
    This helps the agent system understand when this tool is most appropriate.
    
    Args:
        question: Natural language business question
        df: DataFrame to analyze
        
    Returns:
        Dictionary with analysis requirements and recommendations
    """
    requirements = {
        "analysis_type": "general",
        "confidence": 0.7,
        "complexity": "medium",
        "tool_suitability": 0.8,
        "recommended_approach": "python_analysis",
        "requires_full_dataset": False,
        "computation_heavy": False
    }
    
    question_lower = question.lower()
    
    # Detect numerical/statistical analysis needs
    numerical_keywords = ['calculate', 'sum', 'average', 'mean', 'median', 'count', 'total', 
                         'percentage', 'ratio', 'correlation', 'trend', 'growth', 'change',
                         'maximum', 'minimum', 'highest', 'lowest', 'distribution']
    
    if any(keyword in question_lower for keyword in numerical_keywords):
        requirements["analysis_type"] = "statistical"
        requirements["confidence"] = 0.9
        requirements["computation_heavy"] = True
        requirements["tool_suitability"] = 0.95
    
    # Detect time series analysis
    time_keywords = ['over time', 'trend', 'monthly', 'yearly', 'quarterly', 'daily',
                    'before', 'after', 'since', 'until', 'growth', 'change over']
    
    if any(keyword in question_lower for keyword in time_keywords):
        requirements["analysis_type"] = "time_series"
        requirements["requires_full_dataset"] = True
        requirements["tool_suitability"] = 0.9
    
    # Detect aggregation needs
    aggregation_keywords = ['by region', 'by category', 'group by', 'breakdown', 'segment',
                           'per', 'each', 'every', 'all', 'top', 'bottom', 'ranking']
    
    if any(keyword in question_lower for keyword in aggregation_keywords):
        requirements["analysis_type"] = "aggregation"
        requirements["requires_full_dataset"] = True
        requirements["tool_suitability"] = 0.85
    
    # Detect complex analysis that RAG would struggle with
    complex_keywords = ['outliers', 'anomaly', 'pattern', 'cluster', 'segment', 'forecast',
                       'predict', 'model', 'regression', 'variance', 'standard deviation']
    
    if any(keyword in question_lower for keyword in complex_keywords):
        requirements["analysis_type"] = "advanced_analytics"
        requirements["complexity"] = "high"
        requirements["computation_heavy"] = True
        requirements["tool_suitability"] = 0.95
        requirements["recommended_approach"] = "python_required"
    
    # Assess data characteristics
    if len(df) > 1000:
        requirements["requires_full_dataset"] = True
        requirements["complexity"] = "high"
    
    if len(df.select_dtypes(include=[np.number]).columns) > 3:
        requirements["tool_suitability"] += 0.1
    
    # Detect questions better suited for RAG
    text_keywords = ['explain', 'describe', 'what is', 'definition', 'meaning', 'purpose',
                    'background', 'context', 'why', 'how does', 'overview']
    
    if any(keyword in question_lower for keyword in text_keywords) and not requirements["computation_heavy"]:
        requirements["tool_suitability"] = 0.3
        requirements["recommended_approach"] = "rag_better"
    
    return requirements

def generate_smart_schema_summary(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Generate an intelligent schema summary that helps GPT understand the data structure.
    
    Args:
        df: Cleaned pandas DataFrame
        
    Returns:
        Dictionary with smart schema insights
    """
    schema_summary = {
        "columns": {},
        "data_insights": {},
        "business_context": {}
    }
    
    for column in df.columns:
        col_info = {
            "type": str(df[column].dtype),
            "null_count": int(df[column].isna().sum()),
            "unique_count": int(df[column].nunique()),
            "role": "unknown"
        }
        
        # Determine column role
        if df[column].dtype in ['int64', 'float64']:
            col_info["role"] = "numeric"
            col_info["min"] = float(df[column].min()) if not df[column].isna().all() else None
            col_info["max"] = float(df[column].max()) if not df[column].isna().all() else None
            col_info["mean"] = float(df[column].mean()) if not df[column].isna().all() else None
            
            # Check if it might be an ID column
            if col_info["unique_count"] == len(df) and "id" in column.lower():
                col_info["role"] = "identifier"
        
        elif df[column].dtype == 'datetime64[ns]':
            col_info["role"] = "datetime"
            col_info["date_range"] = {
                "start": str(df[column].min()) if not df[column].isna().all() else None,
                "end": str(df[column].max()) if not df[column].isna().all() else None
            }
        
        else:
            col_info["role"] = "categorical"
            # Get top categories for categorical data
            if col_info["unique_count"] <= 20:  # Only if manageable number of categories
                value_counts = df[column].value_counts().head(5)
                col_info["top_values"] = value_counts.to_dict()
        
        schema_summary["columns"][column] = col_info
    
    # Generate data insights
    schema_summary["data_insights"] = {
        "total_rows": len(df),
        "total_columns": len(df.columns),
        "numeric_columns": len([c for c in df.columns if df[c].dtype in ['int64', 'float64']]),
        "categorical_columns": len([c for c in df.columns if df[c].dtype == 'object']),
        "datetime_columns": len([c for c in df.columns if df[c].dtype == 'datetime64[ns]']),
        "missing_data_percentage": float((df.isna().sum().sum() / (len(df) * len(df.columns))) * 100)
    }
    
    return schema_summary

def create_strategic_data_sample(df: pd.DataFrame, max_rows: int = 15) -> pd.DataFrame:
    """
    Create a strategic sample that's more representative than just head().
    
    Args:
        df: DataFrame to sample from
        max_rows: Maximum number of rows to include
        
    Returns:
        Strategic sample of the DataFrame
    """
    if len(df) <= max_rows:
        return df
    
    # Strategy: Mix of head, tail, and random sample to capture patterns
    head_size = max_rows // 3
    tail_size = max_rows // 3
    random_size = max_rows - head_size - tail_size
    
    sample_parts = []
    
    # Get head
    sample_parts.append(df.head(head_size))
    
    # Get tail
    sample_parts.append(df.tail(tail_size))
    
    # Get random sample from middle
    if len(df) > head_size + tail_size:
        middle_df = df.iloc[head_size:-tail_size] if len(df) > head_size + tail_size else df
        if len(middle_df) > 0:
            random_sample = middle_df.sample(min(random_size, len(middle_df)), random_state=42)
            sample_parts.append(random_sample)
    
    # Combine and remove duplicates
    strategic_sample = pd.concat(sample_parts, ignore_index=True).drop_duplicates()
    
    return strategic_sample.head(max_rows)

def generate_sql_query_with_gpt(question: str, schema_summary: Dict[str, Any], table_name: str = "data") -> str:
    """
    Use GPT to generate a SQL query based on the question and schema.
    
    Args:
        question: Natural language business question
        schema_summary: Smart schema summary
        table_name: Name of the table to query
        
    Returns:
        SQL query string
    """
    try:
        # Build schema description for GPT
        schema_desc = []
        for col, info in schema_summary["columns"].items():
            desc = f"- {col}: {info['role']} ({info['type']})"
            if info["role"] == "categorical" and "top_values" in info:
                top_vals = list(info["top_values"].keys())[:3]
                desc += f" - common values: {top_vals}"
            elif info["role"] == "numeric" and info.get("min") is not None:
                desc += f" - range: {info['min']:.2f} to {info['max']:.2f}"
            schema_desc.append(desc)
        
        system_prompt = f"""You are a SQL expert. Generate a SQL query to answer the business question.

Table name: {table_name}
Schema:
{chr(10).join(schema_desc)}

Rules:
1. Use standard SQL syntax compatible with SQLite
2. Return ONLY the SQL query, no explanations
3. Use appropriate aggregations (SUM, COUNT, AVG, etc.)
4. Include GROUP BY and ORDER BY when needed
5. Use LIMIT if asking for "top" results
6. Handle case-insensitive string matching with LOWER()

Examples:
- "What's the total revenue?" → SELECT SUM(revenue) as total_revenue FROM data
- "Top 5 customers by sales" → SELECT customer, SUM(sales) as total_sales FROM data GROUP BY customer ORDER BY total_sales DESC LIMIT 5
"""

        user_prompt = f"Question: {question}\n\nGenerate the SQL query:"

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            max_tokens=300,
            temperature=0.1
        )
        
        sql_query = response.choices[0].message.content.strip()
        
        # Clean up the response - remove markdown formatting if present
        sql_query = re.sub(r'```sql\n?', '', sql_query)
        sql_query = re.sub(r'```\n?', '', sql_query)
        sql_query = sql_query.strip()
        
        logger.info(f"Generated SQL query: {sql_query}")
        return sql_query
        
    except Exception as e:
        logger.error(f"Error generating SQL query: {str(e)}")
        # Fallback to a simple SELECT
        return f"SELECT * FROM {table_name} LIMIT 10"

def execute_sql_on_dataframe(df: pd.DataFrame, sql_query: str) -> pd.DataFrame:
    """
    Execute SQL query on DataFrame using SQLite.
    
    Args:
        df: DataFrame to query
        sql_query: SQL query to execute
        
    Returns:
        Result DataFrame
    """
    try:
        # Create temporary SQLite database
        with tempfile.NamedTemporaryFile(delete=False, suffix='.db') as tmp_file:
            conn = sqlite3.connect(tmp_file.name)
            
            # Write DataFrame to SQLite
            df.to_sql('data', conn, index=False, if_exists='replace')
            
            # Execute query
            result_df = pd.read_sql_query(sql_query, conn)
            
            conn.close()
            os.unlink(tmp_file.name)  # Clean up temp file
            
            logger.info(f"SQL query executed successfully, returned {len(result_df)} rows")
            return result_df
            
    except Exception as e:
        logger.error(f"Error executing SQL query: {str(e)}")
        # Fallback: return the original dataframe head
        return df.head(10)

def analyze_with_gpt_v2(
    question: str, 
    df: pd.DataFrame, 
    file_title: Optional[str] = None,
    schema: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    IMPROVED: Analyze the dataset using a two-stage approach with SQL generation.
    Enhanced for agentic RAG systems with better tool coordination.
    
    Args:
        question: Natural language business question
        df: Cleaned pandas DataFrame
        file_title: Optional metadata about the data source
        schema: Optional column headers for reference
        
    Returns:
        Dictionary with comprehensive analysis results for agent consumption
    """
    try:
        # Stage 0: Analyze requirements and suitability
        requirements = detect_analysis_requirements(question, df)
        
        # Stage 1: Generate smart schema summary
        schema_summary = generate_smart_schema_summary(df)
        
        # Stage 2: Generate SQL query using GPT
        sql_query = generate_sql_query_with_gpt(question, schema_summary)
        
        # Stage 3: Execute SQL query
        query_result = execute_sql_on_dataframe(df, sql_query)
        
        # Stage 4: Get strategic sample for context
        strategic_sample = create_strategic_data_sample(df, max_rows=8)
        
        # Stage 5: Enhanced prompt for agent coordination
        system_prompt = """You are a data analysis specialist working as part of an AI agent team that includes RAG, document retrieval, and SQL tools.

Your role is to provide computational analysis that complements these other tools. Focus on:
1. Statistical calculations and data insights
2. Patterns and trends that require full dataset analysis  
3. Complex aggregations and computations
4. Quantitative answers with specific numbers

Return ONLY valid JSON in this exact format:
{
  "answer": "Direct, specific answer with numbers/data",
  "explanation": "Brief methodology and key findings", 
  "table": "Formatted results table (if helpful, else null)",
  "raw_insights": ["insight1", "insight2", "insight3"],
  "suggested_followup": "One natural follow-up question (or null)"
}

Be precise, quantitative, and actionable. Your insights will be used by other AI tools."""

        # Build enhanced prompt with agent context
        user_prompt = f"""
AGENT CONTEXT:
- Analysis Type: {requirements['analysis_type']}
- Tool Suitability: {requirements['tool_suitability']:.2f}
- This tool specializes in: computational analysis, statistics, full-dataset operations

BUSINESS QUESTION: {question}
File Source: {file_title or 'Unknown'}

DATASET ANALYSIS:
Total Rows: {len(df)}
Columns: {list(df.columns)}
Numeric Columns: {len(df.select_dtypes(include=[np.number]).columns)}

SQL QUERY EXECUTED:
{sql_query}

QUERY RESULTS ({len(query_result)} rows):
{query_result.to_string(index=False) if len(query_result) > 0 else "No results returned"}

SAMPLE DATA PATTERNS:
{strategic_sample.head(5).to_string(index=False)}

STATISTICAL SUMMARY:
{df.describe().to_string() if len(df.select_dtypes(include=[np.number]).columns) > 0 else "No numeric data for statistics"}

Provide computational analysis in the required JSON format:"""

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            max_tokens=1000,
            temperature=0.1
        )
        
        gpt_response = response.choices[0].message.content.strip()
        logger.info("Received enhanced response from GPT for agent system")
        
        # Parse JSON response with enhanced error handling
        try:
            result = json.loads(gpt_response)
            
            # Add agent-specific metadata
            result["sql_query"] = sql_query
            result["confidence_score"] = min(requirements["tool_suitability"], 0.95)
            result["analysis_type"] = requirements["analysis_type"]
            
            # Ensure raw_insights exists
            if "raw_insights" not in result:
                result["raw_insights"] = [result.get("answer", "Analysis completed")]
            
            # Ensure suggested_followup exists
            if "suggested_followup" not in result:
                result["suggested_followup"] = None
                
            return result
            
        except json.JSONDecodeError:
            logger.warning("GPT response was not valid JSON, creating structured fallback")
            
            # Enhanced fallback with agent context
            fallback_insights = []
            if len(query_result) > 0:
                fallback_insights.append(f"Query returned {len(query_result)} results")
                if query_result.select_dtypes(include=[np.number]).columns.any():
                    numeric_cols = query_result.select_dtypes(include=[np.number]).columns
                    for col in numeric_cols[:2]:  # Max 2 columns
                        if len(query_result[col].dropna()) > 0:
                            fallback_insights.append(f"{col}: {query_result[col].sum():.2f} total")
            
            return {
                "answer": gpt_response[:200] + "..." if len(gpt_response) > 200 else gpt_response,
                "explanation": "Analysis completed with computational approach, response formatting adjusted",
                "table": query_result.to_string(index=False) if len(query_result) <= 20 else None,
                "sql_query": sql_query,
                "confidence_score": 0.6,
                "analysis_type": requirements["analysis_type"],
                "suggested_followup": None,
                "raw_insights": fallback_insights if fallback_insights else ["Analysis provided computational results"]
            }
        
    except Exception as e:
        logger.error(f"Error in enhanced GPT analysis: {str(e)}")
        
        # Robust fallback that still provides value to the agent
        return {
            "answer": f"Computational analysis encountered an issue: {str(e)}",
            "explanation": "Python analyzer attempted data processing but encountered technical difficulties. Other tools in the agent system may provide better results for this query.",
            "table": None,
            "sql_query": sql_query if 'sql_query' in locals() else None,
            "confidence_score": 0.1,
            "analysis_type": "error",
            "suggested_followup": "Try rephrasing the question or check if RAG tools might handle this better",
            "raw_insights": ["Tool encountered processing error", "Other agent tools may be more suitable"]
        }

@app.post("/smart-analysis", response_model=AnalysisResponse)
async def smart_analysis(request: AnalysisRequest):
    """
    Analyze tabular business data based on natural language questions.
    
    Enhanced for agentic RAG systems - this tool specializes in computational analysis,
    statistical operations, and data insights that complement RAG and document retrieval tools.
    
    Use this tool when you need:
    - Statistical calculations and aggregations
    - Trend analysis over full datasets  
    - Complex mathematical operations
    - Quantitative insights with specific numbers
    - Analysis that requires seeing all data rows
    """
    try:
        logger.info(f"Processing agentic analysis request: {request.question}")
        
        # Clean and normalize the data
        df = clean_and_normalize_data(request.rows)
        
        # Detect analysis requirements for agent coordination
        requirements = detect_analysis_requirements(request.question, df)
        
        # Log agent guidance
        if requirements["tool_suitability"] < 0.5:
            logger.info(f"Low tool suitability ({requirements['tool_suitability']:.2f}) - agent might prefer RAG/document tools")
        
        # Analyze with enhanced GPT approach
        analysis_result = analyze_with_gpt_v2(
            question=request.question,
            df=df,
            file_title=request.file_title,
            schema=request.schema
        )
        
        # Prepare comprehensive metadata for agent system
        metadata = {
            "total_rows": len(df),
            "columns": list(df.columns),
            "file_title": request.file_title,
            "data_types": df.dtypes.astype(str).to_dict(),
            "analysis_method": "python_analyzer_v2",
            "tool_requirements": requirements,
            "schema_summary": generate_smart_schema_summary(df),
            "agent_guidance": {
                "tool_suitability": requirements["tool_suitability"],
                "recommended_approach": requirements["recommended_approach"],
                "complexity": requirements["complexity"],
                "requires_full_dataset": requirements["requires_full_dataset"]
            }
        }
        
        # Build comprehensive response for agent
        response = AnalysisResponse(
            answer=analysis_result["answer"],
            explanation=analysis_result["explanation"],
            table=analysis_result.get("table"),
            sql_query=analysis_result.get("sql_query"),
            confidence_score=analysis_result.get("confidence_score", 0.7),
            analysis_type=analysis_result.get("analysis_type", "general"),
            suggested_followup=analysis_result.get("suggested_followup"),
            raw_insights=analysis_result.get("raw_insights", []),
            metadata=metadata
        )
        
        logger.info(f"Analysis completed - Type: {response.analysis_type}, Confidence: {response.confidence_score:.2f}")
        return response
        
    except ValueError as e:
        logger.error(f"Validation error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Python analyzer encountered an error - other agent tools may handle this query better"
        )

@app.post("/tool-recommendation")
async def tool_recommendation(request: AnalysisRequest):
    """
    Analyze a question and data to recommend which tool the agent should use.
    
    Returns guidance on whether this Python analyzer, RAG, document retrieval,
    or SQL tools would be most effective for the given question.
    """
    try:
        # Quick data overview without full processing
        df_sample = pd.DataFrame(request.rows[:100])  # Sample for speed
        if df_sample.empty:
            raise ValueError("No data provided")
        
        # Analyze requirements
        requirements = detect_analysis_requirements(request.question, df_sample)
        
        # Build recommendations
        recommendations = {
            "python_analyzer": {
                "suitability": requirements["tool_suitability"],
                "reasons": []
            },
            "rag_tool": {
                "suitability": 0.5,
                "reasons": []
            },
            "document_retrieval": {
                "suitability": 0.3,
                "reasons": []
            },
            "sql_tool": {
                "suitability": 0.4,
                "reasons": []
            }
        }
        
        question_lower = request.question.lower()
        
        # Python analyzer strengths
        if requirements["computation_heavy"]:
            recommendations["python_analyzer"]["reasons"].append("Question requires complex calculations")
        if requirements["requires_full_dataset"]:
            recommendations["python_analyzer"]["reasons"].append("Analysis needs complete dataset view")
        if requirements["analysis_type"] in ["statistical", "advanced_analytics"]:
            recommendations["python_analyzer"]["reasons"].append("Statistical/advanced analysis required")
        
        # RAG tool strengths  
        if any(word in question_lower for word in ['explain', 'describe', 'what is', 'meaning', 'context']):
            recommendations["rag_tool"]["suitability"] = 0.9
            recommendations["rag_tool"]["reasons"].append("Question seeks explanation or context")
        if any(word in question_lower for word in ['document', 'text', 'content', 'summary']):
            recommendations["rag_tool"]["suitability"] = 0.8
            recommendations["rag_tool"]["reasons"].append("Question about document content")
        
        # Document retrieval strengths
        if any(word in question_lower for word in ['meeting notes', 'specific document', 'file']):
            recommendations["document_retrieval"]["suitability"] = 0.9
            recommendations["document_retrieval"]["reasons"].append("Question targets specific documents")
        
        # SQL tool strengths
        if len(df_sample) > 1000 and requirements["analysis_type"] == "aggregation":
            recommendations["sql_tool"]["suitability"] = 0.8
            recommendations["sql_tool"]["reasons"].append("Large dataset with aggregation needs")
        
        # Find best tool
        best_tool = max(recommendations.keys(), key=lambda x: recommendations[x]["suitability"])
        
        return {
            "recommended_tool": best_tool,
            "recommendations": recommendations,
            "analysis_requirements": requirements,
            "confidence": recommendations[best_tool]["suitability"],
            "data_overview": {
                "rows": len(request.rows),
                "columns": len(df_sample.columns) if not df_sample.empty else 0,
                "numeric_columns": len(df_sample.select_dtypes(include=[np.number]).columns) if not df_sample.empty else 0
            }
        }
        
    except Exception as e:
        logger.error(f"Error in tool recommendation: {str(e)}")
        return {
            "recommended_tool": "python_analyzer",
            "confidence": 0.5,
            "error": str(e),
            "fallback": True
        }

@app.get("/health")
async def health_check():
    """Health check endpoint for monitoring"""
    return {"status": "healthy", "service": "smart-analysis-api", "version": "2.0.0"}

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Smart Analysis API v2.0 - Python Analyzer Tool for Agentic RAG Systems", 
        "version": "2.0.0",
        "purpose": "Computational analysis tool designed to complement RAG, document retrieval, and SQL tools in AI agent systems",
        "specializes_in": [
            "Statistical calculations and aggregations",
            "Trend analysis over full datasets",
            "Complex mathematical operations", 
            "Quantitative insights with specific numbers",
            "Pattern detection and outlier analysis",
            "Multi-dimensional data analysis"
        ],
        "works_best_with": [
            "Questions requiring calculations (sum, average, count, etc.)",
            "Trend and time series analysis",
            "Complex aggregations across large datasets",
            "Statistical operations (correlations, distributions)",
            "Data that needs full context (not just chunks)"
        ],
        "agent_coordination": {
            "tool_recommendation_endpoint": "/tool-recommendation",
            "confidence_scoring": "Returns 0.0-1.0 confidence scores",
            "fallback_guidance": "Suggests when other tools might be better",
            "structured_output": "JSON format optimized for agent consumption"
        },
        "improvements_v2": [
            "Enhanced prompt engineering with structured JSON output",
            "Two-stage analysis: SQL generation + interpretation",
            "Smart schema inference and validation",
            "Strategic data sampling for better context",
            "Tool suitability detection for agent coordination",
            "Confidence scoring and fallback recommendations",
            "Raw insights extraction for cross-tool reasoning"
        ],
        "endpoints": {
            "POST /smart-analysis": "Main analysis endpoint - computational data analysis",
            "POST /tool-recommendation": "Helps agents choose the right tool for each question",
            "GET /health": "Health check and system status",
            "GET /docs": "Interactive API documentation"
        },
        "integration_notes": {
            "designed_for": "Multi-tool AI agent systems with RAG, document retrieval, and SQL capabilities",
            "coordinates_with": "Vector databases, document stores, and other analytical tools",
            "provides_fallbacks": "Graceful degradation when other tools might be more suitable"
        }
    }

if __name__ == "__main__":
    # Check for required environment variables
    if not os.getenv("OPENAI_API_KEY"):
        logger.error("OPENAI_API_KEY environment variable is required")
        exit(1)
    
    # Run the application
    uvicorn.run(
        "python_smart_analysis:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", 8000)),
        reload=False
    )