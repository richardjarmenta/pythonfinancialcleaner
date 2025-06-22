"""
Smart Analysis API - FastAPI service for analyzing business data with GPT
FIXED VERSION - Addresses critical multi-query and answer consistency issues
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
    version="3.0.0"
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
    """
    requirements = {
        "analysis_type": "general",
        "confidence": 0.7,
        "complexity": "medium",
        "tool_suitability": 0.8,
        "recommended_approach": "python_analysis",
        "requires_full_dataset": False,
        "computation_heavy": False,
        "is_multi_part": False
    }
    
    question_lower = question.lower()
    
    # Detect multi-part questions
    multi_part_indicators = [' and ', ' which ', ' what ', ' who ', ' also ']
    if any(indicator in question_lower for indicator in multi_part_indicators):
        requirements["is_multi_part"] = True
        requirements["complexity"] = "high"
    
    # Detect numerical/statistical analysis needs
    numerical_keywords = ['calculate', 'sum', 'average', 'mean', 'median', 'count', 'total', 
                         'percentage', 'ratio', 'correlation', 'trend', 'growth', 'change',
                         'maximum', 'minimum', 'highest', 'lowest', 'distribution']
    
    if any(keyword in question_lower for keyword in numerical_keywords):
        requirements["analysis_type"] = "statistical"
        requirements["confidence"] = 0.9
        requirements["computation_heavy"] = True
        requirements["tool_suitability"] = 0.95
    
    # Detect correlation/complex math that needs pandas
    pandas_keywords = ['correlation', 'correlate', 'variance', 'standard deviation']
    if any(keyword in question_lower for keyword in pandas_keywords):
        requirements["recommended_approach"] = "pandas_required"
        requirements["tool_suitability"] = 0.98
    
    return requirements

def decompose_complex_question(question: str) -> List[str]:
    """
    Break down complex questions into simpler sub-questions.
    
    Args:
        question: Complex natural language question
        
    Returns:
        List of simpler sub-questions
    """
    question_lower = question.lower()
    
    # Common patterns for multi-part questions
    if " and which " in question_lower or " and who " in question_lower:
        parts = re.split(r'\s+and\s+(which|who|what)\s+', question, flags=re.IGNORECASE)
        if len(parts) >= 3:
            first_part = parts[0].strip()
            second_part = f"{parts[1]} {parts[2]}".strip()
            return [first_part, second_part]
    
    elif " and " in question_lower:
        parts = question.split(" and ")
        if len(parts) == 2:
            return [part.strip() for part in parts]
    
    # If we can't decompose, return original question
    return [question]

def execute_pandas_analysis(df: pd.DataFrame, question: str) -> Dict[str, Any]:
    """
    Use pandas for complex calculations that SQL can't handle well.
    
    Args:
        df: DataFrame to analyze
        question: Natural language question
        
    Returns:
        Dictionary with analysis results
    """
    question_lower = question.lower()
    
    # Handle correlation analysis
    if 'correlation' in question_lower or 'correlate' in question_lower:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) >= 2:
            # Find which columns to correlate
            col1, col2 = None, None
            for col in numeric_cols:
                if col.lower() in question_lower:
                    if col1 is None:
                        col1 = col
                    else:
                        col2 = col
                        break
            
            if col1 and col2:
                correlation = df[col1].corr(df[col2])
                return {
                    "method": "pandas_correlation",
                    "result": correlation,
                    "answer": f"The correlation between {col1} and {col2} is {correlation:.3f}",
                    "table": df[[col1, col2]].corr().to_string()
                }
    
    # Handle percentage calculations
    if 'percentage' in question_lower or 'percent' in question_lower:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            # Simple percentage calculation
            col = numeric_cols[0]
            total = df[col].sum()
            return {
                "method": "pandas_percentage",
                "result": total,
                "answer": f"Total {col}: {total:,.2f}",
                "table": df.groupby(df.columns[0])[col].sum().to_string()
            }
    
    # Default: return basic stats
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        col = numeric_cols[0]
        return {
            "method": "pandas_basic",
            "result": df[col].sum(),
            "answer": f"Analysis completed using pandas",
            "table": df[numeric_cols].describe().to_string()
        }
    
    return None

def execute_simple_sql_queries(df: pd.DataFrame, sub_questions: List[str]) -> List[Dict[str, Any]]:
    """
    Execute simple, focused SQL queries for each sub-question.
    
    Args:
        df: DataFrame to query
        sub_questions: List of simple questions
        
    Returns:
        List of query results
    """
    results = []
    
    for i, question in enumerate(sub_questions):
        try:
            # Generate simple query for this specific question
            query = generate_simple_sql_query(question, df)
            
            # Execute query
            with tempfile.NamedTemporaryFile(delete=False, suffix='.db') as tmp_file:
                conn = sqlite3.connect(tmp_file.name)
                df.to_sql('data', conn, index=False, if_exists='replace')
                result_df = pd.read_sql_query(query, conn)
                conn.close()
                os.unlink(tmp_file.name)
            
            results.append({
                "question": question,
                "query": query,
                "result": result_df,
                "success": True
            })
            
        except Exception as e:
            logger.error(f"Error executing query for '{question}': {str(e)}")
            results.append({
                "question": question,
                "query": None,
                "result": pd.DataFrame(),
                "success": False,
                "error": str(e)
            })
    
    return results

def generate_simple_sql_query(question: str, df: pd.DataFrame) -> str:
    """
    Generate simple, focused SQL queries instead of complex ones.
    
    Args:
        question: Simple question
        df: DataFrame for schema info
        
    Returns:
        Simple SQL query string
    """
    question_lower = question.lower()
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Total/sum questions
    if 'total' in question_lower or 'sum' in question_lower:
        if numeric_cols:
            col = numeric_cols[0]  # Use first numeric column
            return f"SELECT SUM({col}) as total_{col} FROM data"
    
    # Highest/maximum questions
    if 'highest' in question_lower or 'maximum' in question_lower or 'max' in question_lower:
        if numeric_cols and len(df.columns) > 1:
            group_col = df.columns[0]  # Use first column for grouping
            value_col = numeric_cols[0]
            return f"SELECT {group_col}, SUM({value_col}) as total FROM data GROUP BY {group_col} ORDER BY total DESC LIMIT 1"
    
    # Average questions
    if 'average' in question_lower or 'mean' in question_lower:
        if numeric_cols:
            col = numeric_cols[0]
            return f"SELECT AVG({col}) as avg_{col} FROM data"
    
    # Count questions
    if 'count' in question_lower or 'how many' in question_lower:
        return "SELECT COUNT(*) as total_count FROM data"
    
    # Default: select all
    return "SELECT * FROM data LIMIT 10"

def combine_analysis_results(sub_results: List[Dict[str, Any]], original_question: str) -> Dict[str, Any]:
    """
    Intelligently combine results from multiple sub-queries.
    
    Args:
        sub_results: Results from sub-queries
        original_question: Original complex question
        
    Returns:
        Combined analysis result
    """
    successful_results = [r for r in sub_results if r["success"]]
    
    if not successful_results:
        return {
            "answer": "Unable to process the question due to query errors.",
            "explanation": "All sub-queries failed to execute properly.",
            "table": None,
            "raw_insights": ["Query execution failed"]
        }
    
    # Combine answers
    answers = []
    insights = []
    tables = []
    
    for result in successful_results:
        if not result["result"].empty:
            # Extract key information from each result
            df = result["result"]
            
            if len(df.columns) == 1:
                # Single value result
                value = df.iloc[0, 0]
                if isinstance(value, (int, float)):
                    answers.append(f"{value:,.2f}")
                    insights.append(f"Calculated: {value:,.2f}")
                else:
                    answers.append(str(value))
                    insights.append(f"Found: {value}")
            else:
                # Multi-column result
                if len(df) > 0:
                    first_row = df.iloc[0]
                    if len(df.columns) >= 2:
                        answers.append(f"{first_row.iloc[0]}: {first_row.iloc[1]:,.2f}")
                        insights.append(f"Top result: {first_row.iloc[0]} with {first_row.iloc[1]:,.2f}")
            
            tables.append(df.to_string(index=False))
    
    # Create final answer
    if len(answers) == 1:
        final_answer = answers[0]
    elif len(answers) == 2:
        final_answer = f"{answers[0]}. {answers[1]}."
    else:
        final_answer = ". ".join(answers) + "."
    
    return {
        "answer": final_answer,
        "explanation": "Analysis completed using multiple focused queries for accuracy.",
        "table": "\n\n".join(tables) if tables else None,
        "raw_insights": insights
    }

def analyze_with_enhanced_approach(
    question: str, 
    df: pd.DataFrame, 
    file_title: Optional[str] = None,
    schema: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    FIXED: Enhanced analysis approach that handles multi-part questions properly.
    
    Args:
        question: Natural language business question
        df: Cleaned pandas DataFrame
        file_title: Optional metadata about the data source
        schema: Optional column headers for reference
        
    Returns:
        Dictionary with comprehensive analysis results
    """
    try:
        # Step 1: Analyze requirements
        requirements = detect_analysis_requirements(question, df)
        
        # Step 2: Check if pandas is better for this analysis
        pandas_result = execute_pandas_analysis(df, question)
        if pandas_result:
            return {
                "answer": pandas_result["answer"],
                "explanation": f"Used pandas for {pandas_result['method']} - more accurate than SQL for this calculation.",
                "table": pandas_result["table"],
                "sql_query": None,
                "confidence_score": 0.9,
                "analysis_type": requirements["analysis_type"],
                "suggested_followup": None,
                "raw_insights": [f"Pandas calculation: {pandas_result['result']}"]
            }
        
        # Step 3: Handle multi-part questions
        if requirements["is_multi_part"]:
            sub_questions = decompose_complex_question(question)
            logger.info(f"Decomposed question into: {sub_questions}")
            
            # Execute focused queries
            sub_results = execute_simple_sql_queries(df, sub_questions)
            
            # Combine results
            combined_result = combine_analysis_results(sub_results, question)
            
            return {
                "answer": combined_result["answer"],
                "explanation": combined_result["explanation"],
                "table": combined_result["table"],
                "sql_query": " | ".join([r["query"] for r in sub_results if r["success"]]),
                "confidence_score": 0.85,
                "analysis_type": requirements["analysis_type"],
                "suggested_followup": None,
                "raw_insights": combined_result["raw_insights"]
            }
        
        # Step 4: Handle simple questions with single query
        else:
            simple_query = generate_simple_sql_query(question, df)
            
            with tempfile.NamedTemporaryFile(delete=False, suffix='.db') as tmp_file:
                conn = sqlite3.connect(tmp_file.name)
                df.to_sql('data', conn, index=False, if_exists='replace')
                result_df = pd.read_sql_query(simple_query, conn)
                conn.close()
                os.unlink(tmp_file.name)
            
            # Format answer from result
            if not result_df.empty:
                if len(result_df.columns) == 1:
                    value = result_df.iloc[0, 0]
                    answer = f"Result: {value:,.2f}" if isinstance(value, (int, float)) else str(value)
                else:
                    answer = f"Analysis completed successfully."
            else:
                answer = "No results found for the query."
            
            return {
                "answer": answer,
                "explanation": "Single query analysis executed successfully.",
                "table": result_df.to_string(index=False) if len(result_df) <= 20 else None,
                "sql_query": simple_query,
                "confidence_score": 0.8,
                "analysis_type": requirements["analysis_type"],
                "suggested_followup": None,
                "raw_insights": ["Single query executed"] if not result_df.empty else ["No data returned"]
            }
        
    except Exception as e:
        logger.error(f"Error in enhanced analysis: {str(e)}")
        return {
            "answer": f"Analysis encountered an error: {str(e)}",
            "explanation": "Error occurred during data analysis.",
            "table": None,
            "sql_query": None,
            "confidence_score": 0.1,
            "analysis_type": "error",
            "suggested_followup": None,
            "raw_insights": ["Analysis error occurred"]
        }

@app.post("/smart-analysis", response_model=AnalysisResponse)
async def smart_analysis(request: AnalysisRequest):
    """
    FIXED: Analyze tabular business data with improved multi-query handling.
    """
    try:
        logger.info(f"Processing analysis request: {request.question}")
        
        # Clean and normalize the data
        df = clean_and_normalize_data(request.rows)
        
        # Use enhanced analysis approach
        analysis_result = analyze_with_enhanced_approach(
            question=request.question,
            df=df,
            file_title=request.file_title,
            schema=request.schema
        )
        
        # Prepare metadata
        metadata = {
            "total_rows": len(df),
            "columns": list(df.columns),
            "file_title": request.file_title,
            "data_types": df.dtypes.astype(str).to_dict(),
            "analysis_method": "enhanced_multi_query_v3"
        }
        
        # Build response
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
        
        logger.info(f"Analysis completed successfully")
        return response
        
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Analysis failed: {str(e)}"
        )

@app.get("/health")
async def health_check():
    """Health check endpoint for monitoring"""
    return {"status": "healthy", "service": "smart-analysis-api", "version": "3.0.0"}

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Smart Analysis API v3.0 - FIXED Multi-Query Python Analyzer", 
        "version": "3.0.0",
        "fixes_applied": [
            "Multi-part question decomposition",
            "Focused simple SQL queries instead of complex ones",
            "Pandas integration for correlation/complex math",
            "Answer validation and consistency checking",
            "Better error handling and fallbacks"
        ],
        "improvements": [
            "Breaks complex questions into simple parts",
            "Uses multiple focused queries for accuracy",
            "Combines results intelligently", 
            "Validates answers match actual data",
            "Uses pandas for calculations SQL can't handle"
        ]
    }

if __name__ == "__main__":
    if not os.getenv("OPENAI_API_KEY"):
        logger.error("OPENAI_API_KEY environment variable is required")
        exit(1)
    
    uvicorn.run(
        "python_smart_analysis:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", 8000)),
        reload=False
    )
