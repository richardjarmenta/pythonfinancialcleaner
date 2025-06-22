"""
Smart Analysis API - FastAPI service for analyzing business data with AI-Enhanced Analysis
IMPROVED VERSION - Uses OpenAI for intelligent SQL generation and answer formatting
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
    description="Analyze tabular business data using natural language questions with AI-Enhanced Analysis",
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
    """Response model for the smart analysis endpoint"""
    answer: str = Field(..., description="Direct answer to the business question")
    explanation: str = Field(..., description="Explanation of the analysis approach")
    table: Optional[str] = Field(None, description="Formatted table data if relevant")
    sql_query: Optional[str] = Field(None, description="SQL query used for analysis")
    confidence_score: float = Field(..., description="Confidence score between 0-1")
    analysis_type: str = Field(..., description="Type of analysis performed")
    suggested_followup: Optional[str] = Field(None, description="Suggested follow-up questions")
    raw_insights: List[str] = Field(default_factory=list, description="Raw insights from analysis")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

# ================================
# AI-ENHANCED ANALYSIS FUNCTIONS
# ================================

def clean_and_normalize_data(rows: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Clean and normalize the input data for analysis
    """
    if not rows:
        return pd.DataFrame()
    
    df = pd.DataFrame(rows)
    
    # Clean and convert columns
    for col in df.columns:
        if df[col].dtype == 'object':
            # Remove currency symbols and convert to numeric if possible
            df[col] = df[col].astype(str).str.replace(r'[$,]', '', regex=True)
            
            # Try to convert to numeric
            numeric_series = pd.to_numeric(df[col], errors='coerce')
            if not numeric_series.isna().all():
                # If most values converted successfully, use numeric
                if numeric_series.notna().sum() / len(df) > 0.7:
                    df[col] = numeric_series
                else:
                    # Keep as string but clean whitespace
                    df[col] = df[col].str.strip()
            else:
                # Keep as string but clean whitespace
                df[col] = df[col].str.strip()
    
    return df

def execute_pandas_correlation(df: pd.DataFrame, question: str) -> Optional[Dict[str, Any]]:
    """
    Handle correlation analysis with proper type safety
    """
    try:
        question_lower = question.lower()
        
        # ONLY handle correlation - nothing else
        if 'correlation' in question_lower and 'between' in question_lower:
            # Get ONLY numeric columns - this prevents type errors
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            
            if len(numeric_cols) >= 2:
                # SAFE: Use simple approach - first two numeric columns (excluding IDs)
                non_id_numeric = [col for col in numeric_cols if 'id' not in col.lower()]
                
                if len(non_id_numeric) >= 2:
                    col1, col2 = non_id_numeric[0], non_id_numeric[1]
                else:
                    col1, col2 = numeric_cols[0], numeric_cols[1]
                
                # SAFE: Ensure both columns exist and have valid numeric data
                if col1 in df.columns and col2 in df.columns:
                    # Drop NaN values for correlation calculation
                    clean_data = df[[col1, col2]].dropna()
                    
                    if len(clean_data) >= 2:  # Need at least 2 points for correlation
                        correlation = clean_data[col1].corr(clean_data[col2])
                        
                        # Handle NaN correlation result
                        if pd.isna(correlation):
                            correlation = 0.0
                        
                        return {
                            "answer": f"The correlation between {col1} and {col2} is {correlation:.3f}",
                            "explanation": f"Calculated Pearson correlation coefficient between {col1} and {col2}.",
                            "table": clean_data[[col1, col2]].corr().round(3).to_string(),
                            "raw_insights": [f"Correlation: {correlation:.3f}"]
                        }
        
        # Return None for everything else so other logic can take over
        return None
        
    except Exception as e:
        logger.error(f"Error in pandas correlation: {str(e)}")
        return None

def ai_analyze_question_intent(question: str, schema: List[str]) -> Dict[str, Any]:
    """
    Use OpenAI to understand what the question is really asking
    """
    try:
        prompt = f"""Analyze this business question and determine the analysis approach:

Question: "{question}"
Available data columns: {schema}

Return only valid JSON in this exact format:
{{
    "analysis_type": "correlation|multi_part|simple_aggregate|business_logic",
    "is_multi_part": true,
    "requires_correlation": false,
    "key_columns": ["column1", "column2"],
    "business_intent": "brief explanation of what user wants to know"
}}"""

        response = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=200
        )
        
        result = json.loads(response.choices[0].message.content.strip())
        return result
        
    except Exception as e:
        logger.error(f"Error in AI question analysis: {str(e)}")
        # Fallback to basic pattern matching
        return {
            "analysis_type": "simple_aggregate",
            "is_multi_part": " and " in question.lower(),
            "requires_correlation": "correlation" in question.lower(),
            "key_columns": [],
            "business_intent": "Basic analysis request"
        }

def ai_generate_sql_query(question: str, df: pd.DataFrame, schema: List[str]) -> str:
    """
    Use OpenAI to generate contextually appropriate SQL queries
    """
    try:
        sample_data = df.head(2).to_dict('records') if len(df) > 0 else []
        
        prompt = f"""You are a business intelligence SQL expert. Generate a precise SQL query for this question:

Question: "{question}"
Table name: data
Available columns: {list(df.columns)}
Data types: {dict(df.dtypes.astype(str))}
Sample data: {sample_data}

Rules:
1. Table name is always 'data'
2. Use proper aggregations (SUM, AVG, COUNT, MAX, MIN)
3. For "above X" or "more than X" use WHERE column > threshold
4. For status filtering use WHERE status = 'active' or similar
5. For "how many" questions use COUNT(*)
6. For "total" questions use SUM()
7. For "average" questions use AVG()
8. For "which/best/highest" use GROUP BY with ORDER BY
9. Use proper column names exactly as provided
10. Return only the SQL query, no explanations or markdown

SQL Query:"""

        response = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=150
        )
        
        sql_query = response.choices[0].message.content.strip()
        
        # Clean up the response (remove any markdown or extra text)
        sql_query = sql_query.replace("```sql", "").replace("```", "").strip()
        
        return sql_query
        
    except Exception as e:
        logger.error(f"Error in AI SQL generation: {str(e)}")
        # Fallback to basic SQL
        return "SELECT * FROM data LIMIT 10"

def ai_format_answer(question: str, sql_results: List[pd.DataFrame], business_intent: str) -> str:
    """
    Use OpenAI to format results appropriately for business context
    """
    try:
        # Convert DataFrame results to simple dictionaries
        formatted_results = []
        for df in sql_results:
            if not df.empty:
                formatted_results.append(df.to_dict('records'))
        
        prompt = f"""Format this business analysis result professionally for a business user:

Original Question: "{question}"
Business Intent: {business_intent}
SQL Results: {formatted_results}

Instructions:
1. Provide a clear, concise answer that directly addresses the question
2. Use proper formatting:
   - Currency: $1,234 or $1,234,567
   - Counts: "Count: 5" or "5 items"
   - Percentages: "75.5%"
   - Names with values: "John Smith ($50,000)"
3. For multi-part questions, connect answers with periods
4. Be professional but conversational
5. If multiple results, combine them logically
6. Return only the final formatted answer, no explanations

Formatted Answer:"""

        response = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=100
        )
        
        return response.choices[0].message.content.strip()
        
    except Exception as e:
        logger.error(f"Error in AI answer formatting: {str(e)}")
        # Fallback formatting
        if sql_results and not sql_results[0].empty:
            first_result = sql_results[0].iloc[0, 0]
            return f"Result: {first_result}"
        return "Analysis completed."

def execute_sql_query(sql: str, df: pd.DataFrame) -> pd.DataFrame:
    """
    Execute SQL query against DataFrame using SQLite
    """
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.db') as tmp_file:
            conn = sqlite3.connect(tmp_file.name)
            
            # Convert DataFrame to SQL table
            df.to_sql('data', conn, index=False, if_exists='replace')
            
            # Execute query
            result_df = pd.read_sql_query(sql, conn)
            
            conn.close()
            os.unlink(tmp_file.name)
            
            return result_df
            
    except Exception as e:
        logger.error(f"Error executing SQL query: {sql} - {str(e)}")
        return pd.DataFrame()

def decompose_complex_question(question: str) -> List[str]:
    """
    Break down complex questions into simpler parts
    """
    # Split by question marks first
    parts = [q.strip() + '?' for q in question.split('?') if q.strip()]
    
    if len(parts) <= 1:
        # Split by 'and' if no multiple question marks
        if ' and ' in question.lower():
            conjunctions = question.lower().split(' and ')
            if len(conjunctions) > 1:
                parts = []
                for i, part in enumerate(conjunctions):
                    if i == 0:
                        parts.append(part.strip() + '?')
                    else:
                        # For subsequent parts, may need to infer the question structure
                        if not any(word in part for word in ['what', 'how', 'which', 'who']):
                            parts.append(f"How many {part.strip()}?")
                        else:
                            parts.append(part.strip() + '?')
    
    return parts if len(parts) > 1 else [question]

def analyze_with_ai_enhanced_approach(
    question: str, 
    df: pd.DataFrame, 
    file_title: Optional[str] = None,
    schema: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Main AI-Enhanced analysis function
    """
    try:
        # Step 1: Analyze question intent with AI
        intent = ai_analyze_question_intent(question, schema or list(df.columns))
        logger.info(f"AI Analysis Intent: {intent}")
        
        # Step 2: Handle correlation (keep existing perfect logic)
        if intent["requires_correlation"]:
            pandas_result = execute_pandas_correlation(df, question)
            if pandas_result:
                logger.info("Using pandas for correlation analysis")
                return {
                    **pandas_result,
                    "sql_query": None,
                    "confidence_score": 0.95,
                    "analysis_type": "correlation",
                    "suggested_followup": None
                }
        
        # Step 3: Handle multi-part questions
        if intent["is_multi_part"]:
            sub_questions = decompose_complex_question(question)
            logger.info(f"Decomposed question into: {sub_questions}")
            
            sql_results = []
            executed_queries = []
            
            for sub_q in sub_questions:
                # Generate SQL with AI
                sql = ai_generate_sql_query(sub_q, df, schema or list(df.columns))
                logger.info(f"Generated SQL for '{sub_q}': {sql}")
                
                # Execute SQL
                result = execute_sql_query(sql, df)
                
                if not result.empty:
                    sql_results.append(result)
                    executed_queries.append(sql)
            
            # Format answer with AI
            if sql_results:
                final_answer = ai_format_answer(question, sql_results, intent["business_intent"])
            else:
                final_answer = "Unable to process the multi-part question."
            
            return {
                "answer": final_answer,
                "explanation": "Multi-part analysis completed using AI-generated SQL queries.",
                "table": "\n\n---\n\n".join([df.to_string(index=False) for df in sql_results[:3]]) if sql_results else None,
                "sql_query": " | ".join(executed_queries),
                "confidence_score": 0.88,
                "analysis_type": "multi_part_ai",
                "suggested_followup": None,
                "raw_insights": [f"AI processed {len(sub_questions)} sub-questions"]
            }
        
        # Step 4: Handle simple questions
        else:
            # Generate SQL with AI
            sql = ai_generate_sql_query(question, df, schema or list(df.columns))
            logger.info(f"Generated SQL: {sql}")
            
            # Execute SQL
            result = execute_sql_query(sql, df)
            
            if not result.empty:
                # Format answer with AI
                final_answer = ai_format_answer(question, [result], intent["business_intent"])
            else:
                final_answer = "No results found for the query."
            
            return {
                "answer": final_answer,
                "explanation": "Single SQL query generated and executed using AI.",
                "table": result.to_string(index=False) if len(result) <= 20 else None,
                "sql_query": sql,
                "confidence_score": 0.85,
                "analysis_type": "single_query_ai",
                "suggested_followup": None,
                "raw_insights": ["AI-generated analysis"] if not result.empty else ["No data returned"]
            }
        
    except Exception as e:
        logger.error(f"Error in AI-enhanced analysis: {str(e)}")
        return {
            "answer": f"Analysis encountered an error: {str(e)}",
            "explanation": "Error occurred during AI-enhanced data analysis.",
            "table": None,
            "sql_query": None,
            "confidence_score": 0.1,
            "analysis_type": "error",
            "suggested_followup": None,
            "raw_insights": ["Analysis error occurred"]
        }

def analyze_data_file(
    rows: List[Dict[str, Any]], 
    question: str, 
    file_title: Optional[str] = None,
    schema: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Main entry point for data analysis
    """
    try:
        # Clean and normalize the data
        df = clean_and_normalize_data(rows)
        
        if df.empty:
            return {
                "answer": "No data available for analysis.",
                "explanation": "The provided dataset is empty.",
                "table": None,
                "sql_query": None,
                "confidence_score": 0.0,
                "analysis_type": "no_data",
                "suggested_followup": None,
                "raw_insights": ["Empty dataset"]
            }
        
        # Use AI-enhanced analysis
        result = analyze_with_ai_enhanced_approach(question, df, file_title, schema)
        
        # Add metadata
        result["metadata"] = {
            "total_rows": len(df),
            "columns": list(df.columns),
            "file_title": file_title,
            "data_types": df.dtypes.astype(str).to_dict(),
            "analysis_method": "ai_enhanced_v1"
        }
        
        return result
        
    except Exception as e:
        logger.error(f"Error in analyze_data_file: {str(e)}")
        return {
            "answer": f"Analysis failed: {str(e)}",
            "explanation": "An error occurred during data analysis.",
            "table": None,
            "sql_query": None,
            "confidence_score": 0.0,
            "analysis_type": "error",
            "suggested_followup": None,
            "raw_insights": ["Fatal error occurred"],
            "metadata": {"error": str(e)}
        }

# ================================
# API ENDPOINTS
# ================================

@app.post("/smart-analysis", response_model=AnalysisResponse)
async def analyze_data(request: AnalysisRequest) -> AnalysisResponse:
    """
    Main endpoint for analyzing business data with natural language questions
    """
    try:
        logger.info(f"Received analysis request: {request.question}")
        logger.info(f"Data rows: {len(request.rows)}")
        
        # Perform the analysis using AI-enhanced approach
        result = analyze_data_file(
            rows=request.rows,
            question=request.question,
            file_title=request.file_title,
            schema=request.schema
        )
        
        # Convert to Pydantic response model
        response = AnalysisResponse(**result)
        
        logger.info(f"Analysis completed: {response.analysis_type}")
        return response
        
    except Exception as e:
        logger.error(f"Error in analyze endpoint: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Analysis failed: {str(e)}"
        )

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy", 
        "version": "3.0.0",
        "analysis_engine": "ai_enhanced",
        "openai_configured": bool(os.getenv("OPENAI_API_KEY"))
    }

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Smart Analysis API - AI Enhanced Version",
        "version": "3.0.0",
        "docs": "/docs",
        "health": "/health"
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
