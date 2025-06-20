import os
from dotenv import load_dotenv
from phi.agent import Agent
from phi.model.groq import Groq
from tavily import TavilyClient
from phi.tools.duckduckgo import DuckDuckGo
import yfinance as yf
import json
import pandas as pd # Import pandas for data manipulation

# Load environment variables
load_dotenv()

# Initialize the Groq LLM and Tavily client
model = Groq(id="llama3-70b-8192")
tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

# Custom tool function to perform Tavily search
def tavily_search(query: str, max_results: int = 5) -> str:
    """
    Performs a web search using Tavily and returns the search results as a JSON string.
    Use this tool to find up-to-date information, news, or general data.
    Args:
        query (str): The search query.
        max_results (int): The maximum number of search results to return (default is 5).
    """
    try:
        response = tavily_client.search(query=query, search_depth="advanced", max_results=max_results)
        return json.dumps(response, indent=2)
    except Exception as e:
        return f"Error performing search: {str(e)}"

# --- New yfinance wrapper functions ---

def get_stock_info(ticker_symbol: str) -> str:
    """
    Retrieves general information about a stock using its ticker symbol.
    Returns a JSON string containing key information like longName, sector, industry, marketCap, etc.
    Args:
        ticker_symbol (str): The stock ticker symbol (e.g., "AAPL", "MSFT").
    """
    try:
        ticker = yf.Ticker(ticker_symbol)
        info = ticker.info
        # Filter for relevant info to avoid overwhelming the LLM
        relevant_info = {
            "longName": info.get("longName"),
            "sector": info.get("sector"),
            "industry": info.get("industry"),
            "marketCap": info.get("marketCap"),
            "previousClose": info.get("previousClose"),
            "fiftyTwoWeekLow": info.get("fiftyTwoWeekLow"),
            "fiftyTwoWeekHigh": info.get("fiftyTwoWeekHigh"),
            "beta": info.get("beta"),
            "trailingPE": info.get("trailingPE"),
            "forwardPE": info.get("forwardPE"),
            "returnOnEquity": info.get("returnOnEquity"),
            "debtToEquity": info.get("debtToEquity"),
            "profitMargins": info.get("profitMargins"),
            "revenueGrowth": info.get("revenueGrowth"),
            "dividendYield": info.get("dividendYield"),
            "recommendationKey": info.get("recommendationKey")
        }
        return json.dumps(relevant_info, indent=2)
    except Exception as e:
        return f"Error getting stock info for {ticker_symbol}: {str(e)}. Please check the ticker symbol."

def get_historical_stock_data(ticker_symbol: str, period: str = "1y") -> str:
    """
    Retrieves historical stock data (Open, High, Low, Close, Volume) for a given ticker symbol and period.
    Valid periods: 1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max.
    Returns a JSON string representation of the historical data (last few rows).
    Args:
        ticker_symbol (str): The stock ticker symbol (e.g., "AAPL", "MSFT").
        period (str): The period for which to fetch data (e.g., "1y" for one year).
    """
    try:
        ticker = yf.Ticker(ticker_symbol)
        # Fetch up to 100 rows to keep it manageable for LLM
        data = ticker.history(period=period)
        if data.empty:
            return f"No historical data found for {ticker_symbol} for the period {period}."
        # Return only a relevant summary or head/tail to avoid large outputs
        return data.tail().to_json(indent=2) # Return last 5 rows as JSON
    except Exception as e:
        return f"Error getting historical data for {ticker_symbol}: {str(e)}. Please check the ticker symbol or period."

# -------------------------
# AGENTS SETUP (with token limits)
# -------------------------

# Agent 1: Research Agent
research_agent = Agent(
    name="ResearchAgent",
    model=model,
    tools=[tavily_search, DuckDuckGo()],
    instructions=[
        "You are an investment research agent. Collect concise and relevant financial data using Tavily and DuckDuckGo.",
        "Summarize findings clearly within 1000 tokens. Include only key updates on market trends, stock news, and fundamentals.",
        "Avoid long lists or deep history. Include only top 3 relevant sources and cite them briefly."
    ],
    show_tool_calls=True,
    markdown=True
)

# Agent 2: Finance Agent
finance_agent = Agent(
    name="FinanceAgent",
    model=model,
    # Pass the new yfinance wrapper functions as tools
    tools=[get_stock_info, get_historical_stock_data],
    instructions=[
        "You are a financial analyst. Use the available stock tools to fetch real-time and historical financial data.",
        "When asked for stock information, first use 'get_stock_info' with the ticker symbol.",
        "When asked for historical prices or trends, use 'get_historical_stock_data' with the ticker and appropriate period (e.g., '1y', '3mo').",
        "Summarize key metrics such as P/E, ROE, debt, and margins based on the data you retrieve.",
        "Keep the output short, clear, and under 1000 tokens. Explain only the most significant 3-5 metrics in simple terms.",
        "Avoid repetition or detailed background data unless critical."
    ],
    show_tool_calls=True, # Good to see tool calls for debugging
    markdown=True
)

# Agent 3: Analysis Agent
analysis_agent = Agent(
    name="AnalysisAgent",
    model=model,
    instructions=[
        "You are an investment analyst. Based on financial interpretation, provide a concise Buy, Hold, or Sell recommendation.",
        "Justify your answer with 2-3 solid points. Keep the answer under 1000 tokens and avoid excessive explanation.",
        "Summarize investment risks and opportunities briefly."
    ],
    markdown=True
)

# Agent 4: Editor Agent
editor_agent = Agent(
    name="EditorAgent",
    model=model,
    instructions=[
        "You are a financial editor. Combine all agent outputs into a single professional summary.",
        "Ensure total report stays under 1500 tokens. Use headings, markdown formatting, and avoid repetition.",
        "Keep the language clear, engaging, and suitable for a decision-making investor."
    ],
    markdown=True
)

# -------------------------
# MULTI-STAGE FUNCTION
# -------------------------

def investment_advisor_ai(user_prompt):
    print("\n🔍 Step 1: Research Agent Running...")
    # The research agent might need to identify the ticker from the prompt first
    # For now, let's assume the user_prompt directly implies a search
    research_output = research_agent.run(user_prompt)
    print(f"Research Output: {research_output.content[:200]}...") # Print a snippet for debugging

    # Extract potential ticker from the user_prompt for the FinanceAgent
    # This is a very basic extraction; a more robust solution would be needed
    # For a real agent, the ResearchAgent or an initial parsing step would extract this.
    import re
    match = re.search(r'\b[A-Z]{1,5}\b', user_prompt) # Simple regex for a possible ticker
    ticker_for_finance = match.group(0) if match else None

    if not ticker_for_finance:
        print("Could not identify a ticker symbol from the prompt for the Finance Agent. Skipping Finance and Analysis steps.")
        # Proceed with only research and editor if no ticker
        finance_output_content = "No specific ticker identified for financial analysis."
        analysis_output_content = "No financial data analyzed due to missing ticker."
    else:
        print(f"\n📊 Step 2: Finance Agent Running for {ticker_for_finance}...")
        # The finance agent needs to *know* what ticker to look for.
        # It's better if the prompt to the finance agent explicitly asks for data for a ticker.
        # Here, we combine the research context with a direct instruction for the finance agent.
        finance_prompt = (
            f"Based on the following context about {ticker_for_finance}: \n"
            f"{research_output.content}\n\n"
            f"Now, use your tools to get key financial metrics and information for {ticker_for_finance} "
            f"such as P/E, ROE, debt, margins, and recent stock performance. Summarize them concisely."
        )
        finance_output = finance_agent.run(finance_prompt)
        finance_output_content = finance_output.content
        print(f"Finance Output: {finance_output_content[:200]}...") # Print a snippet for debugging

        print("\n📈 Step 3: Analysis Agent Running...")
        analysis_prompt = (
            f"Based on the financial interpretation provided:\n"
            f"{finance_output_content}\n\n"
            f"And the initial research:\n"
            f"{research_output.content}\n\n"
            f"Provide a concise Buy, Hold, or Sell recommendation for {ticker_for_finance}, "
            f"justifying with 2-3 solid points, investment risks, and opportunities."
        )
        analysis_output = analysis_agent.run(analysis_prompt)
        analysis_output_content = analysis_output.content
        print(f"Analysis Output: {analysis_output_content[:200]}...") # Print a snippet for debugging


    print("\n📝 Step 4: Editor Agent Compiling Final Report...")
    final_report = editor_agent.run(f"""
    Research Summary:
    {research_output.content}

    Financial Interpretation:
    {finance_output_content}

    Investment Analysis:
    {analysis_output_content}
    """)

    return final_report.content

# -------------------------
# CLI LOOP
# -------------------------

if __name__ == "__main__":
    print("💼 Welcome to Investment Analyst AI (Tavily Edition)")
    print("Ask about any company or stock to get a full analysis. Type 'exit' to quit.\n")

    while True:
        user_input = input("📈 You: ")
        if user_input.lower() in ["exit", "quit"]:
            print("👋 Exiting... Happy Investing!")
            break
        report = investment_advisor_ai(user_input)
        print("\n📑 Final Investment Report:\n")
        print(report)