import os
from langchain_core.tools import BaseTool
from langchain_community.utilities.alpha_vantage import AlphaVantageAPIWrapper
from dotenv import load_dotenv

load_dotenv()

class AlphaVantageQueryRun(BaseTool):
    """Tool that queries the Alpha Vantage API."""

    name: str = "alpha_vantage"
    description: str = (
        "A wrapper around Alpha Vantage API. "
        "Useful for getting financial information about stocks, "
        "forex, cryptocurrencies, and economic indicators. "
        "Input should be the name of the stock ticker."
    )
    api_wrapper: AlphaVantageAPIWrapper = AlphaVantageAPIWrapper(alphavantage_api_key=os.environ.get("ALPHAVANTAGE_API_KEY"))

    def _run(self, ticker: str) -> str:
        """Use the tool."""
        return self.api_wrapper._get_time_series_daily(ticker)

alpha_vantage_tool = AlphaVantageQueryRun()
