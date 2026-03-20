"""
NIFTY 50 Stock Tickers List

Contains the current NIFTY 50 constituents with their Yahoo Finance symbols.
Updated as of 2024.
"""

NIFTY_50_TICKERS = [
    "RELIANCE.NS",      # Reliance Industries
    "TCS.NS",           # Tata Consultancy Services
    "HDFCBANK.NS",      # HDFC Bank
    "INFY.NS",          # Infosys
    "ICICIBANK.NS",     # ICICI Bank
    "HINDUNILVR.NS",    # Hindustan Unilever
    "ITC.NS",           # ITC Limited
    "SBIN.NS",          # State Bank of India
    "BHARTIARTL.NS",    # Bharti Airtel
    "KOTAKBANK.NS",     # Kotak Mahindra Bank
    "LT.NS",            # Larsen & Toubro
    "AXISBANK.NS",      # Axis Bank
    "ASIANPAINT.NS",    # Asian Paints
    "MARUTI.NS",        # Maruti Suzuki
    "SUNPHARMA.NS",     # Sun Pharmaceutical
    "TITAN.NS",         # Titan Company
    "BAJFINANCE.NS",    # Bajaj Finance
    "ULTRACEMCO.NS",    # UltraTech Cement
    "NESTLEIND.NS",     # Nestle India
    "WIPRO.NS",         # Wipro
    "HCLTECH.NS",       # HCL Technologies
    "TECHM.NS",         # Tech Mahindra
    "POWERGRID.NS",     # Power Grid Corporation
    "NTPC.NS",          # NTPC Limited
    "M&M.NS",           # Mahindra & Mahindra
    "TATAMOTORS.NS",    # Tata Motors
    "TATASTEEL.NS",     # Tata Steel
    "ONGC.NS",          # Oil & Natural Gas Corporation
    "COALINDIA.NS",     # Coal India
    "BAJAJFINSV.NS",    # Bajaj Finserv
    "ADANIPORTS.NS",    # Adani Ports
    "DIVISLAB.NS",      # Divi's Laboratories
    "BAJAJ-AUTO.NS",    # Bajaj Auto
    "DRREDDY.NS",       # Dr. Reddy's Laboratories
    "APOLLOHOSP.NS",    # Apollo Hospitals
    "BRITANNIA.NS",     # Britannia Industries
    "CIPLA.NS",         # Cipla
    "EICHERMOT.NS",     # Eicher Motors
    "GRASIM.NS",        # Grasim Industries
    "HEROMOTOCO.NS",    # Hero MotoCorp
    "HINDALCO.NS",      # Hindalco Industries
    "INDUSINDBK.NS",    # IndusInd Bank
    "JSWSTEEL.NS",      # JSW Steel
    "SBILIFE.NS",       # SBI Life Insurance
    "SHRIRAMFIN.NS",    # Shriram Finance
    "TATACONSUM.NS",    # Tata Consumer Products
    "ADANIENT.NS",      # Adani Enterprises
    "LTIM.NS",          # LTIMindtree
    "BEL.NS",           # Bharat Electronics
    "BPCL.NS"           # Bharat Petroleum
]


def get_nifty50_tickers():
    """
    Get list of NIFTY 50 ticker symbols for Yahoo Finance.
    
    Returns
    -------
    list of str
        List of Yahoo Finance ticker symbols for NIFTY 50 stocks.
    """
    return NIFTY_50_TICKERS.copy()


def get_ticker_name(ticker_symbol):
    """
    Get company name from ticker symbol.
    
    Parameters
    ----------
    ticker_symbol : str
        Yahoo Finance ticker symbol (e.g., 'RELIANCE.NS')
        
    Returns
    -------
    str
        Company name or ticker if not found.
    """
    names = {
        "RELIANCE.NS": "Reliance Industries",
        "TCS.NS": "TCS",
        "HDFCBANK.NS": "HDFC Bank",
        "INFY.NS": "Infosys",
        "ICICIBANK.NS": "ICICI Bank",
        "HINDUNILVR.NS": "Hindustan Unilever",
        "ITC.NS": "ITC",
        "SBIN.NS": "SBI",
        "BHARTIARTL.NS": "Bharti Airtel",
        "KOTAKBANK.NS": "Kotak Bank",
        "LT.NS": "L&T",
        "AXISBANK.NS": "Axis Bank",
        "ASIANPAINT.NS": "Asian Paints",
        "MARUTI.NS": "Maruti Suzuki",
        "SUNPHARMA.NS": "Sun Pharma",
        "TITAN.NS": "Titan",
        "BAJFINANCE.NS": "Bajaj Finance",
        "ULTRACEMCO.NS": "UltraTech Cement",
        "NESTLEIND.NS": "Nestle India",
        "WIPRO.NS": "Wipro",
        "HCLTECH.NS": "HCL Tech",
        "TECHM.NS": "Tech Mahindra",
        "POWERGRID.NS": "Power Grid",
        "NTPC.NS": "NTPC",
        "M&M.NS": "M&M",
        "TATAMOTORS.NS": "Tata Motors",
        "TATASTEEL.NS": "Tata Steel",
        "ONGC.NS": "ONGC",
        "COALINDIA.NS": "Coal India",
        "BAJAJFINSV.NS": "Bajaj Finserv",
        "ADANIPORTS.NS": "Adani Ports",
        "DIVISLAB.NS": "Divi's Lab",
        "BAJAJ-AUTO.NS": "Bajaj Auto",
        "DRREDDY.NS": "Dr. Reddy's",
        "APOLLOHOSP.NS": "Apollo Hospitals",
        "BRITANNIA.NS": "Britannia",
        "CIPLA.NS": "Cipla",
        "EICHERMOT.NS": "Eicher Motors",
        "GRASIM.NS": "Grasim",
        "HEROMOTOCO.NS": "Hero MotoCorp",
        "HINDALCO.NS": "Hindalco",
        "INDUSINDBK.NS": "IndusInd Bank",
        "JSWSTEEL.NS": "JSW Steel",
        "SBILIFE.NS": "SBI Life",
        "SHRIRAMFIN.NS": "Shriram Finance",
        "TATACONSUM.NS": "Tata Consumer",
        "ADANIENT.NS": "Adani Enterprises",
        "LTIM.NS": "LTIMindtree",
        "BEL.NS": "BEL",
        "BPCL.NS": "BPCL"
    }
    return names.get(ticker_symbol, ticker_symbol.replace(".NS", ""))
