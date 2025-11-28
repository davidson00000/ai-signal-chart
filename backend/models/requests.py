"""
API Request models
"""
from pydantic import BaseModel, Field
from typing import Optional, Literal


class PaperOrderRequest(BaseModel):
    """
    Paper order request body
    
    This model supports JSON body input for /paper-order endpoint
    """
    symbol: str = Field(..., description="Symbol to trade (e.g., AAPL, BTC/USDT)")
    side: Literal["BUY", "SELL"] = Field(..., description="Order side: BUY or SELL")
    quantity: int = Field(..., gt=0, description="Number of shares/units to trade")
    price: Optional[float] = Field(None, description="Limit price (optional, uses market if None)")
    signal_id: Optional[str] = Field(None, description="Associated signal ID")
    order_time: Optional[str] = Field(None, description="Order timestamp (ISO format)")
    mode: str = Field("market", description="Order mode: market or limit")
    
    class Config:
        json_schema_extra = {
            "example": {
                "symbol": "AAPL",
                "side": "BUY",
                "quantity": 10,
                "price": None,
                "mode": "market"
            }
        }
