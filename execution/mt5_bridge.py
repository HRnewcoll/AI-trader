"""
MT5 execution bridge — Python ↔ MetaTrader5.
Falls back to paper trading if MT5 unavailable.
Uses MetaApi cloud bridge as secondary option.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger(__name__)

try:
    import MetaTrader5 as mt5
    _HAS_MT5 = True
except ImportError:
    _HAS_MT5 = False
    logger.info("MetaTrader5 not available — paper trading mode")


@dataclass
class OrderResult:
    order_id: int
    pair: str
    direction: str
    price: float
    size_lots: float
    stop_loss: float
    take_profit: float
    timestamp: str
    mode: str       # "live" | "paper"
    error: str = ""
    success: bool = True


class PaperBroker:
    """
    Simulated broker for paper trading / backtesting.
    Tracks positions and PnL without real money.
    """

    def __init__(self, initial_equity: float = 10_000.0, spread_pips: float = 1.5):
        self.equity = initial_equity
        self.spread_pips = spread_pips
        self.positions: dict[str, dict] = {}
        self.order_counter = 1
        self.trade_log: list[dict] = []

    def place_order(
        self,
        pair: str,
        direction: str,
        size_lots: float,
        price: float,
        stop_loss: float,
        take_profit: float,
    ) -> OrderResult:
        pip = 0.0001 if "JPY" not in pair.upper() else 0.01
        fill_price = price + (self.spread_pips * pip * (1 if direction == "buy" else -1))

        order_id = self.order_counter
        self.order_counter += 1

        self.positions[pair] = {
            "order_id": order_id,
            "direction": direction,
            "size_lots": size_lots,
            "entry_price": fill_price,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "opened_at": datetime.now(tz=timezone.utc).isoformat(),
        }

        logger.info(
            "[PAPER] %s %s %.2f lots @ %.5f SL:%.5f TP:%.5f",
            direction.upper(), pair, size_lots, fill_price, stop_loss, take_profit,
        )

        return OrderResult(
            order_id=order_id,
            pair=pair,
            direction=direction,
            price=fill_price,
            size_lots=size_lots,
            stop_loss=stop_loss,
            take_profit=take_profit,
            timestamp=datetime.now(tz=timezone.utc).isoformat(),
            mode="paper",
        )

    def close_position(self, pair: str, current_price: float) -> dict:
        if pair not in self.positions:
            return {"error": f"No position for {pair}"}

        pos = self.positions.pop(pair)
        pip = 0.0001 if "JPY" not in pair.upper() else 0.01
        direction_mult = 1 if pos["direction"] == "buy" else -1
        pips_gained = (current_price - pos["entry_price"]) * direction_mult / pip
        pnl = pips_gained * pos["size_lots"] * 10.0  # ~$10/pip per lot

        self.equity += pnl
        result = {**pos, "exit_price": current_price, "pnl": round(pnl, 2)}
        self.trade_log.append(result)
        logger.info("[PAPER] CLOSE %s PnL: %+.2f (%.1f pips)", pair, pnl, pips_gained)
        return result

    def update_prices(self, prices: dict[str, float]) -> list[dict]:
        """Check SL/TP for all open positions."""
        closed = []
        for pair, price in prices.items():
            if pair not in self.positions:
                continue
            pos = self.positions[pair]
            direction = pos["direction"]
            hit_sl = (direction == "buy" and price <= pos["stop_loss"]) or \
                     (direction == "sell" and price >= pos["stop_loss"])
            hit_tp = (direction == "buy" and price >= pos["take_profit"]) or \
                     (direction == "sell" and price <= pos["take_profit"])
            if hit_sl or hit_tp:
                reason = "SL" if hit_sl else "TP"
                result = self.close_position(pair, price)
                result["reason"] = reason
                closed.append(result)
        return closed


class MT5Bridge:
    """
    MT5 execution bridge.
    Automatically falls back to PaperBroker if MT5 unavailable.
    """

    def __init__(
        self,
        login: int = 0,
        password: str = "",
        server: str = "",
        mode: str = "paper",
        initial_equity: float = 10_000.0,
    ):
        self.mode = mode
        self._paper = PaperBroker(initial_equity=initial_equity)
        self._connected = False

        if mode == "live" and _HAS_MT5:
            self._connected = mt5.initialize(
                login=login,
                password=password,
                server=server,
            )
            if self._connected:
                info = mt5.account_info()
                logger.info(
                    "MT5 connected: account %s, balance %.2f %s",
                    info.login, info.balance, info.currency,
                )
            else:
                logger.error("MT5 init failed: %s — switching to paper mode", mt5.last_error())
                self.mode = "paper"
        elif mode == "live" and not _HAS_MT5:
            logger.warning("MT5 not available — switching to paper mode")
            self.mode = "paper"

    def place_order(
        self,
        pair: str,
        direction: str,
        size_lots: float,
        price: float,
        stop_loss: float,
        take_profit: float,
    ) -> OrderResult:
        if self.mode == "paper" or not self._connected:
            return self._paper.place_order(pair, direction, size_lots, price, stop_loss, take_profit)

        # Live MT5 order
        action_map = {"buy": mt5.ORDER_TYPE_BUY, "sell": mt5.ORDER_TYPE_SELL}
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": pair,
            "volume": size_lots,
            "type": action_map[direction],
            "price": price,
            "sl": stop_loss,
            "tp": take_profit,
            "comment": "ForexAI",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        result = mt5.order_send(request)

        if result and result.retcode == mt5.TRADE_RETCODE_DONE:
            return OrderResult(
                order_id=result.order,
                pair=pair,
                direction=direction,
                price=result.price,
                size_lots=size_lots,
                stop_loss=stop_loss,
                take_profit=take_profit,
                timestamp=datetime.now(tz=timezone.utc).isoformat(),
                mode="live",
            )
        else:
            error = str(result.retcode) if result else "unknown"
            logger.error("MT5 order failed: %s", error)
            return OrderResult(
                order_id=0, pair=pair, direction=direction, price=price,
                size_lots=size_lots, stop_loss=stop_loss, take_profit=take_profit,
                timestamp=datetime.now(tz=timezone.utc).isoformat(),
                mode="live", error=error, success=False,
            )

    def get_account_equity(self) -> float:
        if self.mode == "live" and self._connected:
            info = mt5.account_info()
            return info.equity if info else 0.0
        return self._paper.equity

    def shutdown(self) -> None:
        if _HAS_MT5 and self._connected:
            mt5.shutdown()
