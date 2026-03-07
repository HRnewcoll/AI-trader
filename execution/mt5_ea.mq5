//+------------------------------------------------------------------+
//| ForexAI Expert Advisor — Python Bridge                           |
//| Reads signals from shared named pipe / file written by Python    |
//+------------------------------------------------------------------+
#property copyright "ForexAI 2026"
#property strict

input string SignalFile        = "forex_ai_signal.json";
input double MaxLots           = 1.0;
input int    MagicNumber       = 20260101;
input double MinConfidence     = 0.65;

struct SignalData {
   string pair;
   int    direction;   // 1 buy, -1 sell, 0 close
   double size;
   double stop_loss;
   double take_profit;
   double confidence;
};

//+------------------------------------------------------------------+
int OnInit() {
   EventSetTimer(5);  // Check signal file every 5 seconds
   Print("ForexAI EA initialised. Watching: ", SignalFile);
   return INIT_SUCCEEDED;
}

void OnTimer() {
   string filepath = TerminalInfoString(TERMINAL_DATA_PATH) + "\\MQL5\\Files\\" + SignalFile;
   if (!FileIsExist(SignalFile)) return;

   int fh = FileOpen(SignalFile, FILE_READ | FILE_TXT | FILE_ANSI);
   if (fh == INVALID_HANDLE) return;

   string json = "";
   while (!FileIsEnding(fh))
      json += FileReadString(fh);
   FileClose(fh);
   FileDelete(SignalFile);  // Consume signal

   if (StringLen(json) < 10) return;

   SignalData sig;
   if (!ParseSignal(json, sig)) return;

   if (sig.confidence < MinConfidence) {
      Print("Low confidence ", sig.confidence, " — skipping signal");
      return;
   }

   ExecuteSignal(sig);
}

bool ParseSignal(string json, SignalData &sig) {
   // Simple JSON field extraction
   sig.pair       = ExtractStr(json, "pair");
   sig.direction  = (int)ExtractNum(json, "direction");
   sig.size       = MathMin(ExtractNum(json, "size"), MaxLots);
   sig.stop_loss  = ExtractNum(json, "stop_loss");
   sig.take_profit= ExtractNum(json, "take_profit");
   sig.confidence = ExtractNum(json, "confidence");
   return (StringLen(sig.pair) > 0);
}

void ExecuteSignal(SignalData &sig) {
   ENUM_ORDER_TYPE order_type;
   if (sig.direction == 1)       order_type = ORDER_TYPE_BUY;
   else if (sig.direction == -1) order_type = ORDER_TYPE_SELL;
   else { CloseAll(sig.pair); return; }

   MqlTradeRequest req  = {};
   MqlTradeResult  res  = {};
   req.action    = TRADE_ACTION_DEAL;
   req.symbol    = sig.pair;
   req.volume    = sig.size;
   req.type      = order_type;
   req.price     = (order_type == ORDER_TYPE_BUY)
                    ? SymbolInfoDouble(sig.pair, SYMBOL_ASK)
                    : SymbolInfoDouble(sig.pair, SYMBOL_BID);
   req.sl        = sig.stop_loss;
   req.tp        = sig.take_profit;
   req.magic     = MagicNumber;
   req.comment   = "ForexAI";

   if (!OrderSend(req, res))
      Print("OrderSend failed: ", GetLastError());
   else
      Print("Order executed: ", sig.pair, " dir:", sig.direction,
            " lots:", sig.size, " retcode:", res.retcode);
}

void CloseAll(string symbol) {
   for (int i = PositionsTotal() - 1; i >= 0; i--) {
      if (PositionGetSymbol(i) == symbol && PositionGetInteger(POSITION_MAGIC) == MagicNumber) {
         MqlTradeRequest req = {};
         MqlTradeResult  res = {};
         req.action   = TRADE_ACTION_DEAL;
         req.symbol   = symbol;
         req.volume   = PositionGetDouble(POSITION_VOLUME);
         req.type     = (PositionGetInteger(POSITION_TYPE) == POSITION_TYPE_BUY)
                         ? ORDER_TYPE_SELL : ORDER_TYPE_BUY;
         req.price    = (req.type == ORDER_TYPE_SELL)
                         ? SymbolInfoDouble(symbol, SYMBOL_BID)
                         : SymbolInfoDouble(symbol, SYMBOL_ASK);
         req.magic    = MagicNumber;
         req.comment  = "ForexAI_close";
         OrderSend(req, res);
      }
   }
}

double ExtractNum(string json, string key) {
   string search = "\"" + key + "\":";
   int pos = StringFind(json, search);
   if (pos < 0) return 0.0;
   pos += StringLen(search);
   while (pos < StringLen(json) && (json[pos] == ' ' || json[pos] == '"')) pos++;
   string val = "";
   while (pos < StringLen(json) && json[pos] != ',' && json[pos] != '}' && json[pos] != '"') {
      val += CharToString(json[pos]);
      pos++;
   }
   return StringToDouble(val);
}

string ExtractStr(string json, string key) {
   string search = "\"" + key + "\":\"";
   int pos = StringFind(json, search);
   if (pos < 0) return "";
   pos += StringLen(search);
   string val = "";
   while (pos < StringLen(json) && json[pos] != '"') {
      val += CharToString(json[pos]);
      pos++;
   }
   return val;
}

void OnDeinit(const int reason) {
   EventKillTimer();
   Print("ForexAI EA stopped. Reason: ", reason);
}
//+------------------------------------------------------------------+
