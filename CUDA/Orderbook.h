
#define SIDE_BUY 1
#define SIDE_SELL 2


typedef struct afOrderbookLevel_t {
  int price;
  int asd;
} afOrderbookLevel_t;

typedef struct afOrderbook_t {
  int       last_trade_price;
  time_t    last_trade_time;

} afOrderbook_t;
