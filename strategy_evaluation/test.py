import pandas as pd

orders_df = pd.DataFrame(columns=["Shares", "Order", "Symbol"])
new_row = pd.DataFrame(
    [[abs(1000), 'sell', 'AAPL'], ],
    columns=["Shares", "Order", "Symbol"],
    index=['2008-01-08', ],
)

orders_df = orders_df._append(new_row)

print(orders_df)