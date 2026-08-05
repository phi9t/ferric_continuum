# Challenge: softmax

Row-wise numerically stable softmax over the last dim: shape `(rows x cols)`.

$$
\text{out}[r, c] = \frac{\exp(x[r,c] - \max_j x[r,j])}{\sum_j \exp(x[r,j] - \max_j x[r,j])}
$$

## Contract

Case fields: `rows`, `cols`, `seed`, `rtol`, `atol`. Output shape: `[rows, cols]`.
