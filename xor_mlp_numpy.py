import numpy as np

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)  # (4, 2)
y = np.array([0, 1, 1, 0], dtype=float)                       # (4,)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# ----------------------------------------------------------------
# ทำไมต้องมี hidden layer?
#   single neuron = เส้นตรง 1 เส้น → แบ่ง XOR ไม่ได้
#   2-2-1 = สร้าง "เส้นตรง 2 เส้น" แล้วรวมกัน → แบ่งได้
# ----------------------------------------------------------------

def forward(x, W1, b1, W2, b2):
    """x: (2,)  →  คืน z1, a1, z2, a2"""
    z1 = W1 @ x + b1          # (2,)   hidden pre-activation
    a1 = sigmoid(z1)           # (2,)   hidden activation
    z2 = W2 @ a1 + b2          # scalar output pre-activation
    a2 = sigmoid(z2)           # scalar output
    return z1, a1, z2, a2

def backward(x, y_true, z1, a1, z2, a2, W2):
    """คืน gradients ทั้งหมด — ไม่ update ใน function นี้"""
    delta2 = (a2 - y_true) * a2 * (1 - a2)    # output delta  (scalar)
    delta1 = delta2 * W2 * a1 * (1 - a1)       # hidden delta  (2,)

    dW2 = delta2 * a1          # (2,)
    db2 = delta2               # scalar
    dW1 = np.outer(delta1, x)  # (2,2)
    db1 = delta1               # (2,)

    return dW1, db1, dW2, db2

def update(W1, b1, W2, b2, dW1, db1, dW2, db2, lr):
    W1 -= lr * dW1
    b1 -= lr * db1
    W2 -= lr * dW2
    b2 -= lr * db2
    return W1, b1, W2, b2

def mse(preds, targets):
    return np.mean((preds - targets) ** 2)

def train(X, y, epochs=10000, lr=1.0, seed=42):
    np.random.seed(seed)
    W1 = np.random.uniform(-1, 1, (2, 2))   # (hidden, input)
    b1 = np.random.uniform(-1, 1, (2,))
    W2 = np.random.uniform(-1, 1, (2,))
    b2 = float(np.random.uniform(-1, 1))

    print("Training XOR MLP (2-2-1)\n")
    print(f"{'Epoch':>7}  {'MSE':>10}")
    print("-" * 22)

    for epoch in range(epochs):
        preds = []
        for xi, yi in zip(X, y):
            z1, a1, z2, a2 = forward(xi, W1, b1, W2, b2)
            preds.append(a2)
            dW1, db1, dW2, db2 = backward(xi, yi, z1, a1, z2, a2, W2)
            W1, b1, W2, b2 = update(W1, b1, W2, b2, dW1, db1, dW2, db2, lr)

        if epoch % 1000 == 0:
            print(f"{epoch:7d}  {mse(np.array(preds), y):10.6f}")

    return W1, b1, W2, b2

W1, b1, W2, b2 = train(X, y)

print("\n--- Test Results ---")
for xi, yi in zip(X, y):
    _, _, _, pred = forward(xi, W1, b1, W2, b2)
    cls = round(pred)
    ok  = cls == int(yi)
    print(f"  {xi.astype(int).tolist()}  ->  {pred:.4f}  (class {cls}, expected {int(yi)})  {'OK' if ok else 'FAIL'}")

final_preds = np.array([forward(xi, W1, b1, W2, b2)[-1] for xi in X])
print(f"\nFinal MSE : {mse(final_preds, y):.6f}")
