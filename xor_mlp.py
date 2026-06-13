import math
import random

# ============================================================
#  Utils
# ============================================================

def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))

def mse(predictions, targets):
    return sum((p - t) ** 2 for p, t in zip(predictions, targets)) / len(predictions)

# ============================================================
#  XOR dataset
# ============================================================

X = [[0, 0], [0, 1], [1, 0], [1, 1]]
y = [0, 1, 1, 0]

# ============================================================
#  Architecture:  2 inputs -> 2 hidden (sigmoid) -> 1 output (sigmoid)
#
#    x0 ─┬─ w1[0][0] ─┬─ h0 ─ w2[0] ─┬─ out
#    x1 ─┴─ w1[1][0] ─┘              │
#         └─ w1[0][1] ─┬─ h1 ─ w2[1] ─┘
#            w1[1][1] ─┘
#
#  W1[i][j]  weight from input i to hidden neuron j
#  W2[j]     weight from hidden neuron j to output
# ============================================================

random.seed(42)

def _w():
    return random.uniform(-1.0, 1.0)

W1 = [[_w(), _w()],   # W1[input_i][hidden_j]
      [_w(), _w()]]
b1 = [_w(), _w()]     # bias for each hidden neuron

W2 = [_w(), _w()]     # W2[hidden_j]
b2 = _w()             # bias for output neuron

# ============================================================
#  Forward pass
# ============================================================

def forward(x):
    # hidden pre-activation:  z1[j] = Σ_i x[i]*W1[i][j] + b1[j]
    z1 = [sum(x[i] * W1[i][j] for i in range(2)) + b1[j]
          for j in range(2)]
    a1 = [sigmoid(z) for z in z1]           # hidden activations

    # output pre-activation: z2 = Σ_j a1[j]*W2[j] + b2
    z2 = sum(a1[j] * W2[j] for j in range(2)) + b2
    a2 = sigmoid(z2)                         # output activation

    return z1, a1, z2, a2

# ============================================================
#  Backward pass  (chain rule / backprop)
#
#  Loss:  L = (a2 - y)^2   (MSE over one sample)
#
#  dL/dz2  = (a2 - y) * sigmoid'(z2)
#           = (a2 - y) * a2*(1-a2)          ← output delta
#
#  dL/dz1[j] = dL/dz2 * W2[j] * a1[j]*(1-a1[j])  ← hidden delta
# ============================================================

def backward(x, y_true, z1, a1, z2, a2, lr):
    # output delta
    delta2 = (a2 - y_true) * (a2 * (1.0 - a2))

    # hidden deltas  (backprop through W2)
    delta1 = [delta2 * W2[j] * (a1[j] * (1.0 - a1[j]))
              for j in range(2)]

    # --- gradient descent update ---

    # output layer
    for j in range(2):
        W2[j] -= lr * delta2 * a1[j]
    global b2
    b2 -= lr * delta2

    # hidden layer
    for i in range(2):
        for j in range(2):
            W1[i][j] -= lr * delta1[j] * x[i]
    for j in range(2):
        b1[j] -= lr * delta1[j]

# ============================================================
#  Training loop
# ============================================================

LR     = 1.0
EPOCHS = 10_000

print("Training XOR MLP (2-2-1)\n")
print(f"{'Epoch':>7}  {'MSE':>10}")
print("-" * 22)

for epoch in range(EPOCHS):
    preds = []
    for xi, yi in zip(X, y):
        z1, a1, z2, a2 = forward(xi)
        preds.append(a2)
        backward(xi, yi, z1, a1, z2, a2, LR)

    if epoch % 1000 == 0:
        print(f"{epoch:7d}  {mse(preds, y):10.6f}")

# ============================================================
#  Final test
# ============================================================

print("\n--- Test Results ---")
all_correct = True
for xi, yi in zip(X, y):
    _, _, _, pred = forward(xi)
    cls = round(pred)
    ok  = cls == yi
    if not ok:
        all_correct = False
    print(f"  {xi}  ->  {pred:.4f}  (class {cls}, expected {yi})  {'OK' if ok else 'FAIL'}")

final_preds = [forward(xi)[-1] for xi in X]
print(f"\nFinal MSE : {mse(final_preds, y):.6f}")
print("All correct:", all_correct)
