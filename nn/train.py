"""Mini-batch training loop, evaluation, and history."""
import time


def batches(x, y, batch_size, rng):
    """Yield shuffled mini-batches using the data-order RNG stream. Covers every sample once."""
    order = rng.permutation(len(x))
    for i in range(0, len(order), batch_size):
        b = order[i:i + batch_size]
        yield x[b], y[b]


def evaluate(model, loss_fn, x, y, chunk=5000):
    """Mean loss and accuracy in eval mode, computed in chunks to keep memory low."""
    total_loss, correct = 0.0, 0
    for i in range(0, len(x), chunk):
        logits = model.forward(x[i:i + chunk], train=False)
        yc = y[i:i + chunk]
        total_loss += float(loss_fn.forward(logits, yc)) * len(yc)
        correct += int((logits.argmax(axis=1) == yc).sum())
    return total_loss / len(x), correct / len(x)


def predict(model, x, chunk=5000):
    """Predicted class ids in eval mode (used later for bootstrap CIs)."""
    out = [model.forward(x[i:i + chunk], train=False).argmax(axis=1)
           for i in range(0, len(x), chunk)]
    return np.concatenate(out)


def train(model, loss_fn, optimizer, data, epochs, rngs,
          batch_size=256, lr_schedule=None, augment=None):
    """Train and return a history dict.

    data = (x_train, y_train, x_val, y_val). optimizer needs .lr and .step().
    lr_schedule(epoch) -> lr (optional). augment(xb, rng) -> xb (optional).
    train_loss is the mean of the batch losses seen during the epoch (training mode).
    """
    x_tr, y_tr, x_va, y_va = data
    history = {"train_loss": [], "val_loss": [], "val_acc": [], "epoch_seconds": []}
    for epoch in range(epochs):
        if lr_schedule is not None:
            optimizer.lr = lr_schedule(epoch)
        start = time.perf_counter()
        losses = []
        for xb, yb in batches(x_tr, y_tr, batch_size, rngs["data"]):
            if augment is not None:
                xb = augment(xb, rngs["aug"])
            losses.append(float(loss_fn.forward(model.forward(xb, train=True), yb)))
            model.backward(loss_fn.backward())
            optimizer.step()
        elapsed = time.perf_counter() - start
        val_loss, val_acc = evaluate(model, loss_fn, x_va, y_va)
        history["train_loss"].append(sum(losses) / len(losses))
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["epoch_seconds"].append(elapsed)
    return history