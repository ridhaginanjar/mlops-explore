def evaluate_model(MODELS, test_gen):
    loss, acc = MODELS.evaluate(test_gen)
    print(f"Test accuracy: {acc: .4f}")

    return loss, acc