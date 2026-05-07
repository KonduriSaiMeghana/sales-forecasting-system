def select_best_model(results):

    best_model = min(results, key=results.get)

    return best_model