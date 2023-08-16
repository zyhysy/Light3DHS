def weightdecay(epoch, max_epochs, weight_decay=3e-5, exponent=0.9):
    if weight_decay < 1e-3:
        return weight_decay * (1 + epoch / max_epochs)**exponent
    else:
        return 1e-3