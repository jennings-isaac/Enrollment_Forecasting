import torch

class EarlyStopping:
    def __init__(self, patience=10, delta=0, verbose=True, save_path="checkpoint.pth"):
        self.patience = patience
        self.delta = delta
        self.verbose = verbose
        self.save_path = save_path
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, validation_loss, model):
        score = -validation_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(validation_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(validation_loss, model)
            self.counter = 0

    def save_checkpoint(self, validation_loss, model):
        if self.verbose:
            print("Validation loss decreased. Saving model...")
        torch.save(model.state_dict(), self.save_path)
