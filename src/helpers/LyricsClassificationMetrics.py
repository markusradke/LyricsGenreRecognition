from sklearn.metrics import f1_score, precision_score, recall_score, cohen_kappa_score


class LyricsClassificationMetrics:
    f1_macro: float
    precision_macro: float
    recall_macro: float
    cohen_kappa: float

    def __init__(self, y_true, y_pred):
        self.cohen_kappa = cohen_kappa_score(y_true, y_pred)
        self.f1_macro = f1_score(y_true, y_pred, average="macro")
        self.precision_macro = precision_score(y_true, y_pred, average="macro")
        self.recall_macro = precision_score(y_true, y_pred, average="macro")

    def __str__(self):
        return (
            f"F1 macro: {self.f1_macro:.3f}\n"
            f"Precision macro: {self.precision_macro:.3f}\n"
            f"Recall macro: {self.recall_macro:.3f}\n"
            f"Cohen's kappa: {self.cohen_kappa:.3f}"
        )
