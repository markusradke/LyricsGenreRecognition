import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import classification_report, f1_score, cohen_kappa_score

from helpers.split_group_stratified_and_join import split_group_stratified_and_join
from helpers.MonroeExtractor import MonroeExtractor
from helpers.STMTopicModeler import STMTopicModeler
from helpers.GenreClassifierTrainer import GenreClassifierTrainer


K_RANGE = (2, 20)

english = pd.read_csv(
    "data/poptrag_lyrics_genres_corpus_filtered_english_lemmatized.csv"
)

labels_and_group = english[["cat32", "track.s.firstartist.name"]].rename(
    columns={"cat32": "label", "track.s.firstartist.name": "group"}
)
X = english[
    [
        "lyrics_lemmatized",
        "cat5",
        "cat12",
        "cat25",
        "cat32",
        "track.s.firstartist.name",
    ]
]
corpus_train, corpus_test, _, _ = split_group_stratified_and_join(
    labels_and_group=labels_and_group, X=X, test_size=0.2, random_state=42
)

monroe_extractor = MonroeExtractor(
    min_artists=20,
    include_unigrams=True,
    use_bigram_boundary_filter=True,
    use_stopword_filter=True,
    p_value=0.001,
    prior_concentration=1.0,
    random_state=42,
    checkpoint_dir="experiments/monroe_extractor_checkpoints",
)
monroe_extractor.fit(
    corpus_train["lyrics_lemmatized"],
    corpus_train["cat32"],
    corpus_train["track.s.firstartist.name"],
)

counts_train = monroe_extractor.transform(corpus_train["lyrics_lemmatized"])
vocabulary = monroe_extractor.vocabulary_

# TEST CTM

stm = STMTopicModeler(
    k_range=K_RANGE,
    use_genre_prevalence=False,  # falls back to regular CTM
    random_state=42,
    model_dir="experiment_outputs/ctm_test",
)
# genre is not used for computation of features
stm.tune_and_fit(counts_train, corpus_train["cat5"], vocab=vocabulary)

X_train = stm.transform(counts_train)
X_test = stm.transform(monroe_extractor.transform(corpus_test["lyrics_lemmatized"]))

y_train_cat5 = corpus_train["cat5"]
y_test_cat5 = corpus_test["cat5"]
y_train_cat12 = corpus_train["cat12"]
y_test_cat12 = corpus_test["cat12"]
y_train_cat25 = corpus_train["cat25"]
y_test_cat25 = corpus_test["cat25"]

print("TEST WITH RF CLASSIFIER ON CTM FEATURES")
pipe = make_pipeline(
    StandardScaler(),
    RandomForestClassifier(
        random_state=42,
        n_jobs=-1,
        min_samples_leaf=30,
        n_estimators=250,
        max_features="sqrt",
        class_weight="balanced",
    ),
)

out = "CTM CLASSIFICATION REPORT RF\n\n"
pipe.fit(X_train, y_train_cat5)
y_pred = pipe.predict(X_test)
out += classification_report(y_test_cat5, y_pred)
out += f"f1_macro: {f1_score(y_test_cat5, y_pred,  average='macro'):.3f}\n"
out += f"Kappa: {cohen_kappa_score(y_test_cat5, y_pred):.3f}\n\n"


pipe.fit(X_train, y_train_cat12)
y_pred = pipe.predict(X_test)
out += classification_report(y_test_cat12, y_pred)
out += f"f1_macro: {f1_score(y_test_cat12, y_pred,  average='macro'):.3f}\n"
out += f"Kappa: {cohen_kappa_score(y_test_cat12, y_pred):.3f}\n\n"


pipe.fit(X_train, y_train_cat25)
y_pred = pipe.predict(X_test)
out += classification_report(y_test_cat25, y_pred)
out += f"f1_macro: {f1_score(y_test_cat25, y_pred,  average='macro'):.3f}\n"
out += f"Kappa: {cohen_kappa_score(y_test_cat25, y_pred):.3f}\n\n"

with open("experiment_outputs/ctm_test/ctm_rf_classification_report.txt", "w") as f:
    f.write(out)
print(out)

out = "CTM CLASSIFICATION REPORT LR\n\n"
trainer5 = GenreClassifierTrainer(
    X_train=X_train, y_train=y_train_cat5, random_state=42
)
trainer5.fit_with_fixed_params()

y_pred = trainer5.best_pipeline_.predict(X_test)
out += classification_report(y_test_cat5, y_pred)
out += f"f1_macro: {f1_score(y_test_cat5, y_pred,  average='macro'):.3f}\n"
out += f"Kappa: {cohen_kappa_score(y_test_cat5, y_pred):.3f}\n\n"


trainer12 = GenreClassifierTrainer(
    X_train=X_train, y_train=y_train_cat12, random_state=42
)
trainer12.fit_with_fixed_params()
y_pred = trainer12.best_pipeline_.predict(X_test)
out += classification_report(y_test_cat12, y_pred)
out += f"f1_macro: {f1_score(y_test_cat12, y_pred,  average='macro'):.3f}\n"
out += f"Kappa: {cohen_kappa_score(y_test_cat12, y_pred):.3f}\n\n"

y_train_cat25 = corpus_train["cat25"]
y_test_cat25 = corpus_test["cat25"]

trainer25 = GenreClassifierTrainer(
    X_train=X_train, y_train=y_train_cat5, random_state=42
)
trainer25.fit_with_fixed_params()
y_pred = trainer25.best_pipeline_.predict(X_test)
out += classification_report(y_test_cat25, y_pred)
out += f"f1_macro: {f1_score(y_test_cat25, y_pred,  average='macro'):.3f}\n"
out += f"Kappa: {cohen_kappa_score(y_test_cat25, y_pred):.3f}\n\n"

with open("experiment_outputs/ctm_test/ctm_lr_classification_report.txt", "w") as f:
    f.write(out)
print(out)


# TEST WITH STM FEATURES WITH GENRE PREVALENCE
stm = STMTopicModeler(
    k_range=K_RANGE,
    use_genre_prevalence=True,
    random_state=42,
    model_dir="experiment_outputs/ctm_test",
)
# use most granular genre for prevalence
stm.tune_and_fit(counts_train, corpus_train["cat32"], vocab=vocabulary)

X_train = stm.transform(counts_train)
X_test = stm.transform(monroe_extractor.transform(corpus_test["lyrics_lemmatized"]))

y_train_cat5 = corpus_train["cat5"]
y_test_cat5 = corpus_test["cat5"]
y_train_cat12 = corpus_train["cat12"]
y_test_cat12 = corpus_test["cat12"]
y_train_cat25 = corpus_train["cat25"]
y_test_cat25 = corpus_test["cat25"]

print("TEST WITH RF CLASSIFIER ON CTM FEATURES")
pipe = make_pipeline(
    StandardScaler(),
    RandomForestClassifier(
        random_state=42,
        n_jobs=-1,
        min_samples_leaf=30,
        n_estimators=250,
        max_features="sqrt",
        class_weight="balanced",
    ),
)

out = "CTM CLASSIFICATION REPORT RF\n\n"
pipe.fit(X_train, y_train_cat5)
y_pred = pipe.predict(X_test)
out += classification_report(y_test_cat5, y_pred)
out += f"f1_macro: {f1_score(y_test_cat5, y_pred,  average='macro'):.3f}\n"
out += f"Kappa: {cohen_kappa_score(y_test_cat5, y_pred):.3f}\n\n"


pipe.fit(X_train, y_train_cat12)
y_pred = pipe.predict(X_test)
out += classification_report(y_test_cat12, y_pred)
out += f"f1_macro: {f1_score(y_test_cat12, y_pred,  average='macro'):.3f}\n"
out += f"Kappa: {cohen_kappa_score(y_test_cat12, y_pred):.3f}\n\n"


pipe.fit(X_train, y_train_cat25)
y_pred = pipe.predict(X_test)
out += classification_report(y_test_cat25, y_pred)
out += f"f1_macro: {f1_score(y_test_cat25, y_pred,  average='macro'):.3f}\n"
out += f"Kappa: {cohen_kappa_score(y_test_cat25, y_pred):.3f}\n\n"

with open("experiment_outputs/ctm_test/ctm_rf_classification_report.txt", "w") as f:
    f.write(out)
print(out)

out = "CTM CLASSIFICATION REPORT LR\n\n"
trainer5 = GenreClassifierTrainer(
    X_train=X_train, y_train=y_train_cat5, random_state=42
)
trainer5.fit_with_fixed_params()

y_pred = trainer5.best_pipeline_.predict(X_test)
out += classification_report(y_test_cat5, y_pred)
out += f"f1_macro: {f1_score(y_test_cat5, y_pred,  average='macro'):.3f}\n"
out += f"Kappa: {cohen_kappa_score(y_test_cat5, y_pred):.3f}\n\n"


trainer12 = GenreClassifierTrainer(
    X_train=X_train, y_train=y_train_cat12, random_state=42
)
trainer12.fit_with_fixed_params()
y_pred = trainer12.best_pipeline_.predict(X_test)
out += classification_report(y_test_cat12, y_pred)
out += f"f1_macro: {f1_score(y_test_cat12, y_pred,  average='macro'):.3f}\n"
out += f"Kappa: {cohen_kappa_score(y_test_cat12, y_pred):.3f}\n\n"

y_train_cat25 = corpus_train["cat25"]
y_test_cat25 = corpus_test["cat25"]

trainer25 = GenreClassifierTrainer(
    X_train=X_train, y_train=y_train_cat5, random_state=42
)
trainer25.fit_with_fixed_params()
y_pred = trainer25.best_pipeline_.predict(X_test)
out += classification_report(y_test_cat25, y_pred)
out += f"f1_macro: {f1_score(y_test_cat25, y_pred,  average='macro'):.3f}\n"
out += f"Kappa: {cohen_kappa_score(y_test_cat25, y_pred):.3f}\n\n"

with open("experiment_outputs/ctm_test/ctm_lr_classification_report.txt", "w") as f:
    f.write(out)
print(out)
