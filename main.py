import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.feature_selection import RFE, VarianceThreshold
from sklearn.model_selection import KFold, GridSearchCV
from kaggle import KaggleApi

train_df = pd.read_csv(filepath_or_buffer="data/train.csv", header=0)
test_df = pd.read_csv(filepath_or_buffer="data/test.csv", header=0)

### Some cleaning and preprocess
train_df["label"] = train_df["label"].astype("category")

# Training set
# Separate the X form the Y
X_train = train_df.iloc[:, 1:]
y_train = train_df.loc[:, ["label"]]

type(train_df["label"])
type(y_train)


################################################################################
#                                              .-')    .-') _
#                                             ( OO ). (  OO) )
#  ,--.      .-'),-----.   ,----.     ,-.-') (_)---\_)/     '._ ,-.-')   .-----.
#  |  |.-') ( OO'  .-.  ' '  .-./-')  |  |OO)/    _ | |'--...__)|  |OO) '  .--./
#  |  | OO )/   |  | |  | |  |_( O- ) |  |  \\  :` `. '--.  .--'|  |  \ |  |('-.
#  |  |`-' |\_) |  |\|  | |  | .--, \ |  |(_/ '..`''.)   |  |   |  |(_//_) |OO
# (|  '---.'  \ |  | |  |(|  | '. (_/,|  |_.'.-._)   \   |  |  ,|  |_.'||  |`-'|
#  |      |    `'  '-'  ' |  '--'  |(_|  |   \       /   |  | (_|  |  (_'  '--'\
#  `------'      `-----'   `------'   `--'    `-----'    `--'   `--'     `-----'
################################################################################


################################################################################
# Logistic Regression
################################################################################

# lre: Logistic Regression Estimator
# todo: fix the warning "fail to converge"
lre = LogisticRegression(
    multi_class="multinomial",
    solver="lbfgs",
).fit(
    X_train,
    y_train.values.ravel()
)

lre.score(X_train, y_train)

# First prediction based on simple logistic regression without neither
# any preprocess nor variable seletion.
prediction = lre.predict(test_df)


# Prediction MUST have (i) 2 columns named ImageId and Label (ii) a total of
# 280000 rows, with ImageId ranging from 1 to 280000.
# todo: Create a function that create the dataframe in the correct format
#  and convert it into csv file.

r, c = test_df.shape

prediction_df = pd.DataFrame({
    "ImageId": range(1, r + 1),
    "Label": prediction
})

prediction_df.to_csv(
    path_or_buf="submissions/01.submission_basic_lre.csv",
    index=False
)

# Directly submit to Kaggle
# todo: use kaggle api directly inside the scipt.
#  A secure file should (must) be created in order to not upload the content
#  of the credential to github


################################################################################
# Remove low variance predictor variance_threshold()
#   * Threshold set to 0.
################################################################################

# Todo: Define sample variance of each bernouilli sample
#  (Should the variance be assessed for every predictors?)
sel = VarianceThreshold(threshold=0).fit(X_train)

predictor_idx = sel.get_support(indices=True)
predictor_names = X_train.columns[predictor_idx]

data_train_tmp = X_train.loc[:, predictor_names]

lre_vt = LogisticRegression(
    multi_class="multinomial",
    solver="lbfgs"
).fit(
    data_train_tmp,
    y_train.values.ravel()
)

# bof: ~0.94
lre_vt.score(data_train_tmp, y_train)


################################################################################
# Logistic Regression Cross Validated
# L1 regularization. Select the most appropriate selector based on a
# logistic lasso (L1) regulatization.
################################################################################

# cross validation validator
fit_control = KFold(
    n_splits=2,
    random_state=1010,
    shuffle=True
)

lre_vt_l1 = LogisticRegressionCV(
    solver="lbfgs",
    multi_class="multinomial",
    cv=fit_control
)
lre_vt_l1__fit = lre_vt_l1.fit(
    X_train,
    y_train.values.ravel()
)

# Bof: 0.94
lre_vt_l1__fit.score(X_train, y_train)
################################################################################
# todo GET THE COEF.
lre_vt_l1__fit.

tuned_parameters = [
    {"Cs": [1, 10, 100]},
]

clf = GridSearchCV(lre_vt_l1, tuned_parameters)
clf.fit(X_train, y_train.values.ravel())

################################################################################
# Feature selection
################################################################################

### Remove low variance predictor variance_threshold()

# Todo: Define sample variance of each bernouilli sample

sel = VarianceThreshold(threshold=0.05).fit(X_train)

predictor_idx = sel.get_support(indices=True)
predictor_names = X_train.columns[predictor_idx]

X_train_nzv = X_train.loc[:, predictor_names]

lr_nzv = LogisticRegression(
    penalty="elasticnet",
    multi_class="multinomial",
    solver="saga",
    l1_ratio=0.5
)
lr_nzv_fit = lr_nzv.fit(X_train_nzv, y_train.values.ravel())

lr_nzv_fit.score(X_train_nzv, y_train.values.ravel())

prediction_nzv_df = lr_nzv_fit.predict(test_df.loc[:, predictor_names])

prediction_nzv_df = pd.DataFrame({
    "ImageId": range(1, len(prediction_nzv_df) + 1),
    "Label": prediction_nzv_df
})

prediction_nzv_df.to_csv("submission_lr_nzv.csv", index=False)

### Group the selectors together

predictor_lst = train_df.columns.values.tolist()
y = ["label"]
X = [p for p in predictor_lst if p not in y]

lr = LogisticRegression(multi_class="multinomial",
                        solver="lbfgs")

rfe = RFE(lr, n_features_to_select=20).fit(X_train, y_train.values.ravel())

