# https://www.nature.com/articles/nmeth.4397
# Graphs to generate
# Pairwise euclidean
# New dose response using powerTransformer
# AUR ROC Curves for confusion matrices
# COnfusion matrix for classifying cells on test data

# %% Imports
from scipy import stats
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import scale
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import euclidean_distances

# from sklearn.metrics import homogeneity_score
from sklearn.cluster import KMeans
import numpy.matlib

# from scipy.stats import kstest
# import scipy.stats
import json

from sklearn.utils import check_matplotlib_support

# from scipy.stats import linregress

sns.set()

# %% Setup
SAVE_FIG = 1
SAVE_CSV = 0
CHANNELS = 1
# DRUG = 'G007'
CELL = "ISO49"
DRUG = "Dabrafenib"
CONC = 10
# CONC = 3.330

FLAGS_TSNSE = 0
FLAG_TISS = 0
FLAG_MAXCORR = 0
FLAG_MEDIAN_PER_EUCLID = 0
FLAG_CORR_CONC_TO_FEATURE = 0
FLAG_IC50 = 0


ZEROISCONTROL = 1
SAVE_CSV = 0
DROP_TEXT_FROM_DF = 1

# %% Read data

# Load data
# root_dir = './_analysed/2colour_largeset/'
# data_folder = '220519 - G007'
data_folder = "230719"
data_folder = "241019 - ISO49+34"
data_folder = "210720 - ISO49+34 - best planes"

root_dir = "./analysed/" + data_folder + "/"


def metadata():
    return "./graphs/" + data_folder + "_channels_" + str(CHANNELS)


# root_dir = './_analysed/190709/'
# root_dir = './_analysed/230719/'
nuclei_file_path = root_dir + "/raw/best_plane_sobel/Secondary.csv"
image_file_path = root_dir + "/raw/best_plane_sobel/Image.csv"
organoid_file_path = root_dir + "/raw/best_plane_sobel/Primary.csv"

# root_dir = './_analysed/230719/'
nuclei_file_path = root_dir + "/raw/projection_XY/Secondary.csv"
image_file_path = root_dir + "/raw/projection_XY/Image.csv"
organoid_file_path = root_dir + "/raw/projection_XY/Primary.csv"


# nuclei_file_path = root_dir + '/raw/best_plane_sobel/Primary.csv'
# image_file_path = root_dir + '/raw/best_plane_sobel/Image.csv'

# organoid_file_path = root_dir + 'MergedCellObjects.csv'


def nucleiPrimaryFilter(df):
    return df.filter(regex=r"^((?!NucleiObjectsPrimary).)*$")


object_headers = ["ImageNumber", "ObjectNumber"]

image_df = pd.read_csv(image_file_path)
nuclei_df = pd.read_csv(nuclei_file_path)
organoid_df = pd.read_csv(organoid_file_path)

organoid_df_idx = organoid_df.set_index(object_headers)
idx_large_organoid = organoid_df_idx["AreaShape_Area"].groupby("ImageNumber").idxmax()
organoid_large_df = organoid_df_idx.loc[idx_large_organoid]

# nuclei_n_organoid_df = pd.merge(organoid_large_df, nuclei_df,
#                                 left_on=['ObjectNumber', 'ImageNumber'],
#                                 right_on=[
#                                     'Parent_MergedCellObjects', 'ImageNumber'],
#                                 how='inner',  # Might be wrong
#                                 # left_index = True,
#                                 suffixes=('_Organoid', '_Nuclei'))
# # nuclei_n_organoid_df = organoid_df


nuclei_n_organoid_df = pd.merge(
    nuclei_df,
    organoid_df,
    left_on=["ObjectNumber", "ImageNumber"],
    right_on=["ObjectNumber", "ImageNumber"],
    how="inner",  # Might be wrong
    # left_index = True,
    suffixes=("", "_Organoid"),
)

image_nuclei_df = pd.merge(
    nuclei_df,
    image_df,
    on="ImageNumber",
    how="left",
    # left_index = True,
    suffixes=("", "_Image"),
)

image_nuclei_n_organoid_df = pd.merge(
    nuclei_n_organoid_df,
    image_df,
    on="ImageNumber",
    how="left",
    # left_index = True,
    suffixes=("", "_Image"),
)


# image_nuclei_n_organoid_df = pd.merge(
#     nuclei_n_organoid_df,
#     image_df,
#     on="ImageNumber",
#     how="left",
#     # left_index = True,
#     suffixes=("", "_Image"),
# )

# nuclei_n_organoid_df_idx = nuclei_n_organoid_df.set_index(object_headers)
# image_nuclei_n_organoid_df_idx = image_nuclei_n_organoid_df.set_index(object_headers)
merged_df = image_nuclei_n_organoid_df
# merged_df = image_df
merged_df = nuclei_df
merged_df = image_nuclei_df

# organoid_df_idx = organoid_df.set_index(object_headers)
# idx_large_organoid = organoid_df_idx["AreaShape_Area"].groupby("ImageNumber").idxmax()
# organoid_large_df = organoid_df_idx.loc[idx_large_organoid]

# nuclei_n_organoid_df_idx = nuclei_n_organoid_df.set_index(object_headers)
# image_nuclei_n_organoid_df_idx = image_nuclei_n_organoid_df.set_index(object_headers)
# merged_df = image_nuclei_n_organoid_df


# %%### Beging metadata extraction
regex_pattern = r"[\/\\](?P<Date>[\d]+)_.+_(?P<Cell>ISO[\d]+)_(?P<Drug>[A-Za-z0-9]+)_(?P<Concentration>[\d\w_-]+uM)(?:.+Position_(?P<Position>[\d]))?"
# filenames_image = merged_df['PathName_Channels'];
filenames_nuclei = nuclei_df["Metadata_FileLocation"]
# filenames_organoid = organoid_df['Metadata_FileLocation']
# filenames_image_nuclei_n_organoid = image_nuclei_n_organoid_df['PathName_Channels'];
filenames_image_nuclei_n_organoid = image_nuclei_n_organoid_df["Metadata_FileLocation"]
raw_df = merged_df
filenames = filenames_image_nuclei_n_organoid

extracted_data = filenames.str.extract(regex_pattern)
extracted_data["filenames"] = filenames
extracted_data["Replicate"] = extracted_data["Position"].fillna(1)
extracted_data["Conc /uM"] = pd.to_numeric(
    extracted_data["Concentration"]
    .str.replace("_", ".")
    .str.replace("-", ".")
    .str.replace("uM", "")
).round(decimals=3)

# extracted_data.to_csv('extracted_data.csv')
filename_headers = ["Date", "Drug", "Cell", "Replicate", "Conc /uM"]
index_headers = object_headers + filename_headers

if ZEROISCONTROL:
    extracted_data["Drug"] = extracted_data["Drug"].where(
        extracted_data["Conc /uM"] != 0, other="Control"
    )

raw_df_indexed = raw_df.join(extracted_data[filename_headers]).set_index(index_headers)
# raw_df_indexed
with open("bad_cols.json", "r") as f:
    # print(*raw_df_indexed.columns,sep='\n')
    bad_cols = json.load(f)

raw_df_indexed_bad_cols = raw_df_indexed.drop(bad_cols, axis=1, errors="ignore")
raw_df_indexed_bad_cols_med = raw_df_indexed_bad_cols.groupby(
    level=["ImageNumber"] + filename_headers
).median()

raw_df_indexed_bad_cols_med = raw_df_indexed_bad_cols.groupby(
    level=["Date", "Drug", "Cell", "Replicate", "Conc /uM", "ImageNumber"]
).median()

# raw_df_indexed_bad_cols_med = raw_df_indexed_bad_cols.groupby(
#     level=["Date", "Drug", "Cell", "Replicate", "Conc /uM", "ImageNumber"]
# ).median()

raw_df_indexed_bad_cols_finite = raw_df_indexed_bad_cols.reindex(
    raw_df_indexed_bad_cols.index.dropna()
)

raw_df_indexed_bad_cols_med_finite = raw_df_indexed_bad_cols_med.reindex(
    raw_df_indexed_bad_cols_med.index.dropna()
)

# raw_df_indexed_bad_cols
# %% Save CSV

SAVE_CSV = 0

csv_name = root_dir + "df_org+nuclei+image"

df = (
    raw_df_indexed_bad_cols_finite.select_dtypes("number")
    .dropna(axis=1)
    .sample(frac=1)
    # .xs("ISO34", level="Cell", drop_level=False)
)
# df = raw_df_indexed_bad_cols
df_median = (
    raw_df_indexed_bad_cols_med_finite.select_dtypes("number")
    .dropna(axis=1)
    .sample(frac=1)
    # .xs("ISO34", level="Cell", drop_level=False)
)

# df_median = raw_df_indexed_bad_cols_med
# df_median_no_nuclei = df_median.drop(
#     df_median.filter(like='Nuclei', axis=1).columns,
#     axis=1)

# df = raw_df_indexed_bad_cols.xs(CELL,level="Cell",drop_level=False)
# df_median = raw_df_indexed_bad_cols_med.xs(CELL,level="Cell",drop_level=False)

if SAVE_CSV:
    df.dropna(axis=1).to_csv(csv_name + ".csv")
    df_median.dropna(axis=1).to_csv(csv_name + "_median_per_org.csv")

# %% ####### Filtering the data #######
df_clean_in = df
df_clean_median_in = df_median

df_clean_in = df_clean_in.select_dtypes("number").dropna(axis=1).astype("float64")
df_clean_median_in = df_clean_median_in.select_dtypes("number").dropna(axis=1)


def drop_sigma(df, sigma):
    return df.mask(df.apply(lambda df: (np.abs(stats.zscore(df)) > sigma))).dropna(
        axis=1
    )


df_clean = df_clean_in
# df_clean = drop_sigma(df_clean_in, 5.0)

df_median_clean = df_clean_median_in
# df_median_clean = drop_sigma(df_clean_median_in, 5.0)


# df.apply(lambda df: (np.abs(stats.zscore(df)) > 3.0))
# df_median_mask = df_median.apply(lambda df: (np.abs(stats.zscore(df)) > 3.0))
# df_median_clean = df_median.mask(df_mask).dropna()

# %% ####### Df summary ##########
PYOD_FLAG = 0
if PYOD_FLAG:
    df_median.apply()
    from pyod.models.knn import KNN  # kNN detector

    clf = KNN()
    clf.fit(df_median)

    # get outlier scores
    y_train_scores = clf.decision_scores_  # raw outlier scores
    y_test_scores = clf.decision_function(df_median)  # outlier scores
    plt.hist(y_test_scores)

    df_median["AreaShape_Area_Organoid"].hist()


# %% Isolation forest

from sklearn.ensemble import IsolationForest

df_clean_isolated = df_clean.loc[IsolationForest().fit(df_clean).predict(df_clean) == 1]

df_median_clean_isolated = df_median_clean.loc[
    IsolationForest().fit(df_median_clean).predict(df_median_clean) == 1
]

# %% ####### Df summary ##########

df_summary = (
    df_median.groupby(
        [
            "Drug",
            "Cell",
            "Conc /uM",
        ]
    )
    .count()
    .rename(columns={df_median.columns[0]: "Organoids"})
)
# df_median.groupby('Cell').count()

sns.catplot(
    y="Conc /uM",
    hue="Drug",
    x="Organoids",
    col="Cell",
    sharex=True,
    kind="bar",
    orient="h",
    data=df_summary["Organoids"].reset_index(),
)
if SAVE_FIG:
    plt.savefig(metadata() + "_summary.pdf")
plt.show()

# %%####### StandardScaler
# index = df.index
# scaled_df=df.copy()
# scaled_df[:]=StandardScaler().fit_transform(df)
# scaled_df
from sklearn.preprocessing import scale, power_transform, robust_scale, normalize

df_scaled_in = df_clean
df_scaled_in = df_clean_isolated

df_median_clean = df_median_clean
df_median_clean = df_median_clean_isolated

# preprocessor = scale
# preprocessor = robust_scale
# preprocessor = normalize
preprocessor = lambda x: power_transform(x, method="yeo-johnson")
preprocessor_standard = lambda x: scale(x)


scaled_df = pd.DataFrame(
    preprocessor(df_scaled_in), index=df_scaled_in.index, columns=df_scaled_in.columns
)

scaled_df_standard = pd.DataFrame(
    preprocessor_standard(df_scaled_in), index=df_scaled_in.index, columns=df_scaled_in.columns
)

scaled_df_median = pd.DataFrame(
    preprocessor(df_median_clean),
    index=df_median_clean.index,
    columns=df_median_clean.columns,
)

# scaled_df_median = df_median_clean.apply(preprocessor, axis=0, result_type="broadcast")

# scaled_clean_df = df_clean.apply(scale, result_type="broadcast")
# scaled_df_median_clean = df_median_clean.apply(scale, result_type="broadcast")

# scaled_df_median_no_nuclei = df_median_no_nuclei.apply(
#     scale, result_type="broadcast")


# %% Feature selection
# %%time
FEATURE_SELECT = 1
if FEATURE_SELECT:
    from sklearn.svm import LinearSVC
    from sklearn.linear_model import SGDClassifier, LinearRegression
    from sklearn.datasets import load_iris
    from sklearn.feature_selection import SelectFromModel, RFECV, RFE
    from sklearn.feature_selection import (
        SelectKBest,
        chi2,
        f_classif,
        SelectPercentile,
        mutual_info_classif,
    )
    from sklearn.model_selection import train_test_split
    from sklearn.neural_network import MLPClassifier
    from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
    from sklearn.multioutput import MultiOutputClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.manifold import TSNE

    # model = LinearSVC()
    model = RandomForestClassifier(
        # random_state=1,
        # n_estimators=750,
        # max_depth=15,
        # min_samples_split=5,
        # min_samples_leaf=1,
    )
    # model = BaggingClassifier()
    # model = SGDClassifier()
    # mlp = MLPClassifier().fit(scaled_df, labels)
    # model = MultiOutputClassifier(model)

    feature_df_in = scaled_df.sample(frac=1)
    feature_df_median_in = scaled_df_median.sample(frac=1)

    labels = feature_df_in.reset_index()[["Date"]]
    # labels = feature_df_in.reset_index()[["Drug", "Cell", "Conc /uM"]]
    labels = labels.apply(lambda x: pd.factorize(x)[0])
    X_train, X_test, y_train, y_test = train_test_split(feature_df_in, labels)
    model.fit(X_train, y_train)
    full_df_pred = pd.DataFrame(model.predict(feature_df_in))
    truth = pd.Series((np.array(full_df_pred) == np.array(labels)).flatten())
    # not(pd.Series(model.predict(feature_df_in))==labels)
    # print(model.score(X_test, y_test))
    # model_select = SelectFromModel(model, prefit=True)
    # model_select = SelectFromModel(model)
    # model_select = RFECV(model, n_jobs=10).fit(X_train, y_train)
    # pipe = Pipeline([('modelselect', SelectFromModel(model)),
    #                     ('rfecv', RFE(model))])

    pipe = Pipeline(
        [
            ("modelselect", SelectFromModel(model)),
            ("kbest", SelectKBest(f_classif, k=20)),
        ]
    )

    # pipe = Pipeline(
    #     [
    #         ("PCA", PCA()),
    #         ("modelselect", SelectFromModel(model)),
    #         ("kbest", SelectKBest(f_classif, k=20)),
    #     ]
    # )

    # #                     ('rfecv', RFE(model))])
    # pipe = Pipeline(
    #     [
    #         ("modelselect", SelectFromModel(model)),
    #         ("kbest", SelectKBest(f_classif, k=20)),
    #     ]
    # )

    # pipe = Pipeline(
    #     [
    #         ("PCA", PCA()),
    #         ("modelselect", SelectFromModel(model)),
    #         ("kbest", SelectKBest(f_classif, k=20)),
    #     ]
    # )

    # pipe = Pipeline(
    #     [
    #         ("PCA", PCA()),
    #         ("modelselect", SelectFromModel(model)),
    #         ("mutual_info_classif", SelectKBest(mutual_info_classif, k=20)),
    #     ]
    # )

    # pipe = Pipeline(
    #     [
    #         ("PCA", PCA()),
    #         ("modelselect", SelectFromModel(model)),
    #         ("mutual_info_classif", SelectPercentile(f_classif)),
    #     ]
    # )

    # pipe = Pipeline(
    #     [
    #         ("PCA", PCA()),
    #         ("mutual_info_classif", SelectPercentile(mutual_info_classif)),
    #         ("modelselect", SelectFromModel(model)),
    #     ]
    # )

    # pipe = Pipeline(
    #     [
    #         # ('mutual_info_classif', SelectPercentile(f_classif)),
    #         ("PCA", PCA()),
    #         # ('mutual_info_classif', SelectPercentile(mutual_info_classif)),
    #         ("modelselect", SelectFromModel(model)),
    #     ]
    # )
    pipe = Pipeline([('PCA', PCA()),
                    ('modelselect', SelectFromModel(model))]) # Best?

    # pipe = Pipeline([('TSNE', TSNE()),
    #                 ('modelselect', SelectFromModel(model)),
    #                 ('kbest', SelectKBest(f_classif, k=20))])

    pipe.fit(X_train, y_train)

    scaled_df_selected_features = pd.DataFrame(
        pipe.transform(feature_df_in), index=feature_df_in.index
    )
    scaled_df_selected_features_median = pd.DataFrame(
        pipe.transform(feature_df_median_in), index=feature_df_median_in.index
    )

    # print(model.score(X_test, y_test))
    print(scaled_df.shape)
    print(scaled_df_selected_features.shape)
# %% RANSAC
# %%timeit
from sklearn import linear_model

RANSAC = 0
if RANSAC:
    ransac_in_df = scaled_df_selected_features.drop(0, level="Conc /uM")
    ransac_in_median_df = scaled_df_selected_features_median.drop(0, level="Conc /uM")

    X = ransac_in_df
    y = ransac_in_df.reset_index()[["Conc /uM"]].apply(np.log10).dropna()
    ransac = linear_model.HuberRegressor()

    ransac.fit(X, y)

    weighted_scaled_df_selected_features = ransac_in_df.multiply(ransac.coef_)

    weighted_scaled_df_selected_features_median = ransac_in_median_df.multiply(
        ransac.coef_
    )

    X = weighted_scaled_df_selected_features

    ransac = linear_model.RANSACRegressor()

    ransac.fit(X, y)

    weighted_scaled_df_selected_features = X.loc[ransac.inlier_mask_]


# outlier_mask = np.logical_not(inlier_mask)
# %% IC50 from classification
from sklearn.model_selection import cross_val_score, KFold, train_test_split
from sklearn.metrics import plot_confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier,VotingClassifier,BaggingClassifier,ExtraTreesClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.metrics import balanced_accuracy_score

import collections
sns.set(rc={'figure.figsize':(11.7/2,8.27/2)})
from sklearn.neural_network import MLPClassifier

CELLS = {"ISO34", "ISO49"}
VARIABLES = ["Conc /uM", "Date", "Drug", "Cell"]
VARIABLES = ["Conc /uM", "Date", "Drug"]

# VARIABLES = ["Conc /uM"]
# VARIABLES = ["Date"]
# VARIABLES = ["Drug"]

# VARIABLE = "Date"
# VARIABLE = "Date"
# VARIABLE = "Drug"
# VARIABLE = "Conc /uM"

CLASS_FRACTIONS = 1

nested_dict = lambda: collections.defaultdict(nested_dict)
confusion_plots = nested_dict()
chart = nested_dict()
report_dict = nested_dict()
report_tall_full = pd.DataFrame()

SAVE_FIG=0

data_in_full = scaled_df.sample(frac=1).xs(CELL, level="Cell").sample(frac=1)
data_in_median = scaled_df_median.xs(CELL, level="Cell").sample(frac=1)
data_in_median_sorted = scaled_df_median.xs(CELL, level="Cell").sort_index(level="Date")
data_in_full_sorted = scaled_df.xs(CELL, level="Cell").sort_index(level="Date")

DATA = {"Median per Organoid unsorted":data_in_median,
        "Per Cell unsorted":data_in_full,
        "Median per Organoid sorted by date":data_in_median_sorted,
        "Per Cell sorted by date":data_in_full_sorted}
# DATA = {"median":data_in_median,
#         "data_in_median_sorted":data_in_median_sorted
#         }
# DATA = {"median":data_in_median}
# report_df_dict = {}
report_tall_list = []

if CLASS_FRACTIONS:
    for VARIABLE in VARIABLES:
        for DATA_KEY in DATA:
            for CELL in CELLS:
                # data_in = scaled_df.sample(frac=1)
                # data_in = df_clean.sample(frac=1).sort_index(level="Date")
                # # data_in = scaled_df.sample(frac=1).xs(CELL, level="Cell").sample(frac=1)
                # data_in = df_clean.sample(frac=1).xs(CELL, level="Cell").sample(frac=1)
                # data_in = scaled_df_median.sample(frac=1).xs(CELL, level="Cell").sample(frac=1)
                data_in = DATA[DATA_KEY]
                # data_in = scaled_df.sample(frac=1).xs(CELL, level="Cell").sample(frac=1)

                # data_in = (
                #     df_median_clean.sample(frac=1)
                #     .xs(CELL, level="Cell")
                #     .xs("Vemurafenib", level="Drug")
                #     .sample(frac=1)
                # )

                # data_in = scaled_df_selected_features.sample(frac=1)

                # data_in_median = scaled_df_median.sample(frac=1)
                # data_in_median = scaled_df_selected_features_median.sample(frac=1)
                # datdata_in_mediana_in = df_clean.sample(frac=1)
                model = RandomForestClassifier()
                # model = GaussianProcessClassifier()
                # model = ExtraTreesClassifier()
                # labels = data_in.reset_index()[["Drug"]]
                # labels = labels.apply(lambda x: pd.factorize(x)[0])\
                #                 .set_index(data_in.index)
                # drugs_df = data_in.reset_index()["Drug"]
                labels, uniques = pd.factorize(data_in.reset_index()[VARIABLE])
                labels = pd.Series(labels, index=data_in.index)
                labels_indexed = (
                    pd.Series(data_in.index.get_level_values(VARIABLE), index=data_in.index)
                    .astype(str)
                    .astype("category")
                )

                X_train, X_test, y_train, y_test = train_test_split(
                    data_in,
                    labels_indexed,
                    test_size=0.5,
                    shuffle=False,
                )

                # corr = data_in.corrwith(labels,axis=1,method="pearson").dropna()
                # corr = data_in.apply(lambda x: x.corr(labels,method="pearson"))
                # corr = data_in.apply(lambda x: x.corr(labels))
                # print(corr.dropna().sort_values(ascending=False))

                # corr = data_in.apply(lambda x: x.corr(labels,axis=1,method="pearson").dropna())

                model.fit(X_train, y_train)
                print(model.score(X_test, y_test))

                scores_train = cross_val_score(model, X_train, y_train, cv=5)
                scores_test = cross_val_score(model, X_test, y_test, cv=5)

                print(
                    "Accuracy: %0.2f (+/- %0.2f)"
                    % (scores_train.mean(), scores_train.std() * 2)
                )



                print(
                    "Blind: %0.2f (+/- %0.2f)" % (scores_test.mean(), scores_test.std() * 2)
                )
                # print(f"Blind : {model.score(X_test, y_test)}")
                importance = pd.Series(
                    model.feature_importances_, index=X_train.columns
                ).sort_values(ascending=False)

                importance_cumsum = (
                    importance.cumsum()
                    .reset_index()
                    .rename(columns={"index": "Feature", 0: "Cumlative importance"})
                )
                fig_dims = (8, 4)
                fig, ax = plt.subplots(figsize=fig_dims)
                sns.barplot(
                    y="Feature", x="Cumlative importance", data=importance_cumsum[:10]
                )
                plt.tight_layout()
                y_pred = pd.Series(model.predict(X_test), index=X_test.index)
                # display(y_pred)

                # confusion_matrix(y_true, y_pred)
                confusion_plots[CELL][VARIABLE][DATA_KEY] = plot_confusion_matrix(
                    model, X_test, y_test, xticks_rotation="vertical", cmap="Greys"
                )

                if SAVE_FIG:
                    plt.grid(None)
                    confusion_plots[CELL][VARIABLE][DATA_KEY].plot(
                        xticks_rotation="vertical", cmap="Greys"
                    )
                    plt.grid(None)
                    plt.tight_layout()
                    # plt.savefig(f"{metadata()}_{CELL}_{VARIABLE}_confusion_matrix.pdf")
                    plt.show()

                # counts = pd.Series(y_pred.groupby(["Cell",
                #                                 "Conc /uM",
                #                                 "Drug"]).value_counts())
                # counts.rename(columns={0: "Counts"}, inplace=True)
                # counts.index.rename("DrugLabel", level=-1, inplace=True)
                # # counts.reset_index(-1,inplace=True)
                # drug_count_table = counts.unstack("DrugLabel", fill_value=0)
                # control_fraction = drug_count_table.divide(drug_count_table[4], axis=0)
                # control_fraction = control_fraction.stack("DrugLabel").rename("Fraction")

                # control_fraction_sns = sns.lmplot(
                #     x="Conc /uM",
                #     y="Fraction",
                #     row="Cell",
                #     col="Drug",
                #     data=control_fraction.reset_index(),
                #     sharex=False,
                #     sharey=False,
                #     x_estimator=np.median,
                #     x_ci="ci",
                #     # x_bins=9,
                #     markers=None,
                #     hue="Drug",
                # )
                from sklearn.metrics import roc_curve, auc
                from sklearn.metrics import roc_auc_score
                from sklearn.preprocessing import label_binarize

                y_score_bin = label_binarize(model.predict(X_test), classes=uniques)
                y_test_bin = label_binarize(y_test, classes=uniques)

                balanced_accuracy_score(model.predict(X_test),y_test)
                fpr = dict()
                tpr = dict()
                roc_auc = dict()
                for i, label in enumerate(uniques):
                    fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_score_bin[:, i])
                    roc_auc[i] = auc(fpr[i], tpr[i])
                from sklearn.metrics import classification_report

                report = pd.DataFrame(
                    classification_report(
                        y_test_bin, y_score_bin, target_names=uniques, output_dict=True
                    )
                )[uniques]
                # report_tall = report.unstack().reset_index()
                report_tall = (
                    report.reset_index()
                    .melt(
                        id_vars="index",
                        var_name=[VARIABLE],
                        value_name="Score",
                        ignore_index=False,
                    )
                    .rename(columns={"index": "Metric"})
                )
                # report_tall["Variable"] = VARIABLE
                # This should be a melt
                chart[CELL][VARIABLE][DATA_KEY] = (
                    sns.catplot(
                        x=VARIABLE,
                        col="Metric",
                        y="Score",
                        # col_wrap=2,
                        data=report_tall,
                        sharey=False,
                        kind="bar",
                    )
                    .set(yscale="log")
                    .set_xticklabels(rotation=45)
                )
                plt.tight_layout()
                plt.suptitle(f"{CELL}",y=1.12)
                if SAVE_FIG:
                    # chart[CELL][VARIABLE]
                    chart[CELL][VARIABLE][DATA_KEY].tight_layout()
                    chart[CELL][VARIABLE][DATA_KEY].savefig(
                        f"{metadata()}_{CELL}_{VARIABLE}_{DATA_KEY}_random_forest_scores.pdf"
                    )
                plt.show()

                report_dict[CELL][VARIABLE][DATA_KEY] = report_tall
                report_tall["Cell type"] = CELL
                report_tall["Population type"] = DATA_KEY
                report_tall["Variable"] = VARIABLE
                report_tall.rename(columns={CELL:"variable_name"})
                report_tall_list.append(report_tall)

SAVE_FIG=1
report_df = pd.concat(report_tall_list)

data= report_df.set_index("Variable").xs("Drug")
plot = sns.catplot(x="Drug",
            y="Score",
            col="Metric",
            row="Cell type",
            hue="Population type",
            data=data,
            sharey=False,
            kind="bar",
            ).set_xticklabels(rotation=45)
if(SAVE_FIG):
    plot.savefig(f"{metadata()}_facet_grid_drug_score_metric_cell_type_population_type.pdf")
    # plot.savefig(f"{metadata()}_facet_grid_drug_score_metric_cell_type_population_type.png")

data = report_df.set_index("Variable").xs("Date")
plot = sns.catplot(x="Date",
            y="Score",
            col="Metric",
            row="Cell type",
            hue="Population type",
            data=data,
            sharey=False,
            kind="bar",
            ).set_xticklabels(rotation=45)
                # report_tall_full = pd.concat([report_tall,report_tall_full])
if(SAVE_FIG):
    plot.savefig(f"{metadata()}_facet_grid_drug_score_metric_date_population_type.pdf")

# # %%
# fractions = y_pred.groupby(["Cell", "Conc /uM", "Drug"]).apply(
#     (lambda x: x.value_counts())
# )
# # y_pred.groupby(["Cell", "Conc /uM", "Drug"]).size().unstack(fill_value=0)

# print("Done")
# def default_to_regular(d):
#     if isinstance(d, collections.defaultdict):
#         d = {k: default_to_regular(v) for k, v in d.items()}
#     return d

# report_dict_true = default_to_regular(report_dict)


# figure_a_report = [report_dict["ISO34"]["Drug"]["full"],
#     report_dict["ISO34"]["Drug"]["median"],
#     report_dict["ISO34"]["Drug"]["data_in_median_sorted"],
#     report_dict["ISO48"]["Drug"]["full"],
#     report_dict["ISO48"]["Drug"]["median"],
#     report_dict["ISO48"]["Drug"]["data_in_median_sorted"]]

# report_dict[:]["Drug"][:]

# VARIABLES = ["Conc /uM", "Date", "Drug"]

# DATA = {"median":data_in_median,
#         "full":data_in_full,
#         "data_in_median_sorted":data_in_median_sorted}
# report_dict["CELL"][VARIABLE][DATA_KEY],
# report_dict[CELL][VARIABLE][DATA_KEY],
# report_dict[CELL][VARIABLE][DATA_KEY],
# report_dict[CELL][VARIABLE][DATA_KEY],
# report_dict[CELL][VARIABLE][DATA_KEY],

# %% Machine learning similarity
ML_SIMILARITY = 1
from sklearn.ensemble import RandomForestClassifier
SAVE_FIG=0
CELL = "ISO49"
if ML_SIMILARITY:
    data_in = scaled_df.sample(frac=1)
    data_in = (
        scaled_df.iloc[
            (scaled_df.index.get_level_values("Conc /uM") == 0)
            | (scaled_df.index.get_level_values("Conc /uM") == 0.041)
        ]
        .xs(CELL, level="Cell")
        .sample(frac=1)
    )
    data_in = (
        scaled_df.iloc[(scaled_df.index.get_level_values("Conc /uM") == 0.041)]
        .xs(CELL, level="Cell")
        .sample(frac=1)
    )
    data_in = scaled_df.sample(frac=1).xs(CELL, level="Cell")
    # labels, uniques = pd.factorize(data_in.reset_index()["Drug"])
    # labels = pd.Series(labels,index=data_in.index)
    labels = pd.Series(data_in.index.get_level_values("Drug"), index=data_in.index)
    # labels = data_in.reset_index()[["Drug"]].\
    #                     apply(lambda x: pd.factorize(x)[0]).\
    #                     set_index(data_in.index)
    drugs = data_in.index.get_level_values("Drug").unique()

    similarity_df = pd.DataFrame()
    models_list = []
    for drug in drugs:
        X_train = data_in.drop(drug, level="Drug")
        y_train = labels.drop(drug, level="Drug")
        X_test = data_in.xs(drug, level="Drug", drop_level=False)
        y_test = labels.xs(drug, level="Drug", drop_level=False)
        model = RandomForestClassifier()
        model.fit(X_train, y_train)
        model.score(X_train, y_train)
        prediction_df = pd.DataFrame(
            model.predict_proba(X_test), index=X_test.index, columns=model.classes_
        )
        predict_current_df = pd.melt(
            prediction_df,
            var_name=["Predicted Drug"],
            value_name="Similarity",
            ignore_index=False,
        )

        # predict_current_df = pd.DataFrame(
        #     model.predict(X_test), index=X_test.index
        #     ).rename(columns=uniques[model.classes_])

        similarity_df = similarity_df.append(predict_current_df)


# similarity_df = pd.DataFrame(similarity_df.index.get_level_values("Drug"),
#                         index=similarity_df.index)
similarity_df_reset = similarity_df.reset_index("Drug")

sns.catplot(
    x="Conc /uM",
    y="Similarity",
    kind="violin",
    row="Predicted Drug",
    col="Drug",
    data=similarity_df.reset_index(),
)

grouped_sim = similarity_df_reset.groupby(["Predicted Drug", "Drug"]).mean()

grouped_sim_square = grouped_sim.reset_index().pivot(
    index="Drug", columns="Predicted Drug", values="Similarity"
)


def draw_heatmap(*args, **kwargs):
    data = kwargs.pop("data")
    # data = data.mask(np.tril(np.ones(data.shape)).astype(np.bool))
    g = sns.heatmap(data, **kwargs)
    plt.yticks(rotation=0)

    # g.set_yticklabels(g.get_yticklabels())


sns.FacetGrid(grouped_sim_square).map_dataframe(
    draw_heatmap, cmap="Spectral", cbar_kws={"label": "Similarity"}
)


def facet_heat(df):
    return sns.FacetGrid(
        df.reset_index(level="Cell"), col="Cell", sharey=False
    ).map_dataframe(draw_heatmap, cmap="Spectral", cbar_kws={"label": "Similarity"})


# %% Machine learning similarity pairwise
ML_SIMILARITY_PAIR = 1
from sklearn.ensemble import RandomForestClassifier

CELL = "ISO49"
CELLS = {"ISO34", "ISO49"}
if ML_SIMILARITY_PAIR:
    for CELL in CELLS:
        data_in = scaled_df.sample(frac=1)
        data_in = (
            scaled_df.iloc[
                (scaled_df.index.get_level_values("Conc /uM") == 0)
                | (scaled_df.index.get_level_values("Conc /uM") == 0.041)
            ]
            .xs(CELL, level="Cell")
            .sample(frac=1)
        )
        data_in = (
            scaled_df.iloc[(scaled_df.index.get_level_values("Conc /uM") == 0.041)]
            .xs(CELL, level="Cell")
            .sample(frac=1)
        )
        data_in = scaled_df.sample(frac=1).xs(CELL, level="Cell")
        # labels, uniques = pd.factorize(data_in.reset_index()["Drug"])
        # labels = pd.Series(labels,index=data_in.index)
        labels = pd.Series(data_in.index.get_level_values("Drug"), index=data_in.index)
        # labels = data_in.reset_index()[["Drug"]].\
        #                     apply(lambda x: pd.factorize(x)[0]).\
        #                     set_index(data_in.index)
        drugs = data_in.index.get_level_values("Drug").unique()

        similarity_df = pd.DataFrame()

        for drug_blind in drugs:
            for drug in drugs:
                drugs_to_drop = drugs.drop(["Control", drug_blind])
                X_train = data_in.drop(drugs_to_drop, level="Drug")
                y_train = labels.drop(drugs_to_drop, level="Drug")
                X_test = data_in.xs(drug, level="Drug", drop_level=False)
                y_test = labels.xs(drug, level="Drug", drop_level=False)
                model = RandomForestClassifier()
                model.fit(X_train, y_train)
                model.score(X_train, y_train)
                prediction_df = pd.DataFrame(
                    model.predict_proba(X_test),
                    index=X_test.index,
                    columns=model.classes_,
                )[[drug_blind]]
                predict_current_df = pd.melt(
                    prediction_df,
                    var_name=["Predicted Drug"],
                    value_name="Similarity",
                    ignore_index=False,
                )
                similarity_df = similarity_df.append(predict_current_df)

        # similarity_df = pd.DataFrame(similarity_df.index.get_level_values("Drug"),
        #                         index=similarity_df.index)
        similarity_df_reset = similarity_df.reset_index("Drug")

        # sns.catplot(x="Conc /uM",
        #         y="Similarity",
        #         kind="violin",
        #         row="Predicted Drug",
        #         col="Drug",
        #         data=similarity_df.reset_index())

        grouped_sim = (
            similarity_df_reset.where(
                similarity_df_reset["Drug"] != similarity_df_reset["Predicted Drug"]
            )
            .groupby(["Predicted Drug", "Drug"])
            .mean()
        )

        grouped_sim_square = grouped_sim.reset_index().pivot(
            index="Drug", columns="Predicted Drug", values="Similarity"
        )

        def draw_heatmap(*args, **kwargs):
            data = kwargs.pop("data")
            # data = data.mask(np.logical_not(np.triu(np.ones(data.shape)).astype(np.bool)))
            g = sns.heatmap(data, **kwargs)
            plt.yticks(rotation=0)

            # g.set_yticklabels(g.get_yticklabels())

        heatmap = sns.FacetGrid(grouped_sim_square).map_dataframe(
            draw_heatmap, cmap="Spectral", cbar_kws={"label": "Similarity"}
        )
        if SAVE_FIG:
            heatmap.savefig(f"{metadata()}_{CELL}_pairwise_similarity.pdf")
            plt.tight_layout()
        plt.show()
    # def facet_heat(df):
    #     return sns.FacetGrid(
    #         df.reset_index(level="Cell"), col="Cell", sharey=False
    #     ).map_dataframe(draw_heatmap, cmap="Spectral", cbar_kws={"label": "Similarity"})
# %% Feature weighting
FEATURE_WEIGHTING = 0
if FEATURE_WEIGHTING:

    def euclid_on_df(df):
        return pd.DataFrame(df.apply(numpy.linalg.norm, axis=1)).rename(
            columns={0: "Euclidean distance"}
        )

    control_data = scaled_df_selected_features.xs("Control", level="Drug")
    control_data_median = pd.Series(
        float(euclid_on_df(control_data).median().astype("float")),
        index=control_data.index,
    )

    control_median_data = scaled_df_selected_features_median.xs("Control", level="Drug")
    control_median_data_median = pd.Series(
        float(euclid_on_df(control_median_data).median().astype("float")),
        index=control_median_data.index,
    )

    reg = LinearRegression(fit_intercept=False)
    reg.fit(control_data, control_data_median)
    reg.score(control_data, control_data_median)

    weighted_scaled_df_selected_features = scaled_df_selected_features.multiply(
        reg.coef_, axis=1
    )

    reg = LinearRegression(fit_intercept=False)
    reg.fit(control_median_data, control_median_data_median)
    reg.score(control_median_data, control_median_data_median)

    weighted_scaled_df_selected_features_median = (
        scaled_df_selected_features_median.multiply(reg.coef_, axis=1)
    )

# %% Ridge regression

RIDGE_REGRESSION = 0
if RIDGE_REGRESSION:
    reg = linear_model.BayesianRidge()
    reg.fit(X, Y)

# reg = LinearRegression()
# reg.fit(scaled_df, labels)
# reg.score(scaled_df, labels)
# scaled_df_selected_features
# %% Ridge regression
# from sklearn import linear_model
# ransac = linear_model.RANSACRegressor()
# ransac.fit(X, y)
# %%####### permutation_importance

# # mlp = MLPClassifier().fit(scaled_df, labels)
# from sklearn.inspection import permutation_importance

# r = permutation_importance(mlp, scaled_df, labels)

# %%####### Dimensionality reduction


def pca_on_df(df):
    return pd.DataFrame(
        PCA(n_components=2, whiten=False).fit_transform(df), index=df.index
    ).rename(columns={0: "PC1", 1: "PC2"})


df_dim_reduce_in = scaled_df
df_dim_reduce_median_in = scaled_df_median

try:
    df_dim_reduce_in = scaled_df_selected_features
    df_dim_reduce_median_in = scaled_df_selected_features_median
except:
    print("No selected features")
try:
    df_dim_reduce_in = weighted_scaled_df_selected_features
    df_dim_reduce_median_in = weighted_scaled_df_selected_features_median
except:
    print("No weighted features")


dimensional_pca_df = scaled_df_reduced = pca_on_df(df_dim_reduce_in)
# dimensional_pca_df = scaled_df_reduced = pca_on_df(scaled_df_standard)

scaled_df_median_reduced = pca_on_df(df_dim_reduce_median_in)
# scaled_df_median_no_nuclei_reduced = pca_on_df(scaled_df_median_no_nuclei)
# scaled_df_median_clean_reduced = pca_on_df(df_median_dim_reduce_clean_in)

# scaled_df_reduced = scaled_df_reduced.groupby(["ImageNumber","Date","Drug","Replicate","Conc /uM","Cell"]).median()

print(
    "Data dimensionality reduced from "
    + str(df_dim_reduce_in.shape[1])
    + " to "
    + str(scaled_df_reduced.shape[1])
)

plot = sns.relplot(
    x="PC1",
    y="PC2",
    col="Cell",
    hue="Drug",
    row="Conc /uM",
    data=dimensional_pca_df.reset_index(),
)
if SAVE_FIG:
    plot.savefig(f"{metadata()}_pca_facet_grid_conc.pdf")

# sns.scatterplot(
#     x="PC1", y="PC2", hue="Drug", size="Conc /uM", data=scaled_df_reduced.reset_index()
# )
# sns.relplot(x=0, y=1, hue="Drug", row="Conc /uM", data=scaled_df_reduced.reset_index())

# sns.relplot(x=0, y=1, hue="Cell", col="Drug", row="Conc /uM", data=scaled_df_reduced.reset_index())

# sns.relplot(x=0, y=1, col="Cell", hue="Drug", row="Conc /uM", data=scaled_df_reduced.reset_index())

# sns.relplot(x=0, y=1, hue="Cell", data=scaled_df_reduced.xs("Control",level="Drug",drop_level=False).reset_index())


# sns.relplot(x="PC1", y="PC2", hue="Cell", data=scaled_df_reduced.xs("Control",level="Drug",drop_level=False).reset_index(),kind="kde")

# sns.relplot(x=0, y=1, col="Cell", hue="Drug",
# data=scaled_df_reduced.xs([0],level="Conc /uM",drop_level=False).reset_index())

dimensional_pca_df = scaled_df_reduced = df_dim_reduce_in
scaled_df_median_reduced = df_dim_reduce_median_in

# sns.pairplot(vars='0', hue="Drug", data=scaled_df_reduced.reset_index())

# sns.distplot(scaled_df_reduced.reset_index(), hue="Drug")

# dimensional_pca_df_unstack = dimensional_pca_df.unstack(level="Drug")
# # dimensional_pca_df_corr = dimensional_pca_df.transpose().corr(method='pearson')
# dimensional_pca_df_median = dimensional_pca_df_unstack.groupby(
#     level="Conc /uM"
# ).median()


# %% ####### Small common DFs ##########

drug_df = scaled_df_reduced.xs(DRUG, level="Drug")
# sns.clustermap(drug_df_corr) # DO NOT RUN
drug_median_per_image = (
    drug_df.unstack(level="Conc /uM").groupby(level="ImageNumber").median().stack()
)
drug_median_per_conc = drug_df.groupby(level="Conc /uM").median()

dimensional_pca_df_median = scaled_df_reduced.groupby(
    ["ImageNumber", "Conc /uM", "Drug"]
).median()
drug_median_per_image_euclid = pd.DataFrame(
    euclidean_distances(drug_median_per_image), index=drug_median_per_image.index
)

drug_median_per_conc_euclid = pd.DataFrame(
    euclidean_distances(drug_median_per_conc), index=drug_median_per_conc.index
)

# df = scaled_df_reduced

# %%####### PCA Analysis ####

FLAG_PCA_OLD = 0  # Buggy
SAVE_FIG = 1
if FLAG_PCA_OLD:
    df_CONC = df.xs(CONC, level="Conc /uM", drop_level=False)
    df_median_CONC = df_median.xs(CONC, level="Conc /uM", drop_level=False)
    df_CONC_control = df.xs(CONC, level="Conc /uM", drop_level=False).append(
        df.xs("Control", level="Drug", drop_level=False)
    )
    df_median_CONC_control = df_median.xs(
        CONC, level="Conc /uM", drop_level=False
    ).append(df_median.xs("Control", level="Drug", drop_level=False))
    pca_variants = [
        (df, "per_cell", "Conc /uM", None),
        (df, "per_cell_size", "Drug", "Conc /uM"),
        (df_median, "per_organoid", "Conc /uM", None),
        (df_CONC, "per_cell_" + str(CONC) + "uM", "Drug", None),
        (df_median_CONC, "per_organoid_" + str(CONC) + "uM", "Drug", None),
        (df_CONC_control, "per_cell_" + str(CONC) + "uM_control", "Drug", None),
        (
            df_median_CONC_control,
            "per_organoid_" + str(CONC) + "uM_control",
            "Drug",
            None,
        ),
    ]
    for j, i in enumerate(pca_variants):
        pca_df, type, hue, size = i
        # print(j)
        index_pca = pca_df.index
        scaled_df_pca = pd.DataFrame(
            StandardScaler().fit_transform(pca_df), index=pca_df.index
        )
        principalComponents = PCA(n_components=2).fit_transform(scaled_df_pca)
        principalDf = pd.DataFrame(
            data=principalComponents,
            columns=["principal component 1", "principal component 2"],
            index=pca_df.index,
        )
        if hue == "Drug":
            cluster_num = len(scaled_df_pca.reset_index()["Drug"].unique())
            cluster_num
            scaled_df_pca_kmeans = KMeans(n_clusters=cluster_num, random_state=0).fit(
                principalComponents
            )
            # kmeans_accuracy = homogeneity_score(scaled_df_pca_kmeans.labels_,scaled_df_pca.reset_index()['Drug'])

            xx, yy = np.meshgrid(
                np.linspace(
                    principalComponents[:, 0].min(),
                    principalComponents[:, 0].max(),
                    500,
                ),
                np.linspace(
                    principalComponents[:, 1].min(),
                    principalComponents[:, 1].max(),
                    500,
                ),
            )
            Z = scaled_df_pca_kmeans.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(
                xx.shape
            )
            plt.imshow(
                Z,
                interpolation="nearest",
                extent=(xx.min(), xx.max(), yy.min(), yy.max()),
                cmap="Pastel2",
                aspect="auto",
                origin="lower",
            )
            # plt.title(f'K means score: {kmeans_accuracy:.2f}');
        sns.scatterplot(
            x="principal component 1",
            y="principal component 2",
            hue=hue,
            size=size,
            data=principalDf.reset_index(),
        )
        if SAVE_FIG:
            plt.savefig(metadata() + "_PCA_" + type + ".pdf")
        plt.show()
        if not (hue == "Drug"):
            g = sns.lmplot(
                data=principalDf.reset_index(),
                x="principal component 1",
                y="principal component 2",
                col="Drug",
                hue=hue,
                scatter=True,
                fit_reg=False,
                sharey=True,
                sharex=True,
                palette=sns.cubehelix_palette(9),
            ).add_legend()
            if SAVE_FIG:
                plt.savefig(metadata() + "_PCA_facet_" + type + ".pdf")
            plt.show()


# %%##### TSNE https://lvdmaaten.github.io/tsne/
# %%time

FLAGS_TSNSE = 0  # Slow
if FLAGS_TSNSE:
    from sklearn.manifold import TSNE

    tsne_in = scaled_df
    # tsne_median_in = scaled_df_median

    try:
        tsne_in = scaled_df_selected_features
        # df_dim_reduce_median_in = scaled_df_selected_features_median
    except:
        print("No selected features")
    try:
        tsne_in = weighted_scaled_df_selected_features
        # df_dim_reduce_median_in = weighted_scaled_df_selected_features_median
    except:
        print("No weighted features")
    # tsne_in = tsne_in.groupby(["ImageNumber","Date","Drug","Replicate","Conc /uM","Cell"]).median()

    X_embedded = TSNE(n_components=2, n_jobs=10).fit_transform(tsne_in)
    # principalComponents = pca.fit_transform(scaled_df)
    X_embeddedDf = pd.DataFrame(
        data=X_embedded,
        columns=["tsne component 1", "tsne component 2"],
        index=tsne_in.index,
    )
    # tnse_df_meta = X_embeddedDf.join(merged_df_extracted[['ImageNumber','Conc /uM','Drug']])
    # sns.scatterplot(x='principal component 1',y="principal component 2",hue="ImageNumber",data=X_embeddedDf)

    sns.scatterplot(
        x="tsne component 1",
        y="tsne component 2",
        hue="Drug",
        size="Conc /uM",
        data=X_embeddedDf.reset_index(),
    )

    if SAVE_FIG:
        plt.savefig(metadata() + "_TSNE.pdf")
        plt.show()
    g = sns.FacetGrid(X_embeddedDf.reset_index(), col="Drug", hue="Conc /uM")
    g.map(sns.scatterplot, "tsne component 1", "tsne component 2").add_legend()
    if SAVE_FIG:
        plt.savefig(metadata() + "_TSNE_facet_Drug_Conc.pdf")
        plt.show()

    g = sns.FacetGrid(X_embeddedDf.reset_index(), col="Conc /uM", hue="Drug")
    g.map(sns.scatterplot, "tsne component 1", "tsne component 2").add_legend()
    if SAVE_FIG:
        plt.savefig(metadata() + "_TSNE_facet_Conc_Drug.pdf")


# %%##### Fingerprints
DF_FINGERPRINTS = 0
if DF_FINGERPRINTS:
    # median_height=5
    def df_to_fingerprints(df, median_height=5, index_by="Drug"):
        # DRUGS = list(df.index.levels[3])
        LABELS = list(set(df.index.dropna().get_level_values(index_by).sort_values()))
        LABELS.sort()
        plt.rcParams["axes.grid"] = False
        fig, axes = plt.subplots(nrows=len(LABELS) * 2, figsize=(8, 7), dpi=150)
        upper = np.mean(df.values.flatten()) + 3 * np.std(df.values.flatten())
        upper
        lower = np.mean(df.values.flatten()) - 3 * np.std(df.values.flatten())
        lower
        for i, ax in enumerate(axes.flat):
            drug = LABELS[int(np.floor(i / 2))]
            drug
            image = df.xs(drug, level=index_by)
            finger_print = image.median(axis=0)
            finger_print_image = np.matlib.repmat(finger_print.values, median_height, 1)

            if i & 1:
                # im = ax.imshow(image, vmin=image.min().min(),
                #                vmax=image.max().max(),cmap='Spectral')
                im = ax.imshow(
                    finger_print_image,
                    vmin=lower,
                    vmax=upper,
                    cmap="Spectral",
                    interpolation="nearest",
                )
                ax.set_xticklabels([])
                ax.set_yticklabels([])
            else:
                im = ax.imshow(image, vmin=lower, vmax=upper, cmap="Spectral")
                ax.title.set_text(drug)
                # sns.heatmap(drug_df.values,ax=ax)
                ax.set(adjustable="box", aspect="auto", autoscale_on=False)
                ax.set_xticklabels([])

        fig.subplots_adjust(right=0.8)
        fig.colorbar(im, ax=axes.ravel().tolist())
        # fig.colorbar(im, cax=cbar_ax)

    dfs_to_finger = {
        "scaled_features_iso34_unsorted_conc": (
            scaled_df.xs("ISO34", level="Cell", drop_level=False),
            5,
            "Conc /uM",
        ),
        "pca_features_iso34_unsorted_conc": (
            dimensional_pca_df.xs("ISO34", level="Cell", drop_level=False),
            1,
            "Conc /uM",
        ),
        "scaled_features_iso34_unsorted": (
            scaled_df.xs("ISO34", level="Cell", drop_level=False),
            5,
            "Drug",
        ),
        "scaled_features_iso49_unsorted": (
            scaled_df.xs("ISO49", level="Cell", drop_level=False),
            5,
            "Drug",
        ),
        "scaled_features_iso49_sorted_conc": (
            scaled_df.xs("ISO49", level="Cell", drop_level=False)
            # .groupby(["ImageNumber","Date","Drug","Replicate","Conc /uM","Cell"]).mean()
            .sort_index(level=["Conc /uM"]),
            5,
            "Drug",
        ),
        "scaled_features_iso34_sorted_conc": (
            scaled_df.xs("ISO34", level="Cell", drop_level=False).sort_index(
                level=["Conc /uM"]
            ),
            5,
            "Drug",
        ),
        # "pca_features_iso34_sorted": dimensional_pca_df
        #     .xs("ISO34",level="Cell",drop_level=False)
        #     .sort_index(level=["Conc /uM"])
        #     ,
        "pca_features_iso34_unsorted": (
            dimensional_pca_df.xs("ISO34", level="Cell", drop_level=False),
            1,
            "Drug",
        ),
        "pca_features_iso49_unsorted": (
            dimensional_pca_df.xs("ISO49", level="Cell", drop_level=False),
            1,
            "Drug",
        ),
    }

    for df_to_finger_name in dfs_to_finger:
        df_to_fingerprints(*dfs_to_finger[df_to_finger_name])
        if SAVE_FIG:
            filename = metadata() + "_" + df_to_finger_name + ".pdf"
            print("Saving " + filename)
            plt.savefig(filename)
        plt.show()

# df_to = dfs_to_finger[next(iter(dfs_to_finger))]

# %%##### Correlations between drugs heirachical clusters, meaningless.

# sns.clustermap(dimensional_pca_df_median_drug.transpose().corr(method='pearson'))

FLAG_CORR = 0
if FLAG_CORR:
    dimensional_pca_df_median = dimensional_pca_df.groupby(
        ["Conc /uM", "Drug"]
    ).median()
    dimensional_pca_df_highconc_drug_corr = (
        dimensional_pca_df_median.xs(10, level="Conc /uM").transpose().corr()
    )

    drug_df_corr = drug_df.transpose().corr(method="pearson")

    drug_median_per_con_corr = drug_median_per_conc.transpose().corr(method="pearson")
    drug_median_per_image_corr = drug_median_per_image.transpose().corr(
        method="pearson"
    )

    sns.clustermap(drug_median_per_con_corr)
    plt.title("Median per conc Pearson")
    if SAVE_FIG:
        plt.savefig(metadata() + "_" + DRUG + "_median_per_con_corr.pdf")

    sns.clustermap(drug_median_per_image_corr)
    plt.title("Median per Organoid Pearson")
    if SAVE_FIG:
        plt.savefig(metadata() + "_" + DRUG + "_median_per_image_corr.pdf")

# %% Groupby does not preserve multindex

FLAG_MEDIAN_PER_EUCLID = 0
if FLAG_MEDIAN_PER_EUCLID:
    # drug_ecluid_df = pd.DataFrame(drug_ecluid,index=drug.index)
    sns.clustermap(drug_median_per_image_euclid)
    plt.title("Median Euclidian distance per Organoid")
    if SAVE_FIG:
        plt.savefig(metadata() + "_" + DRUG + "_median_per_image_euclid.pdf")
        plt.show()
    # sns.heatmap(drug_median_per_image_euclid)
    sns.clustermap(drug_median_per_conc_euclid)
    plt.title("Median Euclidian distance per conc")
    if SAVE_FIG:
        plt.savefig(metadata() + "_" + DRUG + "_median_per_conc_euclid.pdf")
        plt.show()
# %% Corr
FLAG_CORR_CONC_TO_FEATURE = 1
if FLAG_CORR_CONC_TO_FEATURE:
    df_CONC = df.copy()
    corr_feature = "Conc /uM"
    CONC_IDX = df.index.to_frame()[corr_feature]
    df_CONC[corr_feature] = CONC_IDX
    df_corr = (
        df_CONC.groupby("Drug")
        .apply(lambda x: x.corrwith(x[corr_feature]))
        .drop(columns=corr_feature)
    )
    wideform_df_corr = df_corr.T.stack().reset_index()
    wideform_df_corr_named = wideform_df_corr.rename(
        columns={"level_0": "Features", 0: "Correlation"}
    )
    # wideform_df_corr
    # sns.catplot(x='level_0',y=0,col='Drug',kind="bar",legend=False,data=wideform_df_corr);
    # SAVE_FIG = 1
    # wideform_df_corr_named

    kind = "strip"
    for kind in ["bar", "strip"]:
        sns.catplot(
            x="Features",
            y="Correlation",
            col="Drug",
            col_wrap=2,
            legend=False,
            kind=kind,
            data=wideform_df_corr_named,
        )
        plt.xticks([], [])
        metadata() + "_correlation_" + kind + ".pdf"
        if SAVE_FIG:
            plt.savefig(metadata() + "_correlation_" + kind + ".pdf")
            plt.show()

    correlation = pd.DataFrame(df.corrwith(CONC_IDX))
    # correlation.plot(kind='bar')
    # plt.show()
    correlation_scaled = pd.DataFrame(
        StandardScaler().fit_transform(correlation), index=correlation.index
    )

    wideform_df_corr_named_index = wideform_df_corr_named.copy()
    wideform_df_corr_named_index["Features"] = wideform_df_corr_named_index.index

    # wideform_df_corr_named_index
    # sns.catplot(x='Features',y='Correlation',col='Drug',col_wrap=2,legend=False,kind=kind,data=wideform_df_corr_named_index)

    wideform_df_corr_named_feature_index = wideform_df_corr_named.copy().set_index(
        "Features"
    )
    # wideform_df_corr_named.groupby('Drug').max()

    best_correlator = wideform_df_corr_named.sort_values("Correlation").drop_duplicates(
        ["Drug"], keep="last"
    )

    features = wideform_df_corr_named["Features"].unique()
    df_best_correlator = pd.DataFrame()
    best_correlator
    for index, row in best_correlator.iterrows():
        temp_df = pd.DataFrame(
            df_median.stack()
            .xs(row["Drug"], level="Drug", drop_level=False)
            .xs(row["Features"], level=-1, drop_level=False)
        )
        df_best_correlator = pd.concat([df_best_correlator, temp_df])
        # df_best_correlator.append()
    df_best_correlator_sns = df_best_correlator.reset_index().rename(
        columns={"level_6": "Feature", 0: "Magnitude"}
    )

    # df_best_correlator_sns.xs(key)
    sns.lmplot(
        x="Conc /uM",
        y="Magnitude",
        col="Feature",
        col_wrap=None,
        sharey=False,
        hue="Drug",
        data=df_best_correlator_sns,
    )
    if SAVE_FIG:
        plt.savefig(metadata() + "_best_correlator.pdf")

    for best_feature in best_correlator["Features"]:
        sns.lmplot(
            x="Conc /uM",
            col="Drug",
            col_wrap=None,
            hue="Drug",
            y=best_feature,
            sharey=False,
            data=df_median[best_feature].reset_index(),
        )
        # plt.savefig()
        plt.show()

# %% IC50 FULL

# https://www.graphpad.com/guides/prism/7/curve-fitting/reg_classic_dr.htm?toc=0&printWindow
# https://www.sciencegateway.org/protocols/cellbio/drug/hcic50.htm

# '''
# FLAG_IC50_FULL = 0
# if(FLAG_IC50_FULL):
#     drug_median_per_image_euclid_full_zero = df_median_euclid_stacked.\
#         xs(0.0410, level='Conc /uM II').\
#         droplevel(('ImageNumber II'))
#
#     bool_idx = drug_median_per_image_euclid_full_zero.index.get_level_values('Drug')\
#         == drug_median_per_image_euclid_full_zero.index.get_level_values('Drug II')
#
#     drug_to_drug = drug_median_per_image_euclid_full_zero.loc[bool_idx]
#     mean_drug_to_drug = drug_to_drug.groupby(
#         ['Drug', 'Conc /uM', 'ImageNumber']).mean()
#     # mean_drug_to_drug.xs(DRUG,level='Drug').shape
#
#     sns.lmplot(x='Conc /uM', y='Euclidian distance',
#                col='Drug', sharey=False,
#                data=mean_drug_to_drug.reset_index())
# '''


# %% IC50

from sklearn.metrics.pairwise import euclidean_distances

FLAG_IC50 = 1

if FLAG_IC50:

    def euclid_on_df(df):
        return pd.DataFrame(df.apply(numpy.linalg.norm, axis=1)).rename(
            columns={0: "Euclidean distance"}
        )

    data_in_ec50 = scaled_df_reduced
    # data_in_ec50 = dimensional_pca_df.iloc[:, 0:2]
    # data_in_ec50 = weighted_scaled_df_selected_features

    data_in_ec50 = data_in_ec50.groupby(
        ["ImageNumber", "Drug", "Date", "Cell", "Replicate", "Conc /uM"]
    ).median()

    # EC50s = [("scaled_df_median_reduced", scaled_df_median_reduced)]
    EC50s = [("data_in_ec50", data_in_ec50)]

    EC50 = EC50s
    # SAVE_FIG = 1
    for EC50 in EC50s:
        filename, data = EC50s[0]
        print(filename)
        data_euclid_df = euclid_on_df(data)

        # ed = pd.DataFrame(euclidean_distances(data_in_ec50
        #         .xs(["Control"],
        #             level=["Drug"],
        #             drop_level=False)
        #         ,
        #         data_in_ec50
        #         # .xs("ISO34",
        #         #     level="Cell",
        #         #     drop_level=False)
        #         ).T,
        #         index=data_in_ec50\
        #                 # .xs(
        #                 #             "ISO34",
        #                 #             level="Cell",
        #                 #             drop_level=False)
        #                 .index
        #                 )
        # data_euclid_df = ed.median(axis=1)\
        #                     .rename("Euclidean distance")\
        #                     .to_frame()

        """Keep data with 30 entries per point"""
        data_euclid_df_keep = (
            data_euclid_df.groupby(["Drug", "Cell"])
            .count()
            .query("`Euclidean distance` > 30")
            .astype(float)
        )

        # data_euclid_df_keep = data_euclid_df
        # data_euclid_df_keep
        data_euclid_df_subset = data_euclid_df
        # data_euclid_df_subset = pd.merge(
        #     data_euclid_df_keep,
        #     data_euclid_df,
        #     how="left",
        #     left_index=True,
        #     right_index=True,
        #     suffixes=("_drop", ""),
        # ).drop("Euclidean distance_drop", axis="columns")

        LINREGRESS_FLAG = 1
        if LINREGRESS_FLAG:

            data_euclid_df_subset.to_csv(root_dir + "euclid.csv")

            linear_regression = pd.DataFrame(
                data_euclid_df_subset.reset_index()
                .groupby(["Drug", "Cell"])
                .apply(
                    lambda x: pd.Series(
                        stats.linregress(x["Conc /uM"], x["Euclidean distance"]),
                        index=("m", "c", "R", "pvalue", "stderr"),
                    )
                )
            )

            linear_regression
            linear_regression["R2"] = linear_regression["R"] ** 2
            linear_regression["EC50 /uM"] = abs(
                ((0.5 - linear_regression["c"]) / linear_regression["m"])
            )
            linear_regression["CI"] = 2.58 * linear_regression["stderr"]
            # print(linear_regression)
            linear_regression["EC50_err"] = (
                linear_regression["EC50 /uM"] * linear_regression["CI"]
            )
            # linear_regression.reset_index()

            def draw_batplot_err(*args, **kwargs):
                return sns.barplot(x=args[0], y=args[1], yerr=args[2])

            g = sns.FacetGrid(linear_regression.reset_index(), col="Cell")
            g.map(draw_batplot_err, "Drug", "EC50 /uM", "EC50_err")
            if SAVE_FIG:
                plt.savefig(metadata() + "_EC50.pdf")
            plt.show()

            # sns.barplot(
            #     x="Drug",
            #     y="EC50",
            #     data=linear_regression.xs("ISO34", level="Cell").reset_index(),
            #     yerr=linear_regression.xs("ISO34", level="Cell")["EC50_err"] * 1,
            # )
            # plt.show()

        # SAVE_FIG = 1
        cat = sns.catplot(
            x="Conc /uM",
            y="Euclidean distance",
            row="Cell",
            col="Drug",
            data=data_euclid_df_subset.reset_index(),
            sharex=False,
            sharey=False,
            # x_estimator=np.median,
            # x_ci="ci",
            # x_bins=9,
            # markers=None,
            hue="Drug",
            kind="violin",
        )
        if SAVE_FIG:
            plt.savefig(metadata() + "_" + "violin" "_" + filename + ".pdf")

        grid_linear = sns.lmplot(
            x="Conc /uM",
            y="Euclidean distance",
            row="Cell",
            col="Drug",
            data=data_euclid_df_subset.reset_index(),
            sharex=False,
            sharey=False,
            x_estimator=np.median,
            x_ci="ci",
            x_bins=9,
            markers=None,
            hue="Drug",
        )

        grid_linear = sns.lmplot(
            x="Conc /uM",
            y="Euclidean distance",
            row="Cell",
            col="Drug",
            data=data_euclid_df_subset.reset_index(),
            sharex=False,
            sharey=False,
            hue="Drug",
        )

        if SAVE_FIG:
            plt.savefig(metadata() + "_" + "facet_linear" "_" + filename + ".pdf")

        grid = (
            sns.lmplot(
                x="Conc /uM",
                y="Euclidean distance",
                row="Cell",
                col="Drug",
                data=data_euclid_df_subset.reset_index(),
                sharex=False,
                sharey=False,
                x_estimator=np.mean,
                x_ci="ci",
                # x_bins=9,
                markers=None,
                hue="Drug",
            )
        ).set(xscale="log")
        if SAVE_FIG:
            plt.savefig(metadata() + "_" + "facet" "_" + filename + ".pdf")
        plt.show()
        grid = (
            sns.lmplot(
                x="Conc /uM",
                y="Euclidean distance",
                scatter=False,
                col="Cell",
                data=data_euclid_df_subset.reset_index(),
                sharex=True,
                sharey=False,
                x_estimator=np.mean,
                markers=None,
                hue="Drug",
            )
        ).set(xscale="log")

        if SAVE_FIG:
            plt.savefig(metadata() + "_" + "grouped" "_" + filename + ".pdf")
        plt.show()

        # %% Control

        grid = sns.lmplot(
            x="Date",
            y="Euclidean distance",
            scatter=False,
            col="Cell",
            data=data_euclid_df_subset.reset_index(),
            sharex=True,
            sharey=True,
            x_estimator=np.mean,
            markers=None,
            hue="Drug",
        )

# %% Date control
euclid_date_df = data_euclid_df_subset
grid = sns.catplot(
    x="Date",
    y="Euclidean distance",
    hue="Drug",
    # scatter=False,
    col="Drug",
    data=euclid_date_df.reset_index(),
    # sharex=True,
    # sharey=True,
    # x_estimator=np.median,
    # markers=None,
    kind="violin",
)


# %% IC50
# FLAG_IC50_MANUAL = 0
# if(FLAG_IC50_MANUAL):
#     # drug_median_per_image_euclid
#     # drug_median_per_image_euclid.index.to_frame()['Conc /uM'].unique()
#     distance_to_control = drug_median_per_image_euclid.xs(
#         0.0410, level='Conc /uM')
#     distance_to_control_indexed = distance_to_control\
#         .transpose()\
#         .set_index(drug_median_per_image_euclid.index)\
#         .droplevel('ImageNumber')
#     # from scipy.stats.stats import linregress
#     # from sklearn.linear_model import LogisticRegression
#     # logreg = LogisticRegression()
#
#     mean_distance_to_control_indexed = pd.DataFrame(
#         distance_to_control_indexed.mean(axis=1).reset_index())
#
#     x_data = mean_distance_to_control_indexed.reset_index()['Conc /uM'].values
#     y_data = mean_distance_to_control_indexed.reset_index()[0].values
#
#     sns.violinplot(x='Conc /uM', y=0,
#                    data=mean_distance_to_control_indexed).set(ylabel='Euclidean distance')
#     plt.savefig(f'{metadata()}_{DRUG}_violinplot.pdf')
#     plt.show()
#
#     # kstest(scipy.stats.t.rvs(3,size=1000),'norm')
#     from scipy.stats import linregress
#     # kstest(y_data,'norm')
#     for i in [(np.median, 'median', 'linear'), (np.median, 'median', 'log'), (np.mean, 'mean', 'linear'), (np.mean, 'mean', 'log')]:
#         stat, file, axes = i
#         m, c, R, pvalue, stderr = linregress(x_data, y_data)
#         CI = 2.58 * stderr
#         CI * 100
#         # https://www.sciencegateway.org/protocols/cellbio/drug/hcic50.htm
#         EC50 = (0.5 - m) / c
#         title = f'EC50: {EC50:.2g}±{EC50*CI:.2g} | pvalue: {pvalue:.2g} | R: {R:.2g} | stderr: {stderr:.2g}'
#         title = f'95%CI:{CI*100:.2g}% | pvalue: {pvalue:.2g} | R: {R:.2g} | stderr: {stderr:.2g}'
#         print(title)
#         plot_sns = sns.regplot(x=x_data, y=y_data, x_estimator=stat)
#         # plot_sns = sns.regplot(x=x_data,y=y_data)
#         plot_sns.set(xscale=axes, xlabel="Conc /uM",
#                      ylabel="Euclidian distance to Control Organoids")
#         plt.title(title)
#         plt.tight_layout()
#         if(SAVE_FIG):
#             plt.savefig(metadata() + '_' + DRUG + "_" + file +
#                         "_" + axes + "_distance_to_control.pdf")
#         plt.show()

# %% ######### PCA ###############
# sns.scatterplot(x='principal component 1',
#                 y="principal component 2",
#                 hue="Conc /uM",data=principalDf.reset_index())
# # CHANNELS =1
# if(SAVE_FIG):plt.savefig("Channels_"+str(CHANNELS)+"_"+DRUG+"_df_pca.pdf");plt.show()
#
# pca = PCA(n_components=2).fit_transform(drug_median_per_image)
# principalDf = pd.DataFrame(data = pca,
#                             columns = ['principal component 1', 'principal component 2'],
#                             index=drug_median_per_image.index)
#
# sns.scatterplot(x='principal component 1',
#                 y="principal component 2",
#                 hue="Conc /uM",data=principalDf.reset_index())
# if(SAVE_FIG):plt.savefig("Channels_"+str(CHANNELS)+"_"+DRUG+"_median_per_image_pca.pdf");plt.show()
#
# '''
# # TSNE
# TSNE_FLAG = 0
# if(TSNE_FLAG):
#     from sklearn.manifold import TSNE
#
#     TSNE = TSNE(n_components=2).fit_transform(G007_median_per_image)
#     principalDf = pd.DataFrame(data=TSNE,
#                                columns=['principal component 1',
#                                         'principal component 2'],
#                                index=G007_median_per_image.index)
#
#     sns.scatterplot(x='principal component 1',
#                     y="principal component 2",
#                     hue="Conc /uM", data=principalDf.reset_index())
#     if(SAVE_FIG):
#         plt.savefig("G007_median_per_image_tnse.pdf")
#         plt.show()
#     # use dimensionality reduction first
#
# '''
#
# '''
# Maximum correlation. For a set of n doses for each compound, the NxN correlation matrix is computed between all pairs of concentrations, and the maximum value is used as the dose-independent similarity score72.
# '''
# G007_df = df.xs(DRUG, level='Drug')
# FLAG_MAXCORR = 0
# if(FLAG_MAXCORR):
#     G007_df
#     good = G007_df.unstack(level='Conc /uM').corr().stack()
#     G007_df_ImageNumber = G007_df.unstack(level='ImageNumber').corr().stack()
#     good = good.replace(1, np.NaN)
#     yes = good.groupby(level=[1, 2]).max()
#     yes
#     # aaaa = yes.unstack()
#     sns.clustermap(yes)
#     aa = G007_df.transpose().corr()
#     aa
#     aa.unstack()
#     aa.reset_index(col_level=2)
#     aa.stack(level=0)
#
#     G007_df.transpose().corr()
#     Maximum_correlation_g007_2d = (G007_df.transpose().corr()).groupby(
#         'Conc /uM', level='Conc /uM').max()
#     Maximum_correlation_g007_2d
#     Maximum_correlation_g007 = (
#         G007_df.transpose().corr()).groupby(level='Conc /uM').max()
#     Maximum_correlation_g007
#     Maximum_correlation_g007_2 = Maximum_correlation_g007.transpose().groupby(
#         level='Conc /uM').max()
#     # Maximum_correlation_g007 = (g007.transpose().corr()).groupby(level='Conc /uM').max(axis=0).max(axis=1)
#     # (-(g007.transpose().corr().abs()-1)).max().max()
#     # disds = (-(g007.transpose().corr()-1)).max().max();disds
#     sns.clustermap(Maximum_correlation_g007_2)
#     MC = Maximum_correlation_g007_2.replace(1, np.NaN).max().max()
#     Maximum_correlation_g007_2.max()
#     sns.clustermap(Maximum_correlation_g007)
#
# '''Titration-invariant similarity score. First, the titration series of a compound is built by computing the similarity score between each dose and negative controls. Then, the set of scores is sorted by increasing dose and is split into subseries by using a window of certain size (for instance, windows of three doses). Two compounds are compared by computing the correlation between their subwindows, and only the maximum value is retained83.'''


# https://science.sciencemag.org/content/306/5699/1194
#
# FLAG_TISS = 0
# if(FLAG_TISS):
#     control = g007.xs(list(set(extracted_data['Conc /uM']))[1],level='Conc /uM')
#
#     # euclid_fun = lambda(x): euclidean_distances(x,x)
#     g007.groupby(level='Conc /uM').apply(lambda x: euclidean_distances(x,x))
#
#
#     ecluid_g007 = euclidean_distances(g007,g007)
#     g007_ecluid_df = pd.DataFrame(g007_ecluid,index=g007.index)
#     index = g007_ecluid_df.index
#     euclid_df_indexed = g007_ecluid_df.transpose().set_index(index)
#
#
#     doses = list(set(extracted_data['Conc /uM']))
#     doses = euclid_df_indexed.index.levels[1]
#     euclid_df_indexed.xs(doses[0],level='Conc /uM')
#
#     control = g007.xs(doses[0],level='Conc /uM').iloc[0]
#     drug_1 = g007.xs(doses[0],level='Conc /uM').iloc[1]
#
#     control
#
#     from scipy.stats import ks_2samp
#
#     d_value,p_value = ks_2samp(control,drug_1)
#     d_value
#     # Scale D by population
#
#     #The vectors are then individually scaled by a z-score (standard score) by (x_i - mean(x)) / std(x), where x_i is an element in the vector x.
#     from scipy.stats import zscore
#
#     d_scale = zscore(d_value)
#
# from rpy2.robjects.packages import importrs
#     base = importr('base')
#     # import R's "utils" package
#     utils = importr('utils')
#     import rpy2.robjects.packages as rpackages
#
#     # import R's utility package
#     utils = rpackages.importr('utils')
#
#     # select a mirror for R packages
#     utils.chooseCRANmirror(ind=1) # select the first mirror in the list
#     # R package names
#     packnames = ('ggplot2', 'hexbin')
#
#     # R vector of strings
#     from rpy2.robjects.vectors import StrVector
#     from rpy2 import robjects
#     # Selectively install what needs to be install.
#     # We are fancy, just because we can.
#     # names_to_install = [x for packnames if not rpackages.isinstalled(x)]
#     # if len(names_to_install) > 0:
#         # utils.install_packages(StrVector(names_to_install))
#
#
#     test = robjects.r('''
#         # create a function `f`
#             f <- function(r, verbose=FALSE) {
#                 if (verbose) {
#                     cat("I am calling f().\n")
#                 }
#                 2 * pi * r
#             }
#             # call the function `f` with argument value 3
#             f(3)
#     ''')
#     test = robjects.r('''
#         # create a function `f`
#             if (!require(devtools)) install.packages('devtools')
#             devtools::install_github('Swarchal/TISS')
#     ''')
#
#
#     test(0)
#
#     # r_f = robjects.globalenv['f']
#     # print(r_f.r_repr())
#     r_f = robjects.r['f']
#     r_f(2)
#
#     tiss = base = importr('Swarchal/TISS')
#     library('TISS')
#
#     utils.install_packages('devtools')
#

# '''
# import os
# os.environ['R_HOME'] = '/opt/miniconda3/envs/py37/lib/R'
# %load_ext rpy2.ipython
# %%R -i df -w 5 -h 5 --units in -r 200
# # install.packages('devtools')
# library('devtools')
# # # install.packages("ggplot2", repos='http://cran.us.r-project.org', quiet=TRUE)
# # #install.packages("ggplot2",repos='http://cran.us.r-project.org')
# # #if (!require(devtools))
# # #install.packages('devtools',repos='http://cran.us.r-project.org')
# # devtools::install_github('Swarchal/TISS')
# '''

# %%

# %%

# %%

# %%

# %%
