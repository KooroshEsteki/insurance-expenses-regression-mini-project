import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from pandas.plotting import scatter_matrix

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.compose import TransformedTargetRegressor

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR

from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


# 
# STEP 1 + STEP 2: Read the dataset
# 

DATA_PATH = "insurance.csv"

df = pd.read_csv(DATA_PATH)

print("Dataset shape:")
print(df.shape)

print("\nFirst five rows:")
print(df.head())

print("\nColumn names:")
print(df.columns.tolist())

print("\nMissing values per column:")
print(df.isna().sum())

# Drop missing values if any exist
df = df.dropna()

print("\nShape after dropping missing values:")
print(df.shape)

print("\nDataset information:")
print(df.info())

print("\nSummary statistics:")
print(df.describe())


# 
# STEP 3: Preprocessing
# Target variable: charges
#

target = "charges"

X = df.drop(columns=[target])
y = df[target]

numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
categorical_features = X.select_dtypes(include=["object"]).columns.tolist()

print("\nNumeric features:")
print(numeric_features)

print("\nCategorical features:")
print(categorical_features)

print("\nTarget variable:")
print(target)


# 
# STEP 4: Prepare Visualizations
# EDA, pair plots, correlation heat map, and box plots
# 

os.makedirs("visualizations", exist_ok=True)

# 
# Histogram of target variable
# 

plt.figure(figsize=(8, 5))
plt.hist(df["charges"], bins=30)
plt.xlabel("Charges")
plt.ylabel("Frequency")
plt.title("Distribution of Insurance Charges")
plt.tight_layout()
plt.savefig("visualizations/charges_distribution.png")
plt.close()


#
# Box plots for numeric features
# 

for col in numeric_features + [target]:
    plt.figure(figsize=(7, 5))
    plt.boxplot(df[col])
    plt.ylabel(col)
    plt.title(f"Box Plot of {col}")
    plt.tight_layout()
    plt.savefig(f"visualizations/boxplot_{col}.png")
    plt.close()


# 
# Correlation matrix heat map
#

numeric_df = df.select_dtypes(include=["int64", "float64"])
corr_matrix = numeric_df.corr()

plt.figure(figsize=(8, 6))
plt.imshow(corr_matrix, cmap="viridis", aspect="auto")
plt.colorbar(label="Correlation")
plt.xticks(range(len(corr_matrix.columns)), corr_matrix.columns, rotation=45, ha="right")
plt.yticks(range(len(corr_matrix.columns)), corr_matrix.columns)
plt.title("Correlation Matrix Heat Map")
plt.tight_layout()
plt.savefig("visualizations/correlation_heatmap.png")
plt.close()

print("\nCorrelation matrix:")
print(corr_matrix)


# 
# Pair plot using scatter_matrix
# 

scatter_matrix(
    numeric_df,
    figsize=(10, 10),
    diagonal="hist"
)
plt.suptitle("Pair Plot of Numeric Features", y=1.02)
plt.tight_layout()
plt.savefig("visualizations/pair_plot_numeric_features.png")
plt.close()


print("\nVisualizations saved in the visualizations folder.")


# 
# Preprocessing pipeline
# Scaling numeric features
# Encoding categorical features
# 

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
    ]
)


# 
# STEP 5: Split the data
# 80% training set
# 10% validation set
# 10% test set
# 

X_train, X_temp, y_train, y_temp = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.50,
    random_state=42
)

print("\nTraining set:")
print(X_train.shape, y_train.shape)

print("\nValidation set:")
print(X_val.shape, y_val.shape)

print("\nTest set:")
print(X_test.shape, y_test.shape)


# 
# Helper function for model evaluation
# 

def evaluate_regression_model(model_name, model, X_train, X_val, X_test, y_train, y_val, y_test):
    model.fit(X_train, y_train)

    y_val_pred = model.predict(X_val)
    y_test_pred = model.predict(X_test)

    val_r2 = r2_score(y_val, y_val_pred)
    test_r2 = r2_score(y_test, y_test_pred)

    val_mse = mean_squared_error(y_val, y_val_pred)
    test_mse = mean_squared_error(y_test, y_test_pred)

    val_mae = mean_absolute_error(y_val, y_val_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)

    print("\n====================================================")
    print(model_name)
    print("====================================================")

    print("Validation R2:")
    print(round(val_r2, 4))

    print("Test R2:")
    print(round(test_r2, 4))

    print("Validation MSE:")
    print(round(val_mse, 4))

    print("Test MSE:")
    print(round(test_mse, 4))

    print("Validation MAE:")
    print(round(val_mae, 4))

    print("Test MAE:")
    print(round(test_mae, 4))

    return {
        "model": model_name,
        "validation_r2": val_r2,
        "test_r2": test_r2,
        "validation_mse": val_mse,
        "test_mse": test_mse,
        "validation_mae": val_mae,
        "test_mae": test_mae
    }


# 
# STEP 6: Train Regressors
# Decision Tree, Random Forest, and SVR
# Need minimum R2 score of 82%
# 

results = []


# 
# Decision Tree Regressor
# Choose best criterion using validation R2
#

tree_criteria = [
    "squared_error",
    "friedman_mse",
    "absolute_error",
    "poisson"
]

tree_results = []

for criterion in tree_criteria:
    tree_model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("regressor", DecisionTreeRegressor(
                criterion=criterion,
                max_depth=5,
                random_state=42
            ))
        ]
    )

    tree_model.fit(X_train, y_train)
    y_val_pred = tree_model.predict(X_val)
    val_r2 = r2_score(y_val, y_val_pred)

    tree_results.append({
        "criterion": criterion,
        "validation_r2": val_r2
    })

    print(f"Decision Tree criterion = {criterion:15s} | Validation R2 = {val_r2:.4f}")


tree_results_df = pd.DataFrame(tree_results)

print("\nDecision Tree criterion comparison:")
print(tree_results_df)

best_tree_row = tree_results_df.loc[tree_results_df["validation_r2"].idxmax()]
best_tree_criterion = best_tree_row["criterion"]

print("\nBest Decision Tree criterion:")
print(best_tree_criterion)

final_tree_model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("regressor", DecisionTreeRegressor(
            criterion=best_tree_criterion,
            max_depth=5,
            random_state=42
        ))
    ]
)

results.append(
    evaluate_regression_model(
        "Decision Tree Regressor",
        final_tree_model,
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test
    )
)


# 
# Random Forest Regressor
# Choose best criterion using validation R2
# 

forest_criteria = [
    "squared_error",
    "friedman_mse",
    "absolute_error",
    "poisson"
]

forest_results = []

for criterion in forest_criteria:
    forest_model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("regressor", RandomForestRegressor(
                criterion=criterion,
                n_estimators=200,
                max_depth=8,
                random_state=42,
                n_jobs=-1
            ))
        ]
    )

    forest_model.fit(X_train, y_train)
    y_val_pred = forest_model.predict(X_val)
    val_r2 = r2_score(y_val, y_val_pred)

    forest_results.append({
        "criterion": criterion,
        "validation_r2": val_r2
    })

    print(f"Random Forest criterion = {criterion:15s} | Validation R2 = {val_r2:.4f}")


forest_results_df = pd.DataFrame(forest_results)

print("\nRandom Forest criterion comparison:")
print(forest_results_df)

best_forest_row = forest_results_df.loc[forest_results_df["validation_r2"].idxmax()]
best_forest_criterion = best_forest_row["criterion"]

print("\nBest Random Forest criterion:")
print(best_forest_criterion)

final_forest_model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("regressor", RandomForestRegressor(
            criterion=best_forest_criterion,
            n_estimators=200,
            max_depth=8,
            random_state=42,
            n_jobs=-1
        ))
    ]
)

results.append(
    evaluate_regression_model(
        "Random Forest Regressor",
        final_forest_model,
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test
    )
)


# 
# Support Vector Regressor
# Use target transformation because charges are skewed
# 

svr_base_model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("regressor", SVR(
            kernel="rbf",
            C=100,
            gamma="scale",
            epsilon=0.1
        ))
    ]
)

svr_model = TransformedTargetRegressor(
    regressor=svr_base_model,
    func=np.log1p,
    inverse_func=np.expm1
)

results.append(
    evaluate_regression_model(
        "Support Vector Regressor",
        svr_model,
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test
    )
)


# 
# Final comparison
# 

results_df = pd.DataFrame(results)

print("\n====================================================")
print("Final Model Comparison")
print("====================================================")

print(results_df.sort_values(by="test_r2", ascending=False))

best_model = results_df.loc[results_df["test_r2"].idxmax()]

print("\nBest model based on test R2:")
print(best_model)

print("\nMinimum R2 requirement:")
print("The project requires at least 82% R2 score.")

if best_model["test_r2"] >= 0.82:
    print("Requirement satisfied: the best model reaches at least 82% R2.")
else:
    print("Requirement not satisfied: try tuning model parameters.")


print("\nComment:")
print(
    "Decision Tree, Random Forest, and SVR regression models were trained and compared. "
    "The Decision Tree and Random Forest criteria were tuned using validation R2. "
    "The best model was selected based on test R2. MSE and MAE were also reported."
)
