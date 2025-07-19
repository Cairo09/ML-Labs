#Q1
import pandas as pd
import numpy as np

def load_data(file_path):
    df = pd.read_excel(file_path, sheet_name="Purchase data", usecols='A:E')
    return df

def matrices_AC(df):
    A = df.iloc[:, 1:-1].values  
    C = df.iloc[:, -1].values.reshape(-1, 1)  
    return A, C

def analyze(matrix_A):
    vectorspace_dim = matrix_A.shape[1]
    no_of_vectors = matrix_A.shape[0]
    rank = np.linalg.matrix_rank(matrix_A)
    return vectorspace_dim, no_of_vectors, rank

def computeprices(matrix_A, matrix_C):
    pseudo_inv = np.linalg.pinv(matrix_A)
    prices_vector = pseudo_inv @ matrix_C
    return prices_vector

if __name__ == "__main__":
    file_path = "Lab Session Data.xlsx"
    data = load_data(file_path)
    A, C = matrices_AC(data)
    dimension, num_vectors, rank = analyze(A)
    product_prices = computeprices(A, C)

    print("Dimensionality of the vector space:", dimension)
    print("Number of vectors:", num_vectors)
    print("Rank:", rank)
    print("Prices of each products:")
    print(product_prices)


    #Q2
import pandas as pd
import numpy as np

def load_data(file_path):
    df = pd.read_excel(file_path, sheet_name="Purchase data", usecols='A:E')
    return df

def matrices_AC(df):
    A = df.iloc[:, 1:-1].values  
    C = df.iloc[:, -1].values.reshape(-1, 1)  
    return A, C

def analyze(matrix_A):
    vectorspace_dim = matrix_A.shape[1]
    no_of_vectors = matrix_A.shape[0]
    rank = np.linalg.matrix_rank(matrix_A)
    return vectorspace_dim, no_of_vectors, rank

def computeprices(matrix_A, matrix_C):
    pseudo_inv = np.linalg.pinv(matrix_A)
    prices_vector = pseudo_inv @ matrix_C
    return prices_vector

def classify(matrix_C):
    threshold = 200
    labels = np.where(matrix_C > threshold, "Rich", "Poor")
    return labels

if __name__ == "__main__":
    file_path = "Lab Session Data.xlsx"
    data = load_data(file_path)
    A, C = matrices_AC(data)
    dimension, num_vectors, rank = analyze(A)
    product_prices = computeprices(A, C)
    labels = classify(C)

    print("Dimensionality of the vector space:", dimension)
    print("Number of vectors:", num_vectors)
    print("Rank:", rank)
    print("Prices of each products:")
    print(product_prices)
    print("Customer classes (Rich/Poor):")
    print(labels)

#Q3
import pandas as pd
import statistics
import matplotlib.pyplot as plt

def load_irctc_data(file_path):
    df = pd.read_excel(file_path, sheet_name="IRCTC Stock Price", usecols="A:I")
    df["Day"] = pd.to_datetime(df["Date"]).dt.day_name()
    df["Month"] = pd.to_datetime(df["Date"]).dt.month_name()
    return df

def stats_price(df):
    mean_price = statistics.mean(df["Price"])
    var_price = statistics.variance(df["Price"])
    return mean_price, var_price

def mean_wednesday(df):
    wed_prices = df[df["Day"] == "Wednesday"]["Price"]
    return statistics.mean(wed_prices)

def mean_april(df):
    april_prices = df[df["Month"] == "April"]["Price"]
    return statistics.mean(april_prices)

def prob_loss(df):
    chg = df["Chg%"].apply(lambda x: float(str(x).replace('%', '')))
    loss_days = chg.apply(lambda x: x < 0)
    return loss_days.sum() / len(chg)

def prob_profit_wed(df):
    wed = df[df["Day"] == "Wednesday"]
    chg = wed["Chg%"].apply(lambda x: float(str(x).replace('%', '')))
    profit = chg.apply(lambda x: x > 0)
    return profit.sum() / len(chg)

def cond_prob_profit_given_wed(df):
    chg_all = df["Chg%"].apply(lambda x: float(str(x).replace('%', '')))
    profit_all = df[(chg_all > 0)]
    wed_profit = profit_all[profit_all["Day"] == "Wednesday"]
    return len(wed_profit) / len(df[df["Day"] == "Wednesday"])

def plot_chg_vs_day(df):
    chg = df["Chg%"].apply(lambda x: float(str(x).replace('%', '')))
    df["Chg_val"] = chg
    days = df["Day"]
    plt.figure(figsize=(10, 5))
    plt.scatter(days, chg, color='blue')
    plt.title("Chg% vs Day")
    plt.xlabel("Day of Week")
    plt.ylabel("Chg%")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    file_path = "Lab Session Data.xlsx"
    df = load_irctc_data(file_path)

    mean_price, var_price = stats_price(df)
    print("Mean of Price:", mean_price)
    print("Variance of Price:", var_price)

    wed_mean = mean_wednesday(df)
    print("Mean Price on Wednesdays:", wed_mean)

    april_mean = mean_april(df)
    print("Mean Price in April:", april_mean)

    p_loss = prob_loss(df)
    print("Probability of loss:", p_loss)

    p_profit_wed = prob_profit_wed(df)
    print("Probability of profit on Wednesday:", p_profit_wed)

    p_cond = cond_prob_profit_given_wed(df)
    print("P(Profit | Wednesday):", p_cond)

    plot_chg_vs_day(df)


#Q4
import pandas as pd
import numpy as np
import statistics
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns

# Load the Excel sheet into a DataFrame
def load_data(file_path, sheet_name):
    df = pd.read_excel(file_path, sheet_name=sheet_name)
    return df

# Identify types of each column (Nominal, Boolean, Numeric)
def identify_column_types(df):
    column_types = {}
    for col in df.columns:
        if df[col].dtype == 'object':
            unique_vals = df[col].dropna().unique()
            if all(val in ['t', 'f'] for val in unique_vals):
                column_types[col] = 'Boolean'
            else:
                column_types[col] = 'Nominal'
        elif np.issubdtype(df[col].dtype, np.number):
            column_types[col] = 'Numeric'
        else:
            column_types[col] = 'Unknown'
    return column_types

# Suggest encoding based on data type
def suggest_encoding(column_types):
    encodings = {}
    for col, col_type in column_types.items():
        if col_type == 'Nominal':
            encodings[col] = 'One-Hot Encoding'
        elif col_type == 'Boolean':
            encodings[col] = 'Label Encoding (t=1, f=0)'
        elif col_type == 'Numeric':
            encodings[col] = 'No encoding needed'
        else:
            encodings[col] = 'Unknown'
    return encodings

# Return the min and max range for numeric columns
def get_numeric_ranges(df, column_types):
    ranges = {}
    for col, col_type in column_types.items():
        if col_type == 'Numeric':
            ranges[col] = (df[col].min(), df[col].max())
    return ranges

# Count missing values in each column
def get_missing_values(df):
    return df.isnull().sum().to_dict()

# Detect outliers using Z-score method
def detect_outliers(df, column_types, threshold=3.0):
    outliers = {}
    for col, col_type in column_types.items():
        if col_type == 'Numeric':
            mean = df[col].mean()
            std = df[col].std()
            z_scores = (df[col] - mean) / std
            outliers[col] = (np.abs(z_scores) > threshold).sum()
    return outliers

# Calculate mean and variance for numeric columns
def calculate_stats(df, column_types):
    stats = {}
    for col, col_type in column_types.items():
        if col_type == 'Numeric':
            clean_col = df[col].dropna()
            stats[col] = {
                'mean': statistics.mean(clean_col),
                'std_dev': statistics.stdev(clean_col)
            }
    return stats

# Main function
if __name__ == "__main__":
    file = "Lab Session Data.xlsx"
    sheet = "thyroid0387_UCI"

    df = load_data(file, sheet)
    col_types = identify_column_types(df)

    print("Column Types:")
    for col, dtype in col_types.items():
        print(f"{col}: {dtype}")

    encodings = suggest_encoding(col_types)
    print("\nEncoding Suggestions:")
    for col, enc in encodings.items():
        print(f"{col}: {enc}")

    missing = get_missing_values(df)
    print("\nMissing Values:")
    for col, count in missing.items():
        print(f"{col}: {count}")

    ranges = get_numeric_ranges(df, col_types)
    print("\nNumeric Ranges:")
    for col, (min_val, max_val) in ranges.items():
        print(f"{col}: Min = {min_val}, Max = {max_val}")

    outliers = detect_outliers(df, col_types)
    print("\nOutlier Counts (Z-score > 3):")
    for col, count in outliers.items():
        print(f"{col}: {count}")

    stats = calculate_stats(df, col_types)
    print("\nMean and Standard Deviation:")
    for col, stat in stats.items():
        print(f"{col}: Mean = {stat['mean']:.2f}, Std Dev = {stat['std_dev']:.2f}")


#Q5
import pandas as pd

def load_data(file_path, sheet_name):
    return pd.read_excel(file_path, sheet_name=sheet_name)

def get_binary_columns(df):
    # Identify columns with only binary values ('t'/'f', 1/0)
    binary_cols = []
    for col in df.columns:
        unique_vals = df[col].dropna().unique()
        if set(unique_vals).issubset({'t', 'f', 1, 0, '1', '0'}):
            binary_cols.append(col)
    return binary_cols

def convert_to_binary(df, binary_cols):
    # Convert 't'/'f' strings to 1/0 integers
    return df[binary_cols].replace({'t': 1, 'f': 0, '1': 1, '0': 0})

def compute_similarity_metrics(vec1, vec2):
    f11 = sum((vec1 == 1) & (vec2 == 1))
    f00 = sum((vec1 == 0) & (vec2 == 0))
    f10 = sum((vec1 == 1) & (vec2 == 0))
    f01 = sum((vec1 == 0) & (vec2 == 1))

    jc = f11 / (f11 + f10 + f01) if (f11 + f10 + f01) > 0 else 0
    smc = (f11 + f00) / (f11 + f00 + f10 + f01) if (f11 + f00 + f10 + f01) > 0 else 0

    return f11, f00, f10, f01, jc, smc

# Main program
if __name__ == "__main__":
    file_path = "Lab Session Data.xlsx"
    sheet = "thyroid0387_UCI"

    df = load_data(file_path, sheet)
    binary_cols = get_binary_columns(df)
    binary_df = convert_to_binary(df, binary_cols)

    vector1 = binary_df.iloc[0]
    vector2 = binary_df.iloc[1]

    f11, f00, f10, f01, jc, smc = compute_similarity_metrics(vector1, vector2)

    print("Similarity between first two binary vectors:")
    print(f"f11 = {f11}")
    print(f"f00 = {f00}")
    print(f"f10 = {f10}")
    print(f"f01 = {f01}")
    print(f"Jaccard Coefficient (JC) = {jc:.4f}")
    print(f"Simple Matching Coefficient (SMC) = {smc:.4f}")

#Q6
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics.pairwise import cosine_similarity

def read_data(file_path, sheet_name):
    return pd.read_excel(file_path, sheet_name=sheet_name)

def transform_categorical(df):
    df_copy = df.copy()
    encoders = {}
    for col in df_copy.columns:
        if df_copy[col].dtype == 'object':
            le = LabelEncoder()
            df_copy[col] = le.fit_transform(df_copy[col].astype(str))
            encoders[col] = le
    return df_copy

def get_cosine_score(v1, v2):
    v1 = v1.reshape(1, -1)
    v2 = v2.reshape(1, -1)
    score = cosine_similarity(v1, v2)[0][0]
    return score

if __name__ == "__main__":
    data_file = "Lab Session Data.xlsx"
    data_sheet = "thyroid0387_UCI"

    raw_df = read_data(data_file, data_sheet)
    encoded_df = transform_categorical(raw_df)

    first_vec = encoded_df.iloc[0].values
    second_vec = encoded_df.iloc[1].values

    cosine_value = get_cosine_score(first_vec, second_vec)

    print(f"Cosine Similarity Measure between full vectors: {cosine_value:.4f}")


#Q7
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics.pairwise import cosine_similarity

def load_sheet_data(filepath: str, sheet_name: str) -> pd.DataFrame:
    return pd.read_excel(filepath, sheet_name=sheet_name)

def extract_binary_columns(df: pd.DataFrame) -> pd.DataFrame:
    cols = [col for col in df.columns if set(df[col].dropna().unique()).issubset({'t', 'f'})]
    return df[cols].replace({'t': 1, 'f': 0})

def encode_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df_enc = df.copy()
    for col in df_enc.columns:
        if df_enc[col].dtype == 'object':
            df_enc[col] = LabelEncoder().fit_transform(df_enc[col].astype(str))
    return df_enc

def calc_jc_smc(v1, v2):
    a = np.sum((v1 == 1) & (v2 == 1))
    d = np.sum((v1 == 0) & (v2 == 0))
    b = np.sum((v1 == 1) & (v2 == 0))
    c = np.sum((v1 == 0) & (v2 == 1))
    jc = a / (a + b + c) if (a + b + c) else 0
    smc = (a + d) / (a + b + c + d) if (a + b + c + d) else 0
    return jc, smc

def build_jc_smc_matrices(binary_df: pd.DataFrame) -> tuple:
    n = len(binary_df)
    jc_mat = np.zeros((n, n))
    smc_mat = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            jc, smc = calc_jc_smc(binary_df.iloc[i], binary_df.iloc[j])
            jc_mat[i, j] = jc
            smc_mat[i, j] = smc
    return jc_mat, smc_mat

def compute_cosine_sim(encoded_df: pd.DataFrame) -> np.ndarray:
    return cosine_similarity(encoded_df)

def show_heatmap(data: np.ndarray, title: str):
    plt.figure(figsize=(10, 8))
    sns.heatmap(data, annot=True, cmap='coolwarm', square=True, fmt=".2f", linewidths=0.5)
    plt.title(title)
    plt.xlabel("Vector Index")
    plt.ylabel("Vector Index")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    file_path = "Lab Session Data.xlsx"
    sheet = "thyroid0387_UCI"

    df = load_sheet_data(file_path, sheet)
    df_20 = df.head(20)

    bin_df = extract_binary_columns(df_20)
    jc_mat, smc_mat = build_jc_smc_matrices(bin_df)

    enc_df = encode_dataframe(df_20)
    cos_mat = compute_cosine_sim(enc_df)

    show_heatmap(jc_mat, "Jaccard Coefficient Heatmap (First 20 Vectors)")
    show_heatmap(smc_mat, "Simple Matching Coefficient Heatmap (First 20 Vectors)")
    show_heatmap(cos_mat, "Cosine Similarity Heatmap (First 20 Vectors)")


#Q8
import pandas as pd
import numpy as np

def check_fully_missing(df: pd.DataFrame) -> list:
    return [col for col in df.columns if df[col].isnull().sum() == len(df[col])]

def get_imputation_info(df: pd.DataFrame) -> dict:
    strategy_log = {}
    for col in df.columns:
        if df[col].isnull().sum() == len(df[col]):
            strategy_log[col] = "All values missing – skipped"
            continue

        if df[col].dtype == 'object':
            mode_val = df[col].mode()[0]
            df[col].fillna(mode_val, inplace=True)
            strategy_log[col] = f"Categorical → filled with mode: {mode_val}"

        elif df[col].dtype in [np.int64, np.float64]:
            if df[col].nunique() < 10:
                med_val = df[col].median()
                df[col].fillna(med_val, inplace=True)
                strategy_log[col] = f"Numeric (few unique values) → filled with median: {med_val}"
            else:
                mean_val = df[col].mean()
                df[col].fillna(mean_val, inplace=True)
                strategy_log[col] = f"Numeric → filled with mean: {mean_val}"
    return strategy_log

if __name__ == "__main__":
    path = "Lab Session Data.xlsx"
    sheet = "thyroid0387_UCI"

    data = pd.read_excel(path, sheet_name=sheet)
    data.replace('?', np.nan, inplace=True)
    data = data.apply(pd.to_numeric, errors='coerce')

    log = get_imputation_info(data)

    print("Missing value imputation summary:")
    for col, note in log.items():
        print(f"{col}: {note}")



#Q9
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler

def load_dataset(file_path, sheet):
    df_raw = pd.read_excel(file_path, sheet_name=sheet)
    df_raw.replace('?', np.nan, inplace=True)
    df_raw = df_raw.apply(pd.to_numeric, errors='coerce')
    df_raw.dropna(axis=1, how='all', inplace=True)
    df_raw = df_raw.loc[:, df_raw.notna().sum() > 1]
    return df_raw

def impute_missing_values(df):
    df_filled = df.copy()
    df_filled.fillna(df_filled.median(numeric_only=True), inplace=True)
    return df_filled

def apply_normalization(df):
    df_normalized = df.copy()
    numeric_features = df_normalized.select_dtypes(include=['float64', 'int64']).columns.tolist()

    minmax_columns = numeric_features[:10]
    zscore_columns = numeric_features[10:20]
    robust_columns = numeric_features[20:]

    if minmax_columns:
        minmax = MinMaxScaler()
        df_normalized[minmax_columns] = minmax.fit_transform(df_normalized[minmax_columns])
    if zscore_columns:
        zscore = StandardScaler()
        df_normalized[zscore_columns] = zscore.fit_transform(df_normalized[zscore_columns])
    if robust_columns:
        robust = RobustScaler()
        df_normalized[robust_columns] = robust.fit_transform(df_normalized[robust_columns])

    return df_normalized

# --- Main Program ---

file_path = 'Lab Session Data.xlsx'
sheet_name = 'thyroid0387_UCI'

data_raw = load_dataset(file_path, sheet_name)
data_imputed = impute_missing_values(data_raw)
data_scaled = apply_normalization(data_imputed)

print("Preview of normalized dataset:")
print(data_scaled.head())

