import pandas as pd
import numpy as np
import random
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ======================================================
# 1️⃣ GA + Random Forest on StressLevelDataset.csv
# ======================================================

df_stress = pd.read_csv("StressLevelDataset.csv")

print("Dataset shape (Stress):", df_stress.shape)
print("Columns:", df_stress.columns)

# Encode categorical columns if present
categorical_cols = df_stress.select_dtypes(include=["object"]).columns
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df_stress[col] = le.fit_transform(df_stress[col])
    label_encoders[col] = le

X_stress = df_stress.drop("stress_level", axis=1).values
y_stress = df_stress["stress_level"].values
feature_names_stress = df_stress.drop("stress_level", axis=1).columns

X_train_stress, X_test_stress, y_train_stress, y_test_stress = train_test_split(
    X_stress, y_stress, test_size=0.2, random_state=42
)

# Baseline Random Forest
rf_stress = RandomForestRegressor(random_state=42, n_estimators=50)
rf_stress.fit(X_train_stress, y_train_stress)
preds_stress = rf_stress.predict(X_test_stress)
r2_without_ga_stress = r2_score(y_test_stress, preds_stress)
print("Random Forest (Stress Dataset): R² =", round(r2_without_ga_stress, 4))

# ---- Genetic Algorithm Functions for Stress Dataset ----
def fitness_stress(mask):
    mask = np.array(mask, dtype=bool)
    if mask.sum() == 0:
        return -999  # invalid
    X_train_sel, X_test_sel = X_train_stress[:, mask], X_test_stress[:, mask]
    model = RandomForestRegressor(random_state=42, n_estimators=50)
    model.fit(X_train_sel, y_train_stress)
    preds = model.predict(X_test_sel)
    return r2_score(y_test_stress, preds)

def initialize_population(pop_size, n_features):
    return [np.random.randint(0, 2, n_features).tolist() for _ in range(pop_size)]

def crossover(parent1, parent2):
    point = random.randint(1, len(parent1)-1)
    child1 = parent1[:point] + parent2[point:]
    child2 = parent2[:point] + parent1[point:]
    return child1, child2

def mutate(individual, prob=0.1):
    for i in range(len(individual)):
        if random.random() < prob:
            individual[i] = 1 - individual[i]
    return individual

pop_size = 10
generations = 10
n_features_stress = X_stress.shape[1]

population = initialize_population(pop_size, n_features_stress)

for gen in range(generations):
    fitness_scores = [fitness_stress(ind) for ind in population]
    sorted_pop = [ind for _, ind in sorted(zip(fitness_scores, population), reverse=True)]
    population = sorted_pop[:pop_size//2]
    next_gen = []
    while len(next_gen) < pop_size:
        p1, p2 = random.sample(population, 2)
        c1, c2 = crossover(p1, p2)
        next_gen.append(mutate(c1))
        if len(next_gen) < pop_size:
            next_gen.append(mutate(c2))
    population = next_gen

best_ind = max(population, key=fitness_stress)
mask_stress = np.array(best_ind, dtype=bool)
best_features_stress = feature_names_stress[mask_stress]

X_train_sel, X_test_sel = X_train_stress[:, mask_stress], X_test_stress[:, mask_stress]
rf_sel_stress = RandomForestRegressor(random_state=42, n_estimators=50)
rf_sel_stress.fit(X_train_sel, y_train_stress)
preds_sel = rf_sel_stress.predict(X_test_sel)
r2_with_ga_stress = r2_score(y_test_stress, preds_sel)

print("GA selected features (Stress):", list(best_features_stress))
print("Random Forest (GA-selected features, Stress): R² =", round(r2_with_ga_stress, 4))

# ======================================================
# 2️⃣ Decision Tree + ACO on StressLevelDataset.csv
# ======================================================

df_stress_dt = pd.read_csv("StressLevelDataset.csv")
le_target = LabelEncoder()
y_dt = le_target.fit_transform(df_stress_dt['stress_level'].astype(str))
X_dt = df_stress_dt.drop(columns=['stress_level'])

for c in X_dt.select_dtypes(include=['object']).columns:
    X_dt[c] = LabelEncoder().fit_transform(X_dt[c].astype(str))

X_dt = X_dt.fillna(X_dt.median())

X_train_dt, X_test_dt, y_train_dt, y_test_dt = train_test_split(
    X_dt, y_dt, test_size=0.2, stratify=y_dt, random_state=42
)

baseline_dt = DecisionTreeClassifier(random_state=42)
baseline_dt.fit(X_train_dt, y_train_dt)
y_pred_baseline = baseline_dt.predict(X_test_dt)
baseline_acc = accuracy_score(y_test_dt, y_pred_baseline)

print("\nDecision Tree (Baseline) - Stress Dataset")
print("Accuracy:", baseline_acc)

param_space = {
    "criterion": ["gini", "entropy", "log_loss"],
    "max_depth": [3, 5, 10, None],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4]
}

n_ants = 10
n_iterations = 20
alpha = 1.0
rho = 0.1
pheromones = {key: np.ones(len(values)) for key, values in param_space.items()}

def evaluate(params):
    try:
        dt = DecisionTreeClassifier(
            criterion=params["criterion"],
            max_depth=params["max_depth"],
            min_samples_split=params["min_samples_split"],
            min_samples_leaf=params["min_samples_leaf"],
            random_state=42
        )
        dt.fit(X_train_dt, y_train_dt)
        y_pred = dt.predict(X_test_dt)
        return accuracy_score(y_test_dt, y_pred)
    except Exception:
        return 0

best_params = None
best_score = 0

for it in range(n_iterations):
    ant_solutions = []
    ant_scores = []

    for ant in range(n_ants):
        chosen_params = {}
        for param, values in param_space.items():
            probs = pheromones[param] ** alpha
            probs /= probs.sum()
            choice = np.random.choice(len(values), p=probs)
            chosen_params[param] = values[choice]
        score = evaluate(chosen_params)
        ant_solutions.append(chosen_params)
        ant_scores.append(score)

        if score > best_score:
            best_score = score
            best_params = chosen_params

    for param, values in param_space.items():
        pheromones[param] *= (1 - rho)
        for sol, score in zip(ant_solutions, ant_scores):
            idx = values.index(sol[param])
            pheromones[param][idx] += score

    print(f"Iteration {it+1}/{n_iterations}, Best Accuracy so far: {best_score:.4f}, Params: {best_params}")

optimized_dt = DecisionTreeClassifier(
    criterion=best_params["criterion"],
    max_depth=best_params["max_depth"],
    min_samples_split=best_params["min_samples_split"],
    min_samples_leaf=best_params["min_samples_leaf"],
    random_state=42
)
optimized_dt.fit(X_train_dt, y_train_dt)
y_pred_optimized = optimized_dt.predict(X_test_dt)
optimized_acc = accuracy_score(y_test_dt, y_pred_optimized)

print("\nOptimized Decision Tree (After ACO) - Stress Dataset")
print("Accuracy:", optimized_acc)

print("\nAccuracy Comparison (Stress Dataset)")
print(f"Before ACO: {baseline_acc:.4f}")
print(f"After  ACO: {optimized_acc:.4f}")

# ======================================================
# 3️⃣ Decision Tree + ACO on Sleep_health_and_lifestyle_dataset.csv
# ======================================================

df_sleep = pd.read_csv("Sleep_health_and_lifestyle_dataset.csv")

if 'Blood Pressure' in df_sleep.columns:
    bp = df_sleep['Blood Pressure'].astype(str).str.split('/', expand=True)
    df_sleep['BP_systolic'] = pd.to_numeric(bp[0], errors='coerce')
    df_sleep['BP_diastolic'] = pd.to_numeric(bp[1], errors='coerce')
    df_sleep = df_sleep.drop(columns=['Blood Pressure'])

if 'BMI Category' in df_sleep.columns:
    def clean_bmi(x):
        if pd.isna(x): return np.nan
        s = str(x).lower()
        if 'under' in s: return 'Underweight'
        if 'normal' in s: return 'Normal'
        if 'over' in s: return 'Overweight'
        if 'obese' in s: return 'Obese'
        return x.title()
    df_sleep['BMI Category'] = df_sleep['BMI Category'].apply(clean_bmi)

if 'Gender' in df_sleep.columns:
    df_sleep['Gender'] = df_sleep['Gender'].map({'Male': 0, 'Female': 1}).fillna(0)

if 'Occupation' in df_sleep.columns:
    top_occ = df_sleep['Occupation'].value_counts().nlargest(10).index
    df_sleep['Occupation'] = df_sleep['Occupation'].where(df_sleep['Occupation'].isin(top_occ), other='Other')
    df_sleep = pd.get_dummies(df_sleep, columns=['Occupation'], prefix='Occ', drop_first=True)

if 'BMI Category' in df_sleep.columns:
    df_sleep = pd.get_dummies(df_sleep, columns=['BMI Category'], prefix='BMI', drop_first=True)

if 'Sleep Disorder' not in df_sleep.columns:
    raise ValueError("No 'Sleep Disorder' column found in the CSV.")

le = LabelEncoder()
y_sleep = le.fit_transform(df_sleep['Sleep Disorder'].astype(str))
X_sleep = df_sleep.drop(columns=['Sleep Disorder'])

for c in X_sleep.select_dtypes(include=['object']).columns:
    X_sleep = X_sleep.drop(columns=[c])

X_sleep = X_sleep.fillna(X_sleep.median())

X_train_sleep, X_test_sleep, y_train_sleep, y_test_sleep = train_test_split(
    X_sleep, y_sleep, test_size=0.2, stratify=y_sleep, random_state=42
)

baseline_dt_sleep = DecisionTreeClassifier(random_state=42)
baseline_dt_sleep.fit(X_train_sleep, y_train_sleep)
y_pred_baseline_sleep = baseline_dt_sleep.predict(X_test_sleep)
baseline_acc_sleep = accuracy_score(y_test_sleep, y_pred_baseline_sleep)

print("\nDecision Tree (Baseline) - Sleep Dataset")
print("Accuracy:", baseline_acc_sleep)

# ACO for Sleep Dataset
param_space_sleep = param_space.copy()
pheromones_sleep = {key: np.ones(len(values)) for key, values in param_space_sleep.items()}

best_params_sleep = None
best_score_sleep = 0

for it in range(n_iterations):
    ant_solutions = []
    ant_scores = []

    for ant in range(n_ants):
        chosen_params = {}
        for param, values in param_space_sleep.items():
            probs = pheromones_sleep[param] ** alpha
            probs /= probs.sum()
            choice = np.random.choice(len(values), p=probs)
            chosen_params[param] = values[choice]
        score = evaluate(chosen_params)
        ant_solutions.append(chosen_params)
        ant_scores.append(score)

        if score > best_score_sleep:
            best_score_sleep = score
            best_params_sleep = chosen_params

    for param, values in param_space_sleep.items():
        pheromones_sleep[param] *= (1 - rho)
        for sol, score in zip(ant_solutions, ant_scores):
            idx = values.index(sol[param])
            pheromones_sleep[param][idx] += score

    print(f"Iteration {it+1}/{n_iterations}, Best Accuracy so far (Sleep): {best_score_sleep:.4f}, Params: {best_params_sleep}")

optimized_dt_sleep = DecisionTreeClassifier(
    criterion=best_params_sleep["criterion"],
    max_depth=best_params_sleep["max_depth"],
    min_samples_split=best_params_sleep["min_samples_split"],
    min_samples_leaf=best_params_sleep["min_samples_leaf"],
    random_state=42
)
optimized_dt_sleep.fit(X_train_sleep, y_train_sleep)
y_pred_optimized_sleep = optimized_dt_sleep.predict(X_test_sleep)
optimized_acc_sleep = accuracy_score(y_test_sleep, y_pred_optimized_sleep)

print("\nOptimized Decision Tree (After ACO) - Sleep Dataset")
print("Accuracy:", optimized_acc_sleep)

print("\nAccuracy Comparison (Sleep Dataset)")
print(f"Before ACO: {baseline_acc_sleep:.4f}")
print(f"After  ACO: {optimized_acc_sleep:.4f}")

# ======================================================
# 4️⃣ GA + Random Forest on Sleep Dataset
# ======================================================

df_sleep_ga = pd.read_csv("Sleep_health_and_lifestyle_dataset.csv")
df_sleep_ga = df_sleep_ga.drop("Person ID", axis=1)

categorical_cols_sleep = df_sleep_ga.select_dtypes(include=["object"]).columns
label_encoders_sleep = {}
for col in categorical_cols_sleep:
    le = LabelEncoder()
    df_sleep_ga[col] = le.fit_transform(df_sleep_ga[col])
    label_encoders_sleep[col] = le

X_sleep_ga = df_sleep_ga.drop("Stress Level", axis=1).values
y_sleep_ga = df_sleep_ga["Stress Level"].values
feature_names_sleep = df_sleep_ga.drop("Stress Level", axis=1).columns

X_train_sleep_ga, X_test_sleep_ga, y_train_sleep_ga, y_test_sleep_ga = train_test_split(
    X_sleep_ga, y_sleep_ga, test_size=0.2, random_state=42
)

rf_sleep = RandomForestRegressor(random_state=42, n_estimators=50)
rf_sleep.fit(X_train_sleep_ga, y_train_sleep_ga)
preds_sleep = rf_sleep.predict(X_test_sleep_ga)
r2_without_ga_sleep = r2_score(y_test_sleep_ga, preds_sleep)
print("Random Forest (Sleep Dataset): R² =", round(r2_without_ga_sleep, 4))

# GA for Sleep Dataset
def fitness_sleep(mask):
    mask = np.array(mask, dtype=bool)
    if mask.sum() == 0:
        return -999
    X_train_sel, X_test_sel = X_train_sleep_ga[:, mask], X_test_sleep_ga[:, mask]
    model = RandomForestRegressor(random_state=42, n_estimators=50)
    model.fit(X_train_sel, y_train_sleep_ga)
    preds = model.predict(X_test_sel)
    return r2_score(y_test_sleep_ga, preds)

n_features_sleep = X_sleep_ga.shape[1]
population_sleep = initialize_population(pop_size, n_features_sleep)

for gen in range(generations):
    fitness_scores = [fitness_sleep(ind) for ind in population_sleep]
    sorted_pop = [ind for _, ind in sorted(zip(fitness_scores, population_sleep), reverse=True)]
    population_sleep = sorted_pop[:pop_size//2]
    next_gen = []
    while len(next_gen) < pop_size:
        p1, p2 = random.sample(population_sleep, 2)
        c1, c2 = crossover(p1, p2)
        next_gen.append(mutate(c1))
        if len(next_gen) < pop_size:
            next_gen.append(mutate(c2))
    population_sleep = next_gen

best_ind_sleep = max(population_sleep, key=fitness_sleep)
mask_sleep = np.array(best_ind_sleep, dtype=bool)
best_features_sleep = feature_names_sleep[mask_sleep]

X_train_sel, X_test_sel = X_train_sleep_ga[:, mask_sleep], X_test_sleep_ga[:, mask_sleep]
rf_sel_sleep = RandomForestRegressor(random_state=42, n_estimators=50)
rf_sel_sleep.fit(X_train_sel, y_train_sleep_ga)
preds_sel_sleep = rf_sel_sleep.predict(X_test_sel)
r2_with_ga_sleep = r2_score(y_test_sleep_ga, preds_sel_sleep)

print("GA selected features (Sleep):", list(best_features_sleep))
print("Random Forest (GA-selected features, Sleep): R² =", round(r2_with_ga_sleep, 4))
