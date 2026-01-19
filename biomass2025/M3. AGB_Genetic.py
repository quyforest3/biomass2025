import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from deap import base, creator, tools, algorithms
import random
import joblib  # Ensure joblib is imported

# Load the dataset
file_path = r'C:\Users\mn2n23\OneDrive - University of Southampton\Desktop\SC solutions (summer project)\biomass\newforrest\FEI data\opt_means_cleaned.csv'
data = pd.read_csv(file_path)

# Features and target
X = data.drop(columns=['AGB_2017'])
y = data['AGB_2017']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Genetic Algorithm Setup
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))  # We want to minimize the RMSE
creator.create("Individual", list, fitness=creator.FitnessMin)

toolbox = base.Toolbox()
toolbox.register("attr_int", random.randint, 10, 1000)  # for n_estimators
toolbox.register("attr_float", random.uniform, 0.01, 1.0)  # for max_features
toolbox.register("attr_int_split", random.randint, 2, 20)  # for min_samples_split
toolbox.register("attr_int_leaf", random.randint, 1, 20)  # for min_samples_leaf
toolbox.register("attr_int_depth", random.randint, 1, 50)  # for max_depth

toolbox.register("individual", tools.initCycle, creator.Individual,
                 (toolbox.attr_int, toolbox.attr_float, toolbox.attr_int_split, toolbox.attr_int_leaf, toolbox.attr_int_depth), n=1)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

def validate_and_clamp_params(params):
    """Ensure all parameters are within valid ranges."""
    n_estimators = max(1, int(params[0]))  # Ensure n_estimators is at least 1
    max_features = max(0.01, min(1.0, params[1]))  # Ensure max_features is between 0.01 and 1.0
    min_samples_split = max(2, int(params[2]))  # Ensure min_samples_split is at least 2
    min_samples_leaf = max(1, int(params[3]))  # Ensure min_samples_leaf is at least 1
    max_depth = max(1, int(params[4]))  # Ensure max_depth is at least 1
    return [n_estimators, max_features, min_samples_split, min_samples_leaf, max_depth]

def evaluate(individual):
    individual = validate_and_clamp_params(individual)
    n_estimators, max_features, min_samples_split, min_samples_leaf, max_depth = individual

    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_features=max_features,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        max_depth=max_depth,
        random_state=42
    )

    # Perform 5-fold cross-validation and take the mean RMSE
    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='neg_root_mean_squared_error')
    mean_cv_rmse = -np.mean(cv_scores)  # The scores are negative, so we take the negative mean
    
    # Fitness is RMSE from cross-validation
    return mean_cv_rmse,

toolbox.register("mate", tools.cxBlend, alpha=0.5)
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.2)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("evaluate", evaluate)

# Suggested Genetic Algorithm Parameters
population = toolbox.population(n=50)  # Larger population
ngen = 40  # More generations
cxpb = 0.8  # Higher crossover probability
mutpb = 0.4  # Higher mutation probability

# Run the Genetic Algorithm
for gen in range(ngen):
    offspring = algorithms.varAnd(population, toolbox, cxpb, mutpb)
    fits = map(toolbox.evaluate, offspring)
    
    for fit, ind in zip(fits, offspring):
        ind.fitness.values = fit
    
    population = toolbox.select(offspring, k=len(population))
    best_ind = tools.selBest(population, k=1)[0]
    print(f"Generation {gen}: Best CV RMSE: {best_ind.fitness.values[0]}")
    print(f"Best Individual: {best_ind}")

# Final model using the best hyperparameters
best_params = validate_and_clamp_params(best_ind)
print("Best Hyperparameters found by Genetic Algorithm:")
print(f"n_estimators: {best_params[0]}, max_features: {best_params[1]}, min_samples_split: {best_params[2]}, min_samples_leaf: {best_params[3]}, max_depth: {best_params[4]}")

best_rf_model = RandomForestRegressor(
    n_estimators=best_params[0],
    max_features=best_params[1],
    min_samples_split=best_params[2],
    min_samples_leaf=best_params[3],
    max_depth=best_params[4],
    random_state=42
)

best_rf_model.fit(X_train_scaled, y_train)
y_pred_rf = best_rf_model.predict(X_test_scaled)
rmse_rf = mean_squared_error(y_test, y_pred_rf, squared=False)
r2_rf = r2_score(y_test, y_pred_rf)

print(f"Final Random Forest Regressor Performance:")
print(f"  Root Mean Squared Error (RMSE): {rmse_rf}")
print(f"  R-squared (R2): {r2_rf}")

# Save the final model
joblib.dump(best_rf_model, 'AGB_RandomForest_Model_GA.pkl')
