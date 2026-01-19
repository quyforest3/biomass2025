import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, make_scorer
from sklearn.preprocessing import StandardScaler
from deap import base, creator, tools, algorithms
import random
import joblib
import lightgbm as lgb

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
toolbox.register("attr_int", random.randint, 100, 2000)  # for n_estimators
toolbox.register("attr_int_leaves", random.randint, 10, 100)  # for num_leaves
toolbox.register("attr_float_lr", random.uniform, 0.01, 0.3)  # for learning_rate
toolbox.register("attr_float_subsample", random.uniform, 0.5, 1.0)  # for subsample
toolbox.register("attr_float_colsample", random.uniform, 0.5, 1.0)  # for colsample_bytree
toolbox.register("attr_int_depth", random.randint, 3, 20)  # for max_depth

toolbox.register("individual", tools.initCycle, creator.Individual,
                 (toolbox.attr_int, toolbox.attr_int_leaves, toolbox.attr_float_lr, 
                  toolbox.attr_float_subsample, toolbox.attr_float_colsample, toolbox.attr_int_depth), n=1)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

def validate_and_clamp_params(params):
    """Ensure all parameters are within valid ranges."""
    n_estimators = max(100, int(params[0]))  # Ensure n_estimators is at least 100
    num_leaves = max(10, int(params[1]))  # Ensure num_leaves is at least 10
    learning_rate = max(0.01, min(0.3, params[2]))  # Ensure learning_rate is between 0.01 and 0.3
    subsample = max(0.5, min(1.0, params[3]))  # Ensure subsample is between 0.5 and 1.0
    colsample_bytree = max(0.5, min(1.0, params[4]))  # Ensure colsample_bytree is between 0.5 and 1.0
    max_depth = max(3, int(params[5]))  # Ensure max_depth is at least 3
    return [n_estimators, num_leaves, learning_rate, subsample, colsample_bytree, max_depth]

def evaluate(individual):
    individual = validate_and_clamp_params(individual)
    n_estimators, num_leaves, learning_rate, subsample, colsample_bytree, max_depth = individual

    model = lgb.LGBMRegressor(
        n_estimators=n_estimators,
        num_leaves=num_leaves,
        learning_rate=learning_rate,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        max_depth=max_depth,
        random_state=42
    )

    # Perform 5-fold cross-validation and take the mean RMSE
    rmse_scorer = make_scorer(mean_squared_error, squared=False)
    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring=rmse_scorer)
    mean_cv_rmse = np.mean(cv_scores)
    
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
print(f"n_estimators: {best_params[0]}, num_leaves: {best_params[1]}, learning_rate: {best_params[2]}, subsample: {best_params[3]}, colsample_bytree: {best_params[4]}, max_depth: {best_params[5]}")

best_lgbm_model = lgb.LGBMRegressor(
    n_estimators=best_params[0],
    num_leaves=best_params[1],
    learning_rate=best_params[2],
    subsample=best_params[3],
    colsample_bytree=best_params[4],
    max_depth=best_params[5],
    random_state=42
)

best_lgbm_model.fit(X_train_scaled, y_train)
y_pred_lgbm = best_lgbm_model.predict(X_test_scaled)
rmse_lgbm = mean_squared_error(y_test, y_pred_lgbm, squared=False)
r2_lgbm = r2_score(y_test, y_pred_lgbm)

print(f"Final LightGBM Regressor Performance:")
print(f"  Root Mean Squared Error (RMSE): {rmse_lgbm}")
print(f"  R-squared (R2): {r2_lgbm}")

# Save the final model
joblib.dump(best_lgbm_model, 'AGB_LGBM_Model_GA.pkl')
