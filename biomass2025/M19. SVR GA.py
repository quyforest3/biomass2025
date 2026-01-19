import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, make_scorer
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from deap import base, creator, tools, algorithms
import random
import joblib

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
toolbox.register("attr_float_c", random.uniform, 0.1, 1000.0)  # for C
toolbox.register("attr_float_epsilon", random.uniform, 0.01, 1.0)  # for epsilon
toolbox.register("attr_int_degree", random.randint, 2, 5)  # for degree (only relevant for polynomial kernel)
toolbox.register("attr_float_gamma", random.uniform, 0.0001, 1.0)  # for gamma
toolbox.register("attr_choice_kernel", random.choice, ['linear', 'poly', 'rbf', 'sigmoid'])  # for kernel

toolbox.register("individual", tools.initCycle, creator.Individual,
                 (toolbox.attr_float_c, toolbox.attr_float_epsilon, toolbox.attr_int_degree, 
                  toolbox.attr_float_gamma, toolbox.attr_choice_kernel), n=1)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

def validate_and_clamp_params(params):
    """Ensure all parameters are within valid ranges."""
    C = max(0.1, params[0])  # Ensure C is at least 0.1
    epsilon = max(0.01, params[1])  # Ensure epsilon is at least 0.01
    degree = max(2, int(params[2]))  # Ensure degree is at least 2
    gamma = max(0.0001, params[3])  # Ensure gamma is at least 0.0001
    kernel = params[4]  # Kernel doesn't need clamping
    return [C, epsilon, degree, gamma, kernel]

def evaluate(individual):
    individual = validate_and_clamp_params(individual)
    C, epsilon, degree, gamma, kernel = individual

    model = SVR(
        C=C,
        epsilon=epsilon,
        degree=degree,
        gamma=gamma,
        kernel=kernel
    )

    # Perform 5-fold cross-validation and take the mean RMSE
    rmse_scorer = make_scorer(mean_squared_error, squared=False)
    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring=rmse_scorer)
    mean_cv_rmse = np.mean(cv_scores)
    
    return mean_cv_rmse,

def custom_crossover(ind1, ind2):
    """Custom crossover to handle both numeric and categorical values."""
    size = len(ind1)
    for i in range(size):
        if isinstance(ind1[i], (float, int)) and isinstance(ind2[i], (float, int)):
            # Perform crossover on numerical values
            ind1[i] = (ind1[i] + ind2[i]) / 2.0
            ind2[i] = ind1[i]
        else:
            # Randomly swap categorical values (kernel type)
            if random.random() < 0.5:
                ind1[i], ind2[i] = ind2[i], ind1[i]
    return ind1, ind2

def custom_mutate(individual):
    """Custom mutation to handle both numeric and categorical values."""
    for i in range(len(individual)):
        if isinstance(individual[i], (float, int)):
            if random.random() < 0.2:  # mutation probability
                individual[i] += random.gauss(0, 1)  # Apply Gaussian mutation
        else:
            if random.random() < 0.2:  # mutation probability
                individual[i] = random.choice(['linear', 'poly', 'rbf', 'sigmoid'])  # Reassign kernel type
    return individual,

toolbox.register("mate", custom_crossover)
toolbox.register("mutate", custom_mutate)
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
    print(f"Generation {gen}: Best Fitness: {best_ind.fitness.values[0]}")
    print(f"Best Individual: {best_ind}")

# Final model using the best hyperparameters
best_params = validate_and_clamp_params(best_ind)
print("Best Hyperparameters found by Genetic Algorithm:")
print(f"C: {best_params[0]}, epsilon: {best_params[1]}, degree: {best_params[2]}, gamma: {best_params[3]}, kernel: {best_params[4]}")

best_svr_model = SVR(
    C=best_params[0],
    epsilon=best_params[1],
    degree=best_params[2],
    gamma=best_params[3],
    kernel=best_params[4]
)

best_svr_model.fit(X_train_scaled, y_train)
y_pred_svr = best_svr_model.predict(X_test_scaled)
rmse_svr = mean_squared_error(y_test, y_pred_svr, squared=False)
r2_svr = r2_score(y_test, y_pred_svr)

print(f"Final SVR Regressor Performance:")
print(f"  Root Mean Squared Error (RMSE): {rmse_svr}")
print(f"  R-squared (R2): {r2_svr}")

# Save the final model
joblib.dump(best_svr_model, 'AGB_SVR_Model_GA.pkl')
