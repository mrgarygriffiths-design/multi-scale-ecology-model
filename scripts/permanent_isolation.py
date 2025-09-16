import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

np.random.seed(1234)

# Parameters
grid_size = 5
islands = [(0,0), (0,2), (0,4), (1,1), (1,3), (2,0), (2,2), (2,4), (3,1), (3,3)]
N_i = 20
genome_length = 100
generations = 50
mutation_rate = 0.1
seeds = range(1234, 1239)  # 5 seeds

# RLE novelty
def rle_novelty(genome):
    runs = 1
    for i in range(1, len(genome)):
        if genome[i] != genome[i-1]:
            runs += 1
    return runs / len(genome)

# Fitness
def fitness(genome, temp=0.5):  # Fixed temp for isolation
    return 1 - abs(np.mean(genome) - temp)

# Tournament selection
def tournament_selection(pop, temp, k=3):
    indices = np.random.choice(len(pop), k)
    fitnesses = [fitness(pop[i], temp) for i in indices]
    return pop[indices[np.argmax(fitnesses)]].copy()

# Mutation
def mutate(genome, rate):
    mask = np.random.random(genome_length) < rate
    genome[mask] = 1 - genome[mask]
    return genome

# Crossover
def crossover(parent1, parent2):
    point = np.random.randint(1, genome_length)
    return np.concatenate([parent1[:point], parent2[point:]])

results = []
for seed in seeds:
    np.random.seed(seed)
    populations = {pos: np.random.randint(0, 2, (N_i, genome_length)) for pos in islands}

    for gen in range(generations):
        for pos in islands:
            pop = populations[pos]
            new_pop = []
            for _ in range(N_i):
                parent1 = tournament_selection(pop, 0.5)
                parent2 = tournament_selection(pop, 0.5)
                child = crossover(parent1, parent2)
                child = mutate(child, mutation_rate)
                new_pop.append(child)
            populations[pos] = np.array(new_pop)

            rle_scores = [rle_novelty(ind) for ind in pop]
            diversity = np.std([np.mean(ind) for ind in pop])
            results.append({
                'gen': gen,
                'island': pos,
                'rle_novelty': np.mean(rle_scores),
                'diversity': diversity,
                'seed': seed
            })

df = pd.DataFrame(results)
df.to_csv('permanent_isolation_results.csv', index=False)

plt.figure(figsize=(10, 6))
for seed in seeds:
    df_seed = df[df['seed'] == seed]
    plt.plot(df_seed['gen'].unique(), df_seed.groupby('gen')['diversity'].mean(), label=f'Seed {seed}')
plt.xlabel('Generation')
plt.ylabel('Average Diversity (Std)')
plt.title('Permanent Isolation Divergence (5 Seeds)')
plt.legend()
plt.savefig('permanent_isolation_divergence.png')
plt.close()

print("Done! Check 'permanent_isolation_results.csv' and 'permanent_isolation_divergence.png'")
