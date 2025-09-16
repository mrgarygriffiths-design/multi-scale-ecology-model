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
pulse_prob = 0.2
pulse_mag = 0.5
pulse_gen = 10
pulse_decay = 3
migration_prob = 0.1  # Migration probability
seeds = range(1234, 1239)  # 5 seeds

# RLE novelty
def rle_novelty(genome):
    runs = 1
    for i in range(1, len(genome)):
        if genome[i] != genome[i-1]:
            runs += 1
    return runs / len(genome)

# Fitness
def fitness(genome, temp):
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

# Migration between islands
def migrate(populations, migration_prob):
    for i, pos1 in enumerate(islands):
        for pos2 in islands[i+1:]:  # Avoid double-counting pairs
            if np.random.random() < migration_prob:
                pop1 = populations[pos1]
                pop2 = populations[pos2]
                idx1 = np.random.randint(N_i)
                idx2 = np.random.randint(N_i)
                # Swap one individual
                pop1[idx1], pop2[idx2] = pop2[idx2].copy(), pop1[idx1].copy()
                populations[pos1] = pop1
                populations[pos2] = pop2
    return populations

results = []
for seed in seeds:
    np.random.seed(seed)
    populations = {pos: np.random.randint(0, 2, (N_i, genome_length)) for pos in islands}
    temperatures = {pos: 0.5 for pos in islands}

    for gen in range(generations):
        # Apply pulse
        if gen == pulse_gen and np.random.random() < pulse_prob:
            for pos in islands:
                temperatures[pos] += pulse_mag
        if gen > pulse_gen:
            decay_factor = np.exp(-(gen - pulse_gen) / pulse_decay)
            for pos in islands:
                temperatures[pos] = 0.5 + (temperatures[pos] - 0.5) * decay_factor

        # Migration
        populations = migrate(populations, migration_prob)

        for pos in islands:
            pop = populations[pos]
            temp = temperatures[pos]
            new_pop = []
            for _ in range(N_i):
                parent1 = tournament_selection(pop, temp)
                parent2 = tournament_selection(pop, temp)
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
                'temp': temp,
                'seed': seed
            })

df = pd.DataFrame(results)
df.to_csv('migration_hybrid_pulses_results.csv', index=False)

plt.figure(figsize=(10, 6))
for seed in seeds:
    df_seed = df[df['seed'] == seed]
    plt.plot(df_seed['gen'].unique(), df_seed.groupby('gen')['rle_novelty'].mean(), label=f'Seed {seed}')
plt.xlabel('Generation')
plt.ylabel('Average RLE Novelty')
plt.title('Migration-Enhanced Hybrid Pulses Novelty (5 Seeds)')
plt.legend()
plt.savefig('migration_hybrid_pulses_novelty.png')
plt.close()

print("Done! Check 'migration_hybrid_pulses_results.csv' and 'migration_hybrid_pulses_novelty.png'")
