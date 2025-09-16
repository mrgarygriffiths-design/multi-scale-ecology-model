import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor

np.random.seed(1234)

# Parameters
grid_size = 5
islands = [(0,0), (0,2), (0,4), (1,1), (1,3), (2,0), (2,2), (2,4), (3,1), (3,3)]
N_i = 10
bio_length = 50
meme_length = 50
generations = 30
mutation_rate = 0.1
seeds = range(1234, 1236)  # 2 seeds

# RLE novelty
def rle_novelty(genome):
    runs = 1
    for i in range(1, len(genome)):
        if genome[i] != genome[i-1]:
            runs += 1
    return runs / len(genome)

# Fitness
def fitness(bio_genome, temp=0.5):
    return 1 - abs(np.mean(bio_genome) - temp)

# Tournament selection
def tournament_selection(pop_bio, pop_meme, temp, k=3):
    indices = np.random.choice(len(pop_bio), k)
    fitnesses = [fitness(pop_bio[i], temp) for i in indices]
    best_idx = indices[np.argmax(fitnesses)]
    return pop_bio[best_idx].copy(), pop_meme[best_idx].copy()

# Mutation
def mutate(genome, rate):
    mask = np.random.random(len(genome)) < rate
    genome[mask] = 1 - genome[mask]
    return genome

# Crossover
def crossover(parent1, parent2):
    point = np.random.randint(1, len(parent1))
    return np.concatenate([parent1[:point], parent2[point:]])

# Simplified AE
def train_ae(data, hidden_layers=(50, 25, 50)):
    ae = MLPRegressor(hidden_layer_sizes=hidden_layers, max_iter=5, random_state=42)
    ae.fit(data, data)
    return ae

def ae_novelty(ae, data):
    reconstructed = ae.predict(data)
    error = np.mean(np.square(data - reconstructed))
    return 1 / (1 + error)  # Inverse error as novelty

results = []
for seed in seeds:
    np.random.seed(seed)
    populations_bio = {pos: np.random.randint(0, 2, (N_i, bio_length)) for pos in islands}
    populations_meme = {pos: np.random.randint(0, 2, (N_i, meme_length)) for pos in islands}
    temperatures = {pos: 0.5 for pos in islands}
    resources = {pos: 1.0 for pos in islands}
    ae_models = {}

    for gen in range(generations):
        for pos in islands:
            bio_pop = populations_bio[pos]
            meme_pop = populations_meme[pos]
            temp = temperatures[pos]
            res = resources[pos]

            new_bio = []
            new_meme = []
            for _ in range(N_i):
                bio_p1, meme_p1 = tournament_selection(bio_pop, meme_pop, temp)
                bio_p2, meme_p2 = tournament_selection(bio_pop, meme_pop, temp)
                bio_child = crossover(bio_p1, bio_p2)
                meme_child = crossover(meme_p1, meme_p2)

                rle_meme = rle_novelty(meme_child)
                mut_rate = mutation_rate * (1 + 0.1 * rle_meme)
                bio_child = mutate(bio_child, mut_rate)

                bio_div = np.std([np.mean(ind) for ind in bio_pop])
                res -= 0.1 * bio_div
                res = max(0.1, res)

                if temp > 0.5:
                    meme_child = mutate(meme_child, 0.2 * (temp - 0.5))

                new_bio.append(bio_child)
                new_meme.append(meme_child)
            populations_bio[pos] = np.array(new_bio)
            populations_meme[pos] = np.array(new_meme)

            # Train AE on concatenated data
            concat_data = np.hstack([bio_pop, meme_pop])
            if gen == 0 or gen % 5 == 0:  # Train every 5 gens
                ae_models[pos] = train_ae(concat_data)
            ae_nov = ae_novelty(ae_models[pos], concat_data)

            rle_scores = [rle_novelty(ind) for ind in bio_pop]
            diversity = np.std([np.mean(ind) for ind in bio_pop])
            results.append({
                'gen': gen,
                'island': pos,
                'rle_novelty': np.mean(rle_scores),
                'ae_novelty': ae_nov,
                'diversity': diversity,
                'temp': temp,
                'seed': seed
            })

df = pd.DataFrame(results)
df.to_csv('cross_scale_ae_results.csv', index=False)

plt.figure(figsize=(10, 6))
for seed in seeds:
    df_seed = df[df['seed'] == seed]
    plt.plot(df_seed['gen'].unique(), df_seed.groupby('gen')['rle_novelty'].mean(), label=f'Seed {seed} RLE')
    plt.plot(df_seed['gen'].unique(), df_seed.groupby('gen')['ae_novelty'].mean(), label=f'Seed {seed} AE')
plt.xlabel('Generation')
plt.ylabel('Average Novelty')
plt.title('Simplified Cross-Scale AE Novelty (2 Seeds)')
plt.legend()
plt.savefig('cross_scale_ae_novelty.png')
plt.close()

print("Done! Check 'cross_scale_ae_results.csv' and 'cross_scale_ae_novelty.png'")
