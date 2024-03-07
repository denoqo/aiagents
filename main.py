from dotenv import load_dotenv
import os
import pandas as pd


population_path = os.path.join('data', 'population.csv')
population = pd.read_csv(population_path)

load_dotenv()

print(population.head())
