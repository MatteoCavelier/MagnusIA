import pandas as pd
from sklearn.model_selection import train_test_split

def dataprep(data):
    inc = pd.get_dummies(data['increment_code'], prefix='inc')
    eco = pd.get_dummies(data['opening_eco'], prefix='eco')
    opn = pd.get_dummies(data['opening_name'], prefix='opn')
    num_cols = data[['turns', 'white_rating', 'black_rating', 'opening_ply']]
    result = num_cols.join(inc).join(eco).join(opn)
    result = result.dropna()
    return result


def get_train(file_path = '../res/games.csv'):
    dataset = pd.read_csv(file_path)
    dp = dataprep(dataset)
    y = dataset["winner"]
    return train_test_split(dp, y, random_state=42)