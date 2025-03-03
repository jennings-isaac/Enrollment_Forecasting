from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor

class sklearnModel:
    def __init__(self, input_dim, model_type):

        self.model = None
        if model_type == 'mlp_regressor':
            
            self.model = MLPRegressor(
                hidden_layer_sizes=(100,),
                activation='tanh',
                solver='adam',
                learning_rate='adaptive',
                max_iter=1000,
                alpha=0.0001,
                verbose=False
            )

        elif model_type == 'random_forest':
            self.model = RandomForestRegressor(
                max_depth=20,
                min_samples_leaf=1,
                min_samples_split=10,
                n_estimators=200,
                random_state=42
            )


        elif model_type == 'TODO2':
            print("and another sklearn model")

        self.input_dim = input_dim
    
    def train(self, X_train, y_train):
        # Train the model on the entire training dataset
        self.model.fit(X_train, y_train)
    
    def evaluate(self, X):
        # Predict values for given data
        return self.model.predict(X)

