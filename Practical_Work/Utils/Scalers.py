class Scalers:

    def StandardScaler(self, X):
        X_std = X.copy()

        X_std = (X-X.mean())/X.std()

        return X_std

    def MinMaxScaler(self, X):
        X_min_max = X.copy()

        X_min_max = (X-X.min())/(X.max()-X.min())

        return X_min_max
    
    
    