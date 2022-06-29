import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, QuantileTransformer, PowerTransformer

def scaling(scalers: dict, X_train: pd.DataFrame, X_test_lst: list):
   """"
   Apply different scalers to designated columns
   Example:
      scalers = {"Quantile": ['col1', 'col2], "Power": ['col3']}
   """

   for k,cols in scalers.items():
         
      if k == 'Quantile':
         scaler = QuantileTransformer(output_distribution="normal")
         X_train.loc[:,cols] = scaler.fit_transform(X_train.loc[:,cols])
         for df in X_test_lst:
            df.loc[:,cols] = scaler.transform(df.loc[:,cols])

      if k == 'Power':
         scaler = PowerTransformer()
         X_train.loc[:,cols] = scaler.fit_transform(X_train.loc[:,cols])
         for df in X_test_lst:
            df.loc[:,cols] = scaler.transform(df.loc[:,cols])

      if k == "MinMax":
         scaler = MinMaxScaler(feature_range=(0,1))
         X_train.loc[:,cols] = scaler.fit_transform(X_train.loc[:,cols])
         for df in X_test_lst:
            df.loc[:,cols] = scaler.transform(df.loc[:,cols])
   
   return X_train, X_test_lst