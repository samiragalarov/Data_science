import sys
from dataclasses import dataclass
import numpy as np 
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler,LabelEncoder
from sklearn.base import BaseEstimator,TransformerMixin
from sklearn.compose import make_column_selector
from src.exception import CustomException
from src.logger import logging
import os

from src.utils import save_object

def miss_values(df):
    df['salary'].fillna(value=0, inplace=True)
    return df

def drop_features(df):
    df.drop(['ssc_b', 'hsc_b'], axis=1, inplace=True)
    return df

def handling_outlier(df):
    Q1 = df['hsc_p'].quantile(0.25)
    Q3 = df['hsc_p'].quantile(0.75)
    IQR = Q3 - Q1

    filter = (df['hsc_p'] >= Q1 - 1.5 * IQR) & (df['hsc_p'] <= Q3 + 1.5 * IQR)

    df = df.loc[filter]
    df = df.reset_index(drop=True)
    return df


class Label_encoder(BaseEstimator,TransformerMixin):
    def fit(self, X, y=None):
        return self
    def transform(self , X):
        X.drop(['sl_no','ssc_b','hsc_b'], axis = 1,inplace=True) 
        # Make copy to avoid changing original data 
        object_cols=['gender','workex','specialisation','status']

        # Apply label encoder to each column with categorical data
        label_encoder = LabelEncoder()
        for col in object_cols:
            X[col] = label_encoder.fit_transform(X[col])    

        return X   

class Ohe_encoder(BaseEstimator,TransformerMixin):
    def fit(self, X, y=None):
        return self
    def transform(self , X):
        X = pd.get_dummies( X,prefix =['Fuel_Type','Seller_Type'],dtype=int )
        # columns_arr = X.columns.values
        print(X)
        return X  

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('artifacts',"proprocessor.pkl")


class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig(),
   
        
    def get_data_transformer_object(self):
        try:


            transform_pipeline = Pipeline(
                steps=[
                    ("Label_encoder",Label_encoder()),
                    ("Ohe_encoder",Ohe_encoder())                     
                ]
            )

            preprocessor=ColumnTransformer(
                [
                ("transform_pipeline",transform_pipeline,make_column_selector())
                ]
            )

            return preprocessor 

        except Exception as e:
            raise CustomException(e,sys)           
    def initiate_data_transformation(self,train_path,test_path):

        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)

            logging.info("Read train and test data completed")

            logging.info("Obtaining preprocessing object")

            preprocessing_obj=self.get_data_transformer_object()

            # Wrap functions using FunctionTransformer
            miss_values_transformer = FunctionTransformer(miss_values, validate=False)
            drop_features_transformer = FunctionTransformer(drop_features, validate=False)
            handling_outlier_transformer = FunctionTransformer(handling_outlier, validate=False)

            # Create a pipeline
            data_processing_pipeline = Pipeline([
                ('miss_values', miss_values_transformer),
                ('drop_features', drop_features_transformer),
                ('handling_outlier', handling_outlier_transformer)
            ])

            train_processed_data = data_processing_pipeline.fit_transform(train_df)
            test_processed_data = data_processing_pipeline.fit_transform(test_df)


            target_column_name="status"

            input_feature_train_df=train_processed_data.drop(columns=[target_column_name],axis=1)
            target_feature_train_df=test_processed_data[target_column_name]

            input_feature_test_df=test_df.drop(columns=[target_column_name],axis=1)
            target_feature_test_df=test_df[target_column_name]


            input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessing_obj.transform(input_feature_test_df)

            train_arr = np.c_[
                input_feature_train_arr, np.array(target_feature_train_df)
            ]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            logging.info(f"Saved preprocessing object.")
    

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            logging.info(
                f"Applying preprocessing object on training dataframe and testing dataframe."
            )



        except Exception as e:
            raise CustomException(e,sys)    




