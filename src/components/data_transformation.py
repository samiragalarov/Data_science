import sys
from dataclasses import dataclass

import numpy as np 
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler,LabelEncoder

from src.exception import CustomException
from src.logger import logging
import os

from src.utils import save_object


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('artifacts',"proprocessor.pkl")


class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()

    def One_hot_encoder(self,df):

        try:
            ohe = OneHotEncoder()
            feature_arry = ohe.fit_transform(df[["hsc_s", "degree_t"]]).toarray()
            feature_labels = ohe.categories_

         
            np.array(feature_labels).ravel()
            feature_labels = np.array(feature_labels).ravel()
            encoded = pd.DataFrame(feature_arry, columns = feature_labels)
            df_coded = pd.concat([df, encoded],axis=1)
            df_coded.drop(['hsc_s','degree_t'],axis=1, inplace=True)
            return df_coded

        
        except Exception as e:
            raise CustomException(e,sys)
    def Label_enoder(self,df):
        try:
            
            object_cols=['gender','workex','specialisation']
    
            # Apply label encoder to each column with categorical data
            label_encoder = LabelEncoder()
            for col in object_cols:
                df[col] = label_encoder.fit_transform(df[col])
 
            return df
        except Exception as e:
            raise CustomException(e,sys)    
        
    def initiate_data_transformation(self,train_path,test_path):

        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)
  

            logging.info("Read train and test data completed")
            logging.info("Obtaining preprocessing object")       

            target_column_name="status"
            numerical_columns = ['ssc_p','hsc_p','degree_p','etest_p','mba_p']

            train_df['salary'].fillna(value=0, inplace=True)
            test_df['salary'].fillna(value=0, inplace=True)
     
            train_df.drop(['sl_no','ssc_b','hsc_b'], axis = 1,inplace=True) 
            test_df.drop(['sl_no','ssc_b','hsc_b'], axis = 1,inplace=True) 

            Q1 = train_df['hsc_p'].quantile(0.25)
            Q3 = train_df['hsc_p'].quantile(0.75)
            IQR = Q3 - Q1 

            filter = (train_df['hsc_p'] >= Q1 - 1.5 * IQR) & (train_df['hsc_p'] <= Q3 + 1.5 *IQR)
            train_df=train_df.loc[filter].reset_index(drop=True)


            Q1 = test_df['hsc_p'].quantile(0.25)
            Q3 = test_df['hsc_p'].quantile(0.75)
            IQR = Q3 - Q1 

            filter = (test_df['hsc_p'] >= Q1 - 1.5 * IQR) & (test_df['hsc_p'] <= Q3 + 1.5 *IQR)
            test_df=test_df.loc[filter].reset_index(drop=True)

            input_feature_train_df=train_df.drop(columns=[target_column_name],axis=1)
            target_feature_train_df=train_df[target_column_name]

            input_feature_test_df=test_df.drop(columns=[target_column_name],axis=1)
            target_feature_test_df=test_df[target_column_name]

            logging.info(
                f"Applying preprocessing object on training dataframe and testing dataframe."
            )


            # One hot
            ohe = OneHotEncoder()
            input_feature_train_df_One = self.One_hot_encoder(input_feature_train_df)
            input_feature_test_df_One = self.One_hot_encoder(input_feature_test_df)
            
           



            # ////////////////////////

            #lable encodeing
            input_feature_train_df_Label = self.Label_enoder(input_feature_train_df_One)
            input_feature_test_df_Label = self.Label_enoder(input_feature_test_df_One)


            train_arr = np.c_[ input_feature_train_df_Label, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_df_Label, np.array(target_feature_test_df)]


            logging.info(f"Saved preprocessing object.")

            save_object(

                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj='Model'

            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )
        except Exception as e:
            raise CustomException(e,sys)    
