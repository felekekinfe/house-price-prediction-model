import pandas as pd
import pickle
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split



def train_and_save(x_train,y_train,x_test,y_test):

    model=LinearRegression()
    model.fit(x_train,y_train)

    with open('model/_model.pkl','wb') as file:
        pickle.dump(model,file)



if __name__=='__main__':
    df = pd.read_csv('dataset/USA_Housing.csv')

    X = df[['Avg. Area Income','Avg. Area House Age','Avg. Area Number of Rooms','Avg. Area Number of Bedrooms','Area Population']].values
    Y = df['Price'].values
    
  


    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
  

    train_and_save(x_train, y_train, x_test, y_test)




