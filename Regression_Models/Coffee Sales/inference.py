import pickle
import pandas as pd

pred_data = pd.DataFrame(
  [[18,'Mon','Apr','Night','Hot Chocolate']],
   columns = ['hour_of_day'	,'Weekday'	,'Month_name','Time_of_Day','coffee_name']
 )

with open(r"E:\DATA SCIENCE\ML\Regression_Models\Coffee Sales\coffee_sales_model.pkl",'rb') as f:
    model = pickle.load(f)

y_pred = model.predict(pred_data)
print(round(y_pred[0],2))