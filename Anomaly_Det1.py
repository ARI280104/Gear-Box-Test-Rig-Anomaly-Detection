from sklearn.ensemble import IsolationForest
import pandas as pd
import numpy as np
#Anomaly detection data is to be put here 
data = {
    'Gear fault ': [1,2,3,4,5,6,7,8,9,100,10, 11,12,13,14,15,16],
    'Bearing fault ' : [1,2,3,4,5,6,7,8,9,200, 10,11,12,13,14,15,16],
    'Shaft Misalignment' : [1,2,3,4,5,6,7,8,9,101,10,11,12,13,14,15,16],
    'Lubrication Failures' : [1,2,3,4,5,6,7,8,9,200, 10, 11,12,13,14,15,16],
    'Housing Resonance issues' : [1,2,3,4,5,6,7,8,9,300, 10, 11,12,13,14,15,16],

}
df = pd.DataFrame(data)
# isolation forest initiation
iso_forest = IsolationForest(contamination = 1)
iso_forest.fit(df[['Gear fault','Bearing fault' , 'Shaft Misalignment','Lubrication Failures','Housing resonance issues']].values)
#Anomaly Predictions
df['Anomaly_Score'] = iso_forest.predict(df[['Gear fault','Bearing fault' , 'Shaft Misalignment','Lubrication Failures','Housing resonance issues']].values)
# display of the detected anomalies
anomalies = df[df['Anomaly_Score'] == -1]
print("Detected Anomalies")
print(anomalies[['Gear fault','Bearing fault' , 'Shaft Misalignment','Lubrication Failures','Housing resonance issues']])