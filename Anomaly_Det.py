import pandas as pd          
import numpy as np           
from sklearn.ensemble import IsolationForest
data = {
    'Gear Fault':              [1,2,3,4,5,6,7,8,9,100,10,11,12,13,14,15,16],
    'Bearing Fault':           [1,2,3,4,5,6,7,8,9,200,10,11,12,13,14,15,16],
    'Shaft Misalignment':      [1,2,3,4,5,6,7,8,9,101,10,11,12,13,14,15,16],
    'Lubrication Failures':    [1,2,3,4,5,6,7,8,9,200,10,11,12,13,14,15,16],
    'Housing Resonance Issues':[1,2,3,4,5,6,7,8,9,300,10,11,12,13,14,15,16],
}
df = pd.DataFrame(data)
cols = ['Gear Fault', 'Bearing Fault', 'Shaft Misalignment',
        'Lubrication Failures', 'Housing Resonance Issues']
iso_forest = IsolationForest(contamination=0.05, random_state=42)
iso_forest.fit(df[cols].values)
df['Anomaly_Score'] = iso_forest.predict(df[cols].values)
anomalies = df[df['Anomaly_Score'] == -1]
print("Detected Anomalies:")
print(anomalies[cols])