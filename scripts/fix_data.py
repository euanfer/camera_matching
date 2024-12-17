import pandas as pd


dfL = pd.read_csv('data/raw/camL_1_no_smooth.csv')
dfM = pd.read_csv('data/raw/camM_1_no_smooth.csv')
dfR = pd.read_csv('data/raw/camR_1_no_smooth.csv')
print('left data shape old:', dfL.shape)
print('middle data shape old:', dfM.shape)
print('right data shape old:', dfR.shape)

R_offset = 24 * 60
L_offset = 7 * 60

L_end = 50 * 60 * 60 + 50 * 60
R_end = 47*60*60 

# end at same time
dfL = dfL[dfL['frame'] < L_end]
dfR = dfR[dfR['frame'] < R_end]

# start at same time
dfL['frame'] = dfL['frame'] - L_offset
dfR['frame'] = dfR['frame'] - R_offset

dfL = dfL[dfL['frame'] >= 0]
dfR = dfR[dfR['frame'] >= 0]

# same frame rate
dfL = dfL[dfL['frame'] % 2 == 0]
dfR = dfR[dfR['frame'] % 2 == 0]

dfL['frame'] = dfL['frame'] / 2
dfR['frame'] = dfR['frame'] / 2
print('left data shape new:', dfL.shape)
print('right data shape new:', dfR.shape)
print(dfL.head())
dfL.to_csv('data/raw/camL_1_raw_fixed.csv', index=False)
dfR.to_csv('data/raw/camR_1_raw_fixed.csv', index=False)
dfM.to_csv('data/raw/camM_1_raw_fixed.csv', index=False)