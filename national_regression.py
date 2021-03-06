import pandas as pd
import numpy as np
import argparse
import io
import geopandas as gpd
import matplotlib.pyplot as plt
import sklearn
from sklearn import linear_model
import matplotlib.patches as mpatches
from sklearn.feature_selection import RFE

parser = argparse.ArgumentParser()
parser.add_argument('--state', type=str, help="state to map -- no entry means the entire US will be plotted, type \'contiguous\' to exclude Alaska and Hawaii")
parser.add_argument('--exclude_religion', action="store_true")
parser.add_argument('--exclude_urbanization', action="store_true")
parser.add_argument('--exclude_education', action="store_true")
parser.add_argument('--exclude_race', action="store_true")
parser.add_argument('--exclude_income', action="store_true")
parser.add_argument('--exclude_age', action="store_true")
parser.add_argument('--region', type=str, help="region to regress the data against", 
    choices=['National', 'Pacific', 'Great Plains', 'Midwest', 'Sunbelt', 'South', 'Atlantic', 'New England'], default='National')

args = parser.parse_args()

### FONT
plt.rcParams["font.family"] = "Georgia"
pd.set_option("display.max_rows", None, "display.max_columns", None)

national_2020_df = pd.read_csv('2020_demographics_votes_fips.csv', sep=",", header=0)
national_2012_df = pd.read_csv('2012_demographics_votes.csv', sep=",", header=0)

national_df = national_2020_df.merge(national_2012_df, left_on=["gisjoin"], right_on=["gisjoin"], how="inner")

name_cols = ['name10', 'state', 'FIPS']

# source data cols 
partisanship_col = ['Clinton 2-Party Only 2016 Margin']
religion_cols = ['Evangelical Per 1000 (2010)', 'Mainline Protestant Per 1000 (2010)', 'Catholic Per 1000 (2010)', 
'Orthodox Jewish Per 1000 (2010)', 'Reform/Reconstructionist Jewish Per 1000 (2010)', 'Mormon Per 1000 (2010)', 'Orthodox Christian Per 1000 (2010)']
urbanization_cols = ['Rural % (2010)', 'Total Population 2018']
race_cols = ['White CVAP % 2018', 'Black CVAP % 2018', 'Hispanic CVAP % 2018', 'Native CVAP % 2018', 'Asian CVAP % 2018', 
'white_2012', 'black_2012', 'native_2012', 'hispanic_2012', 'asian_2012']
education_col = ['% Bachelor Degree or Above 2018', 'bachelorabove_2012']
income_col = ['Median Household Income 2018', 'medianincome_2012']
age_col = ['Median Age 2018', 'medianage_2012']
raw_vote_2020_col = 'Total Votes 2020 (AK is Rough Estimate)'

vote_cols = [raw_vote_2020_col, '2012votes']
data_cols = vote_cols + partisanship_col

if not args.exclude_religion:
    data_cols += religion_cols
if not args.exclude_urbanization:
    data_cols += urbanization_cols
if not args.exclude_education:
    data_cols += education_col
if not args.exclude_race:
    data_cols += race_cols
if not args.exclude_income:
    data_cols += income_col
if not args.exclude_age:
    data_cols += age_col

# target to regress against
target_col = 'Swing 2016 to 2020'
national_df = national_df[name_cols + data_cols + [target_col] + ['Biden 2-Party Only 2020 Margin', 'Biden 2020 Margin', 'Clinton 2016 Margin']]


# third party
national_df['biden_2020_2party_share'] = (1 - national_df['Biden 2-Party Only 2020 Margin'])/2
national_df['trump_2020_2party_share'] = 1 - national_df['biden_2020_2party_share']
national_df['biden_2020_share'] = national_df['Biden 2020 Margin']/national_df['Biden 2-Party Only 2020 Margin'] * national_df['biden_2020_2party_share']
national_df['2020_third_party_share'] = 1 - (2 * national_df['biden_2020_share'] - national_df['Biden 2020 Margin'])

national_df['clinton_2016_2party_share'] = (1 - national_df['Clinton 2-Party Only 2016 Margin'])/2
national_df['trump_2020_2party_share'] = 1 - national_df['clinton_2016_2party_share']
national_df['clinton_2016_share'] = national_df['Clinton 2016 Margin']/national_df['Clinton 2-Party Only 2016 Margin'] * national_df['clinton_2016_2party_share']
national_df['2016_third_party_share'] = 1 - (2 * national_df['clinton_2016_share'] - national_df['Clinton 2016 Margin'])

data_cols += ['2016_third_party_share', '2020_third_party_share']

# Calculate the *change* in demographics -- we want this for our regression. Don't just use 2012 demographics themselves.
for i in range(len(data_cols)):
    if '2012' in data_cols[i]:
        data_cols[i] = 'changefrom_' + data_cols[i]

national_df['changefrom_2012votes'] = national_df['Total Votes 2020 (AK is Rough Estimate)'] - national_df['2012votes'] 

if not args.exclude_race:
    national_df['changefrom_white_2012'] = national_df['White CVAP % 2018'] - national_df['white_2012']
    national_df['changefrom_black_2012'] = national_df['Black CVAP % 2018'] - national_df['black_2012']
    national_df['changefrom_hispanic_2012'] = national_df['Hispanic CVAP % 2018'] - national_df['hispanic_2012']
    national_df['changefrom_asian_2012'] = national_df['Asian CVAP % 2018'] - national_df['asian_2012']
    national_df['changefrom_native_2012'] = national_df['Native CVAP % 2018'] - national_df['native_2012']
if not args.exclude_income:
    national_df['changefrom_medianincome_2012'] = national_df['Median Household Income 2018'] - national_df['medianincome_2012']
if not args.exclude_age:
    national_df['changefrom_medianage_2012'] = national_df['Median Age 2018'] - national_df['medianage_2012']
if not args.exclude_education:
    national_df['changefrom_bachelorabove_2012'] = national_df['% Bachelor Degree or Above 2018'] - national_df['bachelorabove_2012']

# REGION SELECTION
if args.region == "National":
    region_df = national_df
elif args.region == "New England":
    region_df = national_df[national_df['state'].isin(['Rhode Island', 'Vermont', 'Massachusetts', 'New Hampshire', 'Maine', 'Connecticut'])]
elif args.region == "South":
    region_df = national_df[national_df['state'].isin(['Georgia', 'North Carolina', 'South Carolina', 'Florida', 'Alabama', 'Mississippi', 'Virginia', 'Tennessee', 'Arkansas'])]
elif args.region == "Sunbelt":
    region_df = national_df[national_df['state'].isin(['Texas', 'New Mexico', 'Arizona', 'Nevada', 'Oklahoma', 'California'])]
elif args.region == "Midwest":
    region_df = national_df[national_df['state'].isin(['Michigan', 'Wisconsin', 'Ohio', 'Pennsylvania', 'Iowa', 'Minnesota', 'Indiana', 'Illinois', 'Missouri', 'West Virginia', 'Kentucky'])]
elif args.region == "Great Plains":
    region_df = national_df[national_df['state'].isin(['Kansas', 'Colorado', 'Oklahoma', 'Montana', 'Wyoming', 'Nebraska', 'South Dakota', 'North Dakota'])]
elif args.region == "West":
    region_df = national_df[national_df['state'].isin(['Montana', 'Idaho', "Utah", 'Arizona', 'Nevada', 'California', 'New Mexico', 'Washington', 'Oregon'])]
elif args.region == "Atlantic":
    region_df = national_df[national_df['state'].isin(['Delaware', 'New York', "Virginia", 'Pennsylvania', 'New Jersey', 'Maryland', 'District of Columbia'])]

############################################################
# REGRESSION 

training_data = region_df[data_cols]
target_values = region_df[target_col]
regression_model = linear_model.Ridge(alpha=0.005, normalize=True)
# if nonlinear:
#     regression_model = RFE(regression_model, n_features_to_select=150, step=1)
regression_model.fit(training_data, target_values)
region_df['estimated_target'] = regression_model.predict(training_data)
region_df['difference'] = region_df[target_col] - region_df['estimated_target']
region_df['votes over expected'] = region_df['difference'] * region_df[raw_vote_2020_col]
regression_df = region_df[['state', 'name10', 'FIPS', 'difference', 'votes over expected', raw_vote_2020_col]]
regression_df.to_csv('model_predicted_swing_vs_actual.csv', sep=',')

## NET VOTES
print("AVERAGE ERROR", regression_df['difference'].abs().mean())

## STATE RANKINGS
print(regression_df.groupby(['state']).apply(lambda x: x['votes over expected'].sum()/x[raw_vote_2020_col].sum()))

if args.state is not None:
    if args.state == "contiguous":
        regression_df = regression_df[~regression_df['state'].isin(['Hawaii', 'Alaska'])]
    else:
        regression_df = regression_df[regression_df['state'].isin([args.state])]
        
expected_net_votes = regression_df['votes over expected'].sum()
percent_above_expected = expected_net_votes/regression_df[raw_vote_2020_col].sum()

print(expected_net_votes, percent_above_expected)
r2_score = sklearn.metrics.r2_score(region_df[target_col], region_df['estimated_target'])

#######################################
# PLOT WHAT'S HAPPENING

map_df = gpd.read_file("national/cb_2018_us_county_500k.shp")
map_df = map_df.to_crs(epsg=3857)
map_df['GEOID'] = map_df['GEOID'].astype(int)

# JOIN + TRANSFORM
map_df = map_df.merge(regression_df[['FIPS', 'difference']], left_on=["GEOID"], right_on=["FIPS"], how="inner")
map_df['difference'] = map_df['difference'].astype(float) * 100

# BUCKET THE MARGINS
map_df['performance_bucket'] = 0.0
map_df.loc[map_df['difference'] > -30, 'performance_bucket'] = 1.0
map_df.loc[map_df['difference'] > -20, 'performance_bucket'] = 2.0
map_df.loc[map_df['difference'] > -15, 'performance_bucket'] = 3.0
map_df.loc[map_df['difference'] > -10, 'performance_bucket'] = 4.0
map_df.loc[map_df['difference'] > -5, 'performance_bucket'] = 5.0
map_df.loc[map_df['difference'] > 0, 'performance_bucket'] = 6.0
map_df.loc[map_df['difference'] > 5, 'performance_bucket'] = 7.0
map_df.loc[map_df['difference'] > 10, 'performance_bucket'] = 8.0
map_df.loc[map_df['difference'] > 15, 'performance_bucket'] = 9.0
map_df.loc[map_df['difference'] > 20, 'performance_bucket'] = 10.0
map_df.loc[map_df['difference'] > 30, 'performance_bucket'] = 11.0

for i in range(12):
    map_df.loc['dummy_' + str(i), 'performance_bucket'] = float(i)

# PLOT
f, ax = plt.subplots(1, figsize=(12, 9))

cmap = plt.cm.get_cmap("RdBu", 12)
ax = map_df.plot(column="performance_bucket", cmap=cmap, edgecolor="black", linewidth=0.25, ax=ax)
ax.legend([mpatches.Patch(color=cmap(b)) for b in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]],
           ['R > +30', 'R +20 - R +30', 'R +15 - R +20', 'R +10 - R +15', 'R +5 - R +10',
           'R +5 - EVEN', 'EVEN - D +5', 'D +5 - D +10', 'D +10 - D +15', 'D +15 - D +20', 'D +20 - D +30', 'D > +30'], 
           loc=(1.0, .18), title="Performance vs demographics",)

ax.set_axis_off()

plt.gca().set_axis_off()
plt.subplots_adjust(top = 0.95, bottom = 0.05, right = 0.95, left = 0.05, 
            hspace = 0.05, wspace = 0.05)
plt.margins(0,0)
plt.gca().xaxis.set_major_locator(plt.NullLocator())
plt.gca().yaxis.set_major_locator(plt.NullLocator())

plt.title("Swing relative to Demographics: 2020 Presidential Election -- Regression on Education, Race, Religion, Income, Age, Urbanization, Third Party Voting, and 2016 partisanship")
plt.figtext(0.80, 0.12, 'R^2 = ' + str(np.round(r2_score, 2)), horizontalalignment='right')
plt.figtext(0.80, 0.08, '@lxeagle17', horizontalalignment='right')
plt.figtext(0.80, 0.06, '@Mill226', horizontalalignment='right')
plt.show()