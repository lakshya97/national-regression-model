import pandas as pd
import numpy as np
import argparse
import io
import geopandas as gpd
import matplotlib.pyplot as plt
import sklearn
from sklearn.model_selection import KFold
from sklearn import linear_model
import matplotlib.patches as mpatches
from sklearn.model_selection import cross_val_score

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
parser.add_argument('--nonlinearity', action="store_true")
parser.add_argument('--target', choices=['swing', 'margin'], default='margin')
parser.add_argument('--debug', action="store_true")

args = parser.parse_args()

### FONT
plt.rcParams["font.family"] = "Georgia"
pd.set_option("display.max_rows", None, "display.max_columns", None)

national_2020_df = pd.read_csv('2020_demographics_votes_fips.csv', sep=",", header=0)
national_2012_df = pd.read_csv('2012_demographics_votes.csv', sep=",", header=0)
swing_df = pd.read_csv('2012 to 2016 swing.csv', sep=",", header=0)

national_df = national_2020_df.merge(national_2012_df, left_on=["gisjoin"], right_on=["gisjoin"], how="inner").merge(swing_df, left_on=["gisjoin"], right_on=["gisjoin"], how="inner")

name_cols = ['name10', 'state', 'FIPS']

# source data cols
partisanship_col = ['Clinton 2-Party Only 2016 Margin', '2012 to 2016 2-Party Swing']
religion_cols = ['Evangelical Per 1000 (2010)', 'Mainline Protestant Per 1000 (2010)', 'Catholic Per 1000 (2010)',
'Orthodox Jewish Per 1000 (2010)', 'Mormon Per 1000 (2010)', 'Orthodox Christian Per 1000 (2010)']
urbanization_cols = ['Rural % (2010)', 'Total Population 2018']
race_cols = ['White CVAP % 2018', 'Black CVAP % 2018', 'Hispanic CVAP % 2018', 'Native CVAP % 2018', 'Asian CVAP % 2018']
education_col = ['% Bachelor Degree or Above 2018']
income_col = ['Median Household Income 2018', 'medianincome_2012']
age_col = ['Median Age 2018']
raw_vote_2020_col = 'Total Votes 2020 (AK is Rough Estimate)'

vote_cols = ['2012votes']
additional_cols = ['Trump 2016 %', 'Clinton 2016 %', 'white_2012', 'black_2012', 'hispanic_2012', 'bachelorabove_2012']

if args.target == 'swing':
    vote_cols += [raw_vote_2020_col]
else:
    additional_cols += [raw_vote_2020_col]

data_cols = vote_cols + partisanship_col

if not args.exclude_urbanization:
    data_cols += urbanization_cols
else:
    additional_cols += ['Total Population 2018']

if not args.exclude_religion:
    data_cols += religion_cols
if not args.exclude_education:
    data_cols += education_col
if not args.exclude_race:
    data_cols += race_cols
if not args.exclude_income:
    data_cols += income_col
if not args.exclude_age:
    data_cols += age_col

# target to regress against
if args.target == 'swing':
    target_col = 'Swing 2016 to 2020 (2-Party Margin Only)'
else:
    target_col = 'Biden 2-Party Only 2020 Margin'

national_df = national_df[name_cols + data_cols + [target_col] + additional_cols]

# third party
national_df['2016_third_party_share'] = 1 - (national_df['Trump 2016 %'] + national_df['Clinton 2016 %'])
data_cols += ['2016_third_party_share']


national_df['2012_voting_rate'] = national_df['2012votes']/national_df['Total Population 2018']
data_cols += ['2012_voting_rate']

if not args.exclude_race:
    national_df['changefrom_white_2012'] = national_df['White CVAP % 2018'] - national_df['white_2012']
    national_df['changefrom_black_2012'] = national_df['Black CVAP % 2018'] - national_df['black_2012']
    national_df['changefrom_hispanic_2012'] = national_df['Hispanic CVAP % 2018'] - national_df['hispanic_2012']
if not args.exclude_education:
    data_cols += ['changefrom_bachelorabove_2012']
    national_df['changefrom_bachelorabove_2012'] = (national_df['% Bachelor Degree or Above 2018'] - national_df['bachelorabove_2012'])/national_df['bachelorabove_2012']

## Nonlinearity
if args.nonlinearity:
    if not args.exclude_race:
        data_cols += ['county_diversity_black_white','county_diversity_hispanic_white']
        national_df['county_diversity_black_white'] = national_df['Black CVAP % 2018'] * national_df['White CVAP % 2018']
        national_df['county_diversity_hispanic_white'] = national_df['Hispanic CVAP % 2018'] * national_df['White CVAP % 2018']
    if not args.exclude_income:
        national_df['Median Household Income 2018'] = np.log(national_df['Median Household Income 2018']).replace(-np.inf, -1000)
    if not args.exclude_urbanization:
        national_df['Total Population 2018'] = np.log(national_df['Total Population 2018']).replace(-np.inf, -1000)

## POPULATION CHANGE
if not args.exclude_race:
    data_cols += ['black_pop_change', 'hispanic_pop_change', 'white_pop_change']
    national_df['black_pop_change'] = national_df['changefrom_black_2012'] * national_df['Total Population 2018']
    national_df['white_pop_change'] = national_df['changefrom_white_2012'] * national_df['Total Population 2018']
    national_df['hispanic_pop_change'] = national_df['changefrom_hispanic_2012'] * national_df['Total Population 2018']

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
    region_df = national_df[national_df['state'].isin(['Montana', 'Idaho', "Utah", 'Wyoming', 'Washington', 'Oregon'])]
elif args.region == "Atlantic":
    region_df = national_df[national_df['state'].isin(['Delaware', 'New York', "Virginia", 'Pennsylvania', 'New Jersey', 'Maryland', 'District of Columbia'])]

####################
# REGRESSION       #
####################

training_data = region_df[data_cols]
target_values = region_df[target_col]
regression_model = linear_model.Ridge(alpha=0.05)
regression_model.fit(training_data, target_values)
region_df['estimated_target'] = regression_model.predict(training_data)
region_df['difference'] = region_df[target_col] - region_df['estimated_target']
region_df['votes over expected'] = region_df['difference'] * region_df[raw_vote_2020_col]
regression_df = region_df[['state', 'name10', 'FIPS', 'difference', 'votes over expected', target_col, 'estimated_target', raw_vote_2020_col]]
regression_df.to_csv('model_predicted_vs_actual.csv', sep=',')

if args.debug:
    ## DEBUGGING FOR HELPING US FIGURE OUT WHAT'S HAPPENING
    from regressors import stats
    print("coef_pval:\n", stats.coef_pval(regression_model, training_data, target_values))

    # to print summary table:
    print("\n=========== SUMMARY ===========")
    xlabels = data_cols
    stats.summary(regression_model, training_data, target_values, data_cols)


## NET VOTES
avg_error = regression_df['difference'].abs().mean()

## STATE RANKINGS
print("BIDEN PERFORMANCE ABOVE EXPECTED (NET PERCENT)")
print(regression_df.groupby(['state']).apply(lambda x: np.round(x['votes over expected'].sum()/x[raw_vote_2020_col].sum(), 4) * 100))

if args.state is not None:
    if args.state == "contiguous":
        regression_df = regression_df[~regression_df['state'].isin(['Hawaii', 'Alaska'])]
    else:
        regression_df = regression_df[regression_df['state'].isin([args.state])]
        
expected_net_votes = regression_df['votes over expected'].sum()
percent_above_expected = expected_net_votes/regression_df[raw_vote_2020_col].sum()


r2_score = regression_model.score(training_data, target_values)

if args.debug:
    cv_accuracy = cross_val_score(regression_model, training_data, target_values, cv=4).mean()
    print("Cross Validation accuracy:", cv_accuracy)

print("R^2:", r2_score)
print("AVERAGE ERROR, COUNTY PARTISAN MARGIN:", str(np.round(avg_error * 100, 4)) + '%')
print("Expected Net Votes", np.round(expected_net_votes), "Percent Above Expected", percent_above_expected)

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
map_df.loc[map_df['difference'] > -25, 'performance_bucket'] = 1.0
map_df.loc[map_df['difference'] > -20, 'performance_bucket'] = 2.0
map_df.loc[map_df['difference'] > -15, 'performance_bucket'] = 3.0
map_df.loc[map_df['difference'] > -10, 'performance_bucket'] = 4.0
map_df.loc[map_df['difference'] > -5, 'performance_bucket'] = 5.0
map_df.loc[map_df['difference'] > -2.5, 'performance_bucket'] = 6.0
map_df.loc[map_df['difference'] > 0, 'performance_bucket'] = 7.0
map_df.loc[map_df['difference'] > 2.5, 'performance_bucket'] = 8.0
map_df.loc[map_df['difference'] > 5, 'performance_bucket'] = 9.0
map_df.loc[map_df['difference'] > 10, 'performance_bucket'] = 10.0
map_df.loc[map_df['difference'] > 15, 'performance_bucket'] = 11.0
map_df.loc[map_df['difference'] > 20, 'performance_bucket'] = 12.0
map_df.loc[map_df['difference'] > 25, 'performance_bucket'] = 13.0

for i in range(14):
    map_df.loc['dummy_' + str(i), 'performance_bucket'] = float(i)

# PLOT
f, ax = plt.subplots(1, figsize=(12, 9))

cmap = plt.cm.get_cmap("RdBu", 14)
ax = map_df.plot(column="performance_bucket", cmap=cmap, edgecolor="black", linewidth=0.25, ax=ax)
ax.legend([mpatches.Patch(color=cmap(b)) for b in range(14)],
           ['> R +25', 'R +20 - R +25', 'R +15 - R +20', 'R +10 - R +15', 'R +5 - R +10','R +2.5 - R +5', 'EVEN - R +2.5', 'EVEN - D +2.5', 'D +2.5 - D +5', 'D +5 - D +10', 'D +10 - D +15', 'D+15 - D +20', 'D +20 - D +25', '> D +25'], 
           loc=(1.0, .18), title="Performance vs demographics")

ax.set_axis_off()

plt.gca().set_axis_off()
plt.subplots_adjust(top = 0.95, bottom = 0.05, right = 0.95, left = 0.05, 
            hspace = 0.05, wspace = 0.05)
plt.margins(0,0)
plt.gca().xaxis.set_major_locator(plt.NullLocator())
plt.gca().yaxis.set_major_locator(plt.NullLocator())

plt.title(args.target.title() + " relative to Demographics: 2020 Presidential Election -- Regression on Education, Race, Religion, Income, Age, Urbanization, Trends, and 2016 partisanship")
plt.figtext(0.80, 0.14, 'R^2 = ' + str(np.round(r2_score, 2)), horizontalalignment='left')
plt.figtext(0.80, 0.11, 'Average County Error = ' + str(np.round(avg_error, 4) * 100) + '%' , horizontalalignment='left')
plt.figtext(0.80, 0.08, '@lxeagle17', horizontalalignment='left')
plt.figtext(0.80, 0.05, '@Thorongil16', horizontalalignment='left')
plt.figtext(0.80, 0.02, 'Source: @Mill226', horizontalalignment='left')
plt.show()