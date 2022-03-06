import pandas as pd
import copy
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import dash
from dash import dcc, html, Input, Output
import dash_bootstrap_components as dbc
import plotly.express as px

# -----------------------------------------------------------------------------------------------------
# PREPARING THE DATA FOR ANALYSIS

raw_avocado_data = pd.read_csv('/Users/kathrinhalbich/PycharmProjects/Avocado/avocado_cleaned.csv')
avocado_data = copy.deepcopy(raw_avocado_data)

# Change the 'Date' to a 'datetime' data type
avocado_data['Date'] = pd.to_datetime(avocado_data['Date'])

# Split the Date column into three new columns (day, month, year)
avocado_data['Day'] = avocado_data['Date'].dt.day
avocado_data['Month'] = avocado_data['Date'].dt.month
avocado_data['Year'] = avocado_data['Date'].dt.year
avocado_data['MM-YYYY'] = avocado_data['Date'].dt.strftime('%Y-%m')

# Dropping the columns that are not needed
avocado_data = avocado_data.drop(['Date', 'year', 'plu4046', 'plu4770', 'plu4225'], axis=1)

# convert all entries in the 'type' column to lowercase
avocado_data['type'] = avocado_data['type'].str.lower()

# Creating categorical variables
categorical_features = ['type', 'region']
le = LabelEncoder()

for i in range(2):
    new = le.fit_transform(avocado_data[categorical_features[i]])
    avocado_data[categorical_features[i]] = new


# -----------------------------------------------------------------------------------------------------
# EXPLORATORY DATA ANALYSIS

# Identifying any overarching trend in data over time: How does the Average Price behave throughout the years?
year_clustered = avocado_data.groupby('Year')['AveragePrice'].mean().reset_index()
line_chart_year = px.line(year_clustered, x='Year', y='AveragePrice',
                          title='Average Avocado Prices through the Years', markers=True)
line_chart_year.update_layout(title_x=0.5, plot_bgcolor='#ecf0f1', yaxis_title='Average Price',
                              font_color='#2c3e50')
line_chart_year.update_traces(line_color="#f39c12", line_width=2)

# More detailed: How does the Average Price behave throughout the years? 
# Shows the course over all months of the years 2015 - 2021
year_months_clustered = avocado_data.groupby(['MM-YYYY'])['AveragePrice'].mean().reset_index()
line_chart_year_months = px.line(year_months_clustered, x='MM-YYYY', y='AveragePrice',
                                 title='Through the Years - with more Details', markers=True)
line_chart_year_months.update_layout(title_x=0.5, plot_bgcolor='#ecf0f1', yaxis_title='Average Price',
                                     font_color='#2c3e50')
line_chart_year_months.update_traces(line_color="#f39c12", line_width=2)

# How does the Average Price change throughout the seasons?
months_clustered = avocado_data.groupby('Month')['AveragePrice'].mean().reset_index()
line_chart_month = px.line(months_clustered, x='Month', y='AveragePrice',
                           title='Seasonal Changes of Average Prices (accumulated)', markers=True)
line_chart_month.update_layout(title_x=0.5, plot_bgcolor='white', yaxis_title='Average Price', font_color='#2c3e50')
line_chart_month.update_traces(line_color="#18bc9c", line_width=2)


# -----------------------------------------------------------------------------------------------------
# BUILDING A PREDICTION MODEL

# Create a target object and call it y
y = avocado_data.AveragePrice

# Create X
features = ['Day', 'Month', 'Year', 'TotalVolume', 'SmallBags', 'LargeBags', 'XLargeBags', 'type', 'region']
X = avocado_data[features]

# Split into validation and training data + Specify Model
train_X, test_X, train_y, test_y = train_test_split(X, y, random_state=1)
avocado_model = RandomForestRegressor(random_state=1)

# Fit Model & make predictions
avocado_model.fit(train_X, train_y)
val_predictions = avocado_model.predict(test_X)

# calculate mean absolute error
# round the mean absolute error to four decimal points
val_mae = mean_absolute_error(val_predictions, test_y)


def truncate(num, n):
    integer = int(num * (10**n))/(10**n)
    return float(integer)


val_mae = truncate(val_mae,4)
print(val_mae)

# Clean up the result table
result = test_X
result['Real Price'] = test_y
result['Predicted Price'] = val_predictions.tolist()
result['YYYY-MM-DD'] = pd.to_datetime(result[['Month', 'Year', 'Day']])
result = result.drop(['Day', 'Month', 'Year', 'TotalVolume', 'SmallBags', 'LargeBags', 'XLargeBags',
                      'type', 'region'], axis=1)
result = result.sort_values(by=['YYYY-MM-DD'])
result = result.groupby('YYYY-MM-DD').mean().reset_index()

# Scatter plot: Creating a chart showing the predictions and the real values
scatter_model = px.scatter(result.set_index("YYYY-MM-DD").melt(ignore_index=False), y="value", color="variable",
                           title='Real vs. Predicted Values (using Random Forest)')
scatter_model.update_layout(title_x=0.5, plot_bgcolor='#ecf0f1', yaxis_title='Average Price', xaxis_title='Date',
                            font_color='#2c3e50')


# -----------------------------------------------------------------------------------------------------
# APP LAYOUT

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.FLATLY])

app.layout = dbc.Container(html.Div([

    dbc.Row(
        dbc.Col(html.Div([
            html.H1('Avocado Dashboard', style={'text-align': 'center', 'marginTop': 55, 'color': '#2c3e50'}),
            html.H4('Analysing Average Prices (2015 - 2021)',
                    style={'text-align': 'center', 'marginTop': 10, 'paddingBottom': 50, 'color': '#2c3e50'}),
            html.P('In this Machine Learning Project, I am using an Avocado dataset, '
                   'based on retail sales of Hass avocados in the US. After some exploratory data analysis, '
                   'I am building a forecast model to predict the average prices.',
                   style={'text-align': 'center', 'marginTop': 10, 'paddingBottom': 50, 'color': '#2c3e50'}),
        ]))
    ),

    dbc.Row(
        dbc.Col(html.Div([
            html.H4('Exploratory Data Analysis',
                    style={'text-align': 'center', 'marginTop': 10, 'paddingBottom': 20, 'color': '#2c3e50'})
        ]))
    ),

    dbc.Row(
        [
            dbc.Col(dcc.Graph(figure=line_chart_year)),
            dbc.Col(dcc.Graph(figure=line_chart_year_months))
        ]
    ),

    dbc.Row(
        [
            dbc.Col(dcc.Graph(figure=line_chart_month)),
            dbc.Col(html.Div([
                dcc.Graph(id="prices_per_month_year"),
                html.Br(),
                html.P('Choose a year:', style={'text-align': 'center'}),
                dcc.Dropdown(
                    id='dropdown_year',
                    options=[
                        {'label': "2015", 'value': 2015},
                        {'label': "2016", 'value': 2016},
                        {'label': "2017", 'value': 2017},
                        {'label': "2018", 'value': 2018},
                        {'label': "2019", 'value': 2019},
                        {'label': "2020", 'value': 2020},
                        {'label': "2021", 'value': 2021},
                        ],
                    value=2018,
                    style={'width': '40%', 'display': 'block', 'margin-left': 'auto',
                           'margin-right': 'auto', 'text-align': 'center'}
                ),
            ]))
        ]
    ),

    dbc.Row(
        dbc.Col(html.Div([
            html.H4('Forecast Model',
                    style={'text-align': 'center', 'marginTop': 30, 'paddingBottom': 20, 'color': '#2c3e50'})
        ]))
    ),

    dbc.Row(
        [
            dbc.Col(dcc.Graph(figure=scatter_model)),
        ]
    ),

    dbc.Row(
        [
            html.P('Our cleaned and pre-processed data is ready for prediction and evaluation. I am using '
                   'a random forest algorithm. In our case, our target (y) is the average price. '
                   'After training and testing the model, I visualized the results using a scatter plot. '
                   'On this scatter graph, results looks promising. '
                   'In a next step, I also calculated the mean absolute error. '
                   'The MAE of the model is ' + str(val_mae) + '.',
                   style={'text-align': 'center', 'marginTop': 10, 'paddingBottom': 50, 'color': '#2c3e50'})
        ]
    ),

]))


# -----------------------------------------------------------------------------------------------------
# CONNECT THE GRAPHS WITH DASH COMPONENTS

@app.callback(
    Output(component_id="prices_per_month_year", component_property="figure"),
    [Input(component_id="dropdown_year", component_property="value")]
)
# How does the Average Price change throughout the years/seasons? You can filter by year.
def update_graph(year):
    month_cluster = avocado_data.loc[avocado_data['Year'] == year].groupby('Month').mean().reset_index()
    print(month_cluster)
    fig = px.bar(month_cluster, x="Month", y="AveragePrice",
                 title='Seasonal Changes per Year (filterable)', height=500)
    fig.update_xaxes(tickvals=[1,2,3,4,5,6,7,8,9,10,11,12], range=[0,13],
                     ticktext=["January", "February", "March", "April", "May", "June",
                               "July", "August", "September", "October", "November", "December"])
    fig.update_layout(title_x=0.5, xaxis_tickangle=-45, plot_bgcolor="white", margin_pad=10,
                      yaxis_title='Average Price')
    fig.update_traces(marker_color='#2c3e50')

    return fig


if __name__ == '__main__':
    app.run_server(debug=True)
