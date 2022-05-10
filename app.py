import dash
from dash import dcc, html
import pickle
import pandas as pd
import numpy as np
import plotly.graph_objs as go
from dash.dependencies import Input, Output, State


########### Define your variables ######
myheading1='Predicting Customer Response to Campaigns'
image1='ames_welcome.jpeg'
tabtitle = 'Costumer Behavior Prediction'
sourceurl = 'https://www.kaggle.com/datasets/imakash3011/customer-personality-analysis'
githublink = 'https://github.com/vharkar/504-mortgage-loans-predictor'


########### Model featurse
features = [
 'Kidhome',
 'Teenhome',
 'AcceptedCmp5',
 'Education',
 'Marital_Status',
 'age',
 'Income',
 'Recency',
 'Total_Spent',
 'Yrs_Customer',
 'NumDealsPurchases',
 'NumWebPurchases',
 'NumCatalogPurchases',
 'NumStorePurchases',
 'NumWebVisitsMonth'
]

########### open the pickle files ######
# dataframes for visualization
responded=pd.read_csv('model_components/cust_responded.csv')
ignored=pd.read_csv('model_components/cust_ignored.csv')

# random forest model
filename = open('model_components/cust_response_rf_model.pkl', 'rb')
rf = pickle.load(filename)
filename.close()

# encoder1
filename = open('model_components/cust_response_edu_onehot_encoder.pkl', 'rb')
encoder1 = pickle.load(filename)
filename.close()

# encoder2
filename = open('model_components/cust_response_marital_onehot_encoder.pkl', 'rb')
encoder2 = pickle.load(filename)
filename.close()


filename = open('model_components/age_ss_scaler.pkl', 'rb')
ss_scaler0 = pickle.load(filename)
filename.close()

filename = open('model_components/income_ss_scaler.pkl', 'rb')
ss_scaler1 = pickle.load(filename)
filename.close()

filename = open('model_components/recency_ss_scaler.pkl', 'rb')
ss_scaler2 = pickle.load(filename)
filename.close()

filename = open('model_components/spend_ss_scaler.pkl', 'rb')
ss_scaler3 = pickle.load(filename)
filename.close()

filename = open('model_components/yrs_cust_ss_scaler.pkl', 'rb')
ss_scaler30 = pickle.load(filename)
filename.close()

filename = open('model_components/deals_ss_scaler.pkl', 'rb')
ss_scaler4 = pickle.load(filename)
filename.close()

filename = open('model_components/web_ss_scaler.pkl', 'rb')
ss_scaler40 = pickle.load(filename)
filename.close()

filename = open('model_components/catalog_ss_scaler.pkl', 'rb')
ss_scaler41 = pickle.load(filename)
filename.close()

filename = open('model_components/store_ss_scaler.pkl', 'rb')
ss_scaler42 = pickle.load(filename)
filename.close()

filename = open('model_components/visits_ss_scaler.pkl', 'rb')
ss_scaler43 = pickle.load(filename)
filename.close()

####### FUNCTIONS #######

# Create a function that can take any 8 valid inputs & make a prediction
def make_predictions(listofargs, Threshold):
    try:
        # the order of the arguments must match the order of the features
        df = pd.DataFrame(columns=features)

        df.loc[0] = listofargs

        # convert arguments from integers to floats:
        for var in ['Kidhome', 'Teenhome', 'AcceptedCmp5', 'age', 'Income', 'Recency', 'Total_Spent', 'Yrs_Customer',
                    'NumDealsPurchases', 'NumWebPurchases', 'NumCatalogPurchases', 'NumStorePurchases',
                    'NumWebVisitsMonth']:
            df[var] = int(df[var])

        # transform the categorical variable using the same encoder we trained previously
        ohe = pd.DataFrame(encoder1.transform(df[['Education']]).toarray())
        col_list = ['Education_{}'.format(item) for item in ['Graduation', 'PhD', 'Master', 'Unknown', 'Basic']]
        ohe.columns = col_list
        df = pd.concat([df, ohe], axis=1)

        ohe = pd.DataFrame(encoder2.transform(df[['Marital_Status']]).toarray())
        col_list = ['Marital_Status_{}'.format(item) for item in
                    ['Married', 'Together', 'Single', 'Divorced', 'Widow', 'Unknown']]
        ohe.columns = col_list
        df = pd.concat([df, ohe], axis=1)

        # create new features using the scalers we trained earlier

        df['ln_age'] = ss_scaler0.transform(np.array(np.log(df['age'])).reshape(-1, 1))
        df['ln_Income'] = ss_scaler1.transform(np.array(np.log(df['Income'])).reshape(-1, 1))
        df['ln_Recency'] = ss_scaler2.transform(np.array(np.log(df['Recency'])).reshape(-1, 1))
        df['ln_Spending'] = ss_scaler3.transform(np.array(np.log(df['Total_Spent'])).reshape(-1, 1))
        df['ln_Yrs_Customer'] = ss_scaler3.transform(np.array(np.log(df['Yrs_Customer'])).reshape(-1, 1))

        df['ln_deals'] = ss_scaler4.transform(np.array(df['NumDealsPurchases']).reshape(-1, 1))
        df['ln_web'] = ss_scaler40.transform(np.array(df['NumWebPurchases']).reshape(-1, 1))
        df['ln_catalog'] = ss_scaler41.transform(np.array(df['NumCatalogPurchases']).reshape(-1, 1))
        df['ln_store'] = ss_scaler42.transform(np.array(df['NumStorePurchases']).reshape(-1, 1))
        df['ln_visits'] = ss_scaler43.transform(np.array(df['NumWebVisitsMonth']).reshape(-1, 1))

        # drop & rearrange the columns in the order expected by your trained model!
        df = df[['Kidhome', 'Teenhome', 'AcceptedCmp5', 'Education_Graduation', 'Education_PhD', 'Education_Master',
                 'Education_Unknown', 'Education_Basic',
                 'Marital_Status_Married', 'Marital_Status_Together', 'Marital_Status_Single',
                 'Marital_Status_Divorced', 'Marital_Status_Widow', 'Marital_Status_Unknown',
                 'ln_age', 'ln_Income', 'ln_Recency', 'ln_Spending', 'ln_Yrs_Customer',
                 'ln_deals', 'ln_web', 'ln_catalog', 'ln_store', 'ln_visits']]

        prob = rf.predict_proba(df)
        raw_responded_prob = prob[0][1]
        Threshold = Threshold * .01
        respond_func = lambda y: 'Responded' if raw_responded_prob > Threshold else 'Ignored'
        formatted_ignored_prob = "{:,.1f}%".format(100 * prob[0][0])
        formatted_responded_prob = "{:,.1f}%".format(100 * prob[0][1])
        return respond_func(raw_responded_prob), formatted_responded_prob, formatted_ignored_prob
    except:
        return 'Invalid inputs','Invalid inputs','Invalid inputs'

## FUNCTION FOR VISUALIZATION
def make_loans_cube(*args):
    newdata=pd.DataFrame([args[:15]], columns=features)

    trace0=go.Scatter3d(
        x=responded['Income'],
        y=responded['Total_Spent'],
        z=responded['Recency'],
        name='responded',
        mode='markers',
        text = list(zip(
            ["Marital Status: {}".format(x) for x in responded['Marital_Status']],
            ["<br>Education: {}".format(x) for x in responded['Education']],
            ["<br>Yrs_Customer: {}".format(x) for x in responded['Yrs_Customer']],
            ["<br>Age: {}".format(x) for x in responded['age']],
            ["<br>Kid Home: {}".format(x) for x in responded['Kidhome']],
            ["<br>Teen Home: {}".format(x) for x in responded['Teenhome']],
            ["<br>Accepted Prior Cmp: {}".format(x) for x in responded['AcceptedCmp5']],
            ["<br>Deals Purchased: {}".format(x) for x in responded['NumDealsPurchases']],
            ["<br>Online Purchases: {}".format(x) for x in responded['NumWebPurchases']],
            ["<br>Catalog Purchases: {}".format(x) for x in responded['NumCatalogPurchases']],
            ["<br>Store Purchases: {}".format(x) for x in responded['NumStorePurchases']],
            ["<br>Monthly Web Visits: {}".format(x) for x in responded['NumWebVisitsMonth']]
            )) ,
        hovertemplate =
            '<b>Income: $%{x:.0f}</b>'+
            '<br><b>Total Spent: $%{y:.0f}</b>'+
            '<br><b>Recency: %{z:.0f}</b>'+
            '<br>%{text}',
        hoverinfo='text',
        marker=dict(size=6, color='blue', opacity=0.4))


    trace1=go.Scatter3d(
        x=ignored['Income'],
        y=ignored['Total_Spent'],
        z=ignored['Recency'],
        name='ignored',
        mode='markers',
        text = list(zip(
            ["Marital Status: {}".format(x) for x in ignored['Marital_Status']],
            ["<br>Education: {}".format(x) for x in ignored['Education']],
            ["<br>Yrs_Customer: {}".format(x) for x in ignored['Yrs_Customer']],
            ["<br>Age: {}".format(x) for x in ignored['age']],
            ["<br>Kid Home: {}".format(x) for x in ignored['Kidhome']],
            ["<br>Teen Home: {}".format(x) for x in ignored['Teenhome']],
            ["<br>Accepted Prior Cmp: {}".format(x) for x in ignored['AcceptedCmp5']],
            ["<br>Deals Purchased: {}".format(x) for x in ignored['NumDealsPurchases']],
            ["<br>Online Purchases: {}".format(x) for x in ignored['NumWebPurchases']],
            ["<br>Catalog Purchases: {}".format(x) for x in ignored['NumCatalogPurchases']],
            ["<br>Store Purchases: {}".format(x) for x in ignored['NumStorePurchases']],
            ["<br>Monthly Web Visits: {}".format(x) for x in ignored['NumWebVisitsMonth']]
                )) ,
        hovertemplate =
            '<b>Income: $%{x:.0f}</b>'+
            '<br><b>Total Spent: $%{y:.0f}</b>'+
            '<br><b>Recency: %{z:.0f}</b>'+
            '<br>%{text}',
        hoverinfo='text',
        marker=dict(size=6, color='red', opacity=0.4))

    trace2=go.Scatter3d(
        x=newdata['Income'],
        y=newdata['Total_Spent'],
        z=newdata['Recency'],
        name='Customer',
        mode='markers',
        text = list(zip(
            ["Marital Status: {}".format(x) for x in newdata['Marital_Status']],
            ["<br>Education: {}".format(x) for x in newdata['Education']],
            ["<br>Yrs_Customer: {}".format(x) for x in newdata['Yrs_Customer']],
            ["<br>Age: {}".format(x) for x in newdata['age']],
            ["<br>Kid Home: {}".format(x) for x in newdata['Kidhome']],
            ["<br>Teen Home: {}".format(x) for x in newdata['Teenhome']],
            ["<br>Accepted Prior Cmp: {}".format(x) for x in newdata['AcceptedCmp5']],
            ["<br>Deals Purchased: {}".format(x) for x in newdata['NumDealsPurchases']],
            ["<br>Online Purchases: {}".format(x) for x in newdata['NumWebPurchases']],
            ["<br>Catalog Purchases: {}".format(x) for x in newdata['NumCatalogPurchases']],
            ["<br>Store Purchases: {}".format(x) for x in newdata['NumStorePurchases']],
            ["<br>Monthly Web Visits: {}".format(x) for x in newdata['NumWebVisitsMonth']]
                )) ,
        hovertemplate =
            '<b>Income: $%{x:.0f}</b>'+
            '<br><b>Total Spent: $%{y:.0f}</b>'+
            '<br><b>Recency: %{z:.0f}</b>'+
            '<br>%{text}',
        hoverinfo='text',
        marker=dict(size=15, color='yellow'))


    layout = go.Layout(title="Customer Responses",
                        showlegend=True,
                            scene = dict(
                            xaxis=dict(title='Income'),
                            yaxis=dict(title='Total Spent'),
                            zaxis=dict(title='Recency')
                    ))
    fig=go.Figure([trace0, trace1, trace2], layout)
    return fig


########### Initiate the app
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
server = app.server
app.title=tabtitle

########### Set up the layout
app.layout = html.Div(children=[
    html.H1(myheading1),

    html.Div([
        html.Div(
            [dcc.Graph(id='fig1',style={'width': '90vh', 'height': '90vh'}),
            ], className='eight columns'),
        html.Div([
                html.H3("Features"),
                html.Div('Home With Kids'),
                dcc.Input(id='Kidhome', value=1, type='number', min=0, max=1, step=1),
                html.Div('Home With Teens'),
                dcc.Input(id='Teenhome', value=1, type='number', min=0, max=1, step=1),
                html.Div('Responded to previous campaign'),
                dcc.Input(id='AcceptedCmp5', value=1, type='number', min=0, max=1, step=1),
                html.Div('Age'),
                dcc.Input(id='age', value=25, type='number', min=20, max=80, step=1),
                html.Div('Income'),
                dcc.Input(id='Income', value=25000, type='number', min=2000, max=100000, step=1000),
                html.Div('Days since Most Recent Purchase'),
                dcc.Input(id='Recency', value=30, type='number', min=1, max=100, step=1),
                html.Div('Customer for (in Years)'),
                dcc.Input(id='Yrs_Customer', value=2, type='number', min=1, max=10, step=1),
                html.Div('Prior Spending'),
                dcc.Input(id='Total_Spent', value=120, type='number', min=5, max=2525, step=5),
                html.Div('Education'),
                dcc.Dropdown(id='Education',
                    options=[{'label': i, 'value': i} for i in ['Graduation', 'PhD', 'Master', 'Unknown', 'Basic']],
                    value='Graduation'),
                html.Div('Marital Status'),
                dcc.Dropdown(id='Marital_Status',
                    options=[{'label': i, 'value': i} for i in ['Married', 'Together', 'Single', 'Divorced', 'Widow', 'Unknown']],
                    value='Married'),
                html.Div('Purchases made with Deals (Counts)'),
                dcc.Input(id='NumDealsPurchases', value=1, type='number', min=0, max=15, step=1),
                html.Div('Online Purchases'),
                dcc.Input(id='NumWebPurchases', value=1, type='number', min=0, max=15, step=1),
                html.Div('Catalog Purchases'),
                dcc.Input(id='NumCatalogPurchases', value=1, type='number', min=0, max=15, step=1),
                html.Div('Store Purchases'),
                dcc.Input(id='NumStorePurchases', value=1, type='number', min=0, max=15, step=1),
                html.Div('Web Visits per month'),
                dcc.Input(id='NumWebVisitsMonth', value=1, type='number', min=0, max=30, step=1),
                html.Div('Approval Threshold'),
                dcc.Input(id='Threshold', value=50, type='number', min=0, max=100, step=1),

            ], className='two columns'),
            html.Div([
                html.H3('Predictions'),
                html.Button(children='Submit', id='submit-val', n_clicks=0,
                                style={
                                'background-color': 'red',
                                'color': 'white',
                                'margin-left': '5px',
                                'verticalAlign': 'center',
                                'horizontalAlign': 'center'}
                                ),
                html.Div('Predicted Status:'),
                html.Div(id='PredResults'),
                html.Br(),
                html.Div('Probability of Responding:'),
                html.Div(id='RespondProb'),
                html.Br(),
                html.Div('Probability of Ignoring:'),
                html.Div(id='IgnoreProb')
            ], className='two columns')
        ], className='twelve columns',
    ),

    html.Br(),
    html.A('Code on Github', href=githublink),
    html.Br(),
    html.A("Data Source", href=sourceurl),
    ]
)


######### Define Callback: Predictions
@app.callback(
     Output(component_id='PredResults', component_property='children'),
     Output(component_id='RespondProb', component_property='children'),
     Output(component_id='IgnoreProb', component_property='children'),

     State(component_id='Kidhome', component_property='value'),
     State(component_id='Teenhome', component_property='value'),
     State(component_id='AcceptedCmp5', component_property='value'),
     State(component_id='Education', component_property='value'),
     State(component_id='Marital_Status', component_property='value'),
     State(component_id='age', component_property='value'),
     State(component_id='Income', component_property='value'),
     State(component_id='Recency', component_property='value'),
     State(component_id='Total_Spent', component_property='value'),
     State(component_id='Yrs_Customer', component_property='value'),
     State(component_id='NumDealsPurchases', component_property='value'),
     State(component_id='NumWebPurchases', component_property='value'),
     State(component_id='NumCatalogPurchases', component_property='value'),
     State(component_id='NumStorePurchases', component_property='value'),
     State(component_id='NumWebVisitsMonth', component_property='value'),
     State(component_id='Threshold', component_property='value'),

     Input(component_id='submit-val', component_property='n_clicks'),
    )
def func(*args):
    listofargs=[arg for arg in args[:15]]
    return make_predictions(listofargs, args[15])


######### Define Callback: Visualization

@app.callback(
    Output(component_id='fig1', component_property='figure'),

    State(component_id='Kidhome', component_property='value'),
    State(component_id='Teenhome', component_property='value'),
    State(component_id='AcceptedCmp5', component_property='value'),
    State(component_id='Education', component_property='value'),
    State(component_id='Marital_Status', component_property='value'),
    State(component_id='age', component_property='value'),
    State(component_id='Income', component_property='value'),
    State(component_id='Recency', component_property='value'),
    State(component_id='Total_Spent', component_property='value'),
    State(component_id='Yrs_Customer', component_property='value'),
    State(component_id='NumDealsPurchases', component_property='value'),
    State(component_id='NumWebPurchases', component_property='value'),
    State(component_id='NumCatalogPurchases', component_property='value'),
    State(component_id='NumStorePurchases', component_property='value'),
    State(component_id='NumWebVisitsMonth', component_property='value'),

    Input(component_id='submit-val', component_property='n_clicks'),
    )
def vizfunc(*args):
    return make_loans_cube(*args)


############ Deploy
if __name__ == '__main__':
    app.run_server(debug=True)
