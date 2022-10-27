import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

from pypfopt.expected_returns import mean_historical_return
from pypfopt.risk_models import CovarianceShrinkage
from pypfopt.efficient_frontier import EfficientFrontier

pd.options.plotting.backend = "plotly"

assets = ['SPX Index','DXY Curncy','EHFI451 Index','EHFI453 Index','XAU Curncy']
proxy_assets = dict(
    offense = ['SPX Index'],
    defence = ['EHFI451 Index','EHFI453 Index','XAU Curncy'],
    cash = ['DXY Curncy'],
)

# Load data & format
data_df = pd.read_csv('poc_data.csv')
data_df.columns = ['Date'] + data_df.columns.tolist()[1:]
data_df['Date'] = pd.to_datetime(data_df.Date)
# print(data_df.shape)
data_df = data_df.dropna(how='any')
data_df = data_df.set_index('Date')
# print(data_df.shape)

# Classify regimes
# Resample to quarterly data just to classify regimes
qdata_df = data_df[['EHPIUS Index','GDP CQOQ Index']].resample('Q', convention='start').asfreq()
# print(qdata_df.shape)

# Assign regimes
feat_df = qdata_df.copy()
feat_df['EHPIUS Index'] = feat_df['EHPIUS Index'].diff(1)
feat_df = feat_df.dropna()
feat_df['regime'] = np.nan

feat_df['regime'] = np.where((feat_df['EHPIUS Index']<0) & (feat_df['GDP CQOQ Index']>0),"Quad 1", 0)
feat_df['regime'] = np.where((feat_df['EHPIUS Index']>0) & (feat_df['GDP CQOQ Index']>0),"Quad 2", feat_df['regime'])
feat_df['regime'] = np.where((feat_df['EHPIUS Index']>0) & (feat_df['GDP CQOQ Index']<0),"Quad 3", feat_df['regime'])
feat_df['regime'] = np.where((feat_df['EHPIUS Index']<0) & (feat_df['GDP CQOQ Index']<0),"Quad 4", feat_df['regime'])

# Label monthly data (regime)
data_df1 = data_df[assets]

data_df2 = data_df1.merge(feat_df[['regime']], 'left', left_index=True, right_index=True)
data_df2['regime'] = data_df2['regime'].ffill()
# print(data_df2.shape)

data_df3 = data_df2.dropna(subset=['regime'])
# print(data_df3.shape)

# Returns df (monthly)
ret_df = data_df3.copy()
ret_df[assets] = ret_df[assets].pct_change()
ret_df = ret_df.dropna()


# Estimate distributions conditional on regime
def sumstats(df, regime):
    mu = mean_historical_return(df, returns_data=True, frequency=12, compounding=False)
    S = CovarianceShrinkage(df,  returns_data=True, frequency=12).ledoit_wolf()
    return dict(regime=regime, mean=mu, cov=S)


li_ss = [sumstats(group.drop(columns=['regime']), name) for name, group in ret_df.groupby('regime')]
# len(li_ss)

# Plot asset returns conditional on regime
fig_rets_dic = {}

for a in assets: 
    fig = px.histogram(ret_df, x=a, color="regime",
                   marginal="box", # or violin, rug
                   nbins=30,
                   )
    fig.update_layout(height=500, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    fig_rets_dic[a] = fig


# Optimization conditional on regime
def eff_front(dic):
    ef = EfficientFrontier(dic['mean'], dic['cov'])
    weights = ef.max_sharpe()
    cleaned_weights = ef.clean_weights()
    return dict(regime=dic['regime'], ef=ef, weights=cleaned_weights)

li_eff = [eff_front(i) for i in li_ss]


# Create regime weighting map
rw_df = pd.DataFrame([dict(regime=i['regime'], **i['weights']) for i in li_eff])
rw_df = rw_df.set_index('regime')


# Backtest cond allocations
ts_regimes = ret_df[['regime']]
ts_weights = rw_df.merge(ret_df[['regime']], how='right', left_index=True, right_on='regime').drop(columns=['regime'])

# Porfolio weighted returns
port_rets = ret_df.copy().drop(columns=['regime'])
# print(ts_weights.shape == port_rets.shape)
port_rets['Portfolio'] = (port_rets * ts_weights).sum(axis=1)

# Cum rets
port_crets = port_rets.add(1).cumprod()


# Plotting regimes
ts_regimes1 = ts_regimes.copy()
ts_regimes1['regime_start'] = ts_regimes[ts_regimes.regime != ts_regimes.regime.shift(1)]
ts_regimes1['regime_end'] = ts_regimes[ts_regimes.regime != ts_regimes.regime.shift(-1)]

regime_li = []

for i in list(zip(ts_regimes[ts_regimes.regime != ts_regimes.regime.shift(1)].reset_index().to_records(), \
    ts_regimes[ts_regimes.regime != ts_regimes.regime.shift(-1)].reset_index().to_records())):
    regime_li += [(i[0][2], pd.Timestamp(i[0][1]).replace(day=1).strftime('%Y-%m-%d'), i[1][1])]


fillcolors_regimes = {
    'Quad 1': "green",
    'Quad 2': "blue",
    'Quad 3': "yellow",
    'Quad 4': "LightSalmon",
}

def add_regimes(fig, regimes, fillcolors_regimes=fillcolors_regimes):
    for i in regimes:
        start = i[1]
        end = i[2].astype(str)[:10]

        # Add shape regions
        fig.add_vrect(
            x0=start, x1=end,
            fillcolor=fillcolors_regimes[i[0]], opacity=0.2,
            layer="below", line_width=0,
        )
    return fig


# # Colorcode regimes
# def highlight_rows(val, fillcolors_regimes=fillcolors_regimes):
#     return 'background-color: {}'.format(fillcolors_regimes[val])


# reg_df = ret_df.regime.value_counts().reset_index()
