import streamlit as st
from PIL import Image

import poc_model_script as ms
#import assets, ret_df, li_ss, fig_rets_dic, fillcolors_regimes, regime_li, add_regimes, port_rets

bslogo = Image.open('logo-brightside.png')

# Page configs
st.set_page_config(
    page_title="BS Portfolio Model",
    #page_icon="",
    # layout="wide",
)


# Header
header_col1, header_col2 = st.columns([1, 3], gap='medium')

with header_col1:
    st.image(bslogo, width=100)

with header_col2:
    st.title('Portfolio Allocation Model Proof of Concept')

# Conceptual walkthrough
conceptimg = Image.open('conceptual_flowchart_v2-grey.png')

st.empty()

# _____________________________________________________________________________
#
# Description
# _____________________________________________________________________________

st.empty()
st.markdown(
    """
    ## Intro

    This presents a proof of concept for how to systematize parts of our portfolio allocation methodology going forward

    The intent is to lay out a concise framework on how to quantitatively process both empirical data and discretionary views into a portfolio model that ultimately provides us with a mathematical benchmark for optimal allocation weights. 

    This model is kept intentionally simple. Each of these steps contains a set of non-trivial assumptions that would need to be revisited in detail if we follow through on this approach (see section at end).
    
    **In short, below I'm asking and giving a naive answer to the question:**
    **Given past asset price realizations across different regimes (quads), what portfolio weights *would have* resulted in the optimal return/risk ratio during each regime?**
    """
)

st.image(conceptimg)

st.markdown(
    """
    ## Guide

    #### 1. Model Inputs ####
    - **Empirical Distribution** of risk assets in our universe, for example Indices, Funds, Equities, etc. **(priors)**
    
    &rarr; For this example, I have selected a few 'proxy' assets to broadly mimick how we think about portfolio construction, namely:
        *Offense* &rarr; SPX Index,
        *Defense* &rarr; EHFI451 Index (Long Vol Index) & EHFI453 Index (Tail Risk Index),
        *Cash* &rarr; DXY Curncy & XAU Curncy


    - **Discretionary Views** on risk assets (relative or absolute), for example GIP model, upper/lower return bounds for asset, asset class X will outperform asset class Y, etc.
    
    &rarr; For this example, I'm using a simplified version of **Hedgeyes GIP model** to classify historical timeseries into one of the 4 quadrants. These regimes are determined by two factors only, namely the rate of change in QoQ GDP and QoQ CPI.
    I'm then using these regimes to estimate regime-conditional return distributions for each asset, essentially making expected returns for each asset dependent on the (expected) regime.

    - A priori **portfolio constraints**, for example Equities exposure > 10 % at all times, Gold never < 40 %, etc. These might come from our risk management framework which dictates certain max/min exposures or anything similar.
    
    &rarr; In this case, I'm ommitting any additional portfolio constraints

    #### 2. Construct posterior asset distributions ####
    This is where we combine prior estimates of returns (e.g. market implied returns) with views on certain assets, to produce a **posterior** estimate of expected returns.
    The mathematical model behind this is called **Black-Litterman**.


    #### 3. Solve for optimal portfolio weights according to pre-defined objective function ####
    The traditional metric to optimize for is **mean-variance**, which I'm also using here for ilustrative purposes
    Other objective functions may include max return, min risk, max risk adj return ratio, max utility, mean downside variance, etc. 

    This results in **portfolio weights** optimized for the selected objective.
    
    &rarr; Here we're solving for a max sharpe ratio, i.e. mean-variance optimization (MVO)

    #### 4. Backtesting ####
    Finally, the weights get tested both against historical out of sample or simulated forward looking data (Monte Carlo).
    
    &rarr; Here I'm doing a quick in-sample backtest without any post-processing or stress testing of weights 

    ________________________________________________________________________________________________________________________________
    """
)


# _____________________________________________________________________________
#
# Estimating returns assets
# _____________________________________________________________________________

# Create figure w/ regimes
fig_cret = ms.port_crets.drop(columns=['Portfolio']).plot()
# fig_cret = ms.add_regimes(fig_cret, ms.regime_li)
fig_cret.update_layout(height=500, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", 
                        legend=dict(orientation="h"), 
                        margin=dict(l=0,r=0,b=0,t=0)
                        )


# Returns regime cond
fig_ret = ms.ret_df.drop(columns=['regime']).plot.bar()
fig_ret.update_layout(height=500, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", 
                        legend=dict(orientation="h"),
                        margin=dict(l=0,r=0,b=0,t=0)
                        )


# GIP regime model 
fig_gip_model = ms.feat_df[['EHPIUS Index','GDP CQOQ Index']].plot()
fig_gip_model = ms.add_regimes(fig_gip_model, ms.regime_li)
fig_gip_model.update_layout(height=500, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", 
                        legend=dict(orientation="h"),
                        margin=dict(l=0,r=0,b=0,t=0)
                        )


# Display charts 
st.empty()


st.header('Model Inputs')
st.subheader('Proxy Assets')
st.plotly_chart(fig_cret, use_container_width=True)

model_inputs_cols = st.columns(2) 
with model_inputs_cols[0]:
    st.subheader('Raw returns')
    st.plotly_chart(fig_ret)

with model_inputs_cols[0]:
    st.subheader('GIP Model: Inflation & GDP')
    st.plotly_chart(fig_gip_model)

# Describe return tables (whole data)
hist_returns_cols1, hist_returns_cols2 = st.columns([3, 1], gap='medium')
    
with hist_returns_cols1:
    st.subheader('Returns Summary')
    st.write(ms.ret_df.describe())

with hist_returns_cols2:
    st.subheader('Sample Size')
    st.write(ms.ret_df.regime.value_counts())


# Historical return assets conditional on regime analysis
st.empty()
st.header('Expected Returns')
st.subheader('Return distribution across regimes')

asset_rets_tabs = st.tabs(ms.assets)
for i, name in enumerate(ms.assets):
    tab_content = asset_rets_tabs[i]
    with tab_content:
            st.plotly_chart(ms.fig_rets_dic[name])


# Regime conditional moments
st.subheader('Distribution Moments conditioned on regimes')
reg_names = [i['regime'] for i in ms.li_ss]
regimes_ss_tabs = st.tabs(reg_names)

for i,r in enumerate(ms.li_ss):
    with regimes_ss_tabs[i]:
        st.write(r['regime'])
        dist_mom1, dist_mom2 = st.columns([1, 3], gap='medium')
    
        with dist_mom1:
            st.caption('Mean')
            st.write(r['mean'])

        with dist_mom2:
            st.caption('Covariance')
            st.write(r['cov'])


# _____________________________________________________________________________
# 
# Optimization
# _____________________________________________________________________________
st.empty()
st.header('Optimization Model Results')

st.caption('Objective:') 
st.latex(r'''
    max \quad \mu / \sigma
    ''')

st.caption('Constraints: Apart from non-negativity of weights no constraints were imposed') 

st.subheader('Efficient Frontier across Regimes')
regimes_eff_tabs = st.tabs(reg_names)

for i,r in enumerate(ms.li_eff):
    with regimes_eff_tabs[i]:
        er, vol, sr = r['ef'].portfolio_performance(verbose=True)
        st.write(r['regime'])

        st.caption('Optimizer Solution')
        st.text(
            """
            Expected annual return: {}% 
            Annual volatility: {}%
            Sharpe Ratio: {}%
            """.format(round(er,2)*100, round(vol,2)*100, round(sr,2)*100)
        )

        st.empty()
        st.caption('Optimal weights')
        st.write(r['weights'])

# _____________________________________________________________________________
# 
# Backtest
# _____________________________________________________________________________
st.empty()
st.header('Backtest')

# Create backtest figure w/ regimes
fig_cret = ms.port_crets.plot()

fig_cret = ms.add_regimes(fig_cret, ms.regime_li)
fig_cret.update_layout(height=500, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", 
                        legend=dict(orientation="h"),
                        margin=dict(l=0,r=0,b=0,t=0)
                        )

st.plotly_chart(fig_cret, use_container_width=True)

st.subheader('Weights')
fig_wbtest = ms.ts_weights.plot()
fig_wbtest = ms.add_regimes(fig_wbtest, ms.regime_li)
fig_wbtest.update_layout(height=300, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", 
                        legend=dict(orientation="h"),
                        margin=dict(l=0,r=0,b=0,t=0)
                        )

st.plotly_chart(fig_wbtest, use_container_width=True)


# Tearsheet
# qs.plots.snapshot(port_rets.Portfolio, title='Portfolio Performance')

st.empty()
st.markdown(
    """
    ________________________________________________________________________________________________________________________________

    ## Key Considerations

    #### Ad 1. Model Inputs ####
    - Two major challenges with empirical data
    1. **Proxy returns**: Since we're allocating to funds with limited track records and non-trivial risk-return profiles we will either need to find proxies or replicate their return characteristics (think of long vol)
    2. **Sample size**: As you can see I've included a table with number of observations above for each regime. As an example, 13 data points for Quad4 regime make any statistic/inference about dynamics within that regime superflous given statistical insignificance. In a similar vein, a hedge fund with a 3yr track record of monthly returns provides a very limited basis to understand how it will react in different regimes. Thus, we might have to resort to simulating risk profiles.

    - Translating discretionary views into a language the mathematical model can understand is also not straightforward but key to making forward-looking allocations. 
    (Last 10yrs of market data will tell us nothing about next 10yrs) This is were our true edge comes in.


    #### Ad 3. Optimization ####
    - Mathematical optimization is a very fragile undertaking so understanding objective functions and parameter sensibilities is paramount. Clearly, we will not want to stick to a linear model like MVO, particularly in the light of the highly non-liner payoff profiles that we want in our exposure.
    This will mean iterating through a large amount of objective functions, stress testing results against a series of scenarios, and thinking in bounds instead of point-estimates.


    Let's discuss!

    """
)
