## intelliRefinder
Predicting optimal locations for mortgage refinance business opportunities using\
machine learning algorithm on US OpenStreetMap(OSM) data.\
<img src="example.png" width="600" height="360">\
Example: zip code 98103 related area, King County, WA

## Motivation
According to Zillow Research, mortgage market is about $33.3 trillion dollars in 2018. In the primary mortgage market, the loan origination is the primary revenue generator for the mortgage lenders, both bank and non-bank mortgage lenders. Despite some areas have neutral market, mortgage refinance is a healthy business to maintain cash flow for mortgage lenders, particularly those small, regional mortgage lenders.

## Problem
Because most regional mortgage lenders lack of resource, it is hard for them to identify mortgage refinance opportunities actively and to optimize their marketing strategies.

## Data
* 2008-2017 mortgage transaction data, available via Home Mortgage Disclosure Act;
* Demographic data from 10-year Census American Community Survey data;
* King County, WA selected for the demo project.

## Actionable Insight
The trained logistic model classified mortgage refinance opportunities in King County, WA and was then used to create an interactive maps to help mortgage lenders to actively identify potential mortgage refinance business in the area of interest and to decide how optimize their resource allocations for marketing. Please visit [intelliRefinder](http://bit.ly/IntelliRefinderDemo) to explore the mortgage refinance opportunities of King County, WA.

Important features for classifying the mortgage refinance opportunity:
* Loan status
* Marriage status
* Loaner purchaser
* Income
* Minor (age<18) population
* Work travel time

## Tools and Techs
* Python
  * Scikit-Learn
  * GeoPandas
  * Streamlit
* GIS
* Logistic Regression
* Feature Engineering
* SQL
* AWS EC2, S3

### Contact
meng.chen03(at)gmail.com
