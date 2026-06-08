/****************************************************************************************
 EGX Diversification, Information Asymmetry, Governance, and Earnings Quality
 Optional Stata skeleton

 Before running:
 1. Export data/processed/analysis_panel.csv from the Python pipeline.
 2. Confirm variables: firm_id, year, EQ, DIV, AMIHUD, CGQ, controls.
****************************************************************************************/

clear all
set more off

* Change this path if needed
import delimited using "data/processed/analysis_panel.csv", clear varnames(1) encoding(UTF-8)

* Create numeric panel identifier if firm_id is string
capture confirm numeric variable firm_id
if _rc {
    encode firm_id, gen(firm_num)
}
else {
    gen firm_num = firm_id
}

xtset firm_num year

* Optional: centered interaction term for moderation
capture drop AMIHUD_c CGQ_c IA_x_CGQ
summarize AMIHUD, meanonly
gen AMIHUD_c = AMIHUD - r(mean)
summarize CGQ, meanonly
gen CGQ_c = CGQ - r(mean)
gen IA_x_CGQ = AMIHUD_c * CGQ_c

* Descriptive statistics
summarize EQ DIV AMIHUD CGQ Firm_Size Leverage Firm_Age ROA Sales_Growth Cash_Flow

* H1 direct effect
xtreg EQ DIV Firm_Size Leverage Firm_Age ROA Sales_Growth Cash_Flow i.year, fe vce(cluster firm_num)

* H2 first-stage path
xtreg AMIHUD DIV Firm_Size Leverage Firm_Age ROA Sales_Growth Cash_Flow i.year, fe vce(cluster firm_num)

* H3-H5 moderated outcome model
xtreg EQ DIV AMIHUD CGQ IA_x_CGQ Firm_Size Leverage Firm_Age ROA Sales_Growth Cash_Flow i.year, fe vce(cluster firm_num)

* For H4/H6 moderated mediation, prefer bootstrap in Python because it is easier to store conditional indirect effects.
