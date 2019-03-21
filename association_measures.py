# This module implements several common measures of association between two entities X (Drug), Y (Side effect)
# 
# We implement two classes of measures, one of them is based on contingency table
# 2 x 2 contingency table
#          | y(yes) | y(no) | total
# ---------|--------|-------|------------------
#   x(yes) |   a %  |   b % | a + b %
# ---------|--------|-------|------------------
#   x(no)  |   c %  |   d % | c + d %
# ---------|--------|-------|------------------
#   total  |a + c % |b + d% | a + b + c + d = 1

# The measures based on contingency table include:
# 	+ Chi Squared Test
# 	+ ROR (Reporting Odds Ratio)
# 	+ ROR05
#
# The other mesures
# 	+ Ralative Reporting Ratio
# 	+ Confidence
# 	+ Leverage
# 	+ Unexlev (MUTARA algorithm)
# 	+ RANK_unexlev (HUNT algrithm)
# 	+ Observed to Expected Ratio (OE ratio)

__author__ = 'thai'

import pandas as pd 
import numpy as np 
import operator
import json

def load_counter_data(filename):
	df = pd.read_csv(filename, index_col=0)
	return dict(df.values)

print ('Loading counter data...')
drug_freq = load_counter_data('count/drug_freq.csv')
se_freq = load_counter_data('count/se_freq.csv')
pair_freq = load_counter_data('count/pair_freq.csv')
pair_unex_freq = load_counter_data('count/unex_pair_freq.csv')

print ('Estimating...')


oo = -999.0

# Number of patients
n = 10756

# Number of drugs in the data
d_vocabsize = len(drug_freq)

# Number of side effects in the data
se_vocabsize = len(se_freq)

# Smoothing coefficient
alpha = 0.01

def __concat__(drug, se):
	# concatenate the drug string and
	# the side effect string
	return drug + '/' + se

def build_contingency_table(drug, se):
	a = pair_freq[__concat__(drug, se)]
	b = drug_freq[drug] - a
	c = se_freq[se] - a
	d = n - a - b - c
	return (a, b, c, d)

def chi_squared_test(drug, se):
	(a, b, c, d) = build_contingency_table(drug, se)
	if a == 0 and b == 0 and c == 0 and d == 0:
		return 0
	else:
		chi_squared = (n * (a*d - b*c)**2)/((a+b) * (c+d) * (b+d) * (a+c))
		# H0: variable1 is not associated with variable2
		# H1: variable1 is associated with variable2
		# df = (2-1)(2-1) = 1, alpha = 0.05 so critical value is 3.84
		# If calculated X^2 value > critical value then we reject the null hypothesis (return 1)
		critical_val = 3.84
		return 1 if float(chi_squared) >= critical_val else 0

def ror(drug, se):
	# Estimation of relative odds ratio (ROR)
	(a, b, c, d) = build_contingency_table(drug, se)
	if c == 0 or b == 0 or d == 0:
		return oo
	else:
		return (float(a)/c)/(float(b)/d)

def ror_05(drug, se):
	# 90% confidence intercal of ROR
	(a, b, c, d) = build_contingency_table(drug, se)
	if a == 0 or b == 0 or c == 0 or d == 0:
		return oo
	else:
		return np.exp(np.log(ror(drug, se)) - 1.645 * np.sqrt(1/float(a) + 1/float(b) + 1/float(c) + 1/float(d)))


def pair_supp(drug, se):
	# Estimating support value of drug and se
	# drug-side effect co-occurrence
	if not __concat__(drug, se) in pair_freq:
		return (0.0 + alpha)/(n + alpha * (d_vocabsize + se_vocabsize))
	else:
		val = float(pair_freq[__concat__(drug, se)])
		return (val + alpha)/(n + alpha * (d_vocabsize + se_vocabsize))

def drug_supp(drug):
	if not drug in drug_freq:
		return (0.0 + alpha)/(n + alpha * d_vocabsize)
	else:
		val = drug_freq[drug]
		return (val + alpha)/(n + alpha * d_vocabsize)

def se_supp(se):
	if not se in se_freq:
		return (0.0 + alpha)/(n + alpha * se_vocabsize)
	else:
		val = se_freq[se]
		return (val + alpha)/(n + alpha * se_vocabsize)

def pair_unex_supp(drug, se):
	if not __concat__(drug, se) in pair_unex_freq:
		return (0.0 + alpha)/(n + alpha * (d_vocabsize + se_vocabsize))
	else:
		val = float(pair_unex_freq[__concat__(drug, se)])
		return (val + alpha)/(n + alpha * (d_vocabsize + se_vocabsize))

def se_unex_supp(drug, se):
	return se_supp(se) - pair_supp(drug, se) + pair_unex_supp(drug, se)

def rr(drug, se):
	# Relative Reporting Ratio (RR)
	if drug_supp(drug) == 0 or se_supp(se) == 0:
		return oo
	else:
		return pair_supp(drug, se)/(drug_supp(drug) * se_supp(se))

def confidence(drug, se):
	if drug_supp(drug) == 0:
		return oo
	else:
		return pair_supp(drug, se)/drug_supp(drug)

def leverage(drug, se):
	return pair_supp(drug, se) - (drug_supp(drug) * se_supp(se))

def unexlev(drug, se):
	return pair_unex_supp(drug, se) - (drug_supp(drug) * se_unex_supp(drug, se))

# Construct leverage and unexlev ranked list of whole drug-side effect pairs
pairs = [tuple(k.split('/')) for k in pair_freq.keys()]

# Estimate leverage and unexlev values for all drug-side effect pairs in the data
lev_vals = [(__concat__(drug, se), leverage(drug, se)) for drug, se in pairs]
unexlev_vals = [(__concat__(drug, se), unexlev(drug, se)) for drug, se in pairs]
lev_vals = sorted(lev_vals, key=operator.itemgetter(1), reverse=True)
unexlev_vals = sorted(unexlev_vals, key=operator.itemgetter(1), reverse=True)

# Get leverage and unexlev rank of all pairs
lev_rank = dict([(item[0], r+1) for (r, item) in enumerate(lev_vals)])
unexlev_rank = dict([(item[0], r+1) for (r, item) in enumerate(unexlev_vals)])

def hunt_alg(drug, se):
	# HUNT calculates both unexlev and leverage values, assigns each medical event two ranks
	# based on unexlev and lerverage values respectively in descending order
	lev_order = lev_rank[__concat__(drug, se)] if __concat__(drug, se) in lev_rank else len(lev_rank) + 1
	unexlev_order = unexlev_rank[__concat__(drug, se)] if __concat__(drug, se) in unexlev_rank else len(unexlev_rank) + 1
	return float(lev_order)/float(unexlev_order)

def oe_ratio(drug, se):
	# Observed to Expected ratio compares the number of patients that have the first prescription of drug x
	# followed by event y within a set time t relative to expected number of admissions if drug x and event y
	# are independent.
	#
	# n_xy: Number of patients that have drug x for the first time and event y occur within time period t
	# n_y:  Number of patients that are prescribed any drug for the first time and have event y within t
	# n:    number of patients
	# E_xy: The expected number of patient that have drug x and then event y is in t
	n_xy = pair_freq[__concat__(drug, se)] if __concat__(drug, se) in pair_freq else 0
	n_y = se_freq[se] if se in se_freq else 0
	n_x = drug_freq[drug] if drug in drug_freq else 0
	E_xy = float(n_x * n_y)/n
	return np.log2( (float(n_xy) + 0.5)/(E_xy + 0.5))


if __name__ == '__main__':
	# Load testing set for confirming true causal relation between drugs and sideeffect
	with open('data/testset.json', 'r') as fin:
		testset = json.load(fin)
		sider_ref = list()
		for k, v in testset.items():
			v = [__concat__(k, item) for item in v]
			sider_ref += v
	sider_ref = set(sider_ref)
	
	output = list()
	columns = ['drug', 'se', 'confirm', 'chi_squared', 'ror', 'ror_05', 'rr', 'conf', 'lev', 'unexlev', 'hunt', 'oe_ratio', 'supp']
	numTruePairs = 0
	drugsFoundSe = list()

	for k, v in pair_freq.items():
		score = list()
		drug, se = k.split('/')
		score.append(drug)
		score.append(se)
		if k in sider_ref:
			score.append(1)
			numTruePairs += 1
			drugsFoundSe.append(drug)
		else:
			score.append(0)
		score.append(chi_squared_test(drug, se))
		score.append(ror(drug, se))
		score.append(ror_05(drug, se))
		score.append(rr(drug, se))
		score.append(confidence(drug, se))
		score.append(leverage(drug, se))
		score.append(unexlev(drug, se))
		score.append(hunt_alg(drug, se))
		score.append(oe_ratio(drug, se))
		score.append(pair_supp(drug, se))
		output.append(score)

	drugsFoundSe = set(drugsFoundSe)

	subset = list()
	for row in output:
		if row[0] in drugsFoundSe:
			subset.append(row)

	del(output) 
	
	subset = pd.DataFrame(subset, columns=columns)
	print (subset.count)
	print ('Number of drugs', len(drugsFoundSe))
	print ('Number of real causal pairs', numTruePairs, ', account for', np.round(numTruePairs * 100.0/len(pair_freq), 2), '%')
	subset.to_csv('baseline_result.csv')









