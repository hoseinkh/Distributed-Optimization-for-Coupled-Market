#CASE14    Power flow data for IEEE 14 bus test case.
#   Please see CASEFORMAT for details on the case file format.
#   This data was converted from IEEE Common Data Format
#   (ieee14cdf.txt) on 15-Oct-2014 by cdf2matp, rev. 2393
#   See end of file for warnings generated during conversion.
#
#   Converted from IEEE CDF file from:
#       https://labs.ece.uw.edu/pstca/
#
#  08/19/93 UW ARCHIVE           100.0  1962 W IEEE 14 Bus Test Case

#   MATPOWER

## MATPOWER Case Format : Version 2

import math

def return_data():
	#
	#
	#
	#
    baseMVA = 100
    #
    ## bus data
    #	bus_i	type	Pd	Qd	Gs	Bs	area	Vm	Va	baseKV	zone	Vmax	Vmin
    bus = [
        [1, 2, 11.2, 7.5, 0, 0, 1, 1.07, -14.22, 0, 1, 1.06, 0.94],
        [2, 1, 9, 5.8, 0, 0, 1, 1.051, -15.1, 0, 1, 1.06, 0.94],
        [3, 1, 3.5, 1.8, 0, 0, 1, 1.057, -14.79, 0, 1, 1.06, 0.94],
        [4, 1, 6.1, 1.6, 0, 0, 1, 1.055, -15.07, 0, 1, 1.06, 0.94],
        [5, 1, 13.5, 5.8, 0, 0, 1, 1.05, -15.16, 0, 1, 1.06, 0.94],
        [6, 1, 14.9, 5, 0, 0, 1, 1.036, -16.04, 0, 1, 1.06, 0.94],
    ]

    ## generator data
    #	bus	Pg	Qg	Qmax	Qmin	Vg	mBase	status	Pmax	Pmin	Pc1,	Pc2,	Qc1min	Qc1max	Qc2min	Qc2max	ramp_agc	ramp_10,	ramp_30,	ramp_q	apf
    gen = [
        [4, 0, 0, 0, 0, 1.070, 100, 1, math.inf, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [4, 0, 0, 0, 0, 1.090, 100, 1, math.inf, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [5, 0, 0, 0, 0, 1.090, 100, 1, math.inf, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    ]

    ## branch data
    #	fbus	tbus	r	x	b	rateA	rateB	rateC	ratio	angle	status	angmin	angmax
    branch = [
        [1, 3, 0.09498, 0.1989, 0.0000, 0, 0, 0, 0, 0, 1, -360, 360],  # 10
        [1, 4, 0.12291, 0.25581, 0.0000, 0, 0, 0, 0, 0, 1, -360, 360],  # 11
        [1, 5, 0.06615, 0.13027, 0.0000, 0, 0, 0, 0, 0, 1, -360, 360],  # 12
        [2, 3, 0.08205, 0.19207, 0.0000, 0, 0, 0, 0, 0, 1, -360, 360],  # 15
        [4, 5, 0.22092, 0.19988, 0.0000, 0, 0, 0, 0, 0, 1, -360, 360],  # 16
        [5, 6, 0.17093, 0.34802, 0.0000, 0, 0, 0, 0, 0, 1, -360, 360],  # 17
    ]

    ##-----  OPF Data  -----##
    ## generator cost data
    #	1,	startup	shutdown	n	x1,	y1,	...	xn	yn
    #	2,	startup	shutdown	n	c(n-1)	...	c0
    gencost = [
        [2, 0, 0, 3, 0.010, 30, 0],
        [2, 0, 0, 3, 0.010, 30, 0],
        [2, 0, 0, 3, 0.015, 35, 0],
    ]
    #
    #
    #
    ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ##
    # modify the node numbers (nodes start at 0 in python as opposed to 1 in MATLAB)
    num_of_nodes = len(bus)
    num_of_gens = len(gen)
    num_of_branches = len(branch)
    for i in range(0, num_of_nodes):
        bus[i][0] = int(bus[i][0]) - 1
    #
    for i in range(0, num_of_gens):
        gen[i][0] = int(gen[i][0]) - 1
    #
    for i in range(0, num_of_branches):
        branch[i][0] = int(branch[i][0]) - 1
        branch[i][1] = int(branch[i][1]) - 1
    ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ##
    #
    return bus, branch, gen, gencost, baseMVA


# Warnings from cdf2matp conversion:
#
# ***** check the title format in the first line of the cdf file.
# ***** Qmax = Qmin at generator at bus    1 (Qmax set to Qmin + 10)
# ***** MVA limit of branch 1 - 2 not given, set to 0
# ***** MVA limit of branch 1 - 5 not given, set to 0
# ***** MVA limit of branch 2 - 3 not given, set to 0
# ***** MVA limit of branch 2 - 4 not given, set to 0
# ***** MVA limit of branch 2 - 5 not given, set to 0
# ***** MVA limit of branch 3 - 4 not given, set to 0
# ***** MVA limit of branch 4 - 5 not given, set to 0
# ***** MVA limit of branch 4 - 7 not given, set to 0
# ***** MVA limit of branch 4 - 9 not given, set to 0
# ***** MVA limit of branch 5 - 6 not given, set to 0
# ***** MVA limit of branch 6 - 11 not given, set to 0
# ***** MVA limit of branch 6 - 12 not given, set to 0
# ***** MVA limit of branch 6 - 13 not given, set to 0
# ***** MVA limit of branch 7 - 8 not given, set to 0
# ***** MVA limit of branch 7 - 9 not given, set to 0
# ***** MVA limit of branch 9 - 10 not given, set to 0
# ***** MVA limit of branch 9 - 14 not given, set to 0
# ***** MVA limit of branch 10 - 11 not given, set to 0
# ***** MVA limit of branch 12 - 13 not given, set to 0
# ***** MVA limit of branch 13 - 14 not given, set to 0
