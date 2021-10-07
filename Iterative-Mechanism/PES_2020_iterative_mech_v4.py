"""

In this code, I modified the code that works on the networked case, "simul_cent_new6" to have some simulations!
"""
#
# import modules
if True:
    import numpy as np
    import cvxpy as cp
    import math
    import scipy.stats as dist
    import matplotlib
    matplotlib.use("agg")
    import matplotlib.pyplot as plt
    import pandas as pd
    import seaborn as sns
    # import plotly.graph_objects as go
    # import plotly.express as px
    import ClassHmatpower
#
# setting list of rho_k
if True:
    range_all_num = []
    # range_all_num.extend(list(range(2,1000,10)))
    # range_all_num.extend(list(range(1000,10000,100)))
    # range_all_num.extend(list(range(10000,100000,100)))
    #
    # range_all_num.extend(list(range(2,1000,10)))
    # range_all_num.extend(list(range(1000,10000,100)))
    # range_all_num.extend(list(range(5000000,10000000,10000)))
    range_all_num.extend(list(np.arange(0.1, 2, 0.1).tolist()))
    range_all_num.extend(list(range(10,300000,300)))
    # range_all_num.extend(list(range(10, 300000, 1000)))
    #
    list_rho_k = []
    for i in range_all_num:
        # list_rho_k.append(1/(1+math.log2(i)))
        if i < 2:
            list_rho_k.append(1 / (1 + i))
        else:
            list_rho_k.append(1 / (1 + math.log2(i)))
    #
    dic_indices_probability_thresholds = dict()
    counter = 0
    for i in list_rho_k:
        dic_indices_probability_thresholds[i] = counter
        counter += 1
    #
    #
    num_simul = len(list_rho_k)
#
# set parameters of simulation
if True:
    epsH = 0.000001
    betaa = 0.3
    #
    probability_threshold_r1 = 0.95
    probability_threshold_r2 = 0.95
    probability_threshold_r3 = 0.95
    ratio_std_to_mean_r1 = 0.25
    ratio_std_to_mean_r2 = 0.25
    ratio_std_to_mean_r3 = 0.25
    #
    gen_DA_RT_vector_all = ["D", "D", "R", "D", "R", "R", "D", "R"]
    gen_DA_RT_vector_r1 = ["D", "D", "R"]
    gen_DA_RT_vector_r2 = ["D", "R", "R"]
    gen_DA_RT_vector_r3 = ["D", "R"]
#
# define variables to store results
if True:
    dict_theta_tielines = dict()
    dic_of_delta_signed_rates = dict()
    dic_mu_rate_absol_at_optimum = dict()
    dic_Tie_line_decentralized = dict()
    #
    list_T_r1_to_r2_updated = []
    list_T_r2_to_r1_updated = []
    list_T_r1_to_r3_updated = []
    list_T_r3_to_r1_updated = []
    list_T_r2_to_r3_updated = []
    list_T_r3_to_r2_updated = []
    #
    list_T_r1_to_r2_calculated = []
    list_T_r2_to_r1_calculated = []
    list_T_r1_to_r3_calculated = []
    list_T_r3_to_r1_calculated = []
    list_T_r2_to_r3_calculated = []
    list_T_r3_to_r2_calculated = []
    list_theta_1_tieline_r1_to_r2 = []
    list_theta_1_tieline_r1_to_r3 = []
    list_theta_2_tieline_r2_to_r1 = []
    list_theta_2_tieline_r2_to_r3 = []
    list_theta_3_tieline_r3_to_r1 = []
    list_theta_3_tieline_r3_to_r2 = []
    list_delta_rate_signed_r1_sell_to_r2_decent_r1 = []
    list_delta_rate_signed_r1_sell_to_r3_decent_r1 = []
    list_delta_rate_signed_r2_sell_to_r1_decent_r2 = []
    list_delta_rate_signed_r2_sell_to_r3_decent_r2 = []
    list_delta_rate_signed_r3_sell_to_r1_decent_r3 = []
    list_delta_rate_signed_r3_sell_to_r2_decent_r3 = []
    list_mu_rate_absol_r1_sell_to_r2_decent_r1 = []
    list_mu_rate_absol_r1_sell_to_r3_decent_r1 = []
    list_mu_rate_absol_r2_sell_to_r1_decent_r2 = []
    list_mu_rate_absol_r2_sell_to_r3_decent_r2 = []
    list_mu_rate_absol_r3_sell_to_r1_decent_r3 = []
    list_mu_rate_absol_r3_sell_to_r2_decent_r3 = []
    #
    matrix_P_RT_gen_dec_r1 = np.zeros((len(list_rho_k),))
    matrix_P_RT_gen_dec_r2__first_RT_gen = np.zeros((len(list_rho_k),))
    matrix_P_RT_gen_dec_r2__second_RT_gen = np.zeros((len(list_rho_k),))
    matrix_P_RT_gen_dec_r3 = np.zeros((len(list_rho_k),))
    matrix_rate_signed_r1_sell_to_r2 = np.zeros((len(list_rho_k),))
    matrix_rate_signed_r2_sell_to_r1 = np.zeros((len(list_rho_k),))
    matrix_rate_signed_r1_sell_to_r3 = np.zeros((len(list_rho_k),))
    matrix_rate_signed_r3_sell_to_r1 = np.zeros((len(list_rho_k),))
    matrix_rate_signed_r2_sell_to_r3 = np.zeros((len(list_rho_k),))
    matrix_rate_signed_r3_sell_to_r2 = np.zeros((len(list_rho_k),))
    #
    matrix_absolut_rate_r1_sell_to_r2 = np.zeros((len(list_rho_k),))
    matrix_absolut_rate_r1_sell_to_r3 = np.zeros((len(list_rho_k),))
    matrix_absolut_rate_r2_sell_to_r3 = np.zeros((len(list_rho_k),))
    #
    matrix_Tie_line_r1_r2___r1 = np.zeros((len(list_rho_k),))
    matrix_Tie_line_r2_r1___r2 = np.zeros((len(list_rho_k),))
    matrix_Tie_line_r1_r3___r1 = np.zeros((len(list_rho_k),))
    matrix_Tie_line_r3_r1___r3 = np.zeros((len(list_rho_k),))
    matrix_Tie_line_r2_r3___r2 = np.zeros((len(list_rho_k),))
    matrix_Tie_line_r3_r2___r3 = np.zeros((len(list_rho_k),))
    #
    matrix_Tie_line_r1_r2___cent = np.zeros((len(list_rho_k),))
    matrix_Tie_line_r1_r3___cent = np.zeros((len(list_rho_k),))
    matrix_Tie_line_r2_r3___cent = np.zeros((len(list_rho_k),))
    #
    matrix_RT_and_DA_system_cost = np.zeros((len(list_rho_k),))
#
counter = 0
for current_rho_k in list_rho_k:
    ## Initialization
    if counter == 0:
        updated_theta_1_tieline_r1_to_r2 = 0
        updated_theta_1_tieline_r1_to_r3 = 0
        updated_theta_2_tieline_r2_to_r1 = 0
        updated_theta_2_tieline_r2_to_r3 = 0
        updated_theta_3_tieline_r3_to_r1 = 0
        updated_theta_3_tieline_r3_to_r2 = 0
        dict_theta_tielines["theta_1_tieline_r1_to_r2"] = updated_theta_1_tieline_r1_to_r2
        dict_theta_tielines["theta_1_tieline_r1_to_r3"] = updated_theta_1_tieline_r1_to_r3
        dict_theta_tielines["theta_2_tieline_r2_to_r1"] = updated_theta_2_tieline_r2_to_r1
        dict_theta_tielines["theta_2_tieline_r2_to_r3"] = updated_theta_2_tieline_r2_to_r3
        dict_theta_tielines["theta_3_tieline_r3_to_r1"] = updated_theta_3_tieline_r3_to_r1
        dict_theta_tielines["theta_3_tieline_r3_to_r2"] = updated_theta_3_tieline_r3_to_r2
        #
        updated_delta_rate_signed_r1_sell_to_r2_decent_r1 = 40
        updated_delta_rate_signed_r1_sell_to_r3_decent_r1 = 35
        updated_delta_rate_signed_r2_sell_to_r1_decent_r2 = 25
        updated_delta_rate_signed_r2_sell_to_r3_decent_r2 = 30
        updated_delta_rate_signed_r3_sell_to_r1_decent_r3 = 33
        updated_delta_rate_signed_r3_sell_to_r2_decent_r3 = 33
        dic_of_delta_signed_rates['delta_rate_signed_r1_sell_to_r2'] = updated_delta_rate_signed_r1_sell_to_r2_decent_r1
        dic_of_delta_signed_rates['delta_rate_signed_r1_sell_to_r3'] = updated_delta_rate_signed_r1_sell_to_r3_decent_r1
        dic_of_delta_signed_rates['delta_rate_signed_r2_sell_to_r1'] = updated_delta_rate_signed_r2_sell_to_r1_decent_r2
        dic_of_delta_signed_rates['delta_rate_signed_r2_sell_to_r3'] = updated_delta_rate_signed_r2_sell_to_r3_decent_r2
        dic_of_delta_signed_rates['delta_rate_signed_r3_sell_to_r1'] = updated_delta_rate_signed_r3_sell_to_r1_decent_r3
        dic_of_delta_signed_rates['delta_rate_signed_r3_sell_to_r2'] = updated_delta_rate_signed_r3_sell_to_r2_decent_r3
        #
        updated_mu_rate_absol_r1_sell_to_r2_decent_r1 = 13
        updated_mu_rate_absol_r1_sell_to_r3_decent_r1 = 2.5
        updated_mu_rate_absol_r2_sell_to_r1_decent_r2 = 13
        updated_mu_rate_absol_r2_sell_to_r3_decent_r2 = 0.2
        updated_mu_rate_absol_r3_sell_to_r1_decent_r3 = 2.5
        updated_mu_rate_absol_r3_sell_to_r2_decent_r3 = 0.2
        dic_mu_rate_absol_at_optimum['mu_rate_absol_r1_sell_to_r2'] = updated_mu_rate_absol_r1_sell_to_r2_decent_r1
        dic_mu_rate_absol_at_optimum['mu_rate_absol_r1_sell_to_r3'] = updated_mu_rate_absol_r1_sell_to_r3_decent_r1
        dic_mu_rate_absol_at_optimum['mu_rate_absol_r2_sell_to_r1'] = updated_mu_rate_absol_r2_sell_to_r1_decent_r2
        dic_mu_rate_absol_at_optimum['mu_rate_absol_r2_sell_to_r3'] = updated_mu_rate_absol_r2_sell_to_r3_decent_r2
        dic_mu_rate_absol_at_optimum['mu_rate_absol_r3_sell_to_r1'] = updated_mu_rate_absol_r3_sell_to_r1_decent_r3
        dic_mu_rate_absol_at_optimum['mu_rate_absol_r3_sell_to_r2'] = updated_mu_rate_absol_r3_sell_to_r2_decent_r3
        #
        updated_tie_line_r1_r2_decent = -18*2
        updated_tie_line_r1_r3_decent = -35*2
        updated_tie_line_r2_r1_decent = 18*2
        updated_tie_line_r2_r3_decent = 21*2
        updated_tie_line_r3_r1_decent = 35*2
        updated_tie_line_r3_r2_decent = -21*2
    #
    counter += 1
    #
    # define case objects
    if True:
        # case_number, gen_DA_RT_list, epsH,
        x_14_all = ClassHmatpower.Hmatpower("case14_all_modified_n3", gen_DA_RT_vector_all, epsH, probability_threshold_r1, probability_threshold_r2, probability_threshold_r3, ratio_std_to_mean_r1, ratio_std_to_mean_r2, ratio_std_to_mean_r3)
        x_14_r1  = ClassHmatpower.Hmatpower("case14_region_n3_r1", gen_DA_RT_vector_r1, epsH, probability_threshold_r1, probability_threshold_r2, probability_threshold_r3, ratio_std_to_mean_r1, ratio_std_to_mean_r2, ratio_std_to_mean_r3)
        x_14_r2  = ClassHmatpower.Hmatpower("case14_region_n3_r2", gen_DA_RT_vector_r2, epsH, probability_threshold_r1, probability_threshold_r2, probability_threshold_r3, ratio_std_to_mean_r1, ratio_std_to_mean_r2, ratio_std_to_mean_r3)
        x_14_r3  = ClassHmatpower.Hmatpower("case14_region_n3_r3", gen_DA_RT_vector_r3, epsH, probability_threshold_r1,probability_threshold_r2, probability_threshold_r3, ratio_std_to_mean_r1,ratio_std_to_mean_r2, ratio_std_to_mean_r3)
        #
        tieline_r1_r2_capacity = x_14_all.branch[9, 5]
        tieline_r1_r3_capacity = x_14_all.branch[18, 5]
        tieline_r2_r3_capacity = x_14_all.branch[19, 5]
        #
        P_DA_gen, T_flow_DA_in_MW, angles_DA, LMPs_DA = x_14_all.centralized_DCOPF_DA()
        # P_DA_gen = np.zeros((8,1))
        DA_Schedule_r1 = P_DA_gen[0:3]
        DA_Schedule_r2 = P_DA_gen[3:6]
        DA_Schedule_r3 = P_DA_gen[6:]
    #
    # solve decentralized OPFs
    if True:
        P_RT_gen_values_cent, T_flow_RT_in_MW_cent, angles_RT_values_cent, dict_theta_tielines_central, LMPs_RT_cent, dic_mu_rate_absol_at_optimum_centralized, dic_of_delta_signed_rates_centralized, dic_Tie_line_centralized, RT_and_DA_system_cost_in_RT_dispatch = x_14_all.centralized_DCOPF_RT(P_DA_gen)
        #
        P_RT_gen_values_dec_r1, T_flow_RT_in_MW_dec_r1, angles_RT_values_dec_r1, LMPs_dec_r1, dic_Tie_line_decen_r1, delta_rate_signed_r1_sell_to_r2_decent_r1, delta_rate_signed_r1_sell_to_r3_decent_r1 = x_14_r1.decentralized_DCOPF_RT_r1(DA_Schedule_r1, dict_theta_tielines, dic_mu_rate_absol_at_optimum, dic_of_delta_signed_rates)
        P_RT_gen_values_dec_r2, T_flow_RT_in_MW_dec_r2, angles_RT_values_dec_r2, LMPs_dec_r2, dic_Tie_line_decen_r2, delta_rate_signed_r2_sell_to_r1_decent_r2, delta_rate_signed_r2_sell_to_r3_decent_r2 = x_14_r2.decentralized_DCOPF_RT_r2(DA_Schedule_r2, dict_theta_tielines, dic_mu_rate_absol_at_optimum, dic_of_delta_signed_rates)
        P_RT_gen_values_dec_r3, T_flow_RT_in_MW_dec_r3, angles_RT_values_dec_r3, LMPs_dec_r3, dic_Tie_line_decen_r3, delta_rate_signed_r3_sell_to_r1_decent_r3, delta_rate_signed_r3_sell_to_r2_decent_r3 = x_14_r3.decentralized_DCOPF_RT_r3(DA_Schedule_r3, dict_theta_tielines, dic_mu_rate_absol_at_optimum, dic_of_delta_signed_rates)
        zzz = 5+6
    #
    ## Updating the parameters
    # update tie-line flows!
    if True:
        calculated_tie_line_r1_r2_decent = dic_Tie_line_decen_r1['Tie_line_r1_r2_decent_r1']
        calculated_tie_line_r2_r1_decent = dic_Tie_line_decen_r2['Tie_line_r2_r1_decent_r2']
        calculated_tie_line_r1_r3_decent = dic_Tie_line_decen_r1['Tie_line_r1_r3_decent_r1']
        calculated_tie_line_r3_r1_decent = dic_Tie_line_decen_r3['Tie_line_r3_r1_decent_r3']
        calculated_tie_line_r2_r3_decent = dic_Tie_line_decen_r2['Tie_line_r2_r3_decent_r2']
        calculated_tie_line_r3_r2_decent = dic_Tie_line_decen_r3['Tie_line_r3_r2_decent_r3']
        updated_tie_line_r1_r2_decent = (1 - current_rho_k) * updated_tie_line_r1_r2_decent + current_rho_k * calculated_tie_line_r1_r2_decent
        updated_tie_line_r2_r1_decent = (1 - current_rho_k) * updated_tie_line_r2_r1_decent + current_rho_k * calculated_tie_line_r2_r1_decent
        updated_tie_line_r2_r3_decent = (1 - current_rho_k) * updated_tie_line_r2_r3_decent + current_rho_k * calculated_tie_line_r2_r3_decent
        updated_tie_line_r3_r2_decent = (1 - current_rho_k) * updated_tie_line_r3_r2_decent + current_rho_k * calculated_tie_line_r3_r2_decent
        updated_tie_line_r1_r3_decent = (1 - current_rho_k) * updated_tie_line_r1_r3_decent + current_rho_k * calculated_tie_line_r1_r3_decent
        updated_tie_line_r3_r1_decent = (1 - current_rho_k) * updated_tie_line_r3_r1_decent + current_rho_k * calculated_tie_line_r3_r1_decent
    #
    # update theta-s
    if True:
        calculated_theta_1_tieline_r1_to_r2 = angles_RT_values_dec_r1[4]
        calculated_theta_1_tieline_r1_to_r3 = angles_RT_values_dec_r1[7]
        calculated_theta_2_tieline_r2_to_r1 = angles_RT_values_dec_r2[0]
        calculated_theta_2_tieline_r2_to_r3 = angles_RT_values_dec_r2[1]
        calculated_theta_3_tieline_r3_to_r1 = angles_RT_values_dec_r3[0]
        calculated_theta_3_tieline_r3_to_r2 = angles_RT_values_dec_r3[0]
        updated_theta_1_tieline_r1_to_r2 = (1 - current_rho_k) * updated_theta_1_tieline_r1_to_r2 + current_rho_k * calculated_theta_1_tieline_r1_to_r2
        updated_theta_1_tieline_r1_to_r3 = (1 - current_rho_k) * updated_theta_1_tieline_r1_to_r3 + current_rho_k * calculated_theta_1_tieline_r1_to_r3
        updated_theta_2_tieline_r2_to_r1 = (1 - current_rho_k) * updated_theta_2_tieline_r2_to_r1 + current_rho_k * calculated_theta_2_tieline_r2_to_r1
        updated_theta_2_tieline_r2_to_r3 = (1 - current_rho_k) * updated_theta_2_tieline_r2_to_r3 + current_rho_k * calculated_theta_2_tieline_r2_to_r3
        updated_theta_3_tieline_r3_to_r1 = (1 - current_rho_k) * updated_theta_3_tieline_r3_to_r1 + current_rho_k * calculated_theta_3_tieline_r3_to_r1
        updated_theta_3_tieline_r3_to_r2 = (1 - current_rho_k) * updated_theta_3_tieline_r3_to_r2 + current_rho_k * calculated_theta_3_tieline_r3_to_r2
    #
    # update delta signed rates
    if True:
        calculated_delta_rate_signed_r1_sell_to_r2_decent_r1 = delta_rate_signed_r1_sell_to_r2_decent_r1
        calculated_delta_rate_signed_r1_sell_to_r3_decent_r1 = delta_rate_signed_r1_sell_to_r3_decent_r1
        calculated_delta_rate_signed_r2_sell_to_r1_decent_r2 = delta_rate_signed_r2_sell_to_r1_decent_r2
        calculated_delta_rate_signed_r2_sell_to_r3_decent_r2 = delta_rate_signed_r2_sell_to_r3_decent_r2
        calculated_delta_rate_signed_r3_sell_to_r1_decent_r3 = delta_rate_signed_r3_sell_to_r1_decent_r3
        calculated_delta_rate_signed_r3_sell_to_r2_decent_r3 = delta_rate_signed_r3_sell_to_r2_decent_r3
        updated_delta_rate_signed_r1_sell_to_r2_decent_r1 = (1 - current_rho_k) * updated_delta_rate_signed_r1_sell_to_r2_decent_r1 + current_rho_k * calculated_delta_rate_signed_r1_sell_to_r2_decent_r1
        updated_delta_rate_signed_r1_sell_to_r3_decent_r1 = (1 - current_rho_k) * updated_delta_rate_signed_r1_sell_to_r3_decent_r1 + current_rho_k * calculated_delta_rate_signed_r1_sell_to_r3_decent_r1
        updated_delta_rate_signed_r2_sell_to_r1_decent_r2 = (1 - current_rho_k) * updated_delta_rate_signed_r2_sell_to_r1_decent_r2 + current_rho_k * calculated_delta_rate_signed_r2_sell_to_r1_decent_r2
        updated_delta_rate_signed_r2_sell_to_r3_decent_r2 = (1 - current_rho_k) * updated_delta_rate_signed_r2_sell_to_r3_decent_r2 + current_rho_k * calculated_delta_rate_signed_r2_sell_to_r3_decent_r2
        updated_delta_rate_signed_r3_sell_to_r1_decent_r3 = (1 - current_rho_k) * updated_delta_rate_signed_r3_sell_to_r1_decent_r3 + current_rho_k * calculated_delta_rate_signed_r3_sell_to_r1_decent_r3
        updated_delta_rate_signed_r3_sell_to_r2_decent_r3 = (1 - current_rho_k) * updated_delta_rate_signed_r3_sell_to_r2_decent_r3 + current_rho_k * calculated_delta_rate_signed_r3_sell_to_r2_decent_r3
    #
    # update mu absolute rates!
    if True:
        # updated_mu_rate_absol_r1_sell_to_r2_decent_r1 = max((updated_mu_rate_absol_r1_sell_to_r2_decent_r1 + betaa * (abs(updated_tie_line_r1_r2_decent) + abs(updated_tie_line_r2_r1_decent)/2 - tieline_r1_r2_capacity)), 0)
        # updated_mu_rate_absol_r1_sell_to_r3_decent_r1 = max((updated_mu_rate_absol_r1_sell_to_r3_decent_r1 + betaa * (abs(updated_tie_line_r1_r3_decent) + abs(updated_tie_line_r3_r1_decent)/2 - tieline_r1_r3_capacity)), 0)
        # updated_mu_rate_absol_r2_sell_to_r1_decent_r2 = max((updated_mu_rate_absol_r2_sell_to_r1_decent_r2 + betaa * (abs(updated_tie_line_r2_r1_decent) + abs(updated_tie_line_r1_r2_decent)/2 - tieline_r1_r2_capacity)), 0)
        # updated_mu_rate_absol_r2_sell_to_r3_decent_r2 = max((updated_mu_rate_absol_r2_sell_to_r3_decent_r2 + betaa * (abs(updated_tie_line_r2_r3_decent) + abs(updated_tie_line_r3_r2_decent)/2 - tieline_r2_r3_capacity)), 0)
        # updated_mu_rate_absol_r3_sell_to_r1_decent_r3 = max((updated_mu_rate_absol_r3_sell_to_r1_decent_r3 + betaa * (abs(updated_tie_line_r3_r1_decent) + abs(updated_tie_line_r1_r3_decent)/2 - tieline_r1_r3_capacity)), 0)
        # updated_mu_rate_absol_r3_sell_to_r2_decent_r3 = max((updated_mu_rate_absol_r3_sell_to_r2_decent_r3 + betaa * (abs(updated_tie_line_r3_r2_decent) + abs(updated_tie_line_r2_r3_decent)/2 - tieline_r2_r3_capacity)), 0)
        #
        # updated_mu_rate_absol_r1_sell_to_r2_decent_r1 = max((updated_mu_rate_absol_r1_sell_to_r2_decent_r1 + betaa * (abs(calculated_tie_line_r1_r2_decent) + abs(calculated_tie_line_r2_r1_decent) / 2 - tieline_r1_r2_capacity)), 0)
        # updated_mu_rate_absol_r1_sell_to_r3_decent_r1 = max((updated_mu_rate_absol_r1_sell_to_r3_decent_r1 + betaa * (abs(calculated_tie_line_r1_r3_decent) + abs(calculated_tie_line_r3_r1_decent) / 2 - tieline_r1_r3_capacity)), 0)
        # updated_mu_rate_absol_r2_sell_to_r1_decent_r2 = max((updated_mu_rate_absol_r2_sell_to_r1_decent_r2 + betaa * (abs(calculated_tie_line_r2_r1_decent) + abs(calculated_tie_line_r1_r2_decent) / 2 - tieline_r1_r2_capacity)), 0)
        # updated_mu_rate_absol_r2_sell_to_r3_decent_r2 = max((updated_mu_rate_absol_r2_sell_to_r3_decent_r2 + betaa * (abs(calculated_tie_line_r2_r3_decent) + abs(calculated_tie_line_r3_r2_decent) / 2 - tieline_r2_r3_capacity)), 0)
        # updated_mu_rate_absol_r3_sell_to_r1_decent_r3 = max((updated_mu_rate_absol_r3_sell_to_r1_decent_r3 + betaa * (abs(calculated_tie_line_r3_r1_decent) + abs(calculated_tie_line_r1_r3_decent) / 2 - tieline_r1_r3_capacity)), 0)
        # updated_mu_rate_absol_r3_sell_to_r2_decent_r3 = max((updated_mu_rate_absol_r3_sell_to_r2_decent_r3 + betaa * (abs(calculated_tie_line_r3_r2_decent) + abs(calculated_tie_line_r2_r3_decent) / 2 - tieline_r2_r3_capacity)), 0)
        #
        updated_mu_rate_absol_r1_sell_to_r2_decent_r1 = max((updated_mu_rate_absol_r1_sell_to_r2_decent_r1 + betaa * (abs(updated_tie_line_r1_r2_decent) + abs(updated_tie_line_r2_r1_decent)/2 - tieline_r1_r2_capacity)), 0)
        updated_mu_rate_absol_r1_sell_to_r3_decent_r1 = max((updated_mu_rate_absol_r1_sell_to_r3_decent_r1 + betaa * (abs(updated_tie_line_r1_r3_decent) + abs(updated_tie_line_r3_r1_decent)/2 - tieline_r1_r3_capacity)), 0)
        updated_mu_rate_absol_r2_sell_to_r1_decent_r2 = updated_mu_rate_absol_r1_sell_to_r2_decent_r1
        updated_mu_rate_absol_r2_sell_to_r3_decent_r2 = max((updated_mu_rate_absol_r2_sell_to_r3_decent_r2 + betaa * (abs(updated_tie_line_r2_r3_decent) + abs(updated_tie_line_r3_r2_decent)/2 - tieline_r2_r3_capacity)), 0)
        updated_mu_rate_absol_r3_sell_to_r1_decent_r3 = updated_mu_rate_absol_r1_sell_to_r3_decent_r1
        updated_mu_rate_absol_r3_sell_to_r2_decent_r3 = updated_mu_rate_absol_r2_sell_to_r3_decent_r2
        #
        # updated_mu_rate_absol_r1_sell_to_r2_decent_r1 = max((updated_mu_rate_absol_r1_sell_to_r2_decent_r1 + betaa * (abs(calculated_tie_line_r1_r2_decent) + abs(calculated_tie_line_r2_r1_decent) / 2 - tieline_r1_r2_capacity)), 0)
        # updated_mu_rate_absol_r1_sell_to_r3_decent_r1 = max((updated_mu_rate_absol_r1_sell_to_r3_decent_r1 + betaa * (abs(calculated_tie_line_r1_r3_decent) + abs(calculated_tie_line_r3_r1_decent) / 2 - tieline_r1_r3_capacity)), 0)
        # updated_mu_rate_absol_r2_sell_to_r1_decent_r2 = updated_mu_rate_absol_r1_sell_to_r2_decent_r1
        # updated_mu_rate_absol_r2_sell_to_r3_decent_r2 = max((updated_mu_rate_absol_r2_sell_to_r3_decent_r2 + betaa * (abs(calculated_tie_line_r2_r3_decent) + abs(calculated_tie_line_r3_r2_decent) / 2 - tieline_r2_r3_capacity)), 0)
        # updated_mu_rate_absol_r3_sell_to_r1_decent_r3 = updated_mu_rate_absol_r1_sell_to_r3_decent_r1
        # updated_mu_rate_absol_r3_sell_to_r2_decent_r3 = updated_mu_rate_absol_r2_sell_to_r3_decent_r2
    #
    # store results
    if True:
        list_T_r1_to_r2_updated.append(updated_tie_line_r1_r2_decent)
        list_T_r2_to_r1_updated.append(updated_tie_line_r2_r1_decent)
        list_T_r1_to_r3_updated.append(updated_tie_line_r1_r3_decent)
        list_T_r3_to_r1_updated.append(updated_tie_line_r3_r1_decent)
        list_T_r2_to_r3_updated.append(updated_tie_line_r2_r3_decent)
        list_T_r3_to_r2_updated.append(updated_tie_line_r3_r2_decent)
        #
        list_T_r1_to_r2_calculated.append(calculated_tie_line_r1_r2_decent)
        list_T_r2_to_r1_calculated.append(calculated_tie_line_r2_r1_decent)
        list_T_r1_to_r3_calculated.append(calculated_tie_line_r1_r3_decent)
        list_T_r3_to_r1_calculated.append(calculated_tie_line_r3_r1_decent)
        list_T_r2_to_r3_calculated.append(calculated_tie_line_r2_r3_decent)
        list_T_r3_to_r2_calculated.append(calculated_tie_line_r3_r2_decent)
        #
        list_theta_1_tieline_r1_to_r2.append(updated_theta_1_tieline_r1_to_r2)
        list_theta_1_tieline_r1_to_r3.append(updated_theta_1_tieline_r1_to_r3)
        list_theta_2_tieline_r2_to_r1.append(updated_theta_2_tieline_r2_to_r1)
        list_theta_2_tieline_r2_to_r3.append(updated_theta_2_tieline_r2_to_r3)
        list_theta_3_tieline_r3_to_r1.append(updated_theta_3_tieline_r3_to_r1)
        list_theta_3_tieline_r3_to_r2.append(updated_theta_3_tieline_r3_to_r2)
        #
        list_delta_rate_signed_r1_sell_to_r2_decent_r1.append(updated_delta_rate_signed_r1_sell_to_r2_decent_r1)
        list_delta_rate_signed_r1_sell_to_r3_decent_r1.append(updated_delta_rate_signed_r1_sell_to_r3_decent_r1)
        list_delta_rate_signed_r2_sell_to_r1_decent_r2.append(updated_delta_rate_signed_r2_sell_to_r1_decent_r2)
        list_delta_rate_signed_r2_sell_to_r3_decent_r2.append(updated_delta_rate_signed_r2_sell_to_r3_decent_r2)
        list_delta_rate_signed_r3_sell_to_r1_decent_r3.append(updated_delta_rate_signed_r3_sell_to_r1_decent_r3)
        list_delta_rate_signed_r3_sell_to_r2_decent_r3.append(updated_delta_rate_signed_r3_sell_to_r2_decent_r3)
        #
        list_mu_rate_absol_r1_sell_to_r2_decent_r1.append(updated_mu_rate_absol_r1_sell_to_r2_decent_r1)
        list_mu_rate_absol_r1_sell_to_r3_decent_r1.append(updated_mu_rate_absol_r1_sell_to_r3_decent_r1)
        list_mu_rate_absol_r2_sell_to_r1_decent_r2.append(updated_mu_rate_absol_r2_sell_to_r1_decent_r2)
        list_mu_rate_absol_r2_sell_to_r3_decent_r2.append(updated_mu_rate_absol_r2_sell_to_r3_decent_r2)
        list_mu_rate_absol_r3_sell_to_r1_decent_r3.append(updated_mu_rate_absol_r3_sell_to_r1_decent_r3)
        list_mu_rate_absol_r3_sell_to_r2_decent_r3.append(updated_mu_rate_absol_r3_sell_to_r2_decent_r3)
        #
        dic_Tie_line_decentralized['Tie_line_r1_r2_decentralized'] = updated_tie_line_r1_r2_decent
        dic_Tie_line_decentralized['Tie_line_r2_r1_decentralized'] = updated_tie_line_r2_r1_decent
        dic_Tie_line_decentralized['Tie_line_r1_r3_decentralized'] = updated_tie_line_r1_r3_decent
        dic_Tie_line_decentralized['Tie_line_r3_r1_decentralized'] = updated_tie_line_r3_r1_decent
        dic_Tie_line_decentralized['Tie_line_r2_r3_decentralized'] = updated_tie_line_r2_r3_decent
        dic_Tie_line_decentralized['Tie_line_r3_r2_decentralized'] = updated_tie_line_r3_r2_decent
        #
        dict_theta_tielines["theta_1_tieline_r1_to_r2"] = updated_theta_1_tieline_r1_to_r2
        dict_theta_tielines["theta_1_tieline_r1_to_r3"] = updated_theta_1_tieline_r1_to_r3
        dict_theta_tielines["theta_2_tieline_r2_to_r1"] = updated_theta_2_tieline_r2_to_r1
        dict_theta_tielines["theta_2_tieline_r2_to_r3"] = updated_theta_2_tieline_r2_to_r3
        dict_theta_tielines["theta_3_tieline_r3_to_r1"] = updated_theta_3_tieline_r3_to_r1
        dict_theta_tielines["theta_3_tieline_r3_to_r2"] = updated_theta_3_tieline_r3_to_r2
        #
        dic_of_delta_signed_rates['delta_rate_signed_r1_sell_to_r2'] = updated_delta_rate_signed_r1_sell_to_r2_decent_r1
        dic_of_delta_signed_rates['delta_rate_signed_r1_sell_to_r3'] = updated_delta_rate_signed_r1_sell_to_r3_decent_r1
        dic_of_delta_signed_rates['delta_rate_signed_r2_sell_to_r1'] = updated_delta_rate_signed_r2_sell_to_r1_decent_r2
        dic_of_delta_signed_rates['delta_rate_signed_r2_sell_to_r3'] = updated_delta_rate_signed_r2_sell_to_r3_decent_r2
        dic_of_delta_signed_rates['delta_rate_signed_r3_sell_to_r1'] = updated_delta_rate_signed_r3_sell_to_r1_decent_r3
        dic_of_delta_signed_rates['delta_rate_signed_r3_sell_to_r2'] = updated_delta_rate_signed_r3_sell_to_r2_decent_r3
        #
        dic_mu_rate_absol_at_optimum['mu_rate_absol_r1_sell_to_r2'] = updated_mu_rate_absol_r1_sell_to_r2_decent_r1
        dic_mu_rate_absol_at_optimum['mu_rate_absol_r1_sell_to_r3'] = updated_mu_rate_absol_r1_sell_to_r3_decent_r1
        dic_mu_rate_absol_at_optimum['mu_rate_absol_r2_sell_to_r1'] = updated_mu_rate_absol_r2_sell_to_r1_decent_r2
        dic_mu_rate_absol_at_optimum['mu_rate_absol_r2_sell_to_r3'] = updated_mu_rate_absol_r2_sell_to_r3_decent_r2
        dic_mu_rate_absol_at_optimum['mu_rate_absol_r3_sell_to_r1'] = updated_mu_rate_absol_r3_sell_to_r1_decent_r3
        dic_mu_rate_absol_at_optimum['mu_rate_absol_r3_sell_to_r2'] = updated_mu_rate_absol_r3_sell_to_r2_decent_r3
#
#
## Plots
#
# plot line flows
if True:
    fig1, ax1 = matplotlib.pyplot.subplots()
    # x_data =
    x_data = list(range(1, num_simul + 1))
    y1_data = list_T_r1_to_r2_updated
    y2_data = list_T_r1_to_r2_calculated
    ax1.plot(x_data, y2_data, label="T 1-->2 calculated")
    ax1.plot(x_data, y1_data, label = "T 1-->2 updated")
    ax1.set(xlabel="k", ylabel='Tie Flow 1 to 2', title="Tie-line flow from 1 to 2")
    ax1.grid()
    ax1.set_ylim([min([min(y1_data)*0.5 , min(y1_data)*1.5]) , max([max(y1_data)*0.5 , max(y1_data)*1.5])])
    ax1.legend()
    fig1.savefig("fig1.png")
    #
    fig2, ax2 = matplotlib.pyplot.subplots()
    # x_data = list_rho_k
    x_data = list(range(1, num_simul + 1))
    y1_data = list_T_r2_to_r1_updated
    y2_data = list_T_r2_to_r1_calculated
    ax2.plot(x_data, y2_data, label="T 2-->1 calculated")
    ax2.plot(x_data, y1_data, label="T 2-->1 updated")
    ax2.set(xlabel="k", ylabel='Tie Flow 2 to 1', title="Tie-line flow from 2 to 1")
    ax2.set_ylim([min([min((min(y1_data),min(y2_data)))*0.5 , min(min(y1_data) , min(y2_data))*1.5]) , max([max(max(y1_data),max(y2_data))*0.5 , max(max(y1_data),max(y2_data))*1.5])])
    ax2.grid()
    ax2.legend()
    fig2.savefig("fig2.png")
    #
    fig3, ax3 = matplotlib.pyplot.subplots()
    # x_data = list_rho_k
    x_data = list(range(1, num_simul + 1))
    y1_data = list_T_r1_to_r3_updated
    y2_data = list_T_r1_to_r3_calculated
    ax3.plot(x_data, y2_data, label="T 1-->3 calculated")
    ax3.plot(x_data, y1_data, label="T 1-->3 updated")
    ax3.set(xlabel="k", ylabel='Tie Flow 1 to 3', title="Tie-line flow from 1 to 3")
    ax3.set_ylim([min([min((min(y1_data),min(y2_data)))*0.5 , min(min(y1_data) , min(y2_data))*1.5]) , max([max(max(y1_data),max(y2_data))*0.5 , max(max(y1_data),max(y2_data))*1.5])])
    ax3.grid()
    ax3.legend()
    fig3.savefig("fig3.png")
    #
    fig4, ax4 = matplotlib.pyplot.subplots()
    # x_data = list_rho_k
    x_data = list(range(1, num_simul + 1))
    y1_data = list_T_r3_to_r1_updated
    y2_data = list_T_r3_to_r1_calculated
    ax4.plot(x_data, y2_data, label="T 3-->1 calculated")
    ax4.plot(x_data, y1_data, label="T 3-->1 updated")
    ax4.set(xlabel="k", ylabel='Tie Flow 3 to 1', title="Tie-line flow from 3 to 1")
    ax4.set_ylim([min([min((min(y1_data),min(y2_data)))*0.5 , min(min(y1_data) , min(y2_data))*1.5]) , max([max(max(y1_data),max(y2_data))*0.5 , max(max(y1_data),max(y2_data))*1.5])])
    ax4.grid()
    ax4.legend()
    fig4.savefig("fig4.png")
    #
    fig5, ax5 = matplotlib.pyplot.subplots()
    # x_data = list_rho_k
    x_data = list(range(1, num_simul + 1))
    y1_data = list_T_r2_to_r3_updated
    y2_data = list_T_r2_to_r3_calculated
    ax5.plot(x_data, y2_data, label="T 2-->3 calculated")
    ax5.plot(x_data, y1_data, label="T 2-->3 updated")
    ax5.set(xlabel="k", ylabel='Tie Flow 2 to 3', title="Tie-line flow from 2 to 3")
    ax5.grid()
    ax5.legend()
    ax5.set_ylim([min([min((min(y1_data),min(y2_data)))*0.5 , min(min(y1_data) , min(y2_data))*1.5]) , max([max(max(y1_data),max(y2_data))*0.5 , max(max(y1_data),max(y2_data))*1.5])])
    fig5.savefig("fig5.png")
    #
    fig6, ax6 = matplotlib.pyplot.subplots()
    # x_data = list_rho_k
    x_data = list(range(1, num_simul + 1))
    y_data = list_T_r3_to_r2_updated
    y1_data = list_T_r3_to_r2_updated
    y2_data = list_T_r3_to_r2_calculated
    ax6.plot(x_data, y2_data, label="T 3-->2 calculated")
    ax6.plot(x_data, y1_data, label="T 3-->2 updated")
    ax6.set(xlabel="k", ylabel='Tie Flow 3 to 2', title="Tie-line flow from 3 to 2")
    ax6.set_ylim([min([min(y_data)*0.5 , min(y_data)*1.5]) , max([max(y_data)*0.5 , max(y_data)*1.5])])
    ax6.grid()
    fig6.savefig("fig6.png")
#
#
# plot delta signed rates
if True:
    fig7, ax7 = matplotlib.pyplot.subplots()
    # x_data = list_rho_k
    x_data = list(range(1, num_simul + 1))
    y_data = list_delta_rate_signed_r1_sell_to_r2_decent_r1
    ax7.plot(x_data, y_data)
    ax7.set(xlabel="k", ylabel='delta signed rate r1 sell to r2', title="delta signed rate r1 sell to r2")
    ax7.set_ylim([min([min(y_data) * 0.5, min(y_data) * 1.5]), max([max(y_data) * 0.5, max(y_data) * 1.5])])
    ax7.grid()
    fig7.savefig("fig7.png")
    # #
    fig8, ax8 = matplotlib.pyplot.subplots()
    # x_data = list_rho_k
    x_data = list(range(1, num_simul + 1))
    y_data = list_delta_rate_signed_r2_sell_to_r1_decent_r2
    ax8.plot(x_data, y_data)
    ax8.set(xlabel="k", ylabel='delta signed rate r2 sell to r1', title="delta signed rate r2 sell to r1")
    ax8.set_ylim([min([min(y_data) * 0.5, min(y_data) * 1.5]), max([max(y_data) * 0.5, max(y_data) * 1.5])])
    ax8.grid()
    fig8.savefig("fig8.png")
    #
    fig9, ax9 = matplotlib.pyplot.subplots()
    # x_data = list_rho_k
    x_data = list(range(1, num_simul + 1))
    y_data = list_delta_rate_signed_r1_sell_to_r3_decent_r1
    ax9.plot(x_data, y_data)
    ax9.set(xlabel="k", ylabel='delta signed rate r1 sell to r3', title="delta signed rate r1 sell to r3")
    ax9.set_ylim([min([min(y_data) * 0.5, min(y_data) * 1.5]), max([max(y_data) * 0.5, max(y_data) * 1.5])])
    ax9.grid()
    fig9.savefig("fig9.png")
    # #
    fig10, ax10 = matplotlib.pyplot.subplots()
    # x_data = list_rho_k
    x_data = list(range(1, num_simul + 1))
    y_data = list_delta_rate_signed_r3_sell_to_r1_decent_r3
    ax10.plot(x_data, y_data)
    ax10.set(xlabel="k", ylabel='delta signed rate r3 sell to r1', title="delta signed rate r3 sell to r1")
    ax10.set_ylim([min([min(y_data) * 0.5, min(y_data) * 1.5]), max([max(y_data) * 0.5, max(y_data) * 1.5])])
    ax10.grid()
    fig10.savefig("fig10.png")
    #
    fig11, ax11 = matplotlib.pyplot.subplots()
    # x_data = list_rho_k
    x_data = list(range(1, num_simul + 1))
    y_data = list_delta_rate_signed_r2_sell_to_r3_decent_r2
    ax11.plot(x_data, y_data)
    ax11.set(xlabel="k", ylabel='delta signed rate r2 sell to r3', title="delta signed rate r2 sell to r3")
    ax11.set_ylim([min([min(y_data) * 0.5, min(y_data) * 1.5]), max([max(y_data) * 0.5, max(y_data) * 1.5])])
    ax11.grid()
    fig11.savefig("fig11.png")
    # #
    fig12, ax12 = matplotlib.pyplot.subplots()
    # x_data = list_rho_k
    x_data = list(range(1, num_simul + 1))
    y_data = list_delta_rate_signed_r3_sell_to_r2_decent_r3
    ax12.plot(x_data, y_data)
    ax12.set(xlabel="k", ylabel='delta signed rate r3 sell to r2', title="delta signed rate r3 sell to r2")
    ax12.set_ylim([min([min(y_data) * 0.5, min(y_data) * 1.5]), max([max(y_data) * 0.5, max(y_data) * 1.5])])
    ax12.grid()
    fig12.savefig("fig12.png")
#
# plot mu absolute rates
if True:
    fig13, ax13 = matplotlib.pyplot.subplots()
    # x_data = list_rho_k
    x_data = list(range(1, num_simul + 1))
    y_data = list_mu_rate_absol_r1_sell_to_r2_decent_r1
    ax13.plot(x_data, y_data)
    ax13.set(xlabel="k", ylabel='mu absolute rate r1 sell to r2', title="mu absolute rate r1 sell to r2")
    ax13.set_ylim([min([min(y_data)*0.5 , min(y_data)*1.5]) , max([max(y_data)*0.5 , max(y_data)*1.5])])
    ax13.grid()
    fig13.savefig("fig13.png")
    # #
    fig14, ax14 = matplotlib.pyplot.subplots()
    # x_data = list_rho_k
    x_data = list(range(1, num_simul + 1))
    y_data = list_mu_rate_absol_r1_sell_to_r3_decent_r1
    ax14.plot(x_data, y_data)
    ax14.set(xlabel="k", ylabel='mu absolute rate r1 sell to r3', title="mu absolute rate r1 sell to r3")
    ax14.set_ylim([min([min(y_data)*0.5 , min(y_data)*1.5]) , max([max(y_data)*0.5 , max(y_data)*1.5])])
    ax14.grid()
    fig14.savefig("fig14.png")
    # #
    fig15, ax15 = matplotlib.pyplot.subplots()
    # x_data = list_rho_k
    x_data = list(range(1, num_simul + 1))
    y_data = list_mu_rate_absol_r2_sell_to_r3_decent_r2
    ax15.plot(x_data, y_data)
    ax15.set(xlabel="k", ylabel='mu absolute rate r2 sell to r3', title="mu absolute rate r2 sell to r3")
    ax15.set_ylim([min([min(y_data)*0.5 , min(y_data)*1.5]) , max([max(y_data)*0.5 , max(y_data)*1.5])])
    ax15.grid()
    fig15.savefig("fig15.png")
    # #
# plot theta of tie-lines
if True:
    fig16, ax16 = matplotlib.pyplot.subplots()
    # x_data = list_rho_k
    x_data = list(range(1, num_simul + 1))
    y_data = list_theta_1_tieline_r1_to_r2
    ax16.plot(x_data, y_data)
    ax16.set(xlabel="k", ylabel='theta 1: tieline 1 to 2', title="theta 1: tieline 1 to 2")
    ax16.set_ylim([min([min(y_data) * 0.5, min(y_data) * 1.5]), max([max(y_data) * 0.5, max(y_data) * 1.5])])
    ax16.grid()
    fig16.savefig("fig16.png")
    # #
    fig17, ax17 = matplotlib.pyplot.subplots()
    # x_data = list_rho_k
    x_data = list(range(1, num_simul + 1))
    y_data = list_theta_2_tieline_r2_to_r1
    ax17.plot(x_data, y_data)
    ax17.set(xlabel="k", ylabel='theta 2: tieline 2 to 1', title="theta 2: tieline 2 to 1")
    ax17.set_ylim([min([min(y_data) * 0.5, min(y_data) * 1.5]), max([max(y_data) * 0.5, max(y_data) * 1.5])])
    ax17.grid()
    fig17.savefig("fig17.png")
    # #
    fig18, ax18 = matplotlib.pyplot.subplots()
    # x_data = list_rho_k
    x_data = list(range(1, num_simul + 1))
    y_data = list_theta_1_tieline_r1_to_r3
    ax18.plot(x_data, y_data)
    ax18.set(xlabel="k", ylabel='theta 1: tieline 1 to 3', title="theta 1: tieline 1 to 3")
    ax18.set_ylim([min([min(y_data) * 0.5, min(y_data) * 1.5]), max([max(y_data) * 0.5, max(y_data) * 1.5])])
    ax18.grid()
    fig18.savefig("fig18.png")
    # #
    fig19, ax19 = matplotlib.pyplot.subplots()
    # x_data = list_rho_k
    x_data = list(range(1, num_simul + 1))
    y_data = list_theta_3_tieline_r3_to_r1
    ax19.plot(x_data, y_data)
    ax19.set(xlabel="k", ylabel='theta 3: tieline 3 to 1', title="theta 3: tieline 3 to 1")
    ax19.set_ylim([min([min(y_data) * 0.5, min(y_data) * 1.5]), max([max(y_data) * 0.5, max(y_data) * 1.5])])
    ax19.grid()
    fig19.savefig("fig19.png")
    # #
    fig20, ax20 = matplotlib.pyplot.subplots()
    # x_data = list_rho_k
    x_data = list(range(1, num_simul + 1))
    y_data = list_theta_2_tieline_r2_to_r3
    ax20.plot(x_data, y_data)
    ax20.set(xlabel="k", ylabel='theta 2: tieline 2 to 3', title="theta 2: tieline 2 to 3")
    ax20.set_ylim([min([min(y_data) * 0.5, min(y_data) * 1.5]), max([max(y_data) * 0.5, max(y_data) * 1.5])])
    ax20.grid()
    fig20.savefig("fig20.png")
    # #
    fig21, ax21 = matplotlib.pyplot.subplots()
    # x_data = list_rho_k
    x_data = list(range(1, num_simul + 1))
    y_data = list_theta_3_tieline_r3_to_r2
    ax21.plot(x_data, y_data)
    ax21.set(xlabel="k", ylabel='theta 3: tieline 3 to 2', title="theta 3: tieline 3 to 2")
    ax21.set_ylim([min([min(y_data) * 0.5, min(y_data) * 1.5]), max([max(y_data) * 0.5, max(y_data) * 1.5])])
    ax21.grid()
    fig21.savefig("fig21.png")
    # #
######################################################
# #
print("optimal value of theta for tielines = {}".format(dict_theta_tielines_central))
print("optimal value of delta signed rates = {}".format(dic_of_delta_signed_rates_centralized))
print("optimal value of absol rates = {}".format(dic_mu_rate_absol_at_optimum_centralized))
print("optimal value of tieline flows = {}".format(dic_Tie_line_centralized))
# #
print("list_rho_k[-10:]".format(list_rho_k[-10:]))
# #
print("list_theta_1_tieline_r1_to_r2[-10:] = ".format(list_theta_1_tieline_r1_to_r2[-10:]))
print("list_theta_1_tieline_r1_to_r3[-10:] = ".format(list_theta_1_tieline_r1_to_r3[-10:]))
print("list_theta_2_tieline_r2_to_r1[-10:] = ".format(list_theta_2_tieline_r2_to_r1[-10:]))
print("list_theta_2_tieline_r2_to_r3[-10:] = ".format(list_theta_2_tieline_r2_to_r3[-10:]))
print("list_theta_3_tieline_r3_to_r1[-10:] = ".format(list_theta_3_tieline_r3_to_r1[-10:]))
print("list_theta_3_tieline_r3_to_r2[-10:] = ".format(list_theta_3_tieline_r3_to_r2[-10:]))
print("list_delta_rate_signed_r1_sell_to_r2_decent_r1[-10:] = {}".format(list_delta_rate_signed_r1_sell_to_r2_decent_r1[-10:]))
print("list_delta_rate_signed_r2_sell_to_r1_decent_r2[-10:] = {}".format(list_delta_rate_signed_r2_sell_to_r1_decent_r2[-10:]))
print("list_delta_rate_signed_r1_sell_to_r3_decent_r1[-10:] = {}".format(list_delta_rate_signed_r1_sell_to_r3_decent_r1[-10:]))
print("list_delta_rate_signed_r3_sell_to_r1_decent_r3[-10:] = {}".format(list_delta_rate_signed_r3_sell_to_r1_decent_r3[-10:]))
print("list_delta_rate_signed_r2_sell_to_r3_decent_r2[-10:] = {}".format(list_delta_rate_signed_r2_sell_to_r3_decent_r2[-10:]))
print("list_delta_rate_signed_r3_sell_to_r2_decent_r3[-10:] = {}".format(list_delta_rate_signed_r3_sell_to_r2_decent_r3[-10:]))
print("list_mu_rate_absol_r1_sell_to_r2_decent_r1[-10:] = {}".format(list_mu_rate_absol_r1_sell_to_r2_decent_r1[-10:]))
print("list_mu_rate_absol_r1_sell_to_r3_decent_r1[-10:] = {}".format(list_mu_rate_absol_r1_sell_to_r3_decent_r1[-10:]))
print("list_mu_rate_absol_r2_sell_to_r3_decent_r2[-10:] = {}".format(list_mu_rate_absol_r2_sell_to_r3_decent_r2[-10:]))
print("list_T_r1_to_r2_updated[-10:] = {}".format(list_T_r1_to_r2_updated[-10:]))
print("list_T_r2_to_r1_updated[-10:] = {}".format(list_T_r2_to_r1_updated[-10:]))
print("list_T_r1_to_r3_updated[-10:] = {}".format(list_T_r1_to_r3_updated[-10:]))
print("list_T_r3_to_r1_updated[-10:] = {}".format(list_T_r3_to_r1_updated[-10:]))
print("list_T_r2_to_r3_updated[-10:] = {}".format(list_T_r2_to_r3_updated[-10:]))
print("list_T_r3_to_r2_updated[-10:] = {}".format(list_T_r3_to_r2_updated[-10:]))
# #
# #
zz = 5 + 6



