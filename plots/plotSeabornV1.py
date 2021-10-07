#
#
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
# plt.interactive(False)
plt.interactive(True)
#
# Reading from text file
All_variables = {}
with open("detaileResults.txt") as f:
    for line in f:
        name, value = line.split("=")
        All_variables[name] = eval(value)

for i in All_variables.keys():
    exec(i + ' = ' + str(All_variables[i]))

"""
list_of_variables = [   
    optimal_theta_1_tieline_r1_to_r2, 
    optimal_theta_1_tieline_r1_to_r3, 
    optimal_theta_2_tieline_r2_to_r1, 
    optimal_theta_2_tieline_r2_to_r3, 
    optimal_theta_3_tieline_r3_to_r1, 
    optimal_theta_3_tieline_r3_to_r2, 
    optimal_rate_signed_r1_sell_to_r2, 
    optimal_rate_signed_r1_sell_to_r3, 
    optimal_rate_signed_r2_sell_to_r1, 
    optimal_rate_signed_r2_sell_to_r3, 
    optimal_rate_signed_r3_sell_to_r1, 
    optimal_rate_signed_r3_sell_to_r2, 
    optimal_rate_absol_r1_sell_to_r2, 
    optimal_rate_absol_r1_sell_to_r3, 
    optimal_rate_absol_r2_sell_to_r1, 
    optimal_rate_absol_r2_sell_to_r3, 
    optimal_rate_absol_r3_sell_to_r1, 
    optimal_rate_absol_r3_sell_to_r2, 
    optimal_Tie_line_r1_r2_centralized, 
    optimal_Tie_line_r1_r3_centralized, 
    optimal_Tie_line_r2_r3_centralized, 
    list_rho_k, 
    list_theta_1_tieline_r1_to_r2_in_degree, 
    list_theta_1_tieline_r1_to_r3_in_degree, 
    list_theta_2_tieline_r2_to_r1_in_degree, 
    list_theta_2_tieline_r2_to_r3_in_degree, 
    list_theta_3_tieline_r3_to_r1_in_degree, 
    list_theta_3_tieline_r3_to_r2_in_degree, 
    list_delta_rate_signed_r1_sell_to_r2_decent_r1, 
    list_delta_rate_signed_r2_sell_to_r1_decent_r2, 
    list_delta_rate_signed_r1_sell_to_r3_decent_r1, 
    list_delta_rate_signed_r3_sell_to_r1_decent_r3, 
    list_delta_rate_signed_r2_sell_to_r3_decent_r2, 
    list_delta_rate_signed_r3_sell_to_r2_decent_r3, 
    list_mu_rate_absol_r1_sell_to_r2_decent_r1, 
    list_mu_rate_absol_r1_sell_to_r3_decent_r1, 
    list_mu_rate_absol_r2_sell_to_r3_decent_r2, 
    list_T_r1_to_r2_updated, 
    list_T_r2_to_r1_updated, 
    list_T_r1_to_r3_updated, 
    list_T_r3_to_r1_updated, 
    list_T_r2_to_r3_updated, 
    list_T_r3_to_r2_updated  ]
"""

data = {
    'list_k': list(range(0,len(list_rho_k))),
    'list_rho_k': list_rho_k,
    'optimal_theta_1_tieline_r1_to_r2': optimal_theta_1_tieline_r1_to_r2,
    'optimal_theta_1_tieline_r1_to_r3': optimal_theta_1_tieline_r1_to_r3,
    'optimal_theta_2_tieline_r2_to_r1': optimal_theta_2_tieline_r2_to_r1,
    'optimal_theta_2_tieline_r2_to_r3': optimal_theta_2_tieline_r2_to_r3,
    'optimal_theta_3_tieline_r3_to_r1': optimal_theta_3_tieline_r3_to_r1,
    'optimal_theta_3_tieline_r3_to_r2': optimal_theta_3_tieline_r3_to_r2,
    'optimal_rate_signed_r1_sell_to_r2': optimal_rate_signed_r1_sell_to_r2,
    'optimal_rate_signed_r1_sell_to_r3': optimal_rate_signed_r1_sell_to_r3,
    'optimal_rate_signed_r2_sell_to_r1': optimal_rate_signed_r2_sell_to_r1,
    'optimal_rate_signed_r2_sell_to_r3': optimal_rate_signed_r2_sell_to_r3,
    'optimal_rate_signed_r3_sell_to_r1': optimal_rate_signed_r3_sell_to_r1,
    'optimal_rate_signed_r3_sell_to_r2': optimal_rate_signed_r3_sell_to_r2,
    'optimal_rate_absol_r1_sell_to_r2': optimal_rate_absol_r1_sell_to_r2,
    'optimal_rate_absol_r1_sell_to_r3': optimal_rate_absol_r1_sell_to_r3,
    'optimal_rate_absol_r2_sell_to_r1': optimal_rate_absol_r2_sell_to_r1,
    'optimal_rate_absol_r2_sell_to_r3': optimal_rate_absol_r2_sell_to_r3,
    'optimal_rate_absol_r3_sell_to_r1': optimal_rate_absol_r3_sell_to_r1,
    'optimal_rate_absol_r3_sell_to_r2': optimal_rate_absol_r3_sell_to_r2,
    'optimal_Tie_line_r1_r2_centralized': optimal_Tie_line_r1_r2_centralized,
    'optimal_Tie_line_r1_r3_centralized': optimal_Tie_line_r1_r3_centralized,
    'optimal_Tie_line_r2_r3_centralized': optimal_Tie_line_r2_r3_centralized,
    'list_theta_1_tieline_r1_to_r2_in_degree': list_theta_1_tieline_r1_to_r2_in_degree,
    'list_theta_1_tieline_r1_to_r3_in_degree': list_theta_1_tieline_r1_to_r3_in_degree,
    'list_theta_2_tieline_r2_to_r1_in_degree': list_theta_2_tieline_r2_to_r1_in_degree,
    'list_theta_2_tieline_r2_to_r3_in_degree': list_theta_2_tieline_r2_to_r3_in_degree,
    'list_theta_3_tieline_r3_to_r1_in_degree': list_theta_3_tieline_r3_to_r1_in_degree,
    'list_theta_3_tieline_r3_to_r2_in_degree': list_theta_3_tieline_r3_to_r2_in_degree,
    'list_delta_rate_signed_r1_sell_to_r2_decent_r1': list_delta_rate_signed_r1_sell_to_r2_decent_r1,
    'list_delta_rate_signed_r2_sell_to_r1_decent_r2': list_delta_rate_signed_r2_sell_to_r1_decent_r2,
    'list_delta_rate_signed_r1_sell_to_r3_decent_r1': list_delta_rate_signed_r1_sell_to_r3_decent_r1,
    'list_delta_rate_signed_r3_sell_to_r1_decent_r3': list_delta_rate_signed_r3_sell_to_r1_decent_r3,
    'list_delta_rate_signed_r2_sell_to_r3_decent_r2': list_delta_rate_signed_r2_sell_to_r3_decent_r2,
    'list_delta_rate_signed_r3_sell_to_r2_decent_r3': list_delta_rate_signed_r3_sell_to_r2_decent_r3,
    'list_mu_rate_absol_r1_sell_to_r2_decent_r1': list_mu_rate_absol_r1_sell_to_r2_decent_r1,
    'list_mu_rate_absol_r1_sell_to_r3_decent_r1': list_mu_rate_absol_r1_sell_to_r3_decent_r1,
    'list_mu_rate_absol_r2_sell_to_r3_decent_r2': list_mu_rate_absol_r2_sell_to_r3_decent_r2,
    'list_T_r1_to_r2_updated': list_T_r1_to_r2_updated,
    'list_T_r2_to_r1_updated': list_T_r2_to_r1_updated,
    'list_T_r1_to_r3_updated': list_T_r1_to_r3_updated,
    'list_T_r3_to_r1_updated': list_T_r3_to_r1_updated,
    'list_T_r2_to_r3_updated': list_T_r2_to_r3_updated,
    'list_T_r3_to_r2_updated': list_T_r3_to_r2_updated
}


df = pd.DataFrame(data)
# print(df.head(10))
sns.set_style("whitegrid")
fig1 = sns.barplot(x = 'list_k', y = 'list_T_r1_to_r2_updated', data = data)
figsave1 = fig1.get_figure()
figsave1.savefig('fig101.png', dpi=400)








