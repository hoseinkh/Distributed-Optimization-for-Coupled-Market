#
#
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
# plt.interactive(False)
plt.interactive(True)
#
# Reading from text file
#
#
df = pd.read_csv('Results.csv')
# print(df.head(10))
# sns.set_style("whitegrid")
# fig1 = sns.barplot(x = 'list_k', y = 'list_T_r1_to_r2_updated', data = df)
# figsave1 = fig1.get_figure()
# figsave1.savefig('fig101.png', dpi=400)


numSimul = len(df['list_k'])
fig1, ax1 = plt.subplots()
x_data = df['list_k']
y1_data = df['list_T_r1_to_r2_updated']
y2_data = df['optimal_Tie_line_r1_r2_centralized']
line1, = ax1.plot(x_data, y1_data, label='$T_{1-2}$')
line2, = ax1.plot(x_data, y2_data,'r--', label='$T_{1-2}^{\ *}$')
ax1.plot(x_data, y1_data, color = 'tab:blue')
ax1.plot(x_data, y2_data,'r--')
ax1.set(xlabel="k", ylabel='Tieline Power Flow   $T_{12}$ (MW)')
ax1.set_ylim([min([min(y1_data) * 0.95, min(y1_data) * 1.05]), max([max(y1_data) * 0.5, max(y1_data) * 100])])
ax1.grid()
fig1.legend(handles=[line1, line2], loc=(0.765,0.6955))
fig1.savefig("fig101.png")
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
fig2, ax2 = plt.subplots()
x_data = df['list_k']
y1_data = df['list_T_r1_to_r3_updated']
y2_data = df['optimal_Tie_line_r1_r3_centralized']
line1, = ax2.plot(x_data, y1_data, label='$T_{1-3}$')
line2, = ax2.plot(x_data, y2_data,'r--', label='$T_{1-3}^{\ *}$')
ax2.plot(x_data, y1_data, color = 'tab:blue')
ax2.plot(x_data, y2_data,'r--')
ax2.set(xlabel="k", ylabel='Tieline Power Flow   $T_{13}$ (MW)')
ax2.set_ylim([min([min(y1_data) * 0.5, min(y1_data) * 1.5]), max([max(y1_data) * 0.96, max(y1_data) * 1.04])])
ax2.grid()
fig2.legend(handles=[line1, line2], loc=(0.765,0.7655))
fig2.savefig("fig102.png")
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
fig3, ax3 = plt.subplots()
x_data = df['list_k']
y1_data = df['list_T_r2_to_r3_updated']
y2_data = df['optimal_Tie_line_r2_r3_centralized']
line1, = ax3.plot(x_data, y1_data, label='$T_{2-3}$')
line2, = ax3.plot(x_data, y2_data,'r--', label='$T_{2-3}^{\ *}$')
ax3.set(xlabel="k", ylabel='Tieline Power Flow   $T_{23}$ (MW)')
ax3.set_ylim([min([min(y1_data) * 0.9, min(y1_data) * 1.5]), max([max(y1_data) * 0.5, max(y1_data) * 1.04])])
ax3.grid()
fig3.legend(handles=[line1, line2], loc=(0.765,0.7655))
fig3.savefig("fig103.png")
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #





print("max of list_T_r1_to_r2_updated = {}".format(max(df['list_T_r1_to_r2_updated'])))

7 + 8
