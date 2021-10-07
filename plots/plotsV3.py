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
ax1.set_ylim([min([min(min(y1_data),min(y2_data)) * 0.95, min(min(y1_data),min(y2_data)) * 1.05]), max([max(max(y1_data),max(y2_data)) * 0.5, max(max(y1_data),max(y2_data)) * 100])])
ax1.grid()
fig1.legend(handles=[line1, line2], loc=(0.765,0.6955))
fig1.savefig("fig101.png")
fig1.savefig("fig101.pdf")
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
ax2.set_ylim([min([min(min(y1_data),min(y2_data)) * 0.95, min(min(y1_data),min(y2_data)) * 1.05]), max([max(max(y1_data),max(y2_data)) * 0.5, max(max(y1_data),max(y2_data)) * 1.1])])
ax2.grid()
fig2.legend(handles=[line1, line2], loc=(0.765,0.7655))
fig2.savefig("fig102.png")
fig2.savefig("fig102.pdf")
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
fig3, ax3 = plt.subplots()
x_data = df['list_k']
y1_data = df['list_T_r2_to_r3_updated']
y2_data = df['optimal_Tie_line_r2_r3_centralized']
line1, = ax3.plot(x_data, y1_data, label='$T_{2-3}$')
line2, = ax3.plot(x_data, y2_data,'r--', label='$T_{2-3}^{\ *}$')
ax3.set(xlabel="k", ylabel='Tieline Power Flow   $T_{23}$ (MW)')
ax3.set_ylim([min([min(min(y1_data),min(y2_data)) * 0.95, min(min(y1_data),min(y2_data)) * 1.05]), max([max(max(y1_data),max(y2_data)) * 0.5, max(max(y1_data),max(y2_data)) * 1.1])])
ax3.grid()
fig3.legend(handles=[line1, line2], loc=(0.765,0.7655))
fig3.savefig("fig103.png")
fig3.savefig("fig103.pdf")
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
fig4, ax4 = plt.subplots()
x_data = df['list_k']
y1_data = df['list_delta_rate_signed_r1_sell_to_r2_decent_r1']
y2_data = df['list_delta_rate_signed_r2_sell_to_r1_decent_r2']
y3_data = df['list_mu_rate_absol_r1_sell_to_r2_decent_r1']
line1, = ax4.plot(x_data, y1_data, label='$\delta_{12}$')
line2, = ax4.plot(x_data, y2_data,'b:', label='$\delta_{21}$')
line3, = ax4.plot(x_data, y3_data,'r--', label='$\mu_{12}=\mu_{21}$')
ax4.set(xlabel="k", ylabel='Rates on tie-line between R1 and R2 ($/MWh)')
ax4.set_ylim([min([min([min(y1_data),min(y2_data),min(y3_data)]) * 0.95 - 20, min([min(y1_data),min(y2_data),min(y3_data)]) * 1.05]), max([max([max(y1_data),max(y2_data),max(y3_data)]) * 0.5, max([max(y1_data),max(y2_data),max(y3_data)]) * 1.1])])
ax4.grid()
fig4.legend(handles=[line1, line2, line3], loc=(0.721,0.7275))
fig4.savefig("fig104.png")
fig4.savefig("fig104.pdf")
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
fig5, ax5 = plt.subplots()
x_data = df['list_k']
y1_data = df['list_delta_rate_signed_r1_sell_to_r3_decent_r1']
y2_data = df['list_delta_rate_signed_r3_sell_to_r1_decent_r3']
y3_data = df['list_mu_rate_absol_r1_sell_to_r3_decent_r1']
line1, = ax5.plot(x_data, y1_data, label='$\delta_{13}$')
line2, = ax5.plot(x_data, y2_data,'b:', label='$\delta_{31}$')
line3, = ax5.plot(x_data, y3_data,'r--', label='$\mu_{13}=\mu_{31}$')
ax5.set(xlabel="k", ylabel='Rates on tie-line between R1 and R3 ($/MWh)')
ax5.set_ylim([min([min([min(y1_data),min(y2_data),min(y3_data)]) * 0.95 - 10, min([min(y1_data),min(y2_data),min(y3_data)]) * 1.05]), max([max([max(y1_data),max(y2_data),max(y3_data)]) * 0.5, max([max(y1_data),max(y2_data),max(y3_data)]) * 1.1])])
ax5.grid()
fig5.legend(handles=[line1, line2, line3], loc=(0.721,0.7275))
fig5.savefig("fig105.png")
fig5.savefig("fig105.pdf")
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
fig6, ax6 = plt.subplots()
x_data = df['list_k']
y1_data = df['list_delta_rate_signed_r2_sell_to_r3_decent_r2']
y2_data = df['list_delta_rate_signed_r3_sell_to_r2_decent_r3']
y3_data = df['list_mu_rate_absol_r2_sell_to_r3_decent_r2']
line1, = ax6.plot(x_data, y1_data, label='$\delta_{23}$')
line2, = ax6.plot(x_data, y2_data,'b:', label='$\delta_{32}$')
line3, = ax6.plot(x_data, y3_data,'r--', label='$\mu_{23}=\mu_{32}$')
ax6.set(xlabel="k", ylabel='Rates on tie-line between R2 and R3 ($/MWh)')
ax6.set_ylim([min([min([min(y1_data),min(y2_data),min(y3_data)]) * 0.95 - 10, min([min(y1_data),min(y2_data),min(y3_data)]) * 1.05]), max([max([max(y1_data),max(y2_data),max(y3_data)]) * 0.5, max([max(y1_data),max(y2_data),max(y3_data)]) * 1.1])])
ax6.grid()
fig6.legend(handles=[line1, line2, line3], loc=(0.721,0.7275))
fig6.savefig("fig106.png")
fig6.savefig("fig106.pdf")
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

print("max of list_T_r1_to_r2_updated = {}".format(max(df['list_T_r1_to_r2_updated'])))

7 + 8
