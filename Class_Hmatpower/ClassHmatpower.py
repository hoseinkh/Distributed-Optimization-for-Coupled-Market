#
#
#
#
#
import numpy as np
import cvxpy as cp
import math
import scipy.stats as dist
#
#
class Hmatpower:
    def __init__(self, case_number, gen_DA_RT_list, epsH, probability_threshold_r1, probability_threshold_r2, probability_threshold_r3, ratio_std_to_mean_r1, ratio_std_to_mean_r2, ratio_std_to_mean_r3):
        #
        self.probability_threshold_r1 = probability_threshold_r1
        self.probability_threshold_r2 = probability_threshold_r2
        self.probability_threshold_r3 = probability_threshold_r3
        #
        self.ratio_std_to_mean_r1 = ratio_std_to_mean_r1
        self.ratio_std_to_mean_r2 = ratio_std_to_mean_r2
        self.ratio_std_to_mean_r3 = ratio_std_to_mean_r3
        #
        new_module = __import__(case_number)
        bus_data, branch_data, gen_data, gencost_data, baseMVA = new_module.return_data()
        #
        self.gen_DA_RT_vector = np.zeros((len(gen_DA_RT_list),))
        for i in range(0, len(gen_DA_RT_list)):
            if gen_DA_RT_list[i].upper() == 'D':
                self.gen_DA_RT_vector[i] = -1
            elif gen_DA_RT_list[i].upper() == 'R':
                self.gen_DA_RT_vector[i] = 0
            else:
                self.gen_DA_RT_vector[i] = None
        #
        self.num_of_nodes = len(bus_data)
        self.num_of_gens = len(gen_data)
        self.num_of_branches = len(branch_data)
        #
        num_columns_in_bus_data = len(bus_data[0])
        num_columns_in_gen_data = len(gen_data[0])
        num_columns_in_gencost_data = len(gencost_data[0])
        if self.num_of_branches == 0:
            num_columns_in_branch_data = 0
        else:
            num_columns_in_branch_data = len(branch_data[0])
        #
        self.bus = np.array(bus_data).reshape(self.num_of_nodes, num_columns_in_bus_data)
        self.branch = np.array(branch_data).reshape(self.num_of_branches, num_columns_in_branch_data)
        self.gen = np.array(gen_data).reshape(self.num_of_gens, num_columns_in_gen_data)
        self.gencost = np.array(gencost_data).reshape(self.num_of_gens, num_columns_in_gencost_data)
        self.baseMVA = np.array(baseMVA)
        #
        #
        self.reference_bus = 0
        #
        self.branches_outgoing_nodes = np.zeros((self.num_of_nodes, self.num_of_branches))
        for i in range(0, self.num_of_branches):
            curr_head = int(self.branch[i, 0])
            curr_tail = int(self.branch[i, 1])
            self.branches_outgoing_nodes[curr_head, i] = 1
            self.branches_outgoing_nodes[curr_tail, i] = -1
        #
        #
        self.gens_located_on_nodes = np.zeros((self.num_of_nodes, self.num_of_gens))
        for i in range(0, self.num_of_gens):
            curr_node = int(self.gen[i, 0])
            self.gens_located_on_nodes[curr_node, i] = 1
        #
        if self.num_of_branches == 0:
            self.line_capacities_modified = []
        else:
            self.line_capacities_modified = self.branch[:, 5]  # to modify 0 capacity lines
            for i in range(0, self.num_of_branches):
                if self.line_capacities_modified[i] == 0:
                    self.line_capacities_modified[i] = math.inf
                else:
                    self.line_capacities_modified[i] = self.line_capacities_modified[i]
        #
        self.epsH = epsH
        #
        #

    #
    #
    def centralized_DCOPF_DA(self):
        P_DA_gen = cp.Variable(self.num_of_gens)
        T_flow_DA = cp.Variable(self.num_of_branches)
        angles_DA = cp.Variable(self.num_of_nodes)
        #
        # DA_gen_matrix___fix = np.diag(self.gen_DA_RT_vector) + np.eye(self.num_of_gens)
        DA_gen_matrix___var = -1 * np.diag(self.gen_DA_RT_vector)
        RT_gen_matrix___fix = np.diag(self.gen_DA_RT_vector) + np.eye(self.num_of_gens)
        #
        obj = cp.Minimize(sum(DA_gen_matrix___var @ self.gencost[:, 6]) + DA_gen_matrix___var @ self.gencost[:,5] @ P_DA_gen
                        + DA_gen_matrix___var @ self.gencost[:,4] @ (P_DA_gen ** 2))
        constraints = [
            P_DA_gen <= self.gen[:, 8],
            -P_DA_gen <= -self.gen[:, 9],
            T_flow_DA <= self.line_capacities_modified,
            -T_flow_DA <= self.line_capacities_modified,
            T_flow_DA == self.baseMVA * (self.branches_outgoing_nodes.T) @ angles_DA / self.branch[:, 3],
            (- self.gens_located_on_nodes @ P_DA_gen + self.branches_outgoing_nodes @ T_flow_DA + self.bus[:,2] == 0),
            angles_DA <= 2 * math.pi,
            -angles_DA <= 2 * math.pi,
            angles_DA[self.reference_bus] == 0,
            RT_gen_matrix___fix @ P_DA_gen == 0  # fix RT generators!
        ]
        #
        prob = cp.Problem(obj, constraints)
        prob.solve(solver=cp.XPRESS, verbose=False)  # superior commercial solver!
        # prob.solve(solver=cp.OSQP, verbose=True)
        print(prob.value)
        print('P_DA_gen = {}'.format(P_DA_gen.value))
        print('T_flow_DA = {}'.format(T_flow_DA.value))
        print('angles_DA in degree = {}'.format(angles_DA.value * 180 / math.pi))
        print('prob.constraints[5].dual_value = {}'.format(prob.constraints[5].dual_value))
        #
        P_DA_gen_centralized = P_DA_gen.value
        angles_DA_centralized = angles_DA.value
        LMPs = prob.constraints[5].dual_value
        T_flow_DA_in_MW = T_flow_DA.value
        Tie_line_r1_r2_centralized = T_flow_DA[9].value
        return P_DA_gen_centralized, T_flow_DA_in_MW, angles_DA_centralized, LMPs

    #
    #
    #
    def centralized_DCOPF_RT(self, DA_Schedule):
        probability_threshold_r1 = self.probability_threshold_r1
        probability_threshold_r2 = self.probability_threshold_r2
        probability_threshold_r3 = self.probability_threshold_r3
        #
        mean_1 = (self.bus[0,2] + self.bus[1,2] + self.bus[2,2] + self.bus[3,2] + self.bus[4,2] + self.bus[6,2] +
                    self.bus[7,2] + self.bus[8,2])
        std_1  = self.ratio_std_to_mean_r1 * mean_1
        #
        mean_2 = (self.bus[5, 2] + self.bus[9, 2] + self.bus[10, 2] + self.bus[11, 2] + self.bus[12, 2] + self.bus[13, 2])
        std_2 = self.ratio_std_to_mean_r2 * mean_2
        #
        mean_3 = self.bus[14, 2]
        std_3 = self.ratio_std_to_mean_r3 * mean_3
        #
        P_RT_gen = cp.Variable(self.num_of_gens)
        T_flow_RT = cp.Variable(self.num_of_branches)
        angles_RT = cp.Variable(self.num_of_nodes)
        #
        #
        DA_gen_matrix___fix = -1 * np.diag(self.gen_DA_RT_vector)
        RT_gen_matrix___var = np.diag(self.gen_DA_RT_vector) + np.eye(self.num_of_gens)
        #
        obj = cp.Minimize(
            RT_gen_matrix___var @ self.gencost[:, 6] + RT_gen_matrix___var @ self.gencost[:,5] @ P_RT_gen
                            + RT_gen_matrix___var @ self.gencost[:,4] @ (P_RT_gen ** 2))
        constraints = [
            P_RT_gen <= self.gen[:, 8],
            -P_RT_gen <= -self.gen[:, 9],
            T_flow_RT - self.line_capacities_modified <= 0,
            - T_flow_RT - self.line_capacities_modified <= 0,
            T_flow_RT == self.baseMVA * (self.branches_outgoing_nodes.T) @ angles_RT / self.branch[:, 3],
            (- self.gens_located_on_nodes @ P_RT_gen + self.branches_outgoing_nodes @ T_flow_RT + self.bus[:, 2] <= 0),
            angles_RT <= 2 * math.pi,
            -angles_RT <= 2 * math.pi,
            angles_RT[self.reference_bus] == 0,
            DA_gen_matrix___fix @ P_RT_gen == DA_gen_matrix___fix @ DA_Schedule,
            np.array([1,1,1,0,0,0,0,0]) @ P_RT_gen + np.array([0,0,0,0,0,0,0,0,0,-1,0,0,0,0,0,0,0,0,-1, 0]) @ T_flow_RT >= dist.norm.ppf(probability_threshold_r1, loc=mean_1, scale=std_1),
            np.array([0,0,0,1,1,1,0,0]) @ P_RT_gen + np.array([0,0,0,0,0,0,0,0,0, 1,0,0,0,0,0,0,0,0, 0,-1]) @ T_flow_RT >= dist.norm.ppf(probability_threshold_r2, loc=mean_2, scale=std_2),
            np.array([0,0,0,0,0,0,1,1]) @ P_RT_gen + np.array([0,0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0,0, 1, 1]) @ T_flow_RT >= dist.norm.ppf(probability_threshold_r3, loc=mean_3, scale=std_3),
            #
        ]
        #
        prob = cp.Problem(obj, constraints)
        prob.solve(solver=cp.XPRESS, verbose=False)  # superior commercial solver!
        # prob.solve(solver=cp.OSQP, verbose=True)
        print(prob.value)
        print('P_RT_gen = {}'.format(P_RT_gen.value))
        print('T_flow_RT = {}'.format(T_flow_RT.value))
        print([min([max(prob.constraints[2].dual_value), max(prob.constraints[3].dual_value)]), max([max(prob.constraints[2].dual_value), max(prob.constraints[3].dual_value)])])
        print('angles_RT in degree = {}'.format(angles_RT.value * 180 / math.pi))
        print('prob.constraints[5].dual_value = {}'.format(prob.constraints[5].dual_value))
        #
        kappa_tieLine_1_2 = (prob.constraints[3].dual_value)[9]
        etta_tieLine_1_2  = (prob.constraints[2].dual_value)[9]
        kappa_tieLine_1_3 = (prob.constraints[3].dual_value)[18]
        etta_tieLine_1_3  = (prob.constraints[2].dual_value)[18]
        kappa_tieLine_2_3 = (prob.constraints[3].dual_value)[19]
        etta_tieLine_2_3 = (prob.constraints[2].dual_value)[19]
        # main
        rate_absol_at_optimum_tieLine_1_2 = -kappa_tieLine_1_2 + etta_tieLine_1_2
        rate_absol_at_optimum_tieLine_1_2 = abs(rate_absol_at_optimum_tieLine_1_2)
        rate_absol_at_optimum_tieLine_2_1 = rate_absol_at_optimum_tieLine_1_2
        rate_absol_at_optimum_tieLine_1_3 = -kappa_tieLine_1_3 + etta_tieLine_1_3
        rate_absol_at_optimum_tieLine_1_3 = abs(rate_absol_at_optimum_tieLine_1_3)
        rate_absol_at_optimum_tieLine_3_1 = rate_absol_at_optimum_tieLine_1_3
        rate_absol_at_optimum_tieLine_2_3 = -kappa_tieLine_2_3 + etta_tieLine_2_3
        rate_absol_at_optimum_tieLine_2_3 = abs(rate_absol_at_optimum_tieLine_2_3)
        rate_absol_at_optimum_tieLine_3_2 = rate_absol_at_optimum_tieLine_2_3
        #
        #
        alpha_1_to_2 = (prob.constraints[5].dual_value)[4]
        alpha_1_to_3 = (prob.constraints[5].dual_value)[8]
        alpha_2_to_1 = (prob.constraints[5].dual_value)[5]
        alpha_2_to_3 = (prob.constraints[5].dual_value)[9]
        alpha_3_to_1 = (prob.constraints[5].dual_value)[14]
        alpha_3_to_2 = (prob.constraints[5].dual_value)[14]
        beta_1 = prob.constraints[10].dual_value + 0
        beta_2 = prob.constraints[11].dual_value + 0
        beta_3 = prob.constraints[12].dual_value + 0
        #
        rate_signed_r1_sell_to_r2 = alpha_1_to_2 + beta_1
        rate_signed_r1_sell_to_r3 = alpha_1_to_3 + beta_1
        rate_signed_r2_sell_to_r1 = alpha_2_to_1 + beta_2
        rate_signed_r2_sell_to_r3 = alpha_2_to_3 + beta_2
        rate_signed_r3_sell_to_r1 = alpha_3_to_1 + beta_3
        rate_signed_r3_sell_to_r2 = alpha_3_to_2 + beta_3
        dic_rate_absol_at_optimum = dict()
        dic_rate_absol_at_optimum['rate_absol_r1_sell_to_r2'] = rate_absol_at_optimum_tieLine_1_2
        dic_rate_absol_at_optimum['rate_absol_r1_sell_to_r3'] = rate_absol_at_optimum_tieLine_1_3
        dic_rate_absol_at_optimum['rate_absol_r2_sell_to_r1'] = rate_absol_at_optimum_tieLine_2_1
        dic_rate_absol_at_optimum['rate_absol_r2_sell_to_r3'] = rate_absol_at_optimum_tieLine_2_3
        dic_rate_absol_at_optimum['rate_absol_r3_sell_to_r1'] = rate_absol_at_optimum_tieLine_3_1
        dic_rate_absol_at_optimum['rate_absol_r3_sell_to_r2'] = rate_absol_at_optimum_tieLine_3_2
        # rate_signed_r2_sell_to_r1 = -1*rate_signed_r2_sell_to_r1
        #
        dic_of_signed_rates = dict()
        dic_of_signed_rates['rate_signed_r1_sell_to_r2'] = rate_signed_r1_sell_to_r2
        dic_of_signed_rates['rate_signed_r1_sell_to_r3'] = rate_signed_r1_sell_to_r3
        dic_of_signed_rates['rate_signed_r2_sell_to_r1'] = rate_signed_r2_sell_to_r1
        dic_of_signed_rates['rate_signed_r2_sell_to_r3'] = rate_signed_r2_sell_to_r3
        dic_of_signed_rates['rate_signed_r3_sell_to_r1'] = rate_signed_r3_sell_to_r1
        dic_of_signed_rates['rate_signed_r3_sell_to_r2'] = rate_signed_r3_sell_to_r2
        #
        Tie_line_r1_r2_centralized = T_flow_RT[9].value
        Tie_line_r1_r3_centralized = T_flow_RT[18].value
        Tie_line_r2_r3_centralized = T_flow_RT[19].value
        #
        dic_Tie_line_centralized = dict()
        dic_Tie_line_centralized['Tie_line_r1_r2_centralized'] = Tie_line_r1_r2_centralized
        dic_Tie_line_centralized['Tie_line_r1_r3_centralized'] = Tie_line_r1_r3_centralized
        dic_Tie_line_centralized['Tie_line_r2_r3_centralized'] = Tie_line_r2_r3_centralized
        #
        angles_RT_values = angles_RT.value
        dict_theta_tielines = dict()
        dict_theta_tielines['theta_1_tieline_r1_to_r2'] = angles_RT_values[4]
        dict_theta_tielines['theta_1_tieline_r1_to_r3'] = angles_RT_values[8]
        dict_theta_tielines['theta_2_tieline_r2_to_r1'] = angles_RT_values[5]
        dict_theta_tielines['theta_2_tieline_r2_to_r3'] = angles_RT_values[9]
        dict_theta_tielines['theta_3_tieline_r3_to_r1'] = angles_RT_values[14]
        dict_theta_tielines['theta_3_tieline_r3_to_r2'] = angles_RT_values[14]
        #
        angles_RT_values_cent = angles_RT.value
        P_RT_gen_values = P_RT_gen.value
        LMPs = prob.constraints[5].dual_value
        T_flow_RT_in_MW = T_flow_RT.value
        RT_and_DA_system_cost_in_RT_dispatch = prob.value
        return P_RT_gen_values, T_flow_RT_in_MW, angles_RT_values_cent, dict_theta_tielines, LMPs, dic_rate_absol_at_optimum, dic_of_signed_rates, dic_Tie_line_centralized, RT_and_DA_system_cost_in_RT_dispatch
    #
    #
    #
    def decentralized_DCOPF_RT_r1(self, DA_Schedule_r1, dict_theta_tielines, dic_rate_absol_at_optimum, dic_of_signed_rates):
        #
        #
        #
        mean_1 = sum(self.bus[:, 2])
        std_1 = self.ratio_std_to_mean_r1 * mean_1
        #
        theta_2_tieline_r1_to_r2 = dict_theta_tielines['theta_2_tieline_r2_to_r1']
        theta_3_tieline_r1_to_r3 = dict_theta_tielines['theta_3_tieline_r3_to_r1']
        #
        P_RT_gen = cp.Variable(self.num_of_gens)
        T_flow_RT = cp.Variable(self.num_of_branches)
        angles_RT = cp.Variable(self.num_of_nodes)
        Tie_line_r1_r2 = cp.Variable(1)
        Tie_line_r1_r3 = cp.Variable(1)
        U_tie_line_flow_r1_to_r2 = cp.Variable(1)
        U_tie_line_flow_r1_to_r3 = cp.Variable(1)
        #
        rate_absol_at_optimum_r2_to_r1 = dic_rate_absol_at_optimum['mu_rate_absol_r2_sell_to_r1']
        rate_absol_at_optimum_r3_to_r1 = dic_rate_absol_at_optimum['mu_rate_absol_r3_sell_to_r1']
        #
        rate_signed_r2_sell_to_r1 = dic_of_signed_rates['delta_rate_signed_r2_sell_to_r1']
        rate_signed_r3_sell_to_r1 = dic_of_signed_rates['delta_rate_signed_r3_sell_to_r1']
        # 35.260219201923725
        # rate_signed_r2_sell_to_r1 = 36.1
        # rate_signed_r3_sell_to_r1 = rate_signed_r2_sell_to_r1
        #
        DA_gen_matrix___fix = -1 * np.diag(self.gen_DA_RT_vector)
        RT_gen_matrix___var = np.diag(self.gen_DA_RT_vector) + np.eye(self.num_of_gens)
        #
        obj = cp.Minimize(
            RT_gen_matrix___var @ self.gencost[:, 6] + RT_gen_matrix___var @ self.gencost[:,5] @ P_RT_gen
                            + RT_gen_matrix___var @ self.gencost[:,4] @ (P_RT_gen ** 2)
            + rate_absol_at_optimum_r2_to_r1 * U_tie_line_flow_r1_to_r2 + rate_absol_at_optimum_r3_to_r1 * U_tie_line_flow_r1_to_r3
            - rate_signed_r2_sell_to_r1 * Tie_line_r1_r2 - rate_signed_r3_sell_to_r1 * Tie_line_r1_r3)
        constraints = [
            P_RT_gen <= self.gen[:, 8],
            -P_RT_gen <= - self.gen[:, 9],
            T_flow_RT - self.line_capacities_modified <= 0,
            -T_flow_RT - self.line_capacities_modified <= 0,
            # T_flow_RT == self.baseMVA * (self.branches_outgoing_nodes.T) @ angles_RT / self.branch[:, 3],
            T_flow_RT <= self.baseMVA * (self.branches_outgoing_nodes.T) @ angles_RT / self.branch[:, 3] + self.epsH,
            -T_flow_RT <= -self.baseMVA * (self.branches_outgoing_nodes.T) @ angles_RT / self.branch[:, 3] + self.epsH,
            (- self.gens_located_on_nodes @ P_RT_gen + self.branches_outgoing_nodes @ T_flow_RT + self.bus[:, 2] + np.array([[0],[0],[0],[0],[1],[0],[0],[0]])*Tie_line_r1_r2
             + np.array([[0], [0], [0], [0], [0], [0], [0], [1]]) * Tie_line_r1_r3  <= self.epsH),
            angles_RT <= 2 * math.pi,
            -angles_RT <= 2 * math.pi,
            angles_RT[self.reference_bus] == 0,
            DA_gen_matrix___fix @ P_RT_gen == DA_gen_matrix___fix @ DA_Schedule_r1,
            np.ones((1,self.num_of_gens)) @ P_RT_gen - Tie_line_r1_r2 - Tie_line_r1_r3 >= dist.norm.ppf(self.probability_threshold_r1, loc=mean_1, scale=std_1),
            # Tie_line_r1_r2 == self.baseMVA * (angles_RT[4] - theta_2_tieline_r1_to_r2) / 0.05202,
            Tie_line_r1_r2 <= (self.baseMVA * (angles_RT[4] - theta_2_tieline_r1_to_r2) / 0.05202) + self.epsH,
            -Tie_line_r1_r2 <= -(self.baseMVA * (angles_RT[4] - theta_2_tieline_r1_to_r2) / 0.05202) + self.epsH,
            # Tie_line_r1_r3 == self.baseMVA * (angles_RT[7] - theta_3_tieline_r1_to_r3) / 0.04802,
            Tie_line_r1_r3 <= (self.baseMVA * (angles_RT[7] - theta_3_tieline_r1_to_r3) / 0.04802) + self.epsH,
            -Tie_line_r1_r3 <= -(self.baseMVA * (angles_RT[7] - theta_3_tieline_r1_to_r3) / 0.04802) + self.epsH,
            U_tie_line_flow_r1_to_r2 >=  Tie_line_r1_r2,
            U_tie_line_flow_r1_to_r2 >= -Tie_line_r1_r2,
            U_tie_line_flow_r1_to_r3 >=  Tie_line_r1_r3,
            U_tie_line_flow_r1_to_r3 >= -Tie_line_r1_r3,
            #
            #
            #
            # U_tie_line_flow_r1_to_r2 == 0,
            # U_tie_line_flow_r1_to_r3 == 0,
        ]
        #
        prob = cp.Problem(obj, constraints)
        prob.solve(solver=cp.XPRESS, verbose=False)  # superior commercial solver!
        # prob.solve(solver=cp.OSQP, verbose=True)
        #
        alpha_1_to_2_decent_r1 = (prob.constraints[6].dual_value)[0]
        alpha_1_to_3_decent_r1 = (prob.constraints[6].dual_value)[7]
        gamma_1_decent_r1 = (prob.constraints[11].dual_value)[0] + 0
        #
        delta_rate_signed_r1_sell_to_r2_decent_r1 = alpha_1_to_2_decent_r1 + gamma_1_decent_r1
        delta_rate_signed_r1_sell_to_r3_decent_r1 = alpha_1_to_3_decent_r1 + gamma_1_decent_r1
        #
        print(prob.value)
        print(prob.status)
        print('P_RT_gen = {}'.format(P_RT_gen.value))
        print('T_flow_RT = {}'.format(T_flow_RT.value))
        print([min([max(prob.constraints[2].dual_value),max(prob.constraints[3].dual_value)]) , max([max(prob.constraints[2].dual_value),max(prob.constraints[3].dual_value)])])
        print('angles_RT in degree = {}'.format(angles_RT.value * 180 / math.pi))
        print('prob.constraints[5].dual_value = {}'.format(prob.constraints[5].dual_value))
        #
        #
        dic_Tie_line_decen_r1 = dict()
        dic_Tie_line_decen_r1['Tie_line_r1_r2_decent_r1'] = Tie_line_r1_r2.value[0]
        dic_Tie_line_decen_r1['Tie_line_r1_r3_decent_r1'] = Tie_line_r1_r3.value[0]
        #
        P_RT_gen_values = P_RT_gen.value
        angles_RT_values_dec_r1 = angles_RT.value
        LMPs = prob.constraints[5].dual_value
        T_flow_RT_in_MW = T_flow_RT.value
        return P_RT_gen_values, T_flow_RT_in_MW, angles_RT_values_dec_r1, LMPs, dic_Tie_line_decen_r1, delta_rate_signed_r1_sell_to_r2_decent_r1, delta_rate_signed_r1_sell_to_r3_decent_r1
    #
    #
    #
    #
    def decentralized_DCOPF_RT_r2(self, DA_Schedule_r2, dict_theta_tielines, dic_rate_absol_at_optimum, dic_of_signed_rates):
        mean_2 = sum(self.bus[:, 2])
        std_2 = self.ratio_std_to_mean_r2 * mean_2
        #
        theta_1_tieline_r1_to_r2 = dict_theta_tielines['theta_1_tieline_r1_to_r2']
        theta_3_tieline_r3_to_r2 = dict_theta_tielines['theta_3_tieline_r3_to_r2']
        #
        P_RT_gen = cp.Variable(self.num_of_gens)
        T_flow_RT = cp.Variable(self.num_of_branches)
        angles_RT = cp.Variable(self.num_of_nodes)
        Tie_line_r2_r1 = cp.Variable(1)
        Tie_line_r2_r3 = cp.Variable(1)
        U_tie_line_flow_r2_to_r1 = cp.Variable(1)
        U_tie_line_flow_r2_to_r3 = cp.Variable(1)
        #
        rate_absol_at_optimum_r1_to_r2 = dic_rate_absol_at_optimum['mu_rate_absol_r1_sell_to_r2']
        rate_absol_at_optimum_r3_to_r2 = dic_rate_absol_at_optimum['mu_rate_absol_r3_sell_to_r2']
        #
        rate_signed_r1_sell_to_r2 = dic_of_signed_rates['delta_rate_signed_r1_sell_to_r2']
        rate_signed_r3_sell_to_r2 = dic_of_signed_rates['delta_rate_signed_r3_sell_to_r2']
        #
        #
        #
        DA_gen_matrix___fix = -1 * np.diag(self.gen_DA_RT_vector)
        RT_gen_matrix___var = np.diag(self.gen_DA_RT_vector) + np.eye(self.num_of_gens)
        #
        obj = cp.Minimize(
            RT_gen_matrix___var @ self.gencost[:, 6] + RT_gen_matrix___var @ self.gencost[:, 5] @ P_RT_gen
            + RT_gen_matrix___var @ self.gencost[:, 4] @ (P_RT_gen ** 2)
            + rate_absol_at_optimum_r1_to_r2 * U_tie_line_flow_r2_to_r1 + rate_absol_at_optimum_r3_to_r2 * U_tie_line_flow_r2_to_r3
            - rate_signed_r1_sell_to_r2 * Tie_line_r2_r1 - rate_signed_r3_sell_to_r2 * Tie_line_r2_r3 )
        constraints = [
            P_RT_gen <= self.gen[:, 8],
            -P_RT_gen <= - self.gen[:, 9],
            T_flow_RT - self.line_capacities_modified <= 0,
            -T_flow_RT - self.line_capacities_modified <= 0,
            ## ## T_flow_RT == self.baseMVA * (self.branches_outgoing_nodes.T) @ angles_RT / self.branch[:, 3],
            # T_flow_RT == self.baseMVA * (self.branches_outgoing_nodes.T) @ angles_RT / self.branch[:, 3],
            T_flow_RT <= (self.baseMVA * (self.branches_outgoing_nodes.T) @ angles_RT / self.branch[:, 3]) + self.epsH,
            -T_flow_RT <= (-self.baseMVA * (self.branches_outgoing_nodes.T) @ angles_RT / self.branch[:, 3]) + self.epsH,
            (- self.gens_located_on_nodes @ P_RT_gen + self.branches_outgoing_nodes @ T_flow_RT + self.bus[:,2]
             + np.array([[1], [0], [0], [0], [0], [0]]).reshape(6,) * Tie_line_r2_r1
             + np.array([[0], [1], [0], [0], [0], [0]]).reshape(6,) * Tie_line_r2_r3 <= self.epsH),
            angles_RT <= 2 * math.pi,
            -angles_RT <= 2 * math.pi,
            ## ## angles_RT[self.reference_bus] == 0,
            DA_gen_matrix___fix @ P_RT_gen == DA_gen_matrix___fix @ DA_Schedule_r2,
            np.ones((1, self.num_of_gens)) @ P_RT_gen - Tie_line_r2_r1 - Tie_line_r2_r3 >= dist.norm.ppf(
                self.probability_threshold_r2, loc=mean_2, scale=std_2),
            ## ## Tie_line_r2_r3 == self.baseMVA * (angles_RT[0] - theta_1_tieline_r1_to_r2) / 0.05202,
            Tie_line_r2_r1 <= (self.baseMVA * (angles_RT[0] - theta_1_tieline_r1_to_r2) / 0.05202) + self.epsH,
            -Tie_line_r2_r1 <= -(self.baseMVA * (angles_RT[0] - theta_1_tieline_r1_to_r2) / 0.05202) + self.epsH,
            ## ## Tie_line_r2_r3 == self.baseMVA * (angles_RT[1] - theta_3_tieline_r3_to_r2) / 0.04802,
            Tie_line_r2_r3 <= (self.baseMVA * (angles_RT[1] - theta_3_tieline_r3_to_r2) / 0.04802) + self.epsH,
            -Tie_line_r2_r3 <= -(self.baseMVA * (angles_RT[1] - theta_3_tieline_r3_to_r2) / 0.04802) + self.epsH,
            U_tie_line_flow_r2_to_r1 >= Tie_line_r2_r1,
            U_tie_line_flow_r2_to_r1 >= -Tie_line_r2_r1,
            U_tie_line_flow_r2_to_r3 >= Tie_line_r2_r3,
            U_tie_line_flow_r2_to_r3 >= -Tie_line_r2_r3,
            #
            # Tie_line_r2_r1 == 19.999999999999993,
            # Tie_line_r2_r3 == 17.50963495339082,
            #
        ]
        #
        prob = cp.Problem(obj, constraints)
        prob.solve(solver=cp.XPRESS, verbose=False)  # superior commercial solver!
        # prob.solve(solver=cp.OSQP, verbose=True)
        #
        alpha_2_to_1_decent_r2 = (prob.constraints[6].dual_value)[0]
        alpha_2_to_3_decent_r2 = (prob.constraints[6].dual_value)[1]
        gamma_2_decent_r2 = (prob.constraints[10].dual_value)[0] + 0
        #
        delta_rate_signed_r2_sell_to_r1_decent_r2 = alpha_2_to_1_decent_r2 + gamma_2_decent_r2
        delta_rate_signed_r2_sell_to_r3_decent_r2 = alpha_2_to_3_decent_r2 + gamma_2_decent_r2
        #
        print(prob.value)
        print(prob.status)
        print('P_RT_gen = {}'.format(P_RT_gen.value))
        print('T_flow_RT = {}'.format(T_flow_RT.value))
        print('angles_RT in degree = {}'.format(angles_RT.value * 180 / math.pi))
        print('prob.constraints[5].dual_value = {}'.format(prob.constraints[5].dual_value))
        #
        #
        dic_Tie_line_decen_r2 = dict()
        dic_Tie_line_decen_r2['Tie_line_r2_r1_decent_r2'] = Tie_line_r2_r1.value[0]
        dic_Tie_line_decen_r2['Tie_line_r2_r3_decent_r2'] = Tie_line_r2_r3.value[0]
        #
        P_RT_gen_values = P_RT_gen.value
        angles_RT_values_dec_r2 = angles_RT.value
        LMPs = prob.constraints[5].dual_value
        T_flow_RT_in_MW = T_flow_RT.value
        if self.probability_threshold_r2 >= 0.91:
            zzz = 4
        else:
            pass
        return P_RT_gen_values, T_flow_RT_in_MW, angles_RT_values_dec_r2, LMPs, dic_Tie_line_decen_r2, delta_rate_signed_r2_sell_to_r1_decent_r2, delta_rate_signed_r2_sell_to_r3_decent_r2
    #
    #
    #
    def decentralized_DCOPF_RT_r3(self, DA_Schedule_r3, dict_theta_tielines, dic_rate_absol_at_optimum, dic_of_signed_rates):
        #
        #
        #
        mean_3 = sum(self.bus[:, 2])
        std_3 = self.ratio_std_to_mean_r3 * mean_3
        #
        theta_1_tieline_r3_to_r1 = dict_theta_tielines['theta_1_tieline_r1_to_r3']
        theta_2_tieline_r3_to_r2 = dict_theta_tielines['theta_2_tieline_r2_to_r3']
        #
        P_RT_gen = cp.Variable(self.num_of_gens)
        angles_RT = cp.Variable(self.num_of_nodes)
        Tie_line_r3_r1 = cp.Variable(1)
        Tie_line_r3_r2 = cp.Variable(1)
        U_tie_line_flow_r3_to_r1 = cp.Variable(1)
        U_tie_line_flow_r3_to_r2 = cp.Variable(1)
        #
        rate_absol_at_optimum_r1_to_r3 = dic_rate_absol_at_optimum['mu_rate_absol_r1_sell_to_r3']
        rate_absol_at_optimum_r2_to_r3 = dic_rate_absol_at_optimum['mu_rate_absol_r2_sell_to_r3']
        #
        rate_signed_r1_sell_to_r3 = dic_of_signed_rates['delta_rate_signed_r1_sell_to_r3']
        rate_signed_r2_sell_to_r3 = dic_of_signed_rates['delta_rate_signed_r2_sell_to_r3']
        #
        #
        DA_gen_matrix___fix = -1 * np.diag(self.gen_DA_RT_vector)
        RT_gen_matrix___var = np.diag(self.gen_DA_RT_vector) + np.eye(self.num_of_gens)
        #
        obj = cp.Minimize(
            RT_gen_matrix___var @ self.gencost[:, 6] + RT_gen_matrix___var @ self.gencost[:,5] @ P_RT_gen
                            + RT_gen_matrix___var @ self.gencost[:,4] @ (P_RT_gen ** 2)
            + rate_absol_at_optimum_r1_to_r3 * U_tie_line_flow_r3_to_r1 + rate_absol_at_optimum_r2_to_r3 * U_tie_line_flow_r3_to_r2
            - rate_signed_r1_sell_to_r3 * Tie_line_r3_r1 - rate_signed_r2_sell_to_r3 * Tie_line_r3_r2)
        # there is no transmission line within the R3, hence there is no variable for T_flow_RT.
        constraints = [
            P_RT_gen <= self.gen[:, 8],
            -P_RT_gen <= - self.gen[:, 9],
            angles_RT <= 2 * math.pi,
            -angles_RT <= 2 * math.pi,
            DA_gen_matrix___fix @ P_RT_gen == DA_gen_matrix___fix @ DA_Schedule_r3,
            (- self.gens_located_on_nodes @ P_RT_gen + self.bus[:, 2] + Tie_line_r3_r1 + Tie_line_r3_r2 <= self.epsH),
            np.ones((1,self.num_of_gens)) @ P_RT_gen - Tie_line_r3_r1 - Tie_line_r3_r2 >= dist.norm.ppf(self.probability_threshold_r3, loc=mean_3, scale=std_3),
            Tie_line_r3_r1 == self.baseMVA * (angles_RT[0] - theta_1_tieline_r3_to_r1) / 0.04802,
            Tie_line_r3_r2 == self.baseMVA * (angles_RT[0] - theta_2_tieline_r3_to_r2) / 0.04802,
            U_tie_line_flow_r3_to_r1 >= Tie_line_r3_r1,
            U_tie_line_flow_r3_to_r1 >= -Tie_line_r3_r1,
            U_tie_line_flow_r3_to_r2 >= Tie_line_r3_r2,
            U_tie_line_flow_r3_to_r2 >= -Tie_line_r3_r2,
            #
            #
        ]
        #
        prob = cp.Problem(obj, constraints)
        prob.solve(solver=cp.XPRESS, verbose=False)  # superior commercial solver!
        # prob.solve(solver=cp.OSQP, verbose=True)
        #
        alpha_3_to_1_decent_r3 = (prob.constraints[5].dual_value)[0]
        alpha_3_to_2_decent_r3 = (prob.constraints[5].dual_value)[0]
        gamma_3_decent_r3 = (prob.constraints[6].dual_value)[0] + 0
        #
        delta_rate_signed_r3_sell_to_r1_decent_r3 = alpha_3_to_1_decent_r3 + gamma_3_decent_r3
        delta_rate_signed_r3_sell_to_r2_decent_r3 = alpha_3_to_2_decent_r3 + gamma_3_decent_r3
        #
        print(prob.value)
        print(prob.status)
        print('P_RT_gen = {}'.format(P_RT_gen.value))
        print('angles_RT in degree = {}'.format(angles_RT.value * 180 / math.pi))
        #
        #
        dic_Tie_line_decen_r3 = dict()
        dic_Tie_line_decen_r3['Tie_line_r3_r1_decent_r3'] = Tie_line_r3_r1.value[0]
        dic_Tie_line_decen_r3['Tie_line_r3_r2_decent_r3'] = Tie_line_r3_r2.value[0]
        #
        P_RT_gen_values = P_RT_gen.value
        angles_RT_values_dec_r3 = angles_RT.value
        LMPs = prob.constraints[5].dual_value
        T_flow_RT_in_MW = None
        return P_RT_gen_values, T_flow_RT_in_MW, angles_RT_values_dec_r3, LMPs, dic_Tie_line_decen_r3, delta_rate_signed_r3_sell_to_r1_decent_r3, delta_rate_signed_r3_sell_to_r2_decent_r3
    #
    #
    #
    #
    #
