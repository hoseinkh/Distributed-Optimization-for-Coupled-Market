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
class Hmatpower_generate_scenarios_R1:
    def __init__(self, case_number, gen_DA_RT_list, epsH, probability_threshold, ratio_std_to_mean):
        #
        self.probability_threshold = probability_threshold
        #
        self.ratio_std_to_mean = ratio_std_to_mean
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
    def centralized_DCOPF_RT(self, DA_Schedule, sample, node1, node2):
        #
        #
        #
        probability_threshold = self.probability_threshold
        #
        mean = np.sum(self.bus[:,2])
        std  = self.ratio_std_to_mean * mean
        #
        original_load = self.bus[:,2]
        self.bus[node1, 2] = sample[0]
        self.bus[node2, 2] = sample[1]
        #
        P_RT_gen  = cp.Variable(self.num_of_gens)
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
            np.ones((1,self.num_of_gens)) @ P_RT_gen >= dist.norm.ppf(probability_threshold, loc=mean, scale=std),
            #
        ]
        #
        prob = cp.Problem(obj, constraints)
        prob.solve(solver=cp.XPRESS, verbose=False)  # superior commercial solver!
        # prob.solve(solver=cp.OSQP, verbose=True)
        print(prob.value)
        # print('P_RT_gen = {}'.format(P_RT_gen.value))
        # print('T_flow_RT = {}'.format(T_flow_RT.value))
        # print([min([max(prob.constraints[2].dual_value), max(prob.constraints[3].dual_value)]), max([max(prob.constraints[2].dual_value), max(prob.constraints[3].dual_value)])])
        # print('angles_RT in degree = {}'.format(angles_RT.value * 180 / math.pi))
        # print('prob.constraints[5].dual_value = {}'.format(prob.constraints[5].dual_value))
        #
        #
        kappa_tieLine_1_2 = (prob.constraints[3].dual_value)[node1]
        etta_tieLine_1_2  = (prob.constraints[2].dual_value)[node1]
        kappa_tieLine_1_3 = (prob.constraints[3].dual_value)[node2]
        etta_tieLine_1_3  = (prob.constraints[2].dual_value)[node2]
        # main
        rate_absol_at_optimum_tieLine_1_2 = -kappa_tieLine_1_2 + etta_tieLine_1_2
        rate_absol_at_optimum_tieLine_1_2 = abs(rate_absol_at_optimum_tieLine_1_2)
        rate_absol_at_optimum_tieLine_1_3 = -kappa_tieLine_1_3 + etta_tieLine_1_3
        rate_absol_at_optimum_tieLine_1_3 = abs(rate_absol_at_optimum_tieLine_1_3)
        #
        #
        alpha_1_to_2 = (prob.constraints[5].dual_value)[node1]
        alpha_1_to_3 = (prob.constraints[5].dual_value)[node2]
        beta_1 = prob.constraints[10].dual_value + 0
        #
        rate_signed_r1_sell_to_r2 = alpha_1_to_2 + beta_1
        rate_signed_r1_sell_to_r3 = alpha_1_to_3 + beta_1
        dic_rate_absol_at_optimum = dict()
        dic_rate_absol_at_optimum['rate_absol_r1_sell_to_r2'] = rate_absol_at_optimum_tieLine_1_2
        dic_rate_absol_at_optimum['rate_absol_r1_sell_to_r3'] = rate_absol_at_optimum_tieLine_1_3
        #
        #
        dic_of_signed_rates = dict()
        dic_of_signed_rates['rate_signed_r1_sell_to_r2'] = rate_signed_r1_sell_to_r2
        dic_of_signed_rates['rate_signed_r1_sell_to_r3'] = rate_signed_r1_sell_to_r3
        #
        #
        #
        angles_RT_values = angles_RT.value
        dict_theta_tielines = dict()
        dict_theta_tielines['theta_1_tieline_r1_to_r2'] = angles_RT_values[node1]
        dict_theta_tielines['theta_1_tieline_r1_to_r3'] = angles_RT_values[node2]
        #
        angles_RT_values_cent = angles_RT.value
        P_RT_gen_values = P_RT_gen.value
        LMPs = prob.constraints[5].dual_value
        T_flow_RT_in_MW = T_flow_RT.value
        RT_and_DA_system_cost_in_RT_dispatch = prob.value
        #
        # returning the load to its original form
        self.bus[:, 2] = original_load
        #
        return P_RT_gen_values, T_flow_RT_in_MW, angles_RT_values_cent, dict_theta_tielines, LMPs, dic_rate_absol_at_optimum, dic_of_signed_rates, RT_and_DA_system_cost_in_RT_dispatch
    #
    #
    #
    def generate_scenarios_RT_market_line_and_gen_R1(self, DA_Schedule,num_scenarios, node1, node2):
        #
        #
        probability_threshold = self.probability_threshold
        #
        mean = np.sum(self.bus[:, 2])
        std = self.ratio_std_to_mean * mean
        #
        samples_ALL = np.random.uniform(-0.2 * mean, 0.2 * mean, (num_scenarios, 2))
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
            RT_gen_matrix___var @ self.gencost[:, 6] + RT_gen_matrix___var @ self.gencost[:, 5] @ P_RT_gen
            + RT_gen_matrix___var @ self.gencost[:, 4] @ (P_RT_gen ** 2))
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
            np.ones((1, self.num_of_gens)) @ P_RT_gen >= dist.norm.ppf(probability_threshold, loc=mean, scale=std),
            #
        ]
        #
        prob = cp.Problem(obj, constraints)
        prob.solve(solver=cp.XPRESS, verbose=False)  # superior commercial solver!
        # prob.solve(solver=cp.OSQP, verbose=True)
        print(prob.value)
        print('P_RT_gen = {}'.format(P_RT_gen.value))
        print('T_flow_RT = {}'.format(T_flow_RT.value))
        print([min([max(prob.constraints[2].dual_value), max(prob.constraints[3].dual_value)]),
               max([max(prob.constraints[2].dual_value), max(prob.constraints[3].dual_value)])])
        print('angles_RT in degree = {}'.format(angles_RT.value * 180 / math.pi))
        print('prob.constraints[5].dual_value = {}'.format(prob.constraints[5].dual_value))
        #
        node1 = 9
        node2 = 10
        #
        kappa_tieLine_1_2 = (prob.constraints[3].dual_value)[node1]
        etta_tieLine_1_2 = (prob.constraints[2].dual_value)[node1]
        kappa_tieLine_1_3 = (prob.constraints[3].dual_value)[node2]
        etta_tieLine_1_3 = (prob.constraints[2].dual_value)[node2]
        # main
        rate_absol_at_optimum_tieLine_1_2 = -kappa_tieLine_1_2 + etta_tieLine_1_2
        rate_absol_at_optimum_tieLine_1_2 = abs(rate_absol_at_optimum_tieLine_1_2)
        rate_absol_at_optimum_tieLine_1_3 = -kappa_tieLine_1_3 + etta_tieLine_1_3
        rate_absol_at_optimum_tieLine_1_3 = abs(rate_absol_at_optimum_tieLine_1_3)
        #
        #
        alpha_1_to_2 = (prob.constraints[5].dual_value)[node1]
        alpha_1_to_3 = (prob.constraints[5].dual_value)[node2]
        beta_1 = prob.constraints[10].dual_value + 0
        beta_2 = prob.constraints[11].dual_value + 0
        beta_3 = prob.constraints[12].dual_value + 0
        #
        rate_signed_r1_sell_to_r2 = alpha_1_to_2 + beta_1
        rate_signed_r1_sell_to_r3 = alpha_1_to_3 + beta_1
        dic_rate_absol_at_optimum = dict()
        dic_rate_absol_at_optimum['rate_absol_r1_sell_to_r2'] = rate_absol_at_optimum_tieLine_1_2
        dic_rate_absol_at_optimum['rate_absol_r1_sell_to_r3'] = rate_absol_at_optimum_tieLine_1_3
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
    #
    #
