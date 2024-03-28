#!/usr/bin/env python3
"""
-------------------
MIT License

Copyright (c) 2024  Zeyu Liu

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
-------------------
Description:
    Baseline model for RECOIL
-------------------
"""

# import
import logging
import numpy as np
import gurobipy as grb


class Intermodal():
    """
    Intermodal transportation
    """
    def __init__(self, name, instance):
        """
        `name`: str
        `instance`: dict, containing:
            `G_H/G_R/G_W`: networkx, highway/railway/waterway network,
                nodes are arranged consecutively as H, R, W,
                arcs contain 'tau' as the travel time,
            `P`: list, orders,
                `p` in P, dict, 'i', 'o', 'd', 'v',
            `L`: list, set of locomotives,
            `B`: list, set of barges,
            `C_R`: double, capacity of a locomotive,
            `C_W`: double, capacity of a barge,
            `delta_R`: double, railway transshipment & sorting time,
            `delta_W`: double, waterway transshipment & sorting time
        """
        self.name = name
        # start log
        logging.basicConfig(
            filename=f'{self.name}_unrelaxed_SP.log', filemode='w+',
            format='%(levelname)s - %(message)s', level=logging.INFO
        )
        # networks
        self.G_H = instance['G_H']
        self.G_R = instance['G_R']
        self.G_W = instance['G_W']
        # orders
        self.P = instance['P']
        # vessels
        self.L, self.B = instance['L'], instance['B']
        # capacity
        self.C_R, self.C_W = instance['C_R'], instance['C_W']
        # transshipment time
        self.delta_R = instance['delta_R']
        self.delta_W = instance['delta_W']
        # costs
        self.c_H = instance['c_H']
        self.c_R = instance['c_R']
        self.c_W = instance['c_W']
        self.c_P = instance['c_P']
        # carbon tax
        self.sigma_H = instance['sigma_H']
        self.sigma_R = instance['sigma_R']
        self.sigma_W = instance['sigma_W']
        # a very large number
        self.N = 1e5
        # MP and SP for decomposition
        self.MP, self.LP, self.SP = float('nan'), {}, {}
        # theta
        self.theta, self.theta_prv = {}, {}
        # coupling variables for decomposition
        self.chi_R, self.chi_W = {}, {}
        self.t_R, self.t_W = {}, {}
        self.s_R, self.s_W = {}, {}
        # coupling constraints for decomposition
        self.chi_R_LP, self.chi_W_LP = {}, {}
        self.chi_R_SP, self.chi_W_SP = {}, {}
        self.order_arival_R_LP, self.order_arival_W_LP = {}, {}
        self.order_arival_R_SP, self.order_arival_W_SP = {}, {}
        self.depart_W_R_LP, self.depart_R_W_LP = {}, {}
        self.depart_W_R_SP, self.depart_R_W_SP = {}, {}
        self.depart_O_R_LP, self.depart_O_W_LP = {}, {}
        self.depart_O_R_SP, self.depart_O_W_SP = {}, {}
        self.trans_m_n_R_LP, self.trans_m_n_W_LP = {}, {}
        self.trans_m_n_R_SP, self.trans_m_n_W_SP = {}, {}
        self.trans_m_m_R_LP, self.trans_m_m_W_LP = {}, {}
        self.trans_m_m_R_SP, self.trans_m_m_W_SP = {}, {}
        self.trans_n_n_R_LP, self.trans_n_n_W_LP = {}, {}
        self.trans_n_n_R_SP, self.trans_n_n_W_SP = {}, {}
        self.trans_n_m_R_LP, self.trans_n_m_W_LP = {}, {}
        self.trans_n_m_R_SP, self.trans_n_m_W_SP = {}, {}
        # SP variable
        self.x_R, self.x_W = {}, {}
        # MIP obj for decomposition
        self.obj = {}

    def MIP(self):
        """
        MIP model
        """
        # model
        model = grb.Model(self.name)
        model.setParam("OutputFlag", True)
        # model.setParam("MIPGap", 1e-4)
        # model.setParam("Presolve", -1)
        # model.setParam("MIPFocus", -1)
        # model.setParam("DualReductions", 1)
        # model.setParam("IntFeasTol", 1e-5)
        # model.setParam("NumericFocus", 0)
        # model.setParam("Heuristics", 0.05)
        model.setParam("TimeLimit", 3600)
        # variables, order flow
        x_H, x_R, x_W = {}, {}, {}
        for p in self.P:
            # highway
            for arc in self.G_H.edges:
                x_H[p['i'], arc[0], arc[1]] = model.addVar(
                    lb=0, ub=1,
                    vtype=grb.GRB.BINARY,
                    name=f"x^H_{p['i']}_{arc[0]}_{arc[1]}"
                )
            # railway
            for m in self.L:
                for arc in self.G_R.edges:
                    x_R[p['i'], m, arc[0], arc[1]] = model.addVar(
                        lb=0, ub=1,
                        vtype=grb.GRB.BINARY,
                        name=f"x^R_{p['i']}_{m}_{arc[0]}_{arc[1]}"
                    )
            # waterway
            for m in self.B:
                for arc in self.G_W.edges:
                    x_W[p['i'], m, arc[0], arc[1]] = model.addVar(
                        lb=0, ub=1,
                        vtype=grb.GRB.BINARY,
                        name=f"x^W_{p['i']}_{m}_{arc[0]}_{arc[1]}"
                    )
        # routing
        y_R, y_W, z_R, z_W = {}, {}, {}, {}
        # railway
        for m in self.L:
            for arc in self.G_R.edges:
                y_R[m, arc[0], arc[1]] = model.addVar(
                    lb=0, ub=1,
                    vtype=grb.GRB.BINARY,
                    name=f'y^R_{m}_{arc[0]}_{arc[1]}'
                )
            for i in self.G_R.nodes:
                z_R[m, i] = model.addVar(
                    lb=0, ub=grb.GRB.INFINITY,
                    vtype=grb.GRB.INTEGER,
                    name=f'z^R_{m}_{i}'
                )
            # hub, 0
            for i in self.G_R.nodes:
                y_R[m, 0, i] = model.addVar(
                    lb=0, ub=1,
                    vtype=grb.GRB.BINARY,
                    name=f'y^R_{m}_0_{i}'
                )
            z_R[m, 0] = model.addVar(
                lb=0, ub=grb.GRB.INFINITY,
                vtype=grb.GRB.INTEGER,
                name=f'z^R_{m}_0'
            )
        # waterway
        for m in self.B:
            for arc in self.G_W.edges:
                y_W[m, arc[0], arc[1]] = model.addVar(
                    lb=0, ub=1,
                    vtype=grb.GRB.BINARY,
                    name=f'y^W_{m}_{arc[0]}_{arc[1]}'
                )
            for i in self.G_W.nodes:
                z_W[m, i] = model.addVar(
                    lb=0, ub=grb.GRB.INFINITY,
                    vtype=grb.GRB.INTEGER,
                    name=f'z^W_{m}_{i}'
                )
            # hub, 0
            for i in self.G_W.nodes:
                y_W[m, 0, i] = model.addVar(
                    lb=0, ub=1,
                    vtype=grb.GRB.BINARY,
                    name=f'y^W_{m}_0_{i}'
                )
            z_W[m, 0] = model.addVar(
                lb=0, ub=grb.GRB.INFINITY,
                vtype=grb.GRB.INTEGER,
                name=f'z^W_{m}_0'
            )
        # transshipment
        u_R, u_W = {}, {}
        # railway
        for m in self.L:
            for i in self.G_R.nodes:
                for n in self.L:
                    if m != n:
                        for p in self.P:
                            u_R[p['i'], m, n, i] = model.addVar(
                                lb=0, ub=1,
                                vtype=grb.GRB.BINARY,
                                name=f"u^R_{p['i']}_{m}_{n}_{i}"
                            )
        # waterway
        for m in self.B:
            for i in self.G_W.nodes:
                for n in self.B:
                    if m != n:
                        for p in self.P:
                            u_W[p['i'], m, n, i] = model.addVar(
                                lb=0, ub=1,
                                vtype=grb.GRB.BINARY,
                                name=f"u^W_{p['i']}_{m}_{n}_{i}"
                            )
        # time
        t, t_R, t_W, s_R, s_W = {}, {}, {}, {}, {}
        # highway
        for p in self.P:
            for i in self.G_H.nodes:
                t[p['i'], i] = model.addVar(
                    lb=0, ub=grb.GRB.INFINITY,
                    vtype=grb.GRB.CONTINUOUS,
                    name=f"t_{p['i']}_{i}"
                )
        # railway
        for m in self.L:
            for i in self.G_R.nodes:
                t_R[m, i] = model.addVar(
                    lb=0, ub=grb.GRB.INFINITY,
                    vtype=grb.GRB.CONTINUOUS,
                    name=f't^R_{m}_{i}'
                )
                s_R[m, i] = model.addVar(
                    lb=0, ub=grb.GRB.INFINITY,
                    vtype=grb.GRB.CONTINUOUS,
                    name=f's^R_{m}_{i}'
                )
        # waterway
        for m in self.B:
            for i in self.G_W.nodes:
                t_W[m, i] = model.addVar(
                    lb=0, ub=grb.GRB.INFINITY,
                    vtype=grb.GRB.CONTINUOUS,
                    name=f't^W_{m}_{i}'
                )
                s_W[m, i] = model.addVar(
                    lb=0, ub=grb.GRB.INFINITY,
                    vtype=grb.GRB.CONTINUOUS,
                    name=f's^W_{m}_{i}'
                )
        # if a vehicle is used, for costs
        psi_R, psi_W = {}, {}
        for m in self.L:
            psi_R[m] = model.addVar(
                lb=0, ub=1,
                vtype=grb.GRB.BINARY,
                name=f'psi^R_{m}'
            )
        for m in self.B:
            psi_W[m] = model.addVar(
                lb=0, ub=1,
                vtype=grb.GRB.BINARY,
                name=f'psi^W_{m}'
            )
        # late time, for cost
        T = {}
        for p in self.P:
            T[p['i']] = model.addVar(
                lb=0, ub=grb.GRB.INFINITY,
                vtype=grb.GRB.CONTINUOUS,
                name=f"T_{p['i']}"
            )
        model.update()
        # objective - with cost
        # highway operation cost
        operation_H = grb.quicksum([
            self.c_H * self.G_H.edges[arc]['tau']
            * p['v'] * x_H[p['i'], arc[0], arc[1]]
            for arc in self.G_H.edges
            for p in self.P
        ])
        # railway operation cost
        operation_R = grb.quicksum([
            self.c_R * psi_R[m]
            for m in self.L
        ])
        # waterway operation cost
        operation_W = grb.quicksum([
            self.c_W * psi_W[m]
            for m in self.B
        ])
        # late penalty
        late_penalty = grb.quicksum([
            self.c_P * T[p['i']]
            for p in self.P
        ])
        # Carbon Tax, highway
        GHG_H = grb.quicksum([
            self.sigma_H * x_H[p['i'], arc[0], arc[1]]
            * self.G_H.edges[arc]['d'] * p['v']
            for arc in self.G_H.edges
            for p in self.P
        ])
        # Carbon Tax, railway
        GHG_R = grb.quicksum([
            self.sigma_R * x_R[p['i'], m, arc[0], arc[1]]
            * self.G_R.edges[arc]['d'] * p['w']
            for arc in self.G_R.edges
            for p in self.P
            for m in self.L
        ])
        # Carbon Tax, waterway
        GHG_W = grb.quicksum([
            self.sigma_W * x_W[p['i'], m, arc[0], arc[1]]
            * self.G_W.edges[arc]['d'] * p['w']
            for arc in self.G_W.edges
            for p in self.P
            for m in self.B
        ])
        obj = grb.quicksum([
            operation_H, operation_R, operation_W,
            late_penalty, GHG_H, GHG_R, GHG_W
        ])
        model.setObjective(obj, grb.GRB.MINIMIZE)
        # constraint
        # demand
        for p in self.P:
            # origin
            model.addLConstr(
                lhs=grb.quicksum([
                    grb.quicksum([
                        x_H[p['i'], p['o'], j]
                        for j in self.G_R.nodes
                    ]),
                    grb.quicksum([
                        x_H[p['i'], p['o'], j]
                        for j in self.G_W.nodes
                    ]),
                    x_H[p['i'], p['o'], p['d']]
                ]),
                sense=grb.GRB.EQUAL,
                rhs=1
            )
            # destination
            model.addLConstr(
                lhs=grb.quicksum([
                    grb.quicksum([
                        x_H[p['i'], i, p['d']]
                        for i in self.G_R.nodes
                    ]),
                    grb.quicksum([
                        x_H[p['i'], i, p['d']]
                        for i in self.G_W.nodes
                    ]),
                    x_H[p['i'], p['o'], p['d']]
                ]),
                sense=grb.GRB.EQUAL,
                rhs=1
            )
        # order flow
        for p in self.P:
            # railway
            for j in self.G_R.nodes:
                model.addLConstr(
                    lhs=grb.quicksum([
                        # highway
                        x_H[p['i'], p['o'], j],
                        # waterway
                        grb.quicksum([
                            x_H[p['i'], i, j]
                            for i in self.G_W.nodes
                        ]),
                        # railway
                        grb.quicksum([
                            x_R[p['i'], m, i, j]
                            for m in self.L
                            for i in self.G_R.nodes
                            if (i, j) in self.G_R.edges
                        ])
                    ]),
                    sense=grb.GRB.EQUAL,
                    rhs=grb.quicksum([
                        # highway
                        x_H[p['i'], j, p['d']],
                        # waterway
                        grb.quicksum([
                            x_H[p['i'], j, k]
                            for k in self.G_W.nodes
                        ]),
                        # railway
                        grb.quicksum([
                            x_R[p['i'], m, j, k]
                            for m in self.L
                            for k in self.G_R.nodes
                            if (j, k) in self.G_R.edges
                        ])
                    ])
                )
                # one way out
                model.addLConstr(
                    lhs=grb.quicksum([
                        # highway
                        x_H[p['i'], j, p['d']],
                        # waterway
                        grb.quicksum([
                            x_H[p['i'], j, k]
                            for k in self.G_W.nodes
                        ]),
                        # railway
                        grb.quicksum([
                            x_R[p['i'], m, j, k]
                            for m in self.L
                            for k in self.G_R.nodes
                            if (j, k) in self.G_R.edges
                        ])
                    ]),
                    sense=grb.GRB.LESS_EQUAL,
                    rhs=1
                )
            # waterway
            for j in self.G_W.nodes:
                model.addLConstr(
                    lhs=grb.quicksum([
                        # highway
                        x_H[p['i'], p['o'], j],
                        # railway
                        grb.quicksum([
                            x_H[p['i'], i, j]
                            for i in self.G_R.nodes
                        ]),
                        # waterway
                        grb.quicksum([
                            x_W[p['i'], m, i, j]
                            for m in self.B
                            for i in self.G_W.nodes
                            if (i, j) in self.G_W.edges
                        ])
                    ]),
                    sense=grb.GRB.EQUAL,
                    rhs=grb.quicksum([
                        # highway
                        x_H[p['i'], j, p['d']],
                        # railway
                        grb.quicksum([
                            x_H[p['i'], j, k]
                            for k in self.G_R.nodes
                        ]),
                        # waterway
                        grb.quicksum([
                            x_W[p['i'], m, j, k]
                            for m in self.B
                            for k in self.G_W.nodes
                            if (j, k) in self.G_W.edges
                        ])
                    ])
                )
                # one way out
                model.addLConstr(
                    lhs=grb.quicksum([
                        # highway
                        x_H[p['i'], j, p['d']],
                        # railway
                        grb.quicksum([
                            x_H[p['i'], j, k]
                            for k in self.G_R.nodes
                        ]),
                        # waterway
                        grb.quicksum([
                            x_W[p['i'], m, j, k]
                            for m in self.B
                            for k in self.G_W.nodes
                            if (j, k) in self.G_W.edges
                        ])
                    ]),
                    sense=grb.GRB.LESS_EQUAL,
                    rhs=1
                )
        # routing, railway
        for m in self.L:
            # starts form hub
            model.addLConstr(
                lhs=grb.quicksum([
                    y_R[m, 0, j]
                    for j in self.G_R.nodes
                ]),
                sense=grb.GRB.LESS_EQUAL,
                rhs=1
            )
            # if used, for cost
            model.addLConstr(
                lhs=self.N * psi_R[m],
                sense=grb.GRB.GREATER_EQUAL,
                rhs=grb.quicksum([
                    y_R[m, arc[0], arc[1]]
                    for arc in self.G_R.edges
                ])
            )
            # flow balance
            for j in self.G_R.nodes:
                model.addLConstr(
                    lhs=grb.quicksum([
                        y_R[m, 0, j],
                        grb.quicksum([
                            y_R[m, i, j]
                            for i in self.G_R.predecessors(j)
                        ])
                    ]),
                    sense=grb.GRB.GREATER_EQUAL,
                    rhs=grb.quicksum([
                        y_R[m, j, k]
                        for k in self.G_R.successors(j)
                    ])
                )
            # subtour elimination
            for arc in self.G_R.edges:
                model.addLConstr(
                    lhs=grb.quicksum([
                        z_R[m, arc[0]],
                        -1 * z_R[m, arc[1]],
                        1
                    ]),
                    sense=grb.GRB.LESS_EQUAL,
                    rhs=len(self.G_W.nodes) * grb.quicksum([
                        1, -1 * y_R[m, arc[0], arc[1]]
                    ])
                )
        # routing, waterway
        for m in self.B:
            # starts form hub
            model.addLConstr(
                lhs=grb.quicksum([
                    y_W[m, 0, j]
                    for j in self.G_W.nodes
                ]),
                sense=grb.GRB.LESS_EQUAL,
                rhs=1
            )
            # if used, for cost
            model.addLConstr(
                lhs=self.N * psi_W[m],
                sense=grb.GRB.GREATER_EQUAL,
                rhs=grb.quicksum([
                    y_W[m, arc[0], arc[1]]
                    for arc in self.G_W.edges
                ])
            )
            # flow balance
            for j in self.G_W.nodes:
                model.addLConstr(
                    lhs=grb.quicksum([
                        y_W[m, 0, j],
                        grb.quicksum([
                            y_W[m, i, j]
                            for i in self.G_W.predecessors(j)
                        ])
                    ]),
                    sense=grb.GRB.GREATER_EQUAL,
                    rhs=grb.quicksum([
                        y_W[m, j, k]
                        for k in self.G_W.successors(j)
                    ])
                )
            # subtour elimination
            for arc in self.G_W.edges:
                model.addLConstr(
                    lhs=grb.quicksum([
                        z_W[m, arc[0]],
                        -1 * z_W[m, arc[1]],
                        1
                    ]),
                    sense=grb.GRB.LESS_EQUAL,
                    rhs=len(self.G_W.nodes) * grb.quicksum([
                        1, -1 * y_W[m, arc[0], arc[1]]
                    ])
                )
        # coupling railway
        for m in self.L:
            for arc in self.G_R.edges:
                model.addLConstr(
                    lhs=grb.quicksum([
                        p['v'] * x_R[p['i'], m, arc[0], arc[1]]
                        for p in self.P
                    ]),
                    sense=grb.GRB.LESS_EQUAL,
                    rhs=self.C_R * y_R[m, arc[0], arc[1]]
                )
                # no extra routes
                model.addLConstr(
                    lhs=grb.quicksum([
                        x_R[p['i'], m, arc[0], arc[1]]
                        for p in self.P
                    ]),
                    sense=grb.GRB.GREATER_EQUAL,
                    rhs=y_R[m, arc[0], arc[1]]
                )
        # coupling waterway
        for m in self.B:
            for arc in self.G_W.edges:
                model.addLConstr(
                    lhs=grb.quicksum([
                        p['v'] * x_W[p['i'], m, arc[0], arc[1]]
                        for p in self.P
                    ]),
                    sense=grb.GRB.LESS_EQUAL,
                    rhs=self.C_W * y_W[m, arc[0], arc[1]]
                )
                model.addLConstr(
                    lhs=grb.quicksum([
                        x_W[p['i'], m, arc[0], arc[1]]
                        for p in self.P
                    ]),
                    sense=grb.GRB.GREATER_EQUAL,
                    rhs=y_W[m, arc[0], arc[1]]
                )
        # transshipment & sorting
        for p in self.P:
            # railway
            for j in self.G_R.nodes:
                for m in self.L:
                    for n in self.L:
                        if m != n:
                            model.addLConstr(
                                lhs=u_R[p['i'], m, n, j],
                                sense=grb.GRB.GREATER_EQUAL,
                                rhs=grb.quicksum([
                                    grb.quicksum([
                                        x_R[p['i'], m, i, j]
                                        for i in self.G_R.predecessors(j)
                                    ]),
                                    grb.quicksum([
                                        x_R[p['i'], n, j, k]
                                        for k in self.G_R.successors(j)
                                    ]),
                                    -1 * (3 / 2)
                                ])
                            )
                            model.addLConstr(
                                lhs=u_R[p['i'], m, n, j],
                                sense=grb.GRB.LESS_EQUAL,
                                rhs=(1 / 2) * grb.quicksum([
                                    grb.quicksum([
                                        x_R[p['i'], m, i, j]
                                        for i in self.G_R.predecessors(j)
                                    ]),
                                    grb.quicksum([
                                        x_R[p['i'], n, j, k]
                                        for k in self.G_R.successors(j)
                                    ])
                                ])
                            )
            # waterway
            for j in self.G_W.nodes:
                for m in self.B:
                    for n in self.B:
                        if m != n:
                            model.addLConstr(
                                lhs=u_W[p['i'], m, n, j],
                                sense=grb.GRB.GREATER_EQUAL,
                                rhs=grb.quicksum([
                                    grb.quicksum([
                                        x_W[p['i'], m, i, j]
                                        for i in self.G_W.predecessors(j)
                                    ]),
                                    grb.quicksum([
                                        x_W[p['i'], n, j, k]
                                        for k in self.G_W.successors(j)
                                    ]),
                                    -1 * (3 / 2)
                                ])
                            )
                            model.addLConstr(
                                lhs=u_W[p['i'], m, n, j],
                                sense=grb.GRB.LESS_EQUAL,
                                rhs=(1 / 2) * grb.quicksum([
                                    grb.quicksum([
                                        x_W[p['i'], m, i, j]
                                        for i in self.G_W.predecessors(j)
                                    ]),
                                    grb.quicksum([
                                        x_W[p['i'], n, j, k]
                                        for k in self.G_W.successors(j)
                                    ])
                                ])
                            )
        # arrival time, railway
        for m in self.L:
            for arc in self.G_R.edges:
                model.addLConstr(
                    lhs=t_R[m, arc[1]],
                    sense=grb.GRB.GREATER_EQUAL,
                    rhs=grb.quicksum([
                        s_R[m, arc[0]],
                        self.G_R.edges[arc]['tau'],
                        -self.N * (1 - y_R[m, arc[0], arc[1]])
                    ])
                )
        # arrival time, waterway
        for m in self.B:
            for arc in self.G_W.edges:
                model.addLConstr(
                    lhs=t_W[m, arc[1]],
                    sense=grb.GRB.GREATER_EQUAL,
                    rhs=grb.quicksum([
                        s_W[m, arc[0]],
                        self.G_W.edges[arc]['tau'],
                        -self.N * (1 - y_W[m, arc[0], arc[1]])
                    ])
                )
        # departure time, railway
        for m in self.L:
            for j in self.G_R.nodes:
                # same vehicle, with/without transshipmwnt
                model.addLConstr(
                    lhs=s_R[m, j],
                    sense=grb.GRB.GREATER_EQUAL,
                    rhs=grb.quicksum([
                        t_R[m, j]
                    ])
                )
                # all arriving orders
                for p in self.P:
                    # from waterway
                    for i in self.G_W.nodes:
                        model.addLConstr(
                            lhs=s_R[m, j],
                            sense=grb.GRB.GREATER_EQUAL,
                            rhs=grb.quicksum([
                                t[p['i'], i],
                                self.G_H.edges[(i, j)]['tau'],
                                -self.N * (1 - x_H[p['i'], i, j])
                            ])
                        )
                    # from origin
                    model.addLConstr(
                        lhs=s_R[m, j],
                        sense=grb.GRB.GREATER_EQUAL,
                        rhs=grb.quicksum([
                            t[p['i'], p['o']],
                            self.G_H.edges[(p['o'], j)]['tau'],
                            -self.N * (1 - x_H[p['i'], p['o'], j])
                        ])
                    )
                    # transshipment & sorting
                    for n in self.L:
                        if n != m:
                            model.addLConstr(
                                lhs=s_R[m, j],
                                sense=grb.GRB.GREATER_EQUAL,
                                rhs=grb.quicksum([
                                    t_R[n, j],
                                    self.delta_R,
                                    -self.N * (1 - u_R[p['i'], m, n, j])
                                ])
                            )
                            model.addLConstr(
                                lhs=s_R[m, j],
                                sense=grb.GRB.GREATER_EQUAL,
                                rhs=grb.quicksum([
                                    t_R[m, j],
                                    self.delta_R,
                                    -self.N * (1 - u_R[p['i'], m, n, j])
                                ])
                            )
                            model.addLConstr(
                                lhs=s_R[n, j],
                                sense=grb.GRB.GREATER_EQUAL,
                                rhs=grb.quicksum([
                                    t_R[n, j],
                                    self.delta_R,
                                    -self.N * (1 - u_R[p['i'], m, n, j])
                                ])
                            )
                            model.addLConstr(
                                lhs=s_R[n, j],
                                sense=grb.GRB.GREATER_EQUAL,
                                rhs=grb.quicksum([
                                    t_R[m, j],
                                    self.delta_R,
                                    -self.N * (1 - u_R[p['i'], m, n, j])
                                ])
                            )
        # departure time, waterway
        for m in self.B:
            for j in self.G_W.nodes:
                # same vehicle, with/without transshipmwnt
                model.addLConstr(
                    lhs=s_W[m, j],
                    sense=grb.GRB.GREATER_EQUAL,
                    rhs=grb.quicksum([
                        t_W[m, j]
                    ])
                )
                # all arriving orders
                for p in self.P:
                    # from railway
                    for i in self.G_R.nodes:
                        model.addLConstr(
                            lhs=s_W[m, j],
                            sense=grb.GRB.GREATER_EQUAL,
                            rhs=grb.quicksum([
                                t[p['i'], i],
                                self.G_H.edges[(i, j)]['tau'],
                                -self.N * (1 - x_H[p['i'], i, j])
                            ])
                        )
                    # from origin
                    model.addLConstr(
                        lhs=s_W[m, j],
                        sense=grb.GRB.GREATER_EQUAL,
                        rhs=grb.quicksum([
                            t[p['i'], p['o']],
                            self.G_H.edges[(p['o'], j)]['tau'],
                            -self.N * (1 - x_H[p['i'], p['o'], j])
                        ])
                    )
                    # transshipment & sorting
                    for n in self.B:
                        if n != m:
                            model.addLConstr(
                                lhs=s_W[m, j],
                                sense=grb.GRB.GREATER_EQUAL,
                                rhs=grb.quicksum([
                                    t_W[n, j],
                                    self.delta_W,
                                    -self.N * (1 - u_W[p['i'], m, n, j])
                                ])
                            )
                            model.addLConstr(
                                lhs=s_W[m, j],
                                sense=grb.GRB.GREATER_EQUAL,
                                rhs=grb.quicksum([
                                    t_W[m, j],
                                    self.delta_W,
                                    -self.N * (1 - u_W[p['i'], m, n, j])
                                ])
                            )
                            model.addLConstr(
                                lhs=s_W[n, j],
                                sense=grb.GRB.GREATER_EQUAL,
                                rhs=grb.quicksum([
                                    t_W[n, j],
                                    self.delta_W,
                                    -self.N * (1 - u_W[p['i'], m, n, j])
                                ])
                            )
                            model.addLConstr(
                                lhs=s_W[n, j],
                                sense=grb.GRB.GREATER_EQUAL,
                                rhs=grb.quicksum([
                                    t_W[m, j],
                                    self.delta_W,
                                    -self.N * (1 - u_W[p['i'], m, n, j])
                                ])
                            )
        # order arrival time
        for p in self.P:
            # at railway
            for m in self.L:
                for j in self.G_R.nodes:
                    model.addLConstr(
                        lhs=t[p['i'], j],
                        sense=grb.GRB.GREATER_EQUAL,
                        rhs=grb.quicksum([
                            t_R[m, j],
                            -self.N * (1 - grb.quicksum([
                                x_R[p['i'], m, i, j]
                                for i in self.G_R.predecessors(j)
                            ]))
                        ])
                    )
            # at waterway
            for m in self.B:
                for j in self.G_W.nodes:
                    model.addLConstr(
                        lhs=t[p['i'], j],
                        sense=grb.GRB.GREATER_EQUAL,
                        rhs=grb.quicksum([
                            t_W[m, j],
                            -self.N * (1 - grb.quicksum([
                                x_W[p['i'], m, i, j]
                                for i in self.G_W.predecessors(j)
                            ]))
                        ])
                    )
        # order arrival time at destinations
        for p in self.P:
            for arc in self.G_H.edges:
                if arc[1] != p['o']:
                    model.addLConstr(
                        lhs=t[p['i'], arc[1]],
                        sense=grb.GRB.GREATER_EQUAL,
                        rhs=grb.quicksum([
                            t[p['i'], arc[0]],
                            self.G_H.edges[(arc[0], arc[1])]['tau'],
                            -self.N * (1 - x_H[p['i'], arc[0], arc[1]])
                        ])
                    )
        # late penalty, for cost
        for p in self.P:
            model.addLConstr(
                lhs=T[p['i']],
                sense=grb.GRB.GREATER_EQUAL,
                rhs=t[p['i'], p['d']] - p['T']
            )
        model.update()
        # solve
        model.optimize()
        # results
        file = open(f"results/{self.name}.txt", 'w+')
        try:
            # total cost
            file.write(f"Objective: {model.ObjVal}\n")
            # cost breakdown
            total_operations = np.sum([
                operation_H.getValue(), operation_R.getValue(),
                operation_W.getValue()
            ])
            file.write(f"  Total operations: {total_operations}\n")
            file.write(f"    Highway operations: {operation_H.getValue()}\n")
            file.write(f"    Railway operations: {operation_R.getValue()}\n")
            file.write(f"    Waterway operations: {operation_W.getValue()}\n")
            file.write(f"  Late penalty: {late_penalty.getValue()}\n")
            total_GHG = np.sum([
                GHG_H.getValue(), GHG_R.getValue(), GHG_W.getValue()
            ])
            file.write(f"  Total GHG tax: {total_GHG}\n")
            file.write(f"    Highway GHG tax: {GHG_H.getValue()}\n")
            file.write(f"    Railway GHG tax: {GHG_R.getValue()}\n")
            file.write(f"    Waterway GHG tax: {GHG_W.getValue()}\n")
        except AttributeError:
            file.write("No objective\n")
            file.write(f"Runtime: {model.Runtime}\n")
            file.close()
            return
        # write file
        file.write(f"Model status: {model.Status}\n")
        file.write(f"MIP gap: {model.MIPGap}\n")
        file.write(f"Runtime: {model.Runtime}\n")
        # locomotive route
        file.write("-----------------------\n")
        for m in self.L:
            # not using m
            if psi_R[m].X < 0.99:
                continue
            # using m
            for i in self.G_R.nodes:
                if y_R[m, 0, i].X > 0.99:
                    break
            route = [0, i]
            route_time = [0, t_R[m, i].X]
            depart_time = [0, s_R[m, i].X]
            while np.sum([
                y_R[m, i, j].X
                for j in self.G_R.successors(i)
            ]) > 0.99:
                for j in self.G_R.successors(i):
                    if y_R[m, i, j].X > 0.99:
                        route.append(j)
                        route_time.append(t_R[m, j].X)
                        depart_time.append(s_R[m, j].X)
                        break
                i = j
            file.write(f"Route of locomotive {m}: {route}\n")
            file.write(f"  Arrival time {m}: {route_time}\n")
            file.write(f"  Departure time {m}: {depart_time}\n")
            # transsip
            for i in self.G_R.nodes:
                for n in self.L:
                    if m != n:
                        if np.sum([
                            u_R[p['i'], m, n, i].X for p in self.P
                        ]) > 0.99 or np.sum([
                            u_R[p['i'], n, m, i].X for p in self.P
                        ]) > 0.99:
                            file.write(f"  Transsip at {i} with {n}\n")
        # barge route
        file.write("-----------------------\n")
        for m in self.B:
            # not using m
            if psi_W[m].X < 0.99:
                continue
            # using m
            for i in self.G_W.nodes:
                if y_W[m, 0, i].X > 0.99:
                    break
            route = [0, i]
            route_time = [0, t_R[m, i].X]
            depart_time = [0, s_R[m, i].X]
            while np.sum([
                y_W[m, i, j].X
                for j in self.G_W.successors(i)
            ]) > 0.99:
                for j in self.G_W.successors(i):
                    if y_W[m, i, j].X > 0.99:
                        route.append(j)
                        route_time.append(t_R[m, j].X)
                        depart_time.append(s_R[m, j].X)
                        break
                i = j
            file.write(f"Route of barge {m}: {route}\n")
            file.write(f"  Arrival time {m}: {route_time}\n")
            file.write(f"  Departure time {m}: {depart_time}\n")
            # transsip
            for i in self.G_W.nodes:
                for n in self.B:
                    if m != n:
                        if np.sum([
                            u_W[p['i'], m, n, i].X for p in self.P
                        ]) > 0.99 or np.sum([
                            u_W[p['i'], n, m, i].X for p in self.P
                        ]) > 0.99:
                            file.write(f"  Transsip at {i} with {n}\n")
        file.write("-----------------------\n")
        # order route
        for p in self.P:
            route = []
            route_time = []
            for arc in self.G_H.edges:
                if x_H[p['i'], arc[0], arc[1]].X > 0.99:
                    route.append(('H', arc[0], arc[1]))
                    route_time.append(t[p['i'], arc[1]].X)
            for arc in self.G_R.edges:
                for m in self.L:
                    if x_R[p['i'], m, arc[0], arc[1]].X > 0.99:
                        route.append(('R', m, arc[0], arc[1]))
                        route_time.append(t[p['i'], arc[1]].X)
            for arc in self.G_W.edges:
                for m in self.B:
                    if x_W[p['i'], m, arc[0], arc[1]].X > 0.99:
                        route.append(('W', m, arc[0], arc[1]))
                        route_time.append(t[p['i'], arc[1]].X)
            file.write(f"Route of order {p['i']}: {route}\n")
            file.write(f"  Order arrival time: {route_time}\n")
            file.write(f"  Transport time: {t[p['i'], p['d']].X}\n")
            # for i in self.G_H.nodes:
            #     file.write(f"  {i}: {t[p['i'], i].X}\n")
        file.close()
        return

    def __build_MP(self):
        """
        build MP for decomposition
        """
        # model
        model = grb.Model(self.name)
        model.setParam("OutputFlag", False)
        model.setParam("MIPGap", 1e-3)
        # model.setParam("Presolve", -1)
        # model.setParam("MIPFocus", 0)
        # model.setParam("DualReductions", 1)
        # model.setParam("IntFeasTol", 1e-5)
        # model.setParam("NumericFocus", 0)
        model.setParam("Heuristics", 0.05)
        model.setParam("TimeLimit", 3600)
        # variables
        # routing
        y_R, y_W, z_R, z_W = {}, {}, {}, {}
        # railway
        for m in self.L:
            for arc in self.G_R.edges:
                y_R[m, arc[0], arc[1]] = model.addVar(
                    lb=0, ub=1,
                    vtype=grb.GRB.BINARY,
                    name=f'y^R_{m}_{arc[0]}_{arc[1]}'
                )
            for i in self.G_R.nodes:
                z_R[m, i] = model.addVar(
                    lb=0, ub=grb.GRB.INFINITY,
                    vtype=grb.GRB.INTEGER,
                    name=f'z^R_{m}_{i}'
                )
            # hub, 0
            for i in self.G_R.nodes:
                y_R[m, 0, i] = model.addVar(
                    lb=0, ub=1,
                    vtype=grb.GRB.BINARY,
                    name=f'y^R_{m}_0_{i}'
                )
            z_R[m, 0] = model.addVar(
                lb=0, ub=grb.GRB.INFINITY,
                vtype=grb.GRB.INTEGER,
                name=f'z^R_{m}_0'
            )
        # waterway
        for m in self.B:
            for arc in self.G_W.edges:
                y_W[m, arc[0], arc[1]] = model.addVar(
                    lb=0, ub=1,
                    vtype=grb.GRB.BINARY,
                    name=f'y^W_{m}_{arc[0]}_{arc[1]}'
                )
            for i in self.G_W.nodes:
                z_W[m, i] = model.addVar(
                    lb=0, ub=grb.GRB.INFINITY,
                    vtype=grb.GRB.INTEGER,
                    name=f'z^W_{m}_{i}'
                )
            # hub, 0
            for i in self.G_W.nodes:
                y_W[m, 0, i] = model.addVar(
                    lb=0, ub=1,
                    vtype=grb.GRB.BINARY,
                    name=f'y^W_{m}_0_{i}'
                )
            z_W[m, 0] = model.addVar(
                lb=0, ub=grb.GRB.INFINITY,
                vtype=grb.GRB.INTEGER,
                name=f'z^W_{m}_0'
            )
        # if a vehicle is used, for costs
        psi_R, psi_W = {}, {}
        for m in self.L:
            psi_R[m] = model.addVar(
                lb=0, ub=1,
                vtype=grb.GRB.BINARY,
                name=f'psi^R_{m}'
            )
        for m in self.B:
            psi_W[m] = model.addVar(
                lb=0, ub=1,
                vtype=grb.GRB.BINARY,
                name=f'psi^W_{m}'
            )
        # copying variables, flow capacity for order
        for p in self.P:
            # railway
            for m in self.L:
                for arc in self.G_R.edges:
                    self.chi_R[p['i'], m, arc[0], arc[1]] = model.addVar(
                        lb=0, ub=grb.GRB.INFINITY,
                        vtype=grb.GRB.CONTINUOUS,
                        name=f"chi^R_{p['i']}_{m}_{arc[0]}_{arc[1]}"
                    )
            # waterway
            for m in self.B:
                for arc in self.G_W.edges:
                    self.chi_W[p['i'], m, arc[0], arc[1]] = model.addVar(
                        lb=0, ub=grb.GRB.INFINITY,
                        vtype=grb.GRB.CONTINUOUS,
                        name=f"chi^W_{p['i']}_{m}_{arc[0]}_{arc[1]}"
                    )
        # coupling variables, vehicle arrival (t)/departure (s) time
        # railway
        for m in self.L:
            for i in self.G_R.nodes:
                self.t_R[m, i] = model.addVar(
                    lb=0, ub=100,  # grb.GRB.INFINITY,
                    vtype=grb.GRB.CONTINUOUS,
                    name=f't^R_{m}_{i}'
                )
                self.s_R[m, i] = model.addVar(
                    lb=0, ub=100,  # grb.GRB.INFINITY,
                    vtype=grb.GRB.CONTINUOUS,
                    name=f's^R_{m}_{i}'
                )
        # waterway
        for m in self.B:
            for i in self.G_W.nodes:
                self.t_W[m, i] = model.addVar(
                    lb=0, ub=100,  # grb.GRB.INFINITY,
                    vtype=grb.GRB.CONTINUOUS,
                    name=f't^W_{m}_{i}'
                )
                self.s_W[m, i] = model.addVar(
                    lb=0, ub=100,  # grb.GRB.INFINITY,
                    vtype=grb.GRB.CONTINUOUS,
                    name=f's^W_{m}_{i}'
                )
        # theta for product
        self.theta = {}
        for p in self.P:
            self.theta[p['i']] = model.addVar(
                lb=-1e8, ub=grb.GRB.INFINITY,
                vtype=grb.GRB.CONTINUOUS,
                name=f"theta_{p['i']}"
            )
            self.theta_prv[p['i']] = 0
        model.update()
        # objective - with cost
        # railway operation cost
        operation_R = grb.quicksum([
            self.c_R * psi_R[m]
            for m in self.L
        ])
        # waterway operation cost
        operation_W = grb.quicksum([
            self.c_W * psi_W[m]
            for m in self.B
        ])
        # theta
        sp_obj = grb.quicksum([
            self.theta[p['i']] for p in self.P
        ])
        obj = grb.quicksum([operation_R, operation_W, sp_obj])
        model.setObjective(obj, grb.GRB.MINIMIZE)
        # constraint
        # routing, railway
        for m in self.L:
            # starts form hub
            model.addLConstr(
                lhs=grb.quicksum([
                    y_R[m, 0, j] for j in self.G_R.nodes
                ]),
                sense=grb.GRB.LESS_EQUAL,
                rhs=1
            )
            # if used, for cost
            model.addLConstr(
                lhs=self.N * psi_R[m],
                sense=grb.GRB.GREATER_EQUAL,
                rhs=grb.quicksum([
                    y_R[m, arc[0], arc[1]]
                    for arc in self.G_R.edges
                ])
            )
            # flow balance
            for j in self.G_R.nodes:
                model.addLConstr(
                    lhs=grb.quicksum([
                        y_R[m, 0, j],
                        grb.quicksum([
                            y_R[m, i, j]
                            for i in self.G_R.predecessors(j)
                        ])
                    ]),
                    sense=grb.GRB.GREATER_EQUAL,
                    rhs=grb.quicksum([
                        y_R[m, j, k]
                        for k in self.G_R.successors(j)
                    ])
                )
            # subtour elimination
            for arc in self.G_R.edges:
                model.addLConstr(
                    lhs=grb.quicksum([
                        z_R[m, arc[0]],
                        -1 * z_R[m, arc[1]],
                        1
                    ]),
                    sense=grb.GRB.LESS_EQUAL,
                    rhs=len(self.G_W.nodes) * grb.quicksum([
                        1, -1 * y_R[m, arc[0], arc[1]]
                    ])
                )
        # routing, waterway
        for m in self.B:
            # starts form hub
            model.addLConstr(
                lhs=grb.quicksum([
                    y_W[m, 0, j]
                    for j in self.G_W.nodes
                ]),
                sense=grb.GRB.LESS_EQUAL,
                rhs=1
            )
            # if used, for cost
            model.addLConstr(
                lhs=self.N * psi_W[m],
                sense=grb.GRB.GREATER_EQUAL,
                rhs=grb.quicksum([
                    y_W[m, arc[0], arc[1]]
                    for arc in self.G_W.edges
                ])
            )
            # flow balance
            for j in self.G_W.nodes:
                model.addLConstr(
                    lhs=grb.quicksum([
                        y_W[m, 0, j],
                        grb.quicksum([
                            y_W[m, i, j]
                            for i in self.G_W.predecessors(j)
                        ])
                    ]),
                    sense=grb.GRB.GREATER_EQUAL,
                    rhs=grb.quicksum([
                        y_W[m, j, k]
                        for k in self.G_W.successors(j)
                    ])
                )
            # subtour elimination
            for arc in self.G_W.edges:
                model.addLConstr(
                    lhs=grb.quicksum([
                        z_W[m, arc[0]],
                        -1 * z_W[m, arc[1]],
                        1
                    ]),
                    sense=grb.GRB.LESS_EQUAL,
                    rhs=len(self.G_W.nodes) * grb.quicksum([
                        1, -1 * y_W[m, arc[0], arc[1]]
                    ])
                )
        # time, railway
        for m in self.L:
            for arc in self.G_R.edges:
                # vehicle arival
                model.addLConstr(
                    lhs=self.t_R[m, arc[1]],
                    sense=grb.GRB.GREATER_EQUAL,
                    rhs=grb.quicksum([
                        self.s_R[m, arc[0]],
                        self.G_R.edges[arc]['tau'],
                        -self.N * (1 - y_R[m, arc[0], arc[1]])
                    ])
                )
            for j in self.G_R.nodes:
                # same vehicle, with/without transshipmwnt
                model.addLConstr(
                    lhs=self.s_R[m, j],
                    sense=grb.GRB.GREATER_EQUAL,
                    rhs=grb.quicksum([
                        self.t_R[m, j]
                    ])
                )
        # time, waterway
        for m in self.B:
            for arc in self.G_W.edges:
                # vehicle arival
                model.addLConstr(
                    lhs=self.t_W[m, arc[1]],
                    sense=grb.GRB.GREATER_EQUAL,
                    rhs=grb.quicksum([
                        self.s_W[m, arc[0]],
                        self.G_W.edges[arc]['tau'],
                        -self.N * (1 - y_W[m, arc[0], arc[1]])
                    ])
                )
            for j in self.G_W.nodes:
                # same vehicle, with/without transshipmwnt
                model.addLConstr(
                    lhs=self.s_W[m, j],
                    sense=grb.GRB.GREATER_EQUAL,
                    rhs=grb.quicksum([
                        self.t_W[m, j]
                    ])
                )
        # coupling railway
        for m in self.L:
            for arc in self.G_R.edges:
                model.addLConstr(
                    lhs=grb.quicksum([
                        self.chi_R[p['i'], m, arc[0], arc[1]]
                        for p in self.P
                    ]),
                    sense=grb.GRB.LESS_EQUAL,
                    rhs=self.C_R * y_R[m, arc[0], arc[1]]
                )
                # no extra routes
                model.addLConstr(
                    lhs=self.C_R * grb.quicksum([
                        self.chi_R[p['i'], m, arc[0], arc[1]]
                        for p in self.P
                    ]),
                    sense=grb.GRB.GREATER_EQUAL,
                    rhs=y_R[m, arc[0], arc[1]]
                )
        # coupling waterway
        for m in self.B:
            for arc in self.G_W.edges:
                model.addLConstr(
                    lhs=grb.quicksum([
                        self.chi_W[p['i'], m, arc[0], arc[1]]
                        for p in self.P
                    ]),
                    sense=grb.GRB.LESS_EQUAL,
                    rhs=self.C_W * y_W[m, arc[0], arc[1]]
                )
                model.addLConstr(
                    lhs=self.C_W * grb.quicksum([
                        self.chi_W[p['i'], m, arc[0], arc[1]]
                        for p in self.P
                    ]),
                    sense=grb.GRB.GREATER_EQUAL,
                    rhs=y_W[m, arc[0], arc[1]]
                )
        model.update()
        return model

    def __build_LP(self, p):
        """
        Build LP for p
        """
        # model
        model = grb.Model(self.name)
        model.setParam("OutputFlag", False)
        # model.setParam("MIPGap", 1e-4)
        # model.setParam("Presolve", -1)
        # model.setParam("MIPFocus", -1)
        # model.setParam("DualReductions", 1)
        # model.setParam("IntFeasTol", 1e-5)
        # model.setParam("NumericFocus", 0)
        model.setParam("TimeLimit", 3600)
        # variables, order flow
        x_H, x_R, x_W = {}, {}, {}
        # highway
        for arc in self.G_H.edges:
            x_H[p['i'], arc[0], arc[1]] = model.addVar(
                lb=0, ub=1,
                vtype=grb.GRB.CONTINUOUS,
                name=f"x^H_{p['i']}_{arc[0]}_{arc[1]}"
            )
        # railway
        for m in self.L:
            for arc in self.G_R.edges:
                x_R[p['i'], m, arc[0], arc[1]] = model.addVar(
                    lb=0, ub=1,
                    vtype=grb.GRB.CONTINUOUS,
                    name=f"x^R_{p['i']}_{m}_{arc[0]}_{arc[1]}"
                )
        # waterway
        for m in self.B:
            for arc in self.G_W.edges:
                x_W[p['i'], m, arc[0], arc[1]] = model.addVar(
                    lb=0, ub=1,
                    vtype=grb.GRB.CONTINUOUS,
                    name=f"x^W_{p['i']}_{m}_{arc[0]}_{arc[1]}"
                )
        # transshipment
        u_R, u_W = {}, {}
        # railway
        for m in self.L:
            for i in self.G_R.nodes:
                for n in self.L:
                    if m != n:
                        u_R[p['i'], m, n, i] = model.addVar(
                            lb=0, ub=1,
                            vtype=grb.GRB.CONTINUOUS,
                            name=f"u^R_{p['i']}_{m}_{n}_{i}"
                        )
        # waterway
        for m in self.B:
            for i in self.G_W.nodes:
                for n in self.B:
                    if m != n:
                        u_W[p['i'], m, n, i] = model.addVar(
                            lb=0, ub=1,
                            vtype=grb.GRB.CONTINUOUS,
                            name=f"u^W_{p['i']}_{m}_{n}_{i}"
                        )
        # time
        t = {}
        # highway
        for i in self.G_H.nodes:
            t[p['i'], i] = model.addVar(
                lb=0, ub=grb.GRB.INFINITY,
                vtype=grb.GRB.CONTINUOUS,
                name=f"t_{p['i']}_{i}"
            )
        # late time, for cost
        T = {}
        T[p['i']] = model.addVar(
            lb=0, ub=grb.GRB.INFINITY,
            vtype=grb.GRB.CONTINUOUS,
            name=f"T_{p['i']}"
        )
        model.update()
        # objective - with cost
        # highway operation cost
        operation_H = grb.quicksum([
            self.c_H * self.G_H.edges[arc]['tau']
            * p['v'] * x_H[p['i'], arc[0], arc[1]]
            for arc in self.G_H.edges
        ])
        # late penalty
        late_penalty = grb.quicksum([
            self.c_P * T[p['i']]
        ])
        # Carbon Tax, highway
        GHG_H = grb.quicksum([
            self.sigma_H * x_H[p['i'], arc[0], arc[1]]
            * self.G_H.edges[arc]['d'] * p['v']
            for arc in self.G_H.edges
        ])
        # Carbon Tax, railway
        GHG_R = grb.quicksum([
            self.sigma_R * x_R[p['i'], m, arc[0], arc[1]]
            * self.G_R.edges[arc]['d'] * p['w']
            for arc in self.G_R.edges
            for m in self.L
        ])
        # Carbon Tax, waterway
        GHG_W = grb.quicksum([
            self.sigma_W * x_W[p['i'], m, arc[0], arc[1]]
            * self.G_W.edges[arc]['d'] * p['w']
            for arc in self.G_W.edges
            for m in self.B
        ])
        self.obj[p['i']] = grb.quicksum([
            operation_H, late_penalty, GHG_H, GHG_R, GHG_W
        ])
        model.setObjective(self.obj[p['i']], grb.GRB.MINIMIZE)
        # constraints
        # demand, origin
        model.addLConstr(
            lhs=grb.quicksum([
                grb.quicksum([
                    x_H[p['i'], p['o'], j]
                    for j in self.G_R.nodes
                ]),
                grb.quicksum([
                    x_H[p['i'], p['o'], j]
                    for j in self.G_W.nodes
                ]),
                x_H[p['i'], p['o'], p['d']]
            ]),
            sense=grb.GRB.EQUAL,
            rhs=1
        )
        # demand, destination
        model.addLConstr(
            lhs=grb.quicksum([
                grb.quicksum([
                    x_H[p['i'], i, p['d']]
                    for i in self.G_R.nodes
                ]),
                grb.quicksum([
                    x_H[p['i'], i, p['d']]
                    for i in self.G_W.nodes
                ]),
                x_H[p['i'], p['o'], p['d']]
            ]),
            sense=grb.GRB.EQUAL,
            rhs=1
        )
        # railway order flow
        for j in self.G_R.nodes:
            model.addLConstr(
                lhs=grb.quicksum([
                    # highway
                    x_H[p['i'], p['o'], j],
                    # waterway
                    grb.quicksum([
                        x_H[p['i'], i, j]
                        for i in self.G_W.nodes
                    ]),
                    # railway
                    grb.quicksum([
                        x_R[p['i'], m, i, j]
                        for m in self.L
                        for i in self.G_R.nodes
                        if (i, j) in self.G_R.edges
                    ])
                ]),
                sense=grb.GRB.EQUAL,
                rhs=grb.quicksum([
                    # highway
                    x_H[p['i'], j, p['d']],
                    # waterway
                    grb.quicksum([
                        x_H[p['i'], j, k]
                        for k in self.G_W.nodes
                    ]),
                    # railway
                    grb.quicksum([
                        x_R[p['i'], m, j, k]
                        for m in self.L
                        for k in self.G_R.nodes
                        if (j, k) in self.G_R.edges
                    ])
                ])
            )
            # one way out
            model.addLConstr(
                lhs=grb.quicksum([
                    # highway
                    x_H[p['i'], j, p['d']],
                    # waterway
                    grb.quicksum([
                        x_H[p['i'], j, k]
                        for k in self.G_W.nodes
                    ]),
                    # railway
                    grb.quicksum([
                        x_R[p['i'], m, j, k]
                        for m in self.L
                        for k in self.G_R.nodes
                        if (j, k) in self.G_R.edges
                    ])
                ]),
                sense=grb.GRB.LESS_EQUAL,
                rhs=1
            )
        # waterway order flow
        for j in self.G_W.nodes:
            model.addLConstr(
                lhs=grb.quicksum([
                    # highway
                    x_H[p['i'], p['o'], j],
                    # railway
                    grb.quicksum([
                        x_H[p['i'], i, j]
                        for i in self.G_R.nodes
                    ]),
                    # waterway
                    grb.quicksum([
                        x_W[p['i'], m, i, j]
                        for m in self.B
                        for i in self.G_W.nodes
                        if (i, j) in self.G_W.edges
                    ])
                ]),
                sense=grb.GRB.EQUAL,
                rhs=grb.quicksum([
                    # highway
                    x_H[p['i'], j, p['d']],
                    # railway
                    grb.quicksum([
                        x_H[p['i'], j, k]
                        for k in self.G_R.nodes
                    ]),
                    # waterway
                    grb.quicksum([
                        x_W[p['i'], m, j, k]
                        for m in self.B
                        for k in self.G_W.nodes
                        if (j, k) in self.G_W.edges
                    ])
                ])
            )
            # one way out
            model.addLConstr(
                lhs=grb.quicksum([
                    # highway
                    x_H[p['i'], j, p['d']],
                    # railway
                    grb.quicksum([
                        x_H[p['i'], j, k]
                        for k in self.G_R.nodes
                    ]),
                    # waterway
                    grb.quicksum([
                        x_W[p['i'], m, j, k]
                        for m in self.B
                        for k in self.G_W.nodes
                        if (j, k) in self.G_W.edges
                    ])
                ]),
                sense=grb.GRB.LESS_EQUAL,
                rhs=1
            )
        # coupling railway
        for m in self.L:
            for arc in self.G_R.edges:
                self.chi_R_LP[p['i'], m, arc[0], arc[1]] = model.addLConstr(
                    lhs=p['v'] * x_R[p['i'], m, arc[0], arc[1]],
                    sense=grb.GRB.LESS_EQUAL,
                    # initial build, update before use
                    rhs=0
                )
        # coupling waterway
        for m in self.B:
            for arc in self.G_W.edges:
                self.chi_W_LP[p['i'], m, arc[0], arc[1]] = model.addLConstr(
                    lhs=p['v'] * x_W[p['i'], m, arc[0], arc[1]],
                    sense=grb.GRB.LESS_EQUAL,
                    # initial build, update before use
                    rhs=0
                )
        # railway transshipment & sorting
        for j in self.G_R.nodes:
            for m in self.L:
                for n in self.L:
                    if m != n:
                        model.addLConstr(
                            lhs=u_R[p['i'], m, n, j],
                            sense=grb.GRB.GREATER_EQUAL,
                            rhs=grb.quicksum([
                                grb.quicksum([
                                    x_R[p['i'], m, i, j]
                                    for i in self.G_R.predecessors(j)
                                ]),
                                grb.quicksum([
                                    x_R[p['i'], n, j, k]
                                    for k in self.G_R.successors(j)
                                ]),
                                -1 * (3 / 2)
                            ])
                        )
                        model.addLConstr(
                            lhs=u_R[p['i'], m, n, j],
                            sense=grb.GRB.LESS_EQUAL,
                            rhs=(1 / 2) * grb.quicksum([
                                grb.quicksum([
                                    x_R[p['i'], m, i, j]
                                    for i in self.G_R.predecessors(j)
                                ]),
                                grb.quicksum([
                                    x_R[p['i'], n, j, k]
                                    for k in self.G_R.successors(j)
                                ])
                            ])
                        )
        # waterway transshipment & sorting
        for j in self.G_W.nodes:
            for m in self.B:
                for n in self.B:
                    if m != n:
                        model.addLConstr(
                            lhs=u_W[p['i'], m, n, j],
                            sense=grb.GRB.GREATER_EQUAL,
                            rhs=grb.quicksum([
                                grb.quicksum([
                                    x_W[p['i'], m, i, j]
                                    for i in self.G_W.predecessors(j)
                                ]),
                                grb.quicksum([
                                    x_W[p['i'], n, j, k]
                                    for k in self.G_W.successors(j)
                                ]),
                                -1 * (3 / 2)
                            ])
                        )
                        model.addLConstr(
                            lhs=u_W[p['i'], m, n, j],
                            sense=grb.GRB.LESS_EQUAL,
                            rhs=(1 / 2) * grb.quicksum([
                                grb.quicksum([
                                    x_W[p['i'], m, i, j]
                                    for i in self.G_W.predecessors(j)
                                ]),
                                grb.quicksum([
                                    x_W[p['i'], n, j, k]
                                    for k in self.G_W.successors(j)
                                ])
                            ])
                        )
        # order arrival time at railway, initial build, update before use
        for m in self.L:
            for j in self.G_R.nodes:
                self.order_arival_R_LP[p['i'], m, j] = model.addLConstr(
                    lhs=t[p['i'], j] - self.N * grb.quicksum([
                        x_R[p['i'], m, i, j]
                        for i in self.G_R.predecessors(j)
                    ]),
                    sense=grb.GRB.GREATER_EQUAL,
                    rhs=0
                )
        # order arrival time at waterway, initial build, update before use
        for m in self.B:
            for j in self.G_W.nodes:
                self.order_arival_W_LP[p['i'], m, j] = model.addLConstr(
                    lhs=t[p['i'], j] - self.N * grb.quicksum([
                        x_W[p['i'], m, i, j]
                        for i in self.G_W.predecessors(j)
                    ]),
                    sense=grb.GRB.GREATER_EQUAL,
                    rhs=0
                )
        # departure time, railway, initial build, update before use
        for m in self.L:
            for j in self.G_R.nodes:
                # from waterway
                for i in self.G_W.nodes:
                    self.depart_W_R_LP[p['i'], m, j, i] = model.addLConstr(
                        lhs=grb.quicksum([
                            t[p['i'], i],
                            self.N * x_H[p['i'], i, j]
                        ]),
                        sense=grb.GRB.LESS_EQUAL,
                        rhs=0
                    )
                # from origin
                self.depart_O_R_LP[p['i'], m, j] = model.addLConstr(
                    lhs=grb.quicksum([
                        t[p['i'], p['o']],
                        self.N * x_H[p['i'], p['o'], j]
                    ]),
                    sense=grb.GRB.LESS_EQUAL,
                    rhs=0
                )
                # transshipment & sorting, initial build, update before use
                for n in self.L:
                    if n != m:
                        self.trans_m_n_R_LP[
                            p['i'], m, j, n
                        ] = model.addLConstr(
                            lhs=self.N * u_R[p['i'], m, n, j],
                            sense=grb.GRB.LESS_EQUAL,
                            rhs=0
                        )
                        self.trans_m_m_R_LP[
                            p['i'], m, j, n
                        ] = model.addLConstr(
                            lhs=self.N * u_R[p['i'], m, n, j],
                            sense=grb.GRB.LESS_EQUAL,
                            rhs=0
                        )
                        self.trans_n_n_R_LP[
                            p['i'], m, j, n
                        ] = model.addLConstr(
                            lhs=self.N * u_R[p['i'], m, n, j],
                            sense=grb.GRB.LESS_EQUAL,
                            rhs=0
                        )
                        self.trans_n_m_R_LP[
                            p['i'], m, j, n
                        ] = model.addLConstr(
                            lhs=self.N * u_R[p['i'], m, n, j],
                            sense=grb.GRB.LESS_EQUAL,
                            rhs=0
                        )
        # departure time, waterway
        for m in self.B:
            for j in self.G_W.nodes:
                # from railway
                for i in self.G_R.nodes:
                    self.depart_R_W_LP[p['i'], m, j, i] = model.addLConstr(
                        lhs=grb.quicksum([
                            t[p['i'], i],
                            self.N * x_H[p['i'], i, j]
                        ]),
                        sense=grb.GRB.LESS_EQUAL,
                        rhs=0
                    )
                # from origin
                self.depart_O_W_LP[p['i'], m, j] = model.addLConstr(
                    lhs=grb.quicksum([
                        t[p['i'], p['o']],
                        self.N * x_H[p['i'], p['o'], j]
                    ]),
                    sense=grb.GRB.LESS_EQUAL,
                    rhs=0
                )
                # transshipment & sorting, initial build, update before use
                for n in self.B:
                    if n != m:
                        self.trans_m_n_W_LP[
                            p['i'], m, j, n
                        ] = model.addLConstr(
                            lhs=self.N * u_W[p['i'], m, n, j],
                            sense=grb.GRB.LESS_EQUAL,
                            rhs=0
                        )
                        self.trans_m_m_W_LP[
                            p['i'], m, j, n
                        ] = model.addLConstr(
                            lhs=self.N * u_W[p['i'], m, n, j],
                            sense=grb.GRB.LESS_EQUAL,
                            rhs=0
                        )
                        self.trans_n_n_W_LP[
                            p['i'], m, j, n
                        ] = model.addLConstr(
                            lhs=self.N * u_W[p['i'], m, n, j],
                            sense=grb.GRB.LESS_EQUAL,
                            rhs=0
                        )
                        self.trans_n_m_W_LP[
                            p['i'], m, j, n
                        ] = model.addLConstr(
                            lhs=self.N * u_W[p['i'], m, n, j],
                            sense=grb.GRB.LESS_EQUAL,
                            rhs=0
                        )
        # order arrival time
        for arc in self.G_H.edges:
            if arc[1] != p['o']:
                model.addLConstr(
                    lhs=t[p['i'], arc[1]],
                    sense=grb.GRB.GREATER_EQUAL,
                    rhs=grb.quicksum([
                        t[p['i'], arc[0]],
                        self.G_H.edges[(arc[0], arc[1])]['tau'],
                        -self.N * (1 - x_H[p['i'], arc[0], arc[1]])
                    ])
                )
        # late penalty, for cost
        model.addLConstr(
            lhs=T[p['i']],
            sense=grb.GRB.GREATER_EQUAL,
            rhs=t[p['i'], p['d']] - p['T']
        )
        model.update()
        return model

    def __solve_LP(self, p):
        """
        Solve LP for p
        """
        # rail, chi
        for m in self.L:
            for arc in self.G_R.edges:
                self.chi_R_LP[p['i'], m, arc[0], arc[1]].setAttr(
                    "RHS", self.chi_R[p['i'], m, arc[0], arc[1]].X
                )
        # water, chi
        for m in self.B:
            for arc in self.G_W.edges:
                self.chi_W_LP[p['i'], m, arc[0], arc[1]].setAttr(
                    "RHS", self.chi_W[p['i'], m, arc[0], arc[1]].X
                )
        # order arrival time at railway
        for m in self.L:
            for j in self.G_R.nodes:
                self.order_arival_R_LP[p['i'], m, j].setAttr(
                    "RHS", self.t_R[m, j].X - self.N
                )
        # order arrival time at waterway
        for m in self.B:
            for j in self.G_W.nodes:
                self.order_arival_W_LP[p['i'], m, j].setAttr(
                    "RHS", self.t_W[m, j].X - self.N
                )
        # departure time, railway
        for m in self.L:
            for j in self.G_R.nodes:
                # from waterway
                for i in self.G_W.nodes:
                    self.depart_W_R_LP[p['i'], m, j, i].setAttr(
                        "RHS", np.sum([
                            self.s_R[m, j].X,
                            -self.G_H.edges[(i, j)]['tau'],
                            self.N
                        ])
                    )
                # from origin
                self.depart_O_R_LP[p['i'], m, j].setAttr(
                    "RHS", np.sum([
                        self.s_R[m, j].X,
                        -self.G_H.edges[(p['o'], j)]['tau'],
                        self.N
                    ])
                )
                # transshipment & sorting
                for n in self.L:
                    if n != m:
                        self.trans_m_n_R_LP[p['i'], m, j, n].setAttr(
                            "RHS", np.sum([
                                self.s_R[m, j].X - self.t_R[n, j].X,
                                -self.delta_R + self.N
                            ])
                        )
                        self.trans_m_m_R_LP[p['i'], m, j, n].setAttr(
                            "RHS", np.sum([
                                self.s_R[m, j].X - self.t_R[m, j].X,
                                -self.delta_R + self.N
                            ])
                        )
                        self.trans_n_n_R_LP[p['i'], m, j, n].setAttr(
                            "RHS", np.sum([
                                self.s_R[n, j].X - self.t_R[n, j].X,
                                -self.delta_R + self.N
                            ])
                        )
                        self.trans_n_m_R_LP[p['i'], m, j, n].setAttr(
                            "RHS", np.sum([
                                self.s_R[n, j].X - self.t_R[m, j].X,
                                -self.delta_R + self.N
                            ])
                        )
        # departure time, waterway
        for m in self.B:
            for j in self.G_W.nodes:
                # from railway
                for i in self.G_R.nodes:
                    self.depart_R_W_LP[p['i'], m, j, i].setAttr(
                        "RHS", np.sum([
                            self.s_W[m, j].X,
                            -self.G_H.edges[(i, j)]['tau'],
                            self.N
                        ])
                    )
                # from origin
                self.depart_O_W_LP[p['i'], m, j].setAttr(
                    "RHS", np.sum([
                        self.s_W[m, j].X,
                        -self.G_H.edges[(p['o'], j)]['tau'],
                        self.N
                    ])
                )
                # transshipment & sorting
                for n in self.B:
                    if n != m:
                        self.trans_m_n_W_LP[p['i'], m, j, n].setAttr(
                            "RHS", np.sum([
                                self.s_W[m, j].X - self.t_W[n, j].X,
                                -self.delta_W + self.N
                            ])
                        )
                        self.trans_m_m_W_LP[p['i'], m, j, n].setAttr(
                            "RHS", np.sum([
                                self.s_W[m, j].X - self.t_W[m, j].X,
                                -self.delta_W + self.N
                            ])
                        )
                        self.trans_n_n_W_LP[p['i'], m, j, n].setAttr(
                            "RHS", np.sum([
                                self.s_W[n, j].X - self.t_W[n, j].X,
                                -self.delta_W + self.N
                            ])
                        )
                        self.trans_n_m_W_LP[p['i'], m, j, n].setAttr(
                            "RHS", np.sum([
                                self.s_W[n, j].X - self.t_W[m, j].X,
                                -self.delta_W + self.N
                            ])
                        )
        # update
        self.LP[p['i']].update()
        # solve
        self.LP[p['i']].optimize()
        return

    def __build_SP(self, p):
        """
        Build SP for p
        """
        # model
        model = grb.Model(self.name)
        model.setParam("OutputFlag", False)
        # model.setParam("MIPGap", 1e-4)
        # model.setParam("Presolve", -1)
        # model.setParam("MIPFocus", -1)
        # model.setParam("DualReductions", 1)
        # model.setParam("IntFeasTol", 1e-5)
        # model.setParam("NumericFocus", 0)
        model.setParam("TimeLimit", 3600)
        # variables, order flow
        x_H, x_R, x_W = {}, {}, {}
        # highway
        for arc in self.G_H.edges:
            x_H[p['i'], arc[0], arc[1]] = model.addVar(
                lb=0, ub=1,
                vtype=grb.GRB.BINARY,
                name=f"x^H_{p['i']}_{arc[0]}_{arc[1]}"
            )
        # railway
        for m in self.L:
            for arc in self.G_R.edges:
                x_R[p['i'], m, arc[0], arc[1]] = model.addVar(
                    lb=0, ub=1,
                    vtype=grb.GRB.BINARY,
                    name=f"x^R_{p['i']}_{m}_{arc[0]}_{arc[1]}"
                )
        # waterway
        for m in self.B:
            for arc in self.G_W.edges:
                x_W[p['i'], m, arc[0], arc[1]] = model.addVar(
                    lb=0, ub=1,
                    vtype=grb.GRB.BINARY,
                    name=f"x^W_{p['i']}_{m}_{arc[0]}_{arc[1]}"
                )
        # transshipment
        u_R, u_W = {}, {}
        # railway
        for m in self.L:
            for i in self.G_R.nodes:
                for n in self.L:
                    if m != n:
                        u_R[p['i'], m, n, i] = model.addVar(
                            lb=0, ub=1,
                            vtype=grb.GRB.BINARY,
                            name=f"u^R_{p['i']}_{m}_{n}_{i}"
                        )
        # waterway
        for m in self.B:
            for i in self.G_W.nodes:
                for n in self.B:
                    if m != n:
                        u_W[p['i'], m, n, i] = model.addVar(
                            lb=0, ub=1,
                            vtype=grb.GRB.BINARY,
                            name=f"u^W_{p['i']}_{m}_{n}_{i}"
                        )
        # time
        t = {}
        # highway
        for i in self.G_H.nodes:
            t[p['i'], i] = model.addVar(
                lb=0, ub=grb.GRB.INFINITY,
                vtype=grb.GRB.CONTINUOUS,
                name=f"t_{p['i']}_{i}"
            )
        # late time, for cost
        T = {}
        T[p['i']] = model.addVar(
            lb=0, ub=grb.GRB.INFINITY,
            vtype=grb.GRB.CONTINUOUS,
            name=f"T_{p['i']}"
        )
        model.update()
        # objective - with cost
        # highway operation cost
        operation_H = grb.quicksum([
            self.c_H * self.G_H.edges[arc]['tau']
            * p['v'] * x_H[p['i'], arc[0], arc[1]]
            for arc in self.G_H.edges
        ])
        # late penalty
        late_penalty = grb.quicksum([
            self.c_P * T[p['i']]
        ])
        # Carbon Tax, highway
        GHG_H = grb.quicksum([
            self.sigma_H * x_H[p['i'], arc[0], arc[1]]
            * self.G_H.edges[arc]['d'] * p['v']
            for arc in self.G_H.edges
        ])
        # Carbon Tax, railway
        GHG_R = grb.quicksum([
            self.sigma_R * x_R[p['i'], m, arc[0], arc[1]]
            * self.G_R.edges[arc]['d'] * p['w']
            for arc in self.G_R.edges
            for m in self.L
        ])
        # Carbon Tax, waterway
        GHG_W = grb.quicksum([
            self.sigma_W * x_W[p['i'], m, arc[0], arc[1]]
            * self.G_W.edges[arc]['d'] * p['w']
            for arc in self.G_W.edges
            for m in self.B
        ])
        self.obj[p['i']] = grb.quicksum([
            operation_H, late_penalty, GHG_H, GHG_R, GHG_W
        ])
        model.setObjective(self.obj[p['i']], grb.GRB.MINIMIZE)
        # constraints
        # demand, origin
        model.addLConstr(
            lhs=grb.quicksum([
                grb.quicksum([
                    x_H[p['i'], p['o'], j]
                    for j in self.G_R.nodes
                ]),
                grb.quicksum([
                    x_H[p['i'], p['o'], j]
                    for j in self.G_W.nodes
                ]),
                x_H[p['i'], p['o'], p['d']]
            ]),
            sense=grb.GRB.EQUAL,
            rhs=1
        )
        # demand, destination
        model.addLConstr(
            lhs=grb.quicksum([
                grb.quicksum([
                    x_H[p['i'], i, p['d']]
                    for i in self.G_R.nodes
                ]),
                grb.quicksum([
                    x_H[p['i'], i, p['d']]
                    for i in self.G_W.nodes
                ]),
                x_H[p['i'], p['o'], p['d']]
            ]),
            sense=grb.GRB.EQUAL,
            rhs=1
        )
        # railway order flow
        for j in self.G_R.nodes:
            model.addLConstr(
                lhs=grb.quicksum([
                    # highway
                    x_H[p['i'], p['o'], j],
                    # waterway
                    grb.quicksum([
                        x_H[p['i'], i, j]
                        for i in self.G_W.nodes
                    ]),
                    # railway
                    grb.quicksum([
                        x_R[p['i'], m, i, j]
                        for m in self.L
                        for i in self.G_R.nodes
                        if (i, j) in self.G_R.edges
                    ])
                ]),
                sense=grb.GRB.EQUAL,
                rhs=grb.quicksum([
                    # highway
                    x_H[p['i'], j, p['d']],
                    # waterway
                    grb.quicksum([
                        x_H[p['i'], j, k]
                        for k in self.G_W.nodes
                    ]),
                    # railway
                    grb.quicksum([
                        x_R[p['i'], m, j, k]
                        for m in self.L
                        for k in self.G_R.nodes
                        if (j, k) in self.G_R.edges
                    ])
                ])
            )
            # one way out
            model.addLConstr(
                lhs=grb.quicksum([
                    # highway
                    x_H[p['i'], j, p['d']],
                    # waterway
                    grb.quicksum([
                        x_H[p['i'], j, k]
                        for k in self.G_W.nodes
                    ]),
                    # railway
                    grb.quicksum([
                        x_R[p['i'], m, j, k]
                        for m in self.L
                        for k in self.G_R.nodes
                        if (j, k) in self.G_R.edges
                    ])
                ]),
                sense=grb.GRB.LESS_EQUAL,
                rhs=1
            )
        # waterway order flow
        for j in self.G_W.nodes:
            model.addLConstr(
                lhs=grb.quicksum([
                    # highway
                    x_H[p['i'], p['o'], j],
                    # railway
                    grb.quicksum([
                        x_H[p['i'], i, j]
                        for i in self.G_R.nodes
                    ]),
                    # waterway
                    grb.quicksum([
                        x_W[p['i'], m, i, j]
                        for m in self.B
                        for i in self.G_W.nodes
                        if (i, j) in self.G_W.edges
                    ])
                ]),
                sense=grb.GRB.EQUAL,
                rhs=grb.quicksum([
                    # highway
                    x_H[p['i'], j, p['d']],
                    # railway
                    grb.quicksum([
                        x_H[p['i'], j, k]
                        for k in self.G_R.nodes
                    ]),
                    # waterway
                    grb.quicksum([
                        x_W[p['i'], m, j, k]
                        for m in self.B
                        for k in self.G_W.nodes
                        if (j, k) in self.G_W.edges
                    ])
                ])
            )
            # one way out
            model.addLConstr(
                lhs=grb.quicksum([
                    # highway
                    x_H[p['i'], j, p['d']],
                    # railway
                    grb.quicksum([
                        x_H[p['i'], j, k]
                        for k in self.G_R.nodes
                    ]),
                    # waterway
                    grb.quicksum([
                        x_W[p['i'], m, j, k]
                        for m in self.B
                        for k in self.G_W.nodes
                        if (j, k) in self.G_W.edges
                    ])
                ]),
                sense=grb.GRB.LESS_EQUAL,
                rhs=1
            )
        # coupling railway
        for m in self.L:
            for arc in self.G_R.edges:
                self.chi_R_SP[p['i'], m, arc[0], arc[1]] = model.addLConstr(
                    lhs=p['v'] * x_R[p['i'], m, arc[0], arc[1]],
                    sense=grb.GRB.LESS_EQUAL,
                    # initial build, update before use
                    rhs=0
                )
        # coupling waterway
        for m in self.B:
            for arc in self.G_W.edges:
                self.chi_W_SP[p['i'], m, arc[0], arc[1]] = model.addLConstr(
                    lhs=p['v'] * x_W[p['i'], m, arc[0], arc[1]],
                    sense=grb.GRB.LESS_EQUAL,
                    # initial build, update before use
                    rhs=0
                )
        # railway transshipment & sorting
        for j in self.G_R.nodes:
            for m in self.L:
                for n in self.L:
                    if m != n:
                        model.addLConstr(
                            lhs=u_R[p['i'], m, n, j],
                            sense=grb.GRB.GREATER_EQUAL,
                            rhs=grb.quicksum([
                                grb.quicksum([
                                    x_R[p['i'], m, i, j]
                                    for i in self.G_R.predecessors(j)
                                ]),
                                grb.quicksum([
                                    x_R[p['i'], n, j, k]
                                    for k in self.G_R.successors(j)
                                ]),
                                -1 * (3 / 2)
                            ])
                        )
                        model.addLConstr(
                            lhs=u_R[p['i'], m, n, j],
                            sense=grb.GRB.LESS_EQUAL,
                            rhs=(1 / 2) * grb.quicksum([
                                grb.quicksum([
                                    x_R[p['i'], m, i, j]
                                    for i in self.G_R.predecessors(j)
                                ]),
                                grb.quicksum([
                                    x_R[p['i'], n, j, k]
                                    for k in self.G_R.successors(j)
                                ])
                            ])
                        )
        # waterway transshipment & sorting
        for j in self.G_W.nodes:
            for m in self.B:
                for n in self.B:
                    if m != n:
                        model.addLConstr(
                            lhs=u_W[p['i'], m, n, j],
                            sense=grb.GRB.GREATER_EQUAL,
                            rhs=grb.quicksum([
                                grb.quicksum([
                                    x_W[p['i'], m, i, j]
                                    for i in self.G_W.predecessors(j)
                                ]),
                                grb.quicksum([
                                    x_W[p['i'], n, j, k]
                                    for k in self.G_W.successors(j)
                                ]),
                                -1 * (3 / 2)
                            ])
                        )
                        model.addLConstr(
                            lhs=u_W[p['i'], m, n, j],
                            sense=grb.GRB.LESS_EQUAL,
                            rhs=(1 / 2) * grb.quicksum([
                                grb.quicksum([
                                    x_W[p['i'], m, i, j]
                                    for i in self.G_W.predecessors(j)
                                ]),
                                grb.quicksum([
                                    x_W[p['i'], n, j, k]
                                    for k in self.G_W.successors(j)
                                ])
                            ])
                        )
        # order arrival time at railway, initial build, update before use
        for m in self.L:
            for j in self.G_R.nodes:
                self.order_arival_R_SP[p['i'], m, j] = model.addLConstr(
                    lhs=t[p['i'], j] - self.N * grb.quicksum([
                        x_R[p['i'], m, i, j]
                        for i in self.G_R.predecessors(j)
                    ]),
                    sense=grb.GRB.GREATER_EQUAL,
                    rhs=0
                )
        # order arrival time at waterway, initial build, update before use
        for m in self.B:
            for j in self.G_W.nodes:
                self.order_arival_W_SP[p['i'], m, j] = model.addLConstr(
                    lhs=t[p['i'], j] - self.N * grb.quicksum([
                        x_W[p['i'], m, i, j]
                        for i in self.G_W.predecessors(j)
                    ]),
                    sense=grb.GRB.GREATER_EQUAL,
                    rhs=0
                )
        # departure time, railway, initial build, update before use
        for m in self.L:
            for j in self.G_R.nodes:
                # from waterway
                for i in self.G_W.nodes:
                    self.depart_W_R_SP[p['i'], m, j, i] = model.addLConstr(
                        lhs=grb.quicksum([
                            t[p['i'], i],
                            self.N * x_H[p['i'], i, j]
                        ]),
                        sense=grb.GRB.LESS_EQUAL,
                        rhs=0
                    )
                # from origin
                self.depart_O_R_SP[p['i'], m, j] = model.addLConstr(
                    lhs=grb.quicksum([
                        t[p['i'], p['o']],
                        self.N * x_H[p['i'], p['o'], j]
                    ]),
                    sense=grb.GRB.LESS_EQUAL,
                    rhs=0
                )
                # transshipment & sorting, initial build, update before use
                for n in self.L:
                    if n != m:
                        self.trans_m_n_R_SP[
                            p['i'], m, j, n
                        ] = model.addLConstr(
                            lhs=self.N * u_R[p['i'], m, n, j],
                            sense=grb.GRB.LESS_EQUAL,
                            rhs=0
                        )
                        self.trans_m_m_R_SP[
                            p['i'], m, j, n
                        ] = model.addLConstr(
                            lhs=self.N * u_R[p['i'], m, n, j],
                            sense=grb.GRB.LESS_EQUAL,
                            rhs=0
                        )
                        self.trans_n_n_R_SP[
                            p['i'], m, j, n
                        ] = model.addLConstr(
                            lhs=self.N * u_R[p['i'], m, n, j],
                            sense=grb.GRB.LESS_EQUAL,
                            rhs=0
                        )
                        self.trans_n_m_R_SP[
                            p['i'], m, j, n
                        ] = model.addLConstr(
                            lhs=self.N * u_R[p['i'], m, n, j],
                            sense=grb.GRB.LESS_EQUAL,
                            rhs=0
                        )
        # departure time, waterway
        for m in self.B:
            for j in self.G_W.nodes:
                # from railway
                for i in self.G_R.nodes:
                    self.depart_R_W_SP[p['i'], m, j, i] = model.addLConstr(
                        lhs=grb.quicksum([
                            t[p['i'], i],
                            self.N * x_H[p['i'], i, j]
                        ]),
                        sense=grb.GRB.LESS_EQUAL,
                        rhs=0
                    )
                # from origin
                self.depart_O_W_SP[p['i'], m, j] = model.addLConstr(
                    lhs=grb.quicksum([
                        t[p['i'], p['o']],
                        self.N * x_H[p['i'], p['o'], j]
                    ]),
                    sense=grb.GRB.LESS_EQUAL,
                    rhs=0
                )
                # transshipment & sorting, initial build, update before use
                for n in self.B:
                    if n != m:
                        self.trans_m_n_W_SP[
                            p['i'], m, j, n
                        ] = model.addLConstr(
                            lhs=self.N * u_W[p['i'], m, n, j],
                            sense=grb.GRB.LESS_EQUAL,
                            rhs=0
                        )
                        self.trans_m_m_W_SP[
                            p['i'], m, j, n
                        ] = model.addLConstr(
                            lhs=self.N * u_W[p['i'], m, n, j],
                            sense=grb.GRB.LESS_EQUAL,
                            rhs=0
                        )
                        self.trans_n_n_W_SP[
                            p['i'], m, j, n
                        ] = model.addLConstr(
                            lhs=self.N * u_W[p['i'], m, n, j],
                            sense=grb.GRB.LESS_EQUAL,
                            rhs=0
                        )
                        self.trans_n_m_W_SP[
                            p['i'], m, j, n
                        ] = model.addLConstr(
                            lhs=self.N * u_W[p['i'], m, n, j],
                            sense=grb.GRB.LESS_EQUAL,
                            rhs=0
                        )
        # order arrival time
        for arc in self.G_H.edges:
            if arc[1] != p['o']:
                model.addLConstr(
                    lhs=t[p['i'], arc[1]],
                    sense=grb.GRB.GREATER_EQUAL,
                    rhs=grb.quicksum([
                        t[p['i'], arc[0]],
                        self.G_H.edges[(arc[0], arc[1])]['tau'],
                        -self.N * (1 - x_H[p['i'], arc[0], arc[1]])
                    ])
                )
        # late penalty, for cost
        model.addLConstr(
            lhs=T[p['i']],
            sense=grb.GRB.GREATER_EQUAL,
            rhs=t[p['i'], p['d']] - p['T']
        )
        model.update()
        return model

    def __solve_SP(self, p):
        """
        Build SP for p
        """
        # rail, chi
        for m in self.L:
            for arc in self.G_R.edges:
                self.chi_R_SP[p['i'], m, arc[0], arc[1]].setAttr(
                    "RHS", self.chi_R[p['i'], m, arc[0], arc[1]].X
                )
        # water, chi
        for m in self.B:
            for arc in self.G_W.edges:
                self.chi_W_SP[p['i'], m, arc[0], arc[1]].setAttr(
                    "RHS", self.chi_W[p['i'], m, arc[0], arc[1]].X
                )
        # order arrival time at railway
        for m in self.L:
            for j in self.G_R.nodes:
                self.order_arival_R_SP[p['i'], m, j].setAttr(
                    "RHS", self.t_R[m, j].X - self.N
                )
        # order arrival time at waterway
        for m in self.B:
            for j in self.G_W.nodes:
                self.order_arival_W_SP[p['i'], m, j].setAttr(
                    "RHS", self.t_W[m, j].X - self.N
                )
        # departure time, railway
        for m in self.L:
            for j in self.G_R.nodes:
                # from waterway
                for i in self.G_W.nodes:
                    self.depart_W_R_SP[p['i'], m, j, i].setAttr(
                        "RHS", np.sum([
                            self.s_R[m, j].X,
                            -self.G_H.edges[(i, j)]['tau'],
                            self.N
                        ])
                    )
                # from origin
                self.depart_O_R_SP[p['i'], m, j].setAttr(
                    "RHS", np.sum([
                        self.s_R[m, j].X,
                        -self.G_H.edges[(p['o'], j)]['tau'],
                        self.N
                    ])
                )
                # transshipment & sorting
                for n in self.L:
                    if n != m:
                        self.trans_m_n_R_SP[p['i'], m, j, n].setAttr(
                            "RHS", np.sum([
                                self.s_R[m, j].X - self.t_R[n, j].X,
                                -self.delta_R + self.N
                            ])
                        )
                        self.trans_m_m_R_SP[p['i'], m, j, n].setAttr(
                            "RHS", np.sum([
                                self.s_R[m, j].X - self.t_R[m, j].X,
                                -self.delta_R + self.N
                            ])
                        )
                        self.trans_n_n_R_SP[p['i'], m, j, n].setAttr(
                            "RHS", np.sum([
                                self.s_R[n, j].X - self.t_R[n, j].X,
                                -self.delta_R + self.N
                            ])
                        )
                        self.trans_n_m_R_SP[p['i'], m, j, n].setAttr(
                            "RHS", np.sum([
                                self.s_R[n, j].X - self.t_R[m, j].X,
                                -self.delta_R + self.N
                            ])
                        )
        # departure time, waterway
        for m in self.B:
            for j in self.G_W.nodes:
                # from railway
                for i in self.G_R.nodes:
                    self.depart_R_W_SP[p['i'], m, j, i].setAttr(
                        "RHS", np.sum([
                            self.s_W[m, j].X,
                            -self.G_H.edges[(i, j)]['tau'],
                            self.N
                        ])
                    )
                # from origin
                self.depart_O_W_SP[p['i'], m, j].setAttr(
                    "RHS", np.sum([
                        self.s_W[m, j].X,
                        -self.G_H.edges[(p['o'], j)]['tau'],
                        self.N
                    ])
                )
                # transshipment & sorting
                for n in self.B:
                    if n != m:
                        self.trans_m_n_W_SP[p['i'], m, j, n].setAttr(
                            "RHS", np.sum([
                                self.s_W[m, j].X - self.t_W[n, j].X,
                                -self.delta_W + self.N
                            ])
                        )
                        self.trans_m_m_W_SP[p['i'], m, j, n].setAttr(
                            "RHS", np.sum([
                                self.s_W[m, j].X - self.t_W[m, j].X,
                                -self.delta_W + self.N
                            ])
                        )
                        self.trans_n_n_W_SP[p['i'], m, j, n].setAttr(
                            "RHS", np.sum([
                                self.s_W[n, j].X - self.t_W[n, j].X,
                                -self.delta_W + self.N
                            ])
                        )
                        self.trans_n_m_W_SP[p['i'], m, j, n].setAttr(
                            "RHS", np.sum([
                                self.s_W[n, j].X - self.t_W[m, j].X,
                                -self.delta_W + self.N
                            ])
                        )
        # update
        self.SP[p['i']].update()
        # solve
        self.SP[p['i']].optimize()
        return

    def decomposition(self, epsilon=1e-5):
        """
        Decomposition
        """
        # build MP and SP
        self.MP = self.__build_MP()
        # build LP
        self.LP = {
            p['i']: self.__build_LP(p)
            for p in self.P
        }
        # build SP
        self.SP = {
            p['i']: self.__build_SP(p)
            for p in self.P
        }
        # loop
        iteration = 0
        while True:
            # solve MP
            self.MP.update()
            self.MP.optimize()
            logging.info("-----------")
            logging.info(iteration)
            logging.info(self.MP.ObjVal)
            logging.info(self.MP.Runtime)
            # print({
            #     (m, i): self.s_R[m, i].X
            #     for m in self.L
            #     for i in self.G_R.nodes
            # })
            # print({
            #     (m, i): self.t_R[m, i].X
            #     for m in self.L
            #     for i in self.G_R.nodes
            # })
            # print({
            #     (m, i): self.s_W[m, i].X
            #     for m in self.B
            #     for i in self.G_W.nodes
            # })
            # print({
            #     (m, i): self.t_W[m, i].X
            #     for m in self.B
            #     for i in self.G_W.nodes
            # })
            # for each order subproblem
            optimal = True
            for p in self.P:
                # update & solve LP
                self.__solve_LP(p)
                # update & solve SP
                self.__solve_SP(p)
                # check optimality
                logging.info(
                    f"{self.theta[p['i']].X}, {self.theta_prv[p['i']]}, "
                    f"{self.SP[p['i']].ObjVal}"
                )
                # if np.abs(
                #     self.theta[p['i']].X - self.theta_prv[p['i']]
                # ) >= epsilon and iteration <= 30:
                if np.abs(
                    self.theta[p['i']].X - self.theta_prv[p['i']]
                ) >= epsilon:
                    optimal = False
                    # add Benders cut
                    self.MP.addLConstr(
                        lhs=self.theta[p['i']],
                        sense=grb.GRB.GREATER_EQUAL,
                        rhs=grb.quicksum([
                            self.SP[p['i']].ObjVal,
                            # railway, chi
                            grb.quicksum([
                                self.chi_R_LP[p['i'], m, arc[0], arc[1]].Pi
                                * (
                                    self.chi_R[p['i'], m, arc[0], arc[1]]
                                    - self.chi_R[p['i'], m, arc[0], arc[1]].X
                                )
                                for m in self.L
                                for arc in self.G_R.edges
                            ]),
                            # waterway, chi
                            grb.quicksum([
                                self.chi_W_LP[p['i'], m, arc[0], arc[1]].Pi * (
                                    self.chi_W[p['i'], m, arc[0], arc[1]]
                                    - self.chi_W[p['i'], m, arc[0], arc[1]].X
                                )
                                for m in self.B
                                for arc in self.G_W.edges
                            ]),
                            # railway, arrival
                            grb.quicksum([
                                self.order_arival_R_LP[p['i'], m, j].Pi * (
                                    self.t_R[m, j] - self.t_R[m, j].X
                                )
                                for m in self.L
                                for j in self.G_R.nodes
                            ]),
                            # waterway, arrival
                            grb.quicksum([
                                self.order_arival_W_LP[p['i'], m, j].Pi * (
                                    self.t_W[m, j] - self.t_W[m, j].X
                                )
                                for m in self.B
                                for j in self.G_W.nodes
                            ]),
                            # railway, departure & trans
                            # water
                            grb.quicksum([
                                self.depart_W_R_LP[p['i'], m, j, i].Pi * (
                                    self.s_R[m, j] - self.s_R[m, j].X
                                )
                                for i in self.G_W.nodes
                                for m in self.L
                                for j in self.G_R.nodes
                            ]),
                            # origin
                            grb.quicksum([
                                self.depart_O_R_LP[p['i'], m, j].Pi * (
                                    self.s_R[m, j] - self.s_R[m, j].X
                                )
                                for m in self.L
                                for j in self.G_R.nodes
                            ]),
                            # transshipment
                            grb.quicksum([
                                grb.quicksum([
                                    self.trans_m_n_R_LP[p['i'], m, j, n].Pi * (
                                        self.s_R[m, j] - self.t_R[n, j]
                                        - self.s_R[m, j].X + self.t_R[n, j].X
                                    ),
                                    self.trans_m_m_R_LP[p['i'], m, j, n].Pi * (
                                        self.s_R[m, j] - self.t_R[m, j]
                                        - self.s_R[m, j].X + self.t_R[m, j].X
                                    ),
                                    self.trans_n_n_R_LP[p['i'], m, j, n].Pi * (
                                        self.s_R[n, j] - self.t_R[n, j]
                                        - self.s_R[n, j].X + self.t_R[n, j].X
                                    ),
                                    self.trans_n_m_R_LP[p['i'], m, j, n].Pi * (
                                        self.s_R[n, j] - self.t_R[m, j]
                                        - self.s_R[n, j].X + self.t_R[m, j].X
                                    )
                                ])
                                for m in self.L
                                for j in self.G_R.nodes
                                for n in self.L if n != m
                            ]),
                            # waterway, departure & trans
                            # rail
                            grb.quicksum([
                                self.depart_R_W_LP[p['i'], m, j, i].Pi * (
                                    self.s_W[m, j] - self.s_W[m, j].X
                                )
                                for m in self.B
                                for j in self.G_W.nodes
                                for i in self.G_R.nodes
                            ]),
                            # origin
                            grb.quicksum([
                                self.depart_O_W_LP[p['i'], m, j].Pi * (
                                    self.s_W[m, j] - self.s_W[m, j].X
                                )
                                for m in self.B
                                for j in self.G_W.nodes
                            ]),
                            # transshipment
                            grb.quicksum([
                                grb.quicksum([
                                    self.trans_m_n_W_LP[p['i'], m, j, n].Pi * (
                                        self.s_W[m, j] - self.t_W[n, j]
                                        - self.s_W[m, j].X + self.t_W[n, j].X
                                    ),
                                    self.trans_m_m_W_LP[p['i'], m, j, n].Pi * (
                                        self.s_W[m, j] - self.t_W[m, j]
                                        - self.s_W[m, j].X + self.t_W[m, j].X
                                    ),
                                    self.trans_n_n_W_LP[p['i'], m, j, n].Pi * (
                                        self.s_W[n, j] - self.t_W[n, j]
                                        - self.s_W[n, j].X + self.t_W[n, j].X
                                    ),
                                    self.trans_n_m_W_LP[p['i'], m, j, n].Pi * (
                                        self.s_W[n, j] - self.t_W[m, j]
                                        - self.s_W[n, j].X + self.t_W[m, j].X
                                    )
                                ])
                                for m in self.B
                                for j in self.G_W.nodes
                                for n in self.B if n != m
                            ])
                        ])
                    )
            # all order optimal
            if optimal:
                break
            else:
                # backup theta
                self.theta_prv = {
                    p['i']: self.theta[p['i']].X
                    for p in self.P
                }
                iteration += 1
        return
