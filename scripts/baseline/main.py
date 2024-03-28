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
import pickle
import numpy as np
import pandas as pd
import networkx as nx
import scipy.stats as st
from model import Intermodal


def haversine_dist(A, B):
    """haversine_dist, convert to mile"""
    # Earth radius in km
    R = 6371
    dLat, dLon = np.radians(B[0] - A[0]), np.radians(B[1] - A[1])
    lat1, lat2 = np.radians(A[0]), np.radians(B[0])
    a = np.sin(dLat/2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dLon/2) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    return R * c * 0.621371


def generate_order(V_H, n):
    """
    Generate orders
    """
    P = []
    all_pair = []
    for i in V_H:
        for j in V_H:
            if i != j:
                all_pair.append((i, j))
    chosen_ind = np.random.choice(
        range(len(all_pair)), n, replace=False
    )
    i = 0
    for ind in chosen_ind:
        pair = all_pair[ind]
        P.append({
            'i': i, 'o': pair[0], 'd': pair[1], 'T': 48,
            'v': np.random.choice(range(10, 81, 10))
        })
        # weight max 32.5 ton per container - UNP
        P[-1]['w'] = np.sum(st.uniform.rvs(
            loc=32.5 * 0.3, scale=32.5 * 0.7, size=P[-1]['v']
        ))
        i += 1
    return P


def generate_instance(name, network, P):
    """
    Generate instance
    `network`: str, name of the network
    `P`: orders
    """
    # read data
    data = pd.read_csv(f"data/{network}.csv", index_col=False)
    # nodes
    V_H = data.loc[data['type'] == 'H']['id'].to_list()
    V_R = data.loc[data['type'] == 'R']['id'].to_list()
    V_W = data.loc[data['type'] == 'W']['id'].to_list()
    # capacity, 53 ft containers
    C_R = 120
    C_W = 1000
    # vessels
    L = list(range(1, int(45 * len(P) / (C_R * 1)) + 1, 1))
    B = list(range(1, int(45 * len(P) / (C_R * 2)) + 1, 1))
    # transshipment & sorting time
    delta_R = 3
    delta_W = 3
    # cost of truck per hour
    c_H = 35
    # cost of locomotive
    c_R = 3000000 / (25 * 365)
    # cost of cargo ship
    c_W = 15000000 / (30 * 365)
    # late cost
    c_P = 1000
    # carbon tax $/g CO2, 41.81 per ton CO2, 2024 California
    c_tax = 41.81 / 907185
    # GHG emission, highway, CIDI, Diesel, g/mile * ($/g)
    sigma_H = 1641 * c_tax
    # GHG emission, railway, Diesel, ($/g) *g/ton * mile
    sigma_R = 26 * c_tax
    # GHG emission, waterway, Marine Diesel Oil 0.1, ($/g) * g/ton * mile
    sigma_W = (9.76 / 0.684945) * c_tax
    # network
    G_H, G_R, G_W = nx.DiGraph(), nx.DiGraph(), nx.DiGraph()
    # nodes
    for i in V_H + V_R + V_W:
        G_H.add_node(
            i, lat=data.loc[data['id'] == i]['lat'].to_list()[0],
            lon=data.loc[data['id'] == i]['lon'].to_list()[0]
        )
    for i in V_R:
        G_R.add_node(
            i, lat=data.loc[data['id'] == i]['lat'].to_list()[0],
            lon=data.loc[data['id'] == i]['lon'].to_list()[0]
        )
    for i in V_W:
        G_W.add_node(
            i, lat=data.loc[data['id'] == i]['lat'].to_list()[0],
            lon=data.loc[data['id'] == i]['lon'].to_list()[0]
        )
    # edges
    # highway
    truck_speed = 65
    for i in V_H + V_R + V_W:
        for j in V_H + V_R + V_W:
            if i != j:
                G_H.add_edge(
                    i, j,
                    d=haversine_dist(
                        (G_H.nodes[i]['lat'], G_H.nodes[i]['lon']),
                        (G_H.nodes[j]['lat'], G_H.nodes[j]['lon'])
                    ),
                    tau=haversine_dist(
                        (G_H.nodes[i]['lat'], G_H.nodes[i]['lon']),
                        (G_H.nodes[j]['lat'], G_H.nodes[j]['lon'])
                    ) / truck_speed
                )
    # railway
    locomotive_speed = 75
    # load adj
    adj_R = pd.read_csv(
        f"data/railway-{len(V_H + V_R + V_W)}.csv", index_col=0
    )
    for i in V_R:
        neighbors = []
        # for intermodal-20
        for j in V_R:
            if adj_R.loc[i, f'{j}'] == 1:
                neighbors.append(j)
        for j in neighbors:
            G_R.add_edge(
                i, j,
                d=haversine_dist(
                    (G_R.nodes[i]['lat'], G_R.nodes[i]['lon']),
                    (G_R.nodes[j]['lat'], G_R.nodes[j]['lon'])
                ),
                tau=haversine_dist(
                    (G_R.nodes[i]['lat'], G_R.nodes[i]['lon']),
                    (G_R.nodes[j]['lat'], G_R.nodes[j]['lon'])
                ) / locomotive_speed
            )
    # waterway
    ship_speed = 10
    # load adj
    adj_W = pd.read_csv(
        f"data/waterway-{len(V_H + V_R + V_W)}.csv", index_col=0
    )
    for i in V_W:
        neighbors = []
        # for intermodal-20
        for j in V_W:
            if adj_W.loc[i, f'{j}'] == 1:
                neighbors.append(j)
        for j in neighbors:
            G_W.add_edge(
                i, j,
                d=haversine_dist(
                    (G_W.nodes[i]['lat'], G_W.nodes[i]['lon']),
                    (G_W.nodes[j]['lat'], G_W.nodes[j]['lon'])
                ),
                tau=haversine_dist(
                    (G_W.nodes[i]['lat'], G_W.nodes[i]['lon']),
                    (G_W.nodes[j]['lat'], G_W.nodes[j]['lon'])
                ) / ship_speed
            )
    instance = {
        'G_H': G_H, 'G_R': G_R, 'G_W': G_W,
        'P': P, 'L': L, 'B': B, 'C_R': C_R, 'C_W': C_W,
        'delta_R': delta_R, 'delta_W': delta_W,
        'c_H': c_H, 'c_R': c_R, 'c_W': c_W, 'c_P': c_P,
        'sigma_H': sigma_H, 'sigma_R': sigma_R, 'sigma_W': sigma_W
    }
    # save
    pickle.dump(instance, open(f'data/instances/{name}.pickle', 'wb'))
    return instance


def main():
    """main"""
    # generate instance
    network = "intermodal-20"
    # orders, 20: 6-node, 10/20/30, 30: 10-node, 30/60/90
    P = generate_order(range(1, 7), 5)
    P = [
        {'i': 0, 'o': 4, 'd': 1, 'v': 60, 'T': 48, 'w': 20},
        {'i': 4, 'o': 1, 'd': 5, 'v': 60, 'T': 48, 'w': 20},
        {'i': 6, 'o': 1, 'd': 3, 'v': 60, 'T': 48, 'w': 20},
        {'i': 7, 'o': 2, 'd': 6, 'v': 60, 'T': 48, 'w': 20}
    ]
    name = network + f"-{len(P)}"
    instance = generate_instance(name, network, P)
    # instance = pickle.load(
    #     open('data/instances/intermodal-20-10.pickle', 'rb')
    # )
    # define problem
    problem = Intermodal(name, instance)
    # build and solve MIP
    problem.MIP()
    # solve with decomposition
    # problem.decomposition()
    return


if __name__ == "__main__":
    np.random.seed(1)
    main()
