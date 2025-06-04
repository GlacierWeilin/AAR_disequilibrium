#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  3 11:17:47 2025

@author: wyan0065
"""

import numpy as np
import pandas as pd

years = np.arange(2014, 2024)

random_years_5000 = np.zeros(5000)

for i in range(0,500):
    random_years = np.random.choice(years, size=(10), replace=False)
    random_years_5000[10*i:10*(i+1)]=random_years[:]

df_5000 = pd.DataFrame(random_years_5000, columns=["2014-2023"])
df_5000.index.name = "simulation_years"
df_5000.to_csv("shuffled_years_5000.csv")

df_2000 = pd.DataFrame(random_years_5000[0:2000], columns=["2014-2023"])
df_2000.index.name = "simulation_years"

df_2000.to_csv("shuffled_years_2000.csv")




