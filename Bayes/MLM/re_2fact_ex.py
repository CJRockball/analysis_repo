# -*- coding: utf-8 -*-
"""
Created on Fri Jul 23 14:34:52 2021

@author: PatCa
"""

import pandas as pd                                                                                                        
import statsmodels.api as sm                                                                                               

d = {'y':[8,2,6,4],'x':[1.1,5.2,3.3,6.2],'r1':[1,2,3,4],'r2':[5,6,7,8]}
df = pd.DataFrame(d)                                                                                                          
df["group"] = 1    # all in the case group                                                                                                        

vcf = {"r1": "0 + C(r1)", "r2": "0 + C(r2)"}  # formula                                                        
model = sm.MixedLM.from_formula("y ~ x", groups="group",                                                    
                                vc_formula=vcf, re_formula="~r1", data=df)                                                   
result = model.fit()  























