m0 = MassFunction({'abc':1})


### example with two strong deceptive evidence
m1 = MassFunction({'a':0.7, 'b':0, 'c':0, 'abc':0.3})
m2 = MassFunction({'a':0.6, 'b':0.1, 'c':0, 'abc':0.3})
m3 = MassFunction({'a':0.5, 'b':0.1, 'c':0.2, 'abc':0.2})
m4 = MassFunction({'a':0, 'b':0, 'c':1, 'abc':0})
m5 = MassFunction({'a':0.05, 'b':0.9, 'c':0.05, 'abc':0})


### example of a strong deceptive evidence resulted from several evidence
m1 = MassFunction({'a':0.7, 'b':0, 'c':0, 'abc':0.3})
m2 = MassFunction({'a':0.6, 'b':0.1, 'c':0, 'abc':0.3})
m3 = MassFunction({'a':0.5, 'b':0.1, 'c':0.2, 'abc':0.2})
m4 = MassFunction({'a':0.1, 'b':0.7, 'c':0.2, 'abc':0})
m5 = MassFunction({'a':0.1, 'b':0.4, 'c':0.3, 'abc':0})


m1 = MassFunction({'a':0.7, 'b':0, 'c':0, 'abc':0.3})
m2 = MassFunction({'a':0.6, 'b':0.1, 'c':0, 'abc':0.3})
m3 = MassFunction({'a':0.5, 'b':0.1, 'c':0.2, 'abc':0.2})
m4 = MassFunction({'a':0.01, 'b':0.99, 'c':0})
m5 = MassFunction({'a':0.01, 'b':0, 'c':0.99})


### example from Cui
m1 = MassFunction({'a':0.4, 'b':0.5, 'c':0.1})
m2 = MassFunction({'a':0, 'b':0.9, 'c':0.1})
m3 = MassFunction({'a':0.65, 'b':0.1, 'c':0, 'ac':0.25})
m4 = MassFunction({'a':0.6, 'b':0.2, 'c':0, 'ac':0.2})
m5 = MassFunction({'a':0.5, 'b':0.2, 'c':0, 'ac':0.3})


### example from Zhou
m1 = MassFunction({'a':0.99, 'b':0.01, 'c':0})
m2 = MassFunction({'a':0, 'b':0.01, 'c':0.99})
m3 = MassFunction({'a':0.0, 'b':0.02, 'c':0.98})


### example from Kang (1)
m1 = MassFunction({'a':0.8, 'b':0.2, 'c':0})
m2 = MassFunction({'a':0.6, 'b':0.1, 'c':0.3})
m3 = MassFunction({'a':0.2, 'b':0.35, 'c':0.45}) # m3 is deceptive
m4 = MassFunction({'a':0.7, 'b':0.15, 'c':0.15})
m5 = MassFunction({'a':0.34, 'b':0.34, 'c':0.32})
m5 = MassFunction({'a':0.3, 'b':0.3, 'c':0.4})
m6 = MassFunction({'a':0, 'b':1, 'c':0}) # m6 is deceptive


### example from Kang (2)
m1 = MassFunction({'a':0.3, 'b':0.6, 'c':0, 'abc':0.1})
m2 = MassFunction({'a':0.7, 'b':0, 'c':0, 'abc':0.3})
m3 = MassFunction({'a':0.65, 'b':0.15, 'c':0, 'abc':0.2}) 
m4 = MassFunction({'a':0.75, 'b':0, 'c':0.05, 'abc':0.2})
m5 = MassFunction({'a':0.05, 'b':0.45, 'c':0.5, 'abc':0}) # m5 is deceptive
m6 = MassFunction({'a':0.05, 'b':0.5, 'c':0.45, 'abc':0}) # m6 is deceptive


m1 = MassFunction({'a':0.3, 'b':0.6, 'c':0, 'abc':0.1}) # m1 is deceptive
m2 = MassFunction({'a':0.7, 'b':0, 'c':0, 'abc':0.3})
m3 = MassFunction({'a':0.65, 'b':0.15, 'c':0, 'abc':0.2}) 
m4 = MassFunction({'a':0.75, 'b':0, 'c':0.05, 'abc':0.2})
m5 = MassFunction({'a':0.05, 'b':0.45, 'c':0.5, 'abc':0}) # m5 is deceptive
m6 = MassFunction({'a':0.05, 'b':0.5, 'c':0.45, 'abc':0}) # m6 is deceptive
