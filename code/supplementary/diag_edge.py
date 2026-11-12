import numpy as np, warnings; warnings.filterwarnings("ignore")
from collections import Counter
from sklearn.preprocessing import StandardScaler
import nivel1_benchmark as m, fem_truss as ft, hiper
rng=np.random.default_rng(5); m.RNG=rng
probs={"lin2":m.Linear(2.33,2),"d10":m.Linear(2.5,10),"sph":m.Sphere(3.0,2),"4b":m.FourBranch(6.0),"truss":ft.TrussFatigue(rng)}
for name,p in probs.items():
    c=Counter()
    for n in (12,30,50):
        for k in range(4):
            X=p.sample(n); y=p.g(X); Xs=StandardScaler().fit_transform(X)
            bp=hiper.GridSearchCV(hiper.SVR(kernel="rbf"),hiper.GRID,cv=3,scoring="neg_mean_squared_error").fit(Xs,y).best_params_
            c[(n,"C",bp["C"])]+=1; c[(n,"g",bp["gamma"])]+=1; c[(n,"e",bp["epsilon"])]+=1
    print(name, sorted(c.items(), key=lambda t:(t[0][0],t[0][1],str(t[0][2]))))
