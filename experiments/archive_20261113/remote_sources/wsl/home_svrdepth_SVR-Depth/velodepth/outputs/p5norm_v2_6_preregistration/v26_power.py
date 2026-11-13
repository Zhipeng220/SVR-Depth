import numpy as np,json
from scipy.stats import rankdata,t
rng=np.random.default_rng(20261025);results=[]
for n in [50,60,90,100,120]:
 hit=0;observed=[];R=3000
 for _ in range(R):
  dom=np.arange(n)%3;rng.shuffle(dom);overall=rng.normal(size=n);noise=rng.normal(size=n);layer=.7*overall+np.sqrt(1-.7**2)*noise;u=.35*overall+.3*noise+np.sqrt(1-.35**2-.3**2)*rng.normal(size=n)+np.array([-.25,0,.25])[dom]
  x=rankdata(layer);y=rankdata(u);z=rankdata(overall);D=np.column_stack([np.ones(n),z,dom==1,dom==2]);rx=x-D@np.linalg.lstsq(D,x,rcond=None)[0];ry=y-D@np.linalg.lstsq(D,y,rcond=None)[0];r=np.corrcoef(rx,ry)[0,1];dof=n-D.shape[1]-1;stat=r*np.sqrt(dof/max(1e-15,1-r*r));p=2*t.sf(abs(stat),dof);hit+=p<.05;observed.append(r)
 results.append({'n':n,'replicates':R,'power_two_sided_partial_rank_t_test':hit/R,'mean_partial_rank_r':float(np.mean(observed))})
print(json.dumps(results))
