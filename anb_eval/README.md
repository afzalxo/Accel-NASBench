### ANB Evaluation
We offer NAS optimizers we utilized for uni- and bi-objective optimization, combined with the results, plots and searched models.

#### Random Search
To perform bi-objective accuracy-throughput random search, use the following command (example for vck190, throughput. Please adapt according to device/metric need):

``` bash
python3 search_mo.py --arch_ep 250 --episodes 6 --device vck190 --metric throughput --algorithm RS --simulated
```

This would perform simulated bi-objective random search using `vck190` `throughput` surrogate.

The results are logged in `logs/simulated` directory. Pareto-optimal solutions are plotted and saved as png in the directory. Also, to obtain the pareto optimal set of solutions, go to the logs directory and perform the following operations to load the results pkl file:

``` python3
import pickle
res = pickle.load(open('pareto_designs249.pkl','rb'))
print(res)
``` 

This will print a dictionary of pareto-optimal solutions in the format: accuracy: [throughput, architecture]. E.g., 

``` python3
63.659610748291016: [3228.82421875, [['MB', 1, 3, 1, 32, 16, 2, False], ['MB', 1, 3, 2, 16, 24, 3, True], ['MB', 4, 3, 2, 24, 40, 2, True], ['MB', 1, 5, 2, 40, 80, 3, False], ['MB', 6, 5, 1, 80, 112, 2, True], ['MB', 6, 3, 2, 112, 192, 3, True], ['MB', 6, 3, 1, 192, 320, 3, True]]]
```


#### REINFORCE
Use the following command to run REINFORCE in bi-objective setting:

``` bash
python3 search_mo.py --arch_ep 250 --episodes 6 --device vck190 --metric throughput --algorithm PG --simulated --target_biobj 3000
```

The results are again logged in `logs/simulated` diretory in a similar format as above, however, only pkl files are generated. Please use the same code as above to decode the pkl files contents.


#### Regularized Evolution
TODO
