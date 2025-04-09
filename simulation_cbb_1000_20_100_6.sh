#!/bin/bash

## for n =1000, u =6, find the results for cbb and mcmc when p,r =(20, 100)

#n = $1
n=1000
#p = $2
p=20
#r = $3
r=100
#u = $4
u=1
#nboot = $5
nboot=100
ninit=4
ncore=6
#nsim=1
nsim=25
# Read seeds into an array
mapfile -t seeds < seed.txt

# Loop through the first 25 seeds
for ((i=0; i<$nsim; i++)); do
    seed=${seeds[i]} # Get the i-th seed
    iter_number=$((i+1))
    echo "Running the_script.R iter_number no of times: $iter_number"
    Rscript Opt_boot_futlap\(5\).R "$seed" "$n" "$p" "$r" "$u" "$nboot" "$ninit" "$ncore" "$iter_number"
    
done

#Rscript the_other_script.R "$nsim" "$n" "$p" "$r" "$u"
## calculate the mean and median measures for cbb:
Rscript results_from_simulations_cbb.R "$nsim" "$n" "$p" "$r" "$u" > cbb_measures_1000_2_100_6_with_nsim_1.txt

##cat $(ls cbb*)
#Fatal error: cannot open file 'results_from_simulations_cbb': No such file or directory






