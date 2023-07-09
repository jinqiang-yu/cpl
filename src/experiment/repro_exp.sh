#!/bin/sh

# Go to the `src` directory
cd src/

# Reproducing cmpl experimental results

# Prepare datasets. The prepared datasets are stored in ```datasets/```
./experiment/prepare.sh; python ./experiment/kfold.py

# Generate gradient boosted trees by the following scripts. The generated BT modesl are stored in ```src/temp/```.
./experiment/bt.sh

# Compile boosted trees into decision sets by the following scripts. The logs are stored in ```logs/compile/```
./experiment/cpl_clocal_fsort_fqupdate.sh && ./experiment/cpl_clocal_fsort_fqupdate_reduce_after_maxsat.sh && ./experiment/cpl_clocal_fsort_fqupdate_reduce_after_maxsat_0.005_approx.sh && ./experiment/cpl_clocal_fsort_fqupdate_rrule_wght.sh && ./experiment/cpl_clocal_fsort_fqupdate_reduce_after_maxsat_rrule_wght.sh &&  ./experiment/cpl.sh


# Reproducing experimental results of other competitors

# Preprocess datasets based on the thresholds in the corresponding BT models. The preprocessed datasets are stored in ```datasets/```.
./experiment/prepare_dsdt.sh

# Compute decision sets by selected algorithms. All logs are saved in ```../logs/<competitor>```, e.g. ```../logs/<opt>``` for ```opt```.

# ```opt```: A MaxSAT approach to decision sets agreeing with the trainign data.
./experiment/ds_opt.sh

# ```sparse```: The sparse alternative of ```opt```.
./experiment/ds_0.005_approx.sh

# ```twostg```: A two-stage MaxSAT approach to decision sets agreeing with the trainign data.
./experiment/ds_twostg.sh

# ```imli```: A MaxSAT appraoch to decision sets.
./experiment/ds_imli_1.sh && ./experiment/ds_imli_16.sh

# ```ripper```: heuristic approach to decision sets.
./experiment/ds_ripper.sh

# ```cn2```: another heuristic approach to decision sets.
./experiment/ds_cn2.sh

# ```ids```: 
./experiment/ds_ids.sh

# When the experiments above are done, we can parse the logs and generate the plots.
# The statistics of decision sets is stored in directory ```../stats/```, while the plots are stored in ```../plots/```.

# To parse logs and generate plots:
python parse_logs.py
