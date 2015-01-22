os.path.dirnameGen_Search
==========

Gen_Search : Generic search for patterns using Machine learning.


There are 3 stages to the Gen_search app.
1. Preprocess and generate combination of k items to search for.
2. Find accuracy of each k item pair.
3. Sort the results by accuracy.


Preprocessing
-------------

Currently this stage is done independent of the next two stages as,
this can be done efficiently on a single machine, and used by subsequent
stages without any need for recomputation. It is trivial to run from a
shell script and requires unnecessary complications with the swift version
used to run the steps further down the pipeline.

To run this stage use the preprocess.py script.

```bash
# For help use the -h option
preprocess.py -h

# To run the basic tests:
python preprocess.py -i <Sample> -l <Labels> -n <N> -s <nsteps> -k <K> -m <method> -x <number of chunks> -f <output_prefix>
# Eg:
# python preprocess.py -i data/miRNA_samples.csv -l data/miRNA_labels.csv -n 2 -s 20 -k 5 -m 2 -x 100 -f output/miRNA
```

Parallel run
------------
