type file;

app (file names, file log) preprocess (file wrapper, file pyscript, file samples, file labels, int n, int nsteps, int k, int method, int splitnum, string prefix)
{
    bash @wrapper @pyscript "-i" @samples "-l" @labels "-n" n "-s" nsteps "-k" k "-m" method "-x" splitnum "-f" prefix stdout=@names stderr=@log;
}

file wrapper  <"python_wrapper.sh">;
file pyscript <"preprocess.py">;

file samples <"data/miRNA_samples.csv">;
file labels <"data/miRNA_labels.csv">;

file prescript <"preprocess.py">;

file pre_out <"preprocess.out">;
file pre_log <"preprocess.log">;

(pre_out, pre_log) = preprocess(wrapper, pyscript, samples, labels, 2, 20, 5, 2, 1000, "outtest/miRNA");
