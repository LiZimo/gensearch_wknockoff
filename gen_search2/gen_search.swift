type file;

app (file out, file log, file err) search (file wrapper, file pyscript, file samples, file labels, file set, int n, int nsteps, int k, int method, file extra)
{
    bash @wrapper @pyscript "-i" @samples "-l" @labels "-p" @set "-n" n "-s" nsteps "-k" k "-m" method "-o" @out stdout=@log stderr=@err;
}


app (file result, file err) sort (file script, file parts[])
{
    bash @script filenames(parts) stdout=@result stderr=@err;
}

int n      = toInt(arg("n",       "2"));
int nsteps = toInt(arg("nsteps",  "20"));
int kval   = toInt(arg("kfold",   "5"));
int method = toInt(arg("method",  "2"));

file samples <single_file_mapper; file=arg("sample", "data/miRNA_samples.csv")>;
file labels  <single_file_mapper; file=arg("labels", "data/miRNA_labels.csv" )>;

string input_folder=arg("folder", "test");
string input_prefix=arg("prefix", "miRNA");

file splits[] <filesys_mapper; location=input_folder, prefix=input_prefix>;


file wrapper  <"python_wrapper.sh">;
file pyscript <"gen_search.py">;
file extra    <"classifier.py">;

file out[] <simple_mapper; prefix="results/search_", suffix=".csv">;
file log[] <simple_mapper; prefix="results/search_", suffix=".out">;
file err[] <simple_mapper; prefix="results/search_", suffix=".err">;

//foreach split,index in splits{
foreach f_split, index in splits
{
    tracef("Filename  = %s \n", filename(f_split));
    (out[index], log[index], err[index]) = search(wrapper, pyscript, samples, labels, f_split, n,nsteps, kval, method, extra);
}


file final <"final.out">;
file f_err <"final.err">;
file postscript <"postprocess.sh">;
(final, f_err) = sort (postscript, out);
