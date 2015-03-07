These runs are for 3d data using f1, f2, and f3.

Each run has three pickle files:

- data_f3_full_X.pkl: Non-stacked data. Each entry in the list is an array of
    examples from a single phoneme. Columns are dimensions. These data are
    exactly the Teacher object's data attribute.

- samples_f3_full_X.pkl: Stacked data. A single array of data. Each row in
    an example; each column is a dimension. Gotten using t.get_stacked_data()

- z_f3_full_X.pkl: Key to samples. Each entry, i, is an integer designating to
    which phoneme samples[i, :] belongs.