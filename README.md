
Introduction
------------

Genotate is a tool to annotate prokaryotic and phage genomes.  It uses scrolling amino-acid
windows in all six frames to distinguish between windows that belong to protein coding gene
regions and those that belong to noncoding regions, in order to determine the coding frame
at every position along the genome.

To install `Genotate`,
```sh
 git clone https://github.com/deprekate/Genotate.git
 cd Genotate
```

To run `Genotate` you only need to specify the FASTA formatted genome file, and which
model to use.  To run on the provided phiX174 genome, use the command:
```
 python3 classify.py phiX174.fna --model models/single.ckpt > predictions.gb
```

The output of the script are individual single codon predictions, in GenBank format, that
represent the predicted coding frame.  In future versions adjacent single predictions will
be merged into one long gene CDS feature.
