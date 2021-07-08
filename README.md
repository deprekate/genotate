
Introduction
------------

Genotate is a tool to annotate prokaryotic and phage genomes.  It uses scrolling amino-acid
windows in all six frames to distinguish between windows that belong to protein coding gene
regions and those that belong to noncoding regions, in order to determine the coding frame
at every position along the genome.

To install `Genotate`,
```sh
 git clone https://github.com/deprekate/genotate.git
 cd genotate
```

And to run `Genotate` you only need to specify the FASTA formatted genome file, and which
model to use (provided is the most basic model trained on the first 800 phage genomes from
RefSeq). To run on the provided phiX174 genome, use the command:
```
 python3 classify.py phiX174.fna --model models/single.ckpt > predictions.gb
```

The output of the script are individual single codon predictions, in GenBank format, that
represent the predicted coding frame.  In future versions adjacent single predictions will
be merged into one long gene CDS feature.


Currently the best way to visualize the predictions is in a Genome Viewer application, such
as Artemis by Sanger. The example phiX174.gb GenBank file loaded into Artemis shows the 
gene layout:
![](https://github.com/deprekate/genotate/blob/main/src/genes.png)

The predictions.gb file can then be loaded using the 'File>Read An Entry' menu, and the
predictions will be overlaid as black/grey single codon CDS features in the gene layout
window:
![](https://github.com/deprekate/genotate/blob/main/src/predictions.png)
