
Introduction
------------

Genotate is a tool to annotate prokaryotic<sup>*</sup> and phage genomes.  It uses scrolling amino-acid
windows in all six frames to distinguish between windows that belong to protein coding gene
regions and those that belong to noncoding regions, in order to determine the coding frame
at every position along the genome.
*(the bacteria/archaea model is still being trained)

To install `Genotate`,
```sh
 pip install genotate
```

And to run `Genotate` you only need to specify the FASTA formatted genome file
The command to run using the phage models on the provided phiX174 genome is:
```
 genotate.py test/phiX174.fasta -o predictions.gb
```
The command to run using the partially trained bacterial/archaeal models needs the --bacterial flag. Instead of
a FASTA formatted file, you can provide a Genbank formatted file and Genotate will use only the genomic sequence.
```
 genotate.py test/mycoplasma.gbff.gz -o predictions.gb --bacteria
```
**It is recommended to use a GPU to run Genotate since it will take a long time to for prokaryotic 
genomes.**  Genotate will automatically try to run on GPU, if one isn't found it will run on a CPU.

---

The output of `Genotate` are 'coding region' predictions in GenBank format.  They *should*
match with the true coding gene regions, but are not genes per say, since they are not based
on start and stop codons. Though they have all been trimmed to a stop codon after Genotate
determines which transation table the genome uses (i.e. if it performs stop codon readthrough).

There are three main phases to the Genotate workflow
1. window classification 
2. change-point detection
3. refinement
   * analyze stop codons
   * merge adjacent regions
   * split regions on stop
   * adjust ends to a stop

Genotate determines the translation table by analyzing the initial coding gene region 
predictions.  There are two outcomes for a stop codon that is readthrough: either the stop 
codon appears in the middle of a coding gene region or the region is broken into two pieces at 
the stop codon. If one of the three known stop codons is significantly over represented in the 
middle AND between predicted gene regions, that stop codon can be assumed to be read through. 
With the stop codon usage now known, same frame adjacent coding regions are merged if there is 
not a stop codon between them. Then the regions are split on any internal stop codons and the 
ends adjusted to the nearest stop codon.

** The opposite end is not adjusted to valid start codon since Genotate does not have a translation
initiation site detection method yet, so the beginning of a gene call may be off by a few codons


Currently the best way to visualize the predictions is in a Genome Viewer application, such
as Artemis by Sanger. The example phiX174.gb GenBank file loaded into Artemis shows the 
gene layout:

![](https://github.com/deprekate/genotate/blob/main/src/genes.png)

The predictions.gb file can then be loaded using the 'File>Read An Entry' menu, and the
predictions will be overlaid as grey 'coding regions' in the gene layout window:

![](https://github.com/deprekate/genotate/blob/main/src/predictions.png)
