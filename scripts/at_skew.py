import os
import io
import sys

def read_fasta(filepath, base_trans=str.maketrans('','')):
    contigs_dict = dict()
    name = seq = ''

    lib = gzip if filepath.endswith(".gz") else io
    with lib.open(filepath, mode="rb") as f:
        for line in f:
            if line.startswith(b'>'):
                contigs_dict[name] = seq
                name = line[1:].decode("utf-8").split()[0]
                seq = ''
            else:
                seq += line.decode("utf-8").rstrip().upper()
        contigs_dict[name] = seq.translate(base_trans)
    if '' in contigs_dict: del contigs_dict['']
    return contigs_dict


def at_skew(dna):
	a = dna.count('A')
	t = dna.count('T')
	return (a-t) / (a+t)


dna = list(read_fasta('genomes/fna/AB008550.fna').values())[0]

c = [0]
for n in range(len(dna)):
	window = dna[ max( 0 , n-27) : n+30]
	#c.append( c[n-1] + at_skew(window))
	c.append( at_skew(window))

for n, item in enumerate(c):
	window = c[ max( 0 , n-27) : n+30]
	print(n, sum(window))
