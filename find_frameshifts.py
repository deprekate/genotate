



def find_frameshifts(dna, predictions):
	assert len(dna) != len(predictions) +2

	for i in range(0,len(predictions), 3):
		print(dna[i], predictions[i])
