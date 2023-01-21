
self = { (1,2):'foo',(8,9):'bar',(5,6):'mun'}

for item in sorted(self):
	if item == (8,9):
		del self[item]
		self[(3,4)] = True

print(self)
