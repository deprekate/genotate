#include <stdio.h>
#include <limits.h>
#include <Python.h>

/*
 // commented out because these are gcc specific
#define min(a,b) \
   ({ __typeof__ (a) _a = (a); \
      __typeof__ (b) _b = (b); \
     _a < _b ? _a : _b; })
#define max(a,b) \
   ({ __typeof__ (a) _a = (a); \
      __typeof__ (b) _b = (b); \
     _a > _b ? _a : _b; })
*/
#define MAX(x, y) (((x) > (y)) ? (x) : (y))
#define MIN(x, y) (((x) < (y)) ? (x) : (y))
#define SWAP(a, b)   \
do {                 \
    int temp = a;    \
    a = b;           \
    b = temp;        \
} while(0)

#define VAL_1X     -128
#define VAL_2X     VAL_1X,  VAL_1X
#define VAL_3X     VAL_1X,  VAL_1X, VAL_1X
#define VAL_4X     VAL_2X,  VAL_2X
#define VAL_8X     VAL_4X,  VAL_4X
#define VAL_16X    VAL_8X,  VAL_8X
#define VAL_32X    VAL_16X, VAL_16X
#define VAL_64X    VAL_32X, VAL_32X
#define VAL_128X   VAL_64X, VAL_64X
static const char nuc_table[256] = { VAL_64X, VAL_32X, VAL_1X, 0, VAL_1X, 1, VAL_3X, 2, VAL_8X, VAL_4X, 3, VAL_128X, VAL_8X, VAL_3X };

unsigned char compl[256] = "nnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnntngnnncnnnnnnnnnnnnannnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnn";

unsigned char aa_table[65] = "KNKNTTTTRSRSIIMIQHQHPPPPRRRRLLLLEDEDAAAAGGGGVVVV#Y+YSSSS*CWCLFLFX";


typedef struct {
	PyObject_HEAD
	const unsigned char* dna;
	unsigned int len;
	unsigned int i;
	unsigned int f;
	float gc;
} windows_Iterator;

unsigned char get_chr(const unsigned char *dna, unsigned int i, unsigned int f) {
    // unsigned char* so high-ASCII -> 128..255, not negative,
    // and works as an index into a 256 entry table
    unsigned int idx;
	if(f){
		idx	= nuc_table[dna[i]];
		idx = idx*4 + nuc_table[dna[i+1]];
		idx = idx*4 + nuc_table[dna[i+2]];
	}else{
		idx	= nuc_table[compl[dna[i+2]]];
		idx = idx*4 + nuc_table[compl[dna[i+1]]];
		idx = idx*4 + nuc_table[compl[dna[i]]];
	}
	//printf("i:%i c:%c%c%c idx: %i aa: %c\n", i, dna[i], dna[i+1], dna[i+2], idx, aa_table[idx]);
	if(idx > 63){
		return aa_table[64];
	}else{
    	return aa_table[idx];
	}
}

PyObject* windows_Iterator_iter(PyObject *self){
	Py_INCREF(self);
	return self;
}

PyObject* windows_Iterator_iternext(PyObject *self){
	windows_Iterator *p = (windows_Iterator *)self;
	char aa[90] = {0};
	unsigned int nuc[5] = {0};
	unsigned int i, j, k, t;
	float total;

	if( p->i < p->len - 2 ){
		//printf("%i %i %c%c%c = %u\n", p->len, p->i, p->dna[p->i], p->dna[p->i+1], p->dna[p->i+2], get_chr(p->dna, p->i, p->f));
		t = 0;
		j =    (p->i > 56)       ? p->i-57  : p->i%3;
		k = (p->i+60 > p->len-2) ? p->len-2 : p->i+60;
		//j = MAX( p->i  ,  57 + p->i % 3) - 57;
		//k = MIN( p->len - 2 , p->i + 60);
		//-----------------------------------------------
		for (i = j; i < k; i += 3){
			//printf("%c\t", get_chr(p->dna, i, p->f) );
			aa[get_chr(p->dna, i, p->f)]++;
			nuc[ nuc_table[p->dna[i  ]] % 6 ]++;
			nuc[ nuc_table[p->dna[i+1]] % 6 ]++;
			nuc[ nuc_table[p->dna[i+2]] % 6 ]++;
			t++;
		}
		//printf("\n");

		if(!p->f){
			SWAP(nuc[0] , nuc[3]);
			SWAP(nuc[1] , nuc[2]);
		}
		total = (float) t;
		// ADD IN DIV ZERO HANDLING IN CASE BAD SEQUENCE
		//
		//PyObject *aa_list = Py_BuildValue("[fffffffffffffffffffffffff]",
		PyObject *aa_list = Py_BuildValue("[fffffffffffffffffffff]",
									p->gc,
									//nuc[0] / (3.0*t),
									//nuc[1] / (3.0*t),
									//nuc[2] / (3.0*t),
									//nuc[3] / (3.0*t),
                                    aa['A'] / total,
                                    aa['C'] / total,
                                    aa['D'] / total,
                                    aa['E'] / total,
                                    aa['F'] / total,
                                    aa['G'] / total,
                                    aa['H'] / total,
                                    aa['I'] / total,
                                    aa['K'] / total,
                                    aa['L'] / total,
                                    aa['M'] / total,
                                    aa['N'] / total,
                                    aa['P'] / total,
                                    aa['Q'] / total,
                                    aa['R'] / total,
                                    aa['S'] / total,
                                    aa['T'] / total,
                                    aa['V'] / total,
                                    aa['W'] / total,
                                    aa['Y'] / total
									);
		//
		
		p->f ^= 1;
		p->i += p->f;
		return aa_list;
	}else{
		PyErr_SetNone(PyExc_StopIteration);
		return NULL;
	}
}

static void Iter_dealloc(windows_Iterator *self){ PyObject_Del(self); }

static PyTypeObject IterableType = {
	PyVarObject_HEAD_INIT(NULL, 0)
	.tp_name = "Iter",
	.tp_doc = "Custom objects",
	.tp_basicsize = sizeof(windows_Iterator),
	.tp_itemsize = 0,
	.tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,
	.tp_dealloc = (destructor) Iter_dealloc,
	.tp_iter	  = windows_Iterator_iter,
	.tp_iternext  = windows_Iterator_iternext
};

static PyObject * get_windows(PyObject *self, PyObject *args){
	unsigned int i;
	unsigned int nuc[5] = {0};
	windows_Iterator *p;
	p = PyObject_New(windows_Iterator, &IterableType);
	if (!p) return NULL;
	
	if (!PyArg_ParseTuple(args, "s", &p->dna)) {
	//if (!PyArg_ParseTuple(args, "s#", &p->dna, &p->len)) {
		return NULL;
	}
	
	p->i = 0;
	p->f = 1;
	p->len = strlen( (const char*) p->dna);

	for (i=0; p->dna[i] ; i++){
		nuc[ nuc_table[(p->dna[i])] % 6 ]++;
	}
	p->gc =  (float)( nuc[1] + nuc[2] ) / ( nuc[0] + nuc[1] + nuc[2] + nuc[3] );

	/* I'm not sure if it's strictly necessary. */
	if (!PyObject_Init((PyObject *)p, &IterableType)) {
		Py_DECREF(p);
		return NULL;
	}

	return (PyObject *)p;
}

// Module method definitions
static PyObject* no_args(PyObject *self, PyObject *args) {
	Py_RETURN_NONE;
}

// Method definition object for this extension, these argumens mean:
static PyMethodDef windows_methods[] = { 
	{"get_windows",    get_windows, METH_VARARGS, "Gets the aminoacid frequency windows."},  
	{"no_args",            no_args, METH_NOARGS,  "Empty for now."},  
	{NULL, NULL, 0, NULL}
};

// Module definition
static struct PyModuleDef windows_definition = { 
	PyModuleDef_HEAD_INIT,
	"windows",
	"A Python module that get aminoacid windows.",
	-1, 
	windows_methods
};

// Module initialization
// Python calls this function when importing your extension. It is important
// that this function is named PyInit_[[your_module_name]] exactly, and matches
// the name keyword argument in setup.py's setup() call.
PyMODINIT_FUNC PyInit_windows(void) {
	//Py_Initialize();
	return PyModule_Create(&windows_definition);
}





