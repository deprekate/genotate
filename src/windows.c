#include <stdio.h>
#include <limits.h>
#include <Python.h>

#define min(a,b) \
   ({ __typeof__ (a) _a = (a); \
      __typeof__ (b) _b = (b); \
     _a < _b ? _a : _b; })
#define max(a,b) \
   ({ __typeof__ (a) _a = (a); \
      __typeof__ (b) _b = (b); \
     _a > _b ? _a : _b; })

#define BAD_NUC -128
static const int8_t nuc_table[UCHAR_MAX+1] = {
    [0 ... 255] = BAD_NUC,  // ranges are a GNU extension
    // last init takes precedence  https://gcc.gnu.org/onlinedocs/gcc/Designated-Inits.html
    ['a'] = 0,
    ['c'] = 1,
    ['g'] = 2,
    ['t'] = 3,
};

unsigned get_int(const char *dna, long int i) {
    // unsigned char* so high-ASCII -> 128..255, not negative,
    // and works as an index into a 256 entry table
    const char aa[64] = "KNKNTTTTRSRSIIMIQHQHPPPPRRRRLLLLEDEDAAAAGGGGVVVV#Y+YSSSS*CWCLFLF";
    unsigned idx = nuc_table[dna[i]];
    idx = idx*4 + nuc_table[dna[i+1]];
    idx = idx*4 + nuc_table[dna[i+2]];
    return aa[idx];
}
typedef struct {
	PyObject_HEAD
	const char* dna;
	long int i;
} windows_Iterator;

PyObject* windows_Iterator_iter(PyObject *self){
	Py_INCREF(self);
	return self;
}

PyObject* windows_Iterator_iternext(PyObject *self){
	windows_Iterator *p = (windows_Iterator *)self;
	int aa[54] = {0};
	long int i, j, k;
	//if(p->dna[p->i] != '\0') {
	if(p->i < strlen(p->dna)-3 ) {
		i = max(   p->i - 57    , p->i % 3);
		j = min( strlen(p->dna)-2 , p->i + 60);
		//printf("%li - %li\n", i, j);
		for (k = i; k < j; k += 3){
			//printf("%c", get_int(p->dna, k) );
			aa[get_int(p->dna, k)]++;
		}
		//printf("\n");


		//PyObject *tmp = Py_BuildValue("c", p->dna[p->i]);
		PyObject *aa_list = PyList_New(0);
		PyList_Append(aa_list, Py_BuildValue("i", 42));
		(p->i)++;
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
	windows_Iterator *p;
	p = PyObject_New(windows_Iterator, &IterableType);
	if (!p) return NULL;

	if (!PyArg_ParseTuple(args, "s", &p->dna)) {
		return NULL;
	}
	p->i = 0;

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

/*
static PyObject* get_windows(PyObject *self, PyObject *args) {
	const char* name;
	if (!PyArg_ParseTuple(args, "s", &name)) {
		return NULL;
	}
	printf("%s!\n", name);
	Py_RETURN_NONE;
}
*/

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





