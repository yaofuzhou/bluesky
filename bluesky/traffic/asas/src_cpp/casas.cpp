#include <iostream>
#include <vector>
#include <algorithm>
#define NPY_NO_DEPRECATED_API NPY_1_10_API_VERSION
#include <Python.h>
#include "structmember.h"
#include "numpy/arrayobject.h"
#include "asas.hpp"
#define DEG2RAD 0.017453292519943295
#define RAD2DEG 57.29577951308232
#define M2NM 0.0005399568034557236
#define NM2M 1852.0
#define KTS2MS 0.514444
#define FT2M 0.3048
#define FPM2MS 0.00508

static PyObject* casas_detect(PyObject* self, PyObject* args)
{
    PyObject      *ownship = NULL,
                  *intruder = NULL;
    PyArrayObject *RPZobj = NULL,
                  *HPZobj = NULL,
                  *tlookaheadobj = NULL;

    if (!PyArg_ParseTuple(args, "OOO!O!O!", &ownship, &intruder, &PyArray_Type, &RPZobj, &PyArray_Type, &HPZobj, &PyArray_Type, &tlookaheadobj))
        Py_RETURN_NONE;

    PyArrayObject *lat1 = (PyArrayObject *)PyObject_GetAttrString(ownship, "lat"),
                  *lon1 = (PyArrayObject *)PyObject_GetAttrString(ownship, "lon"),
                  *trk1 = (PyArrayObject *)PyObject_GetAttrString(ownship, "trk"),
                  *gs1 = (PyArrayObject *)PyObject_GetAttrString(ownship, "gs"),
                  *alt1 = (PyArrayObject *)PyObject_GetAttrString(ownship, "alt"),
                  *vs1 = (PyArrayObject *)PyObject_GetAttrString(ownship, "vs"),
                  *lat2 = (PyArrayObject *)PyObject_GetAttrString(intruder, "lat"),
                  *lon2 = (PyArrayObject *)PyObject_GetAttrString(intruder, "lon"),
                  *trk2 = (PyArrayObject *)PyObject_GetAttrString(intruder, "trk"),
                  *gs2 = (PyArrayObject *)PyObject_GetAttrString(intruder, "gs"),
                  *alt2 = (PyArrayObject *)PyObject_GetAttrString(intruder, "alt"),
                  *vs2 = (PyArrayObject *)PyObject_GetAttrString(intruder, "vs");

    PyListAttr  acid(ownship, "id");

    // Only continue if all arrays exist
    if (lat1 && lon1 && trk1 && gs1  && alt1 && vs1  && lat2 && lon2 && trk2 && gs2  && alt2 && vs2 && RPZ && HPZ && tlookahead)
    {
        // Assume all arrays are the same size; only get the size of lat1
        npy_intp size = PyArray_SIZE(lat1);

        // Loop over all combinations of aircraft to detect conflicts
        conflict confhor, confver;
        double tin, tout;
        double dalt, dvs;

        npy_bool acinconf = NPY_FALSE;
        double tcpamax_ac = 0.0;

        // Return values
        int nd = 1;
        npy_intp dims[] = {size};
        PyObject *tcpamax = PyArray_SimpleNew(nd, dims, NPY_DOUBLE),
                 *inconf = PyArray_SimpleNew(nd, dims, NPY_BOOL),
        
                 *confpairs = PyList_New(0),
                 *lospairs = PyList_New(0),
                 *qdr = PyList_New(0),
                 *dist = PyList_New(0),
                 *dcpa = PyList_New(0),
                 *tcpa = PyList_New(0),
                 *tinconf = PyList_New(0);

        double *RPZ1 = RPZ.ptr_start;
        double *HPZ1 = HPZ.ptr_start;
        double *tlookahead1 = tlookahead.ptr_start;
        for (unsigned int i = 0; i < size; ++i) {
            std::cout << "test " << i << ": " << *RPZ1++ << ", " << *HPZ1++ << ", " << *tlookahead1++ << std::endl;
            continue;
            acinconf = NPY_FALSE;
            double *RPZ2 = RPZ.ptr_start;
            double *HPZ2 = HPZ.ptr_start;
            for (unsigned int j = 0; j < size; ++j) {
                if (i != j) {
                    // Vectical detection first
                    dalt = *alt1.ptr - *alt2.ptr;
                    dvs  = *vs1.ptr  - *vs2.ptr;
                    if (detect_ver(confver, std::max(*HPZ1, *HPZ2), *tlookahead1, dalt, dvs)) {
                        // Horizontal detection
                        if (detect_hor(confhor, std::max(*RPZ1, *RPZ2), *tlookahead1,
                                       *lat1.ptr * DEG2RAD, *lon1.ptr * DEG2RAD, *gs1.ptr, *trk1.ptr * DEG2RAD,
                                       *lat2.ptr * DEG2RAD, *lon2.ptr * DEG2RAD, *gs2.ptr, *trk2.ptr * DEG2RAD))
                        {
                            tin  = std::max(confhor.tin, confver.tin);
                            tout = std::min(confhor.tout, confver.tout);
                            // Combined conflict?
                            if (tin <= tlookahead && tin < tout && tout > 0.0) {
                                // Add AC id to conflict list
                                PyObject* pair = PyTuple_Pack(2, acid[i], acid[j]);
                                confpairs.append(pair);
                                tcpamax_ac = std::max(confhor.tcpa, tcpamax_ac);
                                acinconf = NPY_TRUE; // This aircraft is in conflict
                                if (confver.LOS && confhor.LOS) {
                                    // Add to lospairs if this is also a LoS
                                    lospairs.append(pair);
                                }
                                Py_DECREF(pair);
                                qdr.append(confhor.q * RAD2DEG);
                                dist.append(confhor.d);
                                dcpa.append(confhor.dcpa);
                                tcpa.append(confhor.tcpa);
                                tinconf.append(tin);
                            }
                        }
                    }
                }
                lat2.ptr++; lon2.ptr++; trk2.ptr++; gs2.ptr++; alt2.ptr++; vs2.ptr++;
                RPZ2++; HPZ2++;
            }
            *inconf.ptr = acinconf;
            *tcpamax.ptr = tcpamax_ac;
            inconf.ptr++;
            tcpamax.ptr++;
            acinconf = NPY_FALSE;
            tcpamax_ac = 0.0;
            lat2.ptr = lat2.ptr_start; lon2.ptr = lon2.ptr_start;
            trk2.ptr = trk2.ptr_start; gs2.ptr  = gs2.ptr_start;
            alt2.ptr = alt2.ptr_start; vs2.ptr  = vs2.ptr_start;
            lat1.ptr++; lon1.ptr++; trk1.ptr++; gs1.ptr++; alt1.ptr++; vs1.ptr++;
            RPZ1++; HPZ1++; tlookahead1++;
        }

        return PyTuple_Pack(9, confpairs.attr, lospairs.attr, inconf.arr, tcpamax.arr, qdr.attr, dist.attr, dcpa.attr, tcpa.attr, tinconf.attr);
    }

    Py_RETURN_NONE;
};

static PyMethodDef methods[] = {
    {"detect", casas_detect, METH_VARARGS, "Detect conflicts for traffic"},
    {NULL}  /* Sentinel */
};

#ifndef PyMODINIT_FUNC  /* declarations for DLL import/export */
#define PyMODINIT_FUNC void
#endif
#if PY_MAJOR_VERSION >= 3
static struct PyModuleDef casasdef =
{
    PyModuleDef_HEAD_INIT,
    "casas",     /* name of module */
    "",          /* module documentation, may be NULL */
    -1,          /* size of per-interpreter state of the module, or -1 if the module keeps state in global variables. */
    methods
};

PyMODINIT_FUNC PyInit_casas(void)
{
    import_array();
    return PyModule_Create(&casasdef);
};
#else
PyMODINIT_FUNC initcasas(void)
{
    Py_InitModule("casas", methods);
    import_array();
};
#endif
