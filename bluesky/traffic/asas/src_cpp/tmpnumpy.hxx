#define NPY_NO_DEPRECATED_API NPY_1_10_API_VERSION
#include <Python.h>
#include "structmember.h"
#include "numpy/arrayobject.h"

class PyIterable {
    PyObject *pyobj;

    PyIterable(PyObject* obj=NULL): pyobj(obj) {}
};

class PyList: public PyIterable {
    PyList(int size=0) : PyIterable(PyList_New(size)) {}
    PyList(PyObject* obj) : PyIterable(obj) {}
    PyList(PyObject* parent, const char* name) : PyIterable(parent, name) {}

    PyObject* operator[](Py_ssize_t idx) const {return PyList_GetItem(attr, idx);}
    inline int setItem(const Py_ssize_t& idx, const int& item) {return PyList_SetItem(attr, idx, PyLong_FromLong(item));}
    inline int setItem(const Py_ssize_t& idx, const double& item) {return PyList_SetItem(attr, idx, PyFloat_FromDouble(item));}
    inline int setItem(const Py_ssize_t& idx, PyObject* item) {return PyList_SetItem(attr, idx, item);}
    inline int append(const int& item) {
        PyObject* o = PyLong_FromLong(item);
        int index = PyList_Append(attr, o);
        Py_DECREF(o);
        return index;}
    inline int append(const double& item) {
        PyObject* o = PyFloat_FromDouble(item);
        int index = PyList_Append(attr, o);
        Py_DECREF(o);
        return index;}
    inline int append(PyObject* item) {return PyList_Append(attr, item);}
};

class PyArray: public PyIterable {

};