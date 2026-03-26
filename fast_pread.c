/*
 * fast_pread.c — C extension for fast expert weight loading from SSD.
 *
 * Eliminates Python overhead in the expert loading hot path:
 * - Direct pread() to pre-allocated buffer (no Python object creation)
 * - Parallel loading of multiple experts via pthreads
 * - Returns a pointer that can be wrapped as numpy array with zero copy
 *
 * Build: python3 setup.py build_ext --inplace
 */

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <numpy/arrayobject.h>
#include <fcntl.h>
#include <unistd.h>
#include <pthread.h>
#include <string.h>
#include <stdlib.h>

/* Thread argument for parallel pread */
typedef struct {
    int fd;
    void *buf;
    size_t size;
    off_t offset;
    ssize_t result;
} PreadTask;

static void *pread_thread(void *arg) {
    PreadTask *task = (PreadTask *)arg;
    task->result = pread(task->fd, task->buf, task->size, task->offset);
    return NULL;
}

/*
 * load_experts(fd, expert_ids, base_offset, expert_bytes, expert_shape, dtype)
 *
 * Loads multiple experts from a file descriptor in parallel.
 * Returns a numpy array of shape [K, *expert_shape] with the loaded data.
 */
static PyObject *py_load_experts(PyObject *self, PyObject *args) {
    int fd;
    PyObject *expert_ids_obj;
    long long base_offset;
    int expert_bytes;
    PyObject *shape_obj;
    int dtype_num;

    if (!PyArg_ParseTuple(args, "iOLiOi",
                          &fd, &expert_ids_obj, &base_offset,
                          &expert_bytes, &shape_obj, &dtype_num)) {
        return NULL;
    }

    /* Parse expert IDs */
    Py_ssize_t k = PyList_Size(expert_ids_obj);
    if (k <= 0) {
        PyErr_SetString(PyExc_ValueError, "expert_ids must be non-empty list");
        return NULL;
    }

    int *expert_ids = malloc(k * sizeof(int));
    for (Py_ssize_t i = 0; i < k; i++) {
        expert_ids[i] = (int)PyLong_AsLong(PyList_GetItem(expert_ids_obj, i));
    }

    /* Allocate output buffer */
    size_t total_bytes = (size_t)k * expert_bytes;
    void *buf = malloc(total_bytes);
    if (!buf) {
        free(expert_ids);
        return PyErr_NoMemory();
    }

    /* Parallel pread */
    PreadTask *tasks = malloc(k * sizeof(PreadTask));
    pthread_t *threads = malloc(k * sizeof(pthread_t));

    for (Py_ssize_t i = 0; i < k; i++) {
        tasks[i].fd = fd;
        tasks[i].buf = (char *)buf + i * expert_bytes;
        tasks[i].size = expert_bytes;
        tasks[i].offset = base_offset + (off_t)expert_ids[i] * expert_bytes;
        tasks[i].result = 0;
    }

    /* Launch threads (up to 4) */
    int n_threads = k < 4 ? k : 4;
    for (Py_ssize_t i = 0; i < k; i++) {
        if (i < n_threads) {
            pthread_create(&threads[i], NULL, pread_thread, &tasks[i]);
        } else {
            /* Wait for a thread to finish and reuse */
            int slot = i % n_threads;
            pthread_join(threads[slot], NULL);
            pthread_create(&threads[slot], NULL, pread_thread, &tasks[i]);
        }
    }
    for (int i = 0; i < n_threads && i < k; i++) {
        pthread_join(threads[i], NULL);
    }

    free(expert_ids);
    free(tasks);
    free(threads);

    /* Build numpy shape: [k, *expert_shape] */
    Py_ssize_t shape_len = PyTuple_Size(shape_obj);
    npy_intp *dims = malloc((1 + shape_len) * sizeof(npy_intp));
    dims[0] = k;
    for (Py_ssize_t i = 0; i < shape_len; i++) {
        dims[i + 1] = PyLong_AsLong(PyTuple_GetItem(shape_obj, i));
    }

    /* Create numpy array that OWNS the buffer */
    PyObject *arr = PyArray_SimpleNewFromData(1 + (int)shape_len, dims, dtype_num, buf);
    if (arr) {
        PyArray_ENABLEFLAGS((PyArrayObject *)arr, NPY_ARRAY_OWNDATA);
    }

    free(dims);
    return arr;
}

static PyMethodDef methods[] = {
    {"load_experts", py_load_experts, METH_VARARGS,
     "Load multiple experts from file descriptor via parallel pread.\n"
     "Args: fd, expert_ids, base_offset, expert_bytes, expert_shape, numpy_dtype_num\n"
     "Returns: numpy array [K, *expert_shape]"},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef module = {
    PyModuleDef_HEAD_INIT, "fast_pread", NULL, -1, methods
};

PyMODINIT_FUNC PyInit_fast_pread(void) {
    import_array();
    return PyModule_Create(&module);
}
