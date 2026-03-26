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

/*
 * remap_indices(indices_flat, num_elements)
 *
 * Given a flat int32 array of expert indices, returns:
 *   (unique_ids: list, mapped: numpy int32 array same shape as input)
 *
 * This replaces the Python:
 *   unique_ids = np.unique(flat)
 *   remap[unique_ids] = np.arange(len(unique_ids))
 *   mapped = remap[idx_np]
 */
static PyObject *py_remap_indices(PyObject *self, PyObject *args) {
    PyArrayObject *indices_arr;

    if (!PyArg_ParseTuple(args, "O!", &PyArray_Type, &indices_arr)) {
        return NULL;
    }

    if (PyArray_TYPE(indices_arr) != NPY_INT32) {
        PyErr_SetString(PyExc_TypeError, "indices must be int32");
        return NULL;
    }

    int32_t *data = (int32_t *)PyArray_DATA(indices_arr);
    npy_intp n = PyArray_SIZE(indices_arr);

    /* Find unique IDs and max */
    int32_t max_id = 0;
    for (npy_intp i = 0; i < n; i++) {
        if (data[i] > max_id) max_id = data[i];
    }

    /* Mark which IDs are present */
    int32_t *present = calloc(max_id + 1, sizeof(int32_t));
    for (npy_intp i = 0; i < n; i++) {
        present[data[i]] = 1;
    }

    /* Build unique list and remap table */
    int32_t *remap = malloc((max_id + 1) * sizeof(int32_t));
    PyObject *unique_list = PyList_New(0);
    int32_t next_id = 0;
    for (int32_t i = 0; i <= max_id; i++) {
        if (present[i]) {
            remap[i] = next_id++;
            PyList_Append(unique_list, PyLong_FromLong(i));
        }
    }
    free(present);

    /* Apply remap to create output array (same shape as input) */
    int ndim = PyArray_NDIM(indices_arr);
    npy_intp *dims = PyArray_DIMS(indices_arr);
    PyObject *mapped = PyArray_SimpleNew(ndim, dims, NPY_INT32);
    int32_t *out = (int32_t *)PyArray_DATA((PyArrayObject *)mapped);
    for (npy_intp i = 0; i < n; i++) {
        out[i] = remap[data[i]];
    }
    free(remap);

    /* Return (unique_ids_list, mapped_array) */
    PyObject *result = PyTuple_Pack(2, unique_list, mapped);
    Py_DECREF(unique_list);
    Py_DECREF(mapped);
    return result;
}

static PyMethodDef methods[] = {
    {"load_experts", py_load_experts, METH_VARARGS,
     "Load multiple experts from file descriptor via parallel pread.\n"
     "Args: fd, expert_ids, base_offset, expert_bytes, expert_shape, numpy_dtype_num\n"
     "Returns: numpy array [K, *expert_shape]"},
    {"remap_indices", py_remap_indices, METH_VARARGS,
     "Remap expert indices to contiguous IDs.\n"
     "Args: indices (numpy int32 array)\n"
     "Returns: (unique_ids_list, mapped_numpy_array)"},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef module = {
    PyModuleDef_HEAD_INIT, "fast_pread", NULL, -1, methods
};

PyMODINIT_FUNC PyInit_fast_pread(void) {
    import_array();
    return PyModule_Create(&module);
}
