/*
 * fast_streaming.c — Full C expert streaming: pread + remap + array creation
 *
 * Combines expert loading AND index remapping in one C call,
 * eliminating all Python overhead in the hot path.
 *
 * Before: Python indices.tolist() → set() → dict → vectorize → mx.array  (~5ms)
 * After:  Single C call that returns (weight, scales, biases, mapped_indices)  (~0.5ms)
 */

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <numpy/arrayobject.h>
#include <fcntl.h>
#include <unistd.h>
#include <pthread.h>
#include <string.h>
#include <stdlib.h>

typedef struct {
    int fd;
    void *buf;
    size_t size;
    off_t offset;
} PreadTask;

static void *pread_worker(void *arg) {
    PreadTask *t = (PreadTask *)arg;
    pread(t->fd, t->buf, t->size, t->offset);
    return NULL;
}

/*
 * stream_experts(indices_flat, w_fd, w_offset, w_expert_bytes, w_shape,
 *                s_fd, s_offset, s_expert_bytes, s_shape,
 *                b_fd, b_offset, b_expert_bytes, b_shape)
 *
 * All-in-one: find unique experts, remap indices, parallel-load weights.
 * Returns (weight_array, scales_array, biases_array, mapped_indices)
 */
static PyObject *py_stream_experts(PyObject *self, PyObject *args) {
    PyArrayObject *indices_obj;
    int w_fd, s_fd, b_fd;
    long long w_offset, s_offset, b_offset;
    int w_expert_bytes, s_expert_bytes, b_expert_bytes;
    PyObject *w_shape_obj, *s_shape_obj, *b_shape_obj;

    if (!PyArg_ParseTuple(args, "O!iLiOiLiOiLiO",
            &PyArray_Type, &indices_obj,
            &w_fd, &w_offset, &w_expert_bytes, &w_shape_obj,
            &s_fd, &s_offset, &s_expert_bytes, &s_shape_obj,
            &b_fd, &b_offset, &b_expert_bytes, &b_shape_obj)) {
        return NULL;
    }

    /* Get flat indices */
    npy_intp n_indices = PyArray_SIZE(indices_obj);
    int *flat = (int *)PyArray_DATA(indices_obj);

    /* Find unique expert IDs and build remap */
    int max_id = 0;
    for (npy_intp i = 0; i < n_indices; i++) {
        if (flat[i] > max_id) max_id = flat[i];
    }

    int *seen = calloc(max_id + 1, sizeof(int));
    int *unique = malloc((max_id + 1) * sizeof(int));
    int n_unique = 0;

    for (npy_intp i = 0; i < n_indices; i++) {
        if (!seen[flat[i]]) {
            seen[flat[i]] = 1;
            unique[n_unique++] = flat[i];
        }
    }

    /* Sort unique IDs */
    for (int i = 0; i < n_unique - 1; i++)
        for (int j = i + 1; j < n_unique; j++)
            if (unique[i] > unique[j]) { int t = unique[i]; unique[i] = unique[j]; unique[j] = t; }

    /* Build remap table */
    int *remap = calloc(max_id + 1, sizeof(int));
    for (int i = 0; i < n_unique; i++) {
        remap[unique[i]] = i;
    }

    /* Create mapped indices array (same shape as input) */
    npy_intp *ind_dims = PyArray_DIMS(indices_obj);
    int ind_ndim = PyArray_NDIM(indices_obj);
    PyObject *mapped = PyArray_SimpleNew(ind_ndim, ind_dims, NPY_INT32);
    int *mapped_data = (int *)PyArray_DATA((PyArrayObject *)mapped);
    for (npy_intp i = 0; i < n_indices; i++) {
        mapped_data[i] = remap[flat[i]];
    }

    free(seen);
    free(remap);

    /* Parallel pread all unique experts for weight, scales, biases */
    int n_tasks = n_unique * 3; /* w + s + b per expert */
    PreadTask *tasks = malloc(n_tasks * sizeof(PreadTask));
    pthread_t *threads = malloc(n_tasks * sizeof(pthread_t));

    size_t w_total = (size_t)n_unique * w_expert_bytes;
    size_t s_total = (size_t)n_unique * s_expert_bytes;
    size_t b_total = (b_fd >= 0) ? (size_t)n_unique * b_expert_bytes : 0;

    void *w_buf = malloc(w_total);
    void *s_buf = malloc(s_total);
    void *b_buf = (b_total > 0) ? malloc(b_total) : NULL;

    int task_idx = 0;
    for (int i = 0; i < n_unique; i++) {
        int eid = unique[i];
        tasks[task_idx] = (PreadTask){w_fd, (char*)w_buf + i*w_expert_bytes, w_expert_bytes, w_offset + (off_t)eid * w_expert_bytes};
        tasks[task_idx + 1] = (PreadTask){s_fd, (char*)s_buf + i*s_expert_bytes, s_expert_bytes, s_offset + (off_t)eid * s_expert_bytes};
        if (b_buf) {
            tasks[task_idx + 2] = (PreadTask){b_fd, (char*)b_buf + i*b_expert_bytes, b_expert_bytes, b_offset + (off_t)eid * b_expert_bytes};
        }
        task_idx += (b_buf ? 3 : 2);
    }
    n_tasks = task_idx;

    /* Launch threads (up to 8) */
    int max_threads = (n_tasks < 8) ? n_tasks : 8;
    for (int i = 0; i < n_tasks; i++) {
        if (i < max_threads) {
            pthread_create(&threads[i], NULL, pread_worker, &tasks[i]);
        } else {
            int slot = i % max_threads;
            pthread_join(threads[slot], NULL);
            pthread_create(&threads[slot], NULL, pread_worker, &tasks[i]);
        }
    }
    for (int i = 0; i < max_threads && i < n_tasks; i++) {
        pthread_join(threads[i], NULL);
    }

    free(unique);
    free(tasks);
    free(threads);

    /* Build numpy arrays */
    Py_ssize_t w_ndim = PyTuple_Size(w_shape_obj);
    npy_intp *w_dims = malloc((1 + w_ndim) * sizeof(npy_intp));
    w_dims[0] = n_unique;
    for (Py_ssize_t i = 0; i < w_ndim; i++) w_dims[i+1] = PyLong_AsLong(PyTuple_GetItem(w_shape_obj, i));
    PyObject *w_arr = PyArray_SimpleNewFromData(1 + (int)w_ndim, w_dims, NPY_UINT32, w_buf);
    PyArray_ENABLEFLAGS((PyArrayObject *)w_arr, NPY_ARRAY_OWNDATA);
    free(w_dims);

    Py_ssize_t s_ndim = PyTuple_Size(s_shape_obj);
    npy_intp *s_dims = malloc((1 + s_ndim) * sizeof(npy_intp));
    s_dims[0] = n_unique;
    for (Py_ssize_t i = 0; i < s_ndim; i++) s_dims[i+1] = PyLong_AsLong(PyTuple_GetItem(s_shape_obj, i));
    PyObject *s_arr = PyArray_SimpleNewFromData(1 + (int)s_ndim, s_dims, NPY_UINT16, s_buf);
    PyArray_ENABLEFLAGS((PyArrayObject *)s_arr, NPY_ARRAY_OWNDATA);
    free(s_dims);

    PyObject *b_arr = Py_None;
    if (b_buf) {
        Py_ssize_t b_ndim = PyTuple_Size(b_shape_obj);
        npy_intp *b_dims = malloc((1 + b_ndim) * sizeof(npy_intp));
        b_dims[0] = n_unique;
        for (Py_ssize_t i = 0; i < b_ndim; i++) b_dims[i+1] = PyLong_AsLong(PyTuple_GetItem(b_shape_obj, i));
        b_arr = PyArray_SimpleNewFromData(1 + (int)b_ndim, b_dims, NPY_UINT16, b_buf);
        PyArray_ENABLEFLAGS((PyArrayObject *)b_arr, NPY_ARRAY_OWNDATA);
        free(b_dims);
    } else {
        Py_INCREF(Py_None);
    }

    return Py_BuildValue("(OOOO)", w_arr, s_arr, b_arr, mapped);
}

static PyMethodDef methods[] = {
    {"stream_experts", py_stream_experts, METH_VARARGS,
     "All-in-one expert streaming: find unique IDs, remap indices, parallel pread.\n"
     "Returns (weight_np, scales_np, biases_np, mapped_indices_np)"},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef module = {
    PyModuleDef_HEAD_INIT, "fast_streaming", NULL, -1, methods
};

PyMODINIT_FUNC PyInit_fast_streaming(void) {
    import_array();
    return PyModule_Create(&module);
}
