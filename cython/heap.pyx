cdef Py_ssize_t left(Py_ssize_t idx):
    return (idx * 2 + 1)

cdef void sift_down(
        list idxs,
        float[:,:] uu,
        Py_ssize_t idx_start,
        Py_ssize_t idx_end
):
    cdef Py_ssize_t idx_root, idx_swap, idx_left, idx_right
    cdef Py_ssize_t swap_ix, swap_iy, left_ix, left_iy, right_ix, right_iy

    idx_root = idx_start

    while left(idx_root) <= idx_end:
        idx_left = left(idx_root)
        left_ix, left_iy = idxs[idx_left][0], idxs[idx_left][1]
        idx_swap = idx_root
        swap_ix, swap_iy = idxs[idx_swap][0], idxs[idx_swap][1]
        if uu[swap_ix, swap_iy] < uu[left_ix, left_iy]:
            idx_swap, swap_ix, swap_iy = idx_left, left_ix, left_iy

        idx_right = idx_left + 1
        if idx_right <= idx_end:
            right_ix, right_iy = idxs[idx_right][0], idxs[idx_right][1]
            if uu[swap_ix, swap_iy] < uu[right_ix, right_iy]:
                idx_swap = idx_right
        if idx_swap == idx_root:
            return
        else:
            idxs[idx_root], idxs[idx_swap] = idxs[idx_swap], idxs[idx_root]
            idx_root = idx_swap

def heapify(list idxs, float[:,:] uu):
    cdef int count
    cdef Py_ssize_t idx_start

    count = len(idxs)
    idx_start = count - 1

    while idx_start >= 0:
        sift_down(idxs, uu, idx_start, count - 1)
        idx_start -= 1


def heap_sort(list idxs, float[:,:] uu):
    cdef int count
    cdef Py_ssize_t idx_end

    count = len(idxs)
    idx_end = count - 1
    while idx_end > 0:
        idxs[0], idxs[idx_end] = idxs[idx_end], idxs[0]
        idx_end = idx_end - 1
        sift_down(idxs, uu, 0, idx_end)
