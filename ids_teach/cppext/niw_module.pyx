from libcpp.vector cimport vector

cdef extern from "niw_mmml.hpp":
    double lgniwmmml(vector[vector[double]] X,
                     vector[vector[double]] lambda_0,
                     vector[double] mu_0,
                     double kappa_0,
                     double nu_0,
                     double crp_alpha,
                     vector[size_t] Z_start,
                     vector[size_t] k_start,
                     vector[double] hist_start,
                     size_t do_n_calculations)


cdef vector[double] conv_to_double_vec(a):
    cdef vector[double] ret
    for val in enumerate(a):
        tmp = val
        ret.push_back(tmp)
    return ret


cdef vector[vector[double]] np_array_to_double_vec2(a):
    cdef vector[vector[double]] ret
    for i in range(a.shape[0]):
        ret.push_back(conv_to_double_vec(a[i,:]))
    return ret


cdef vector[size_t] conv_to_size_t_vec(a):
    cdef vector[size_t] ret
    for val in enumerate(a):
        tmp = val
        ret.push_back(tmp)
    return ret


def niw_mmml_mp(args):
    return niw_mmml(*args)

def niw_mmml(X, lambda_0, mu_0, kappa_0, nu_0, crp_alpha, Z_start, k_start, hist_start, ncalc):
    val = lgniwmmml(X, lambda_0, mu_0, kappa_0, nu_0, crp_alpha, Z_start, k_start, hist_start, ncalc)
    return val
