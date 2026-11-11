#ifndef CADO_ROPT_STAGE2_HPP
#define CADO_ROPT_STAGE2_HPP

#include <cstdint>      // int16_t
#include "ropt_str.hpp"    // ropt_param ...
#include "ropt_tree.h" // MurphyE_pq


/**
 * Sieving array 
 */
struct ropt_sieve_op;
struct sievearray_s {
  int16_t *array;
  unsigned int len_i;
  unsigned int len_j;
  /* if not NULL, rootsieve_run_line records its lines here instead of
     sieving (they are applied on the GPU) */
  struct ropt_sieve_oplist * ops;
};
typedef struct sievearray_s sievearray_t[1];


/* -- declarations -- */

void
ropt_stage2 ( ropt_poly const & poly,
              ropt_s2param & s2param,
              ropt_param_srcptr param,
              ropt_info_ptr info,
              MurphyE_pq *global_E_pqueue,
              int w );


#endif /* CADO_ROPT_STAGE2_HPP */
