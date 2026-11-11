#ifndef CADO_FACUL_ECM_H
#define CADO_FACUL_ECM_H

#include "bytecode.h"
#include "stage2.h"

typedef enum {
  BRENT12 = 1,
  MONTY12 = 2,
  MONTY16 = 4,
  MONTYTWED12 = 8,
  MONTYTWED16 = 16      /* named, but not implemented: there is no curve
                         * construction for it anywhere. See
                         * ec_parameterization_is_implemented(). */
} ec_parameterization_t;

#define FULLMONTY (BRENT12 | MONTY12 | MONTY16)
#define FULLMONTYTWED (MONTYTWED12 | MONTYTWED16)
#define ECM_TORSION16 (MONTY16 | MONTYTWED16)
#define ECM_TORSION12 (BRENT12 | MONTY12 | MONTYTWED12)


typedef struct {
  bytecode bc;          /* Bytecode for stage 1 */
  unsigned int exp2;    /* Exponent of 2 in stage 1 primes */
  unsigned int B1;
  ec_parameterization_t parameterization;
  unsigned long parameter;  /* Used to compute curve coefficients with the
                             * above parameterization. */
  stage2_plan_t stage2;
} ecm_plan_t;


#ifdef __cplusplus
extern "C" {
#endif

void ecm_make_plan (ecm_plan_t *, unsigned int, unsigned int, 
		    ec_parameterization_t, unsigned long, int, int);

void ecm_clear_plan (ecm_plan_t *);

/* Whether this parameterization has a curve construction at all. The
 * enum names one that does not, and everything that offers a choice of
 * parameterizations should ask, rather than carry its own pruned copy of
 * the list. */
int ec_parameterization_is_implemented (ec_parameterization_t);

/* How many of the parameters of this parameterization are exceptional,
 * that is, yield better smoothness probabilities than the rest --
 * sigma = 11 for Brent-Suyama, and the curve isomorphic to it in the Z6
 * family (see Barbulescu, Bos, Bouvier, Kleinjung, Montgomery,
 * hal-00671948, and #30033). They are the first values
 * ec_valid_parameter_from_sequence() hands out; every parameter past
 * them is generic, and generic parameters are indistinguishable from
 * one another as far as we know. */
unsigned long ec_parameterization_nexceptional (ec_parameterization_t);

int ec_parameter_is_valid (ec_parameterization_t, unsigned long);
unsigned long
ec_valid_parameter_from_sequence (ec_parameterization_t parameterization,
                       unsigned long sequence_value);

#ifdef __cplusplus
}
#endif

#endif	/* CADO_FACUL_ECM_H */
