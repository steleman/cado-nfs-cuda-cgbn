#!/usr/bin/env bash

if [ "$CADO_DEBUG" ] ; then set -x ; fi

#[fbb_lpb]
fbb=$1
lpb=$2

#proba
./bench_cofactoring_methods -in "Data_PP1-65_"$fbb"_"$lpb"/Data_PP1-65["$fbb"_"$lpb"]" -p -lb $fbb  -out  "Data_PP1-65_"$fbb"_"$lpb"/Data_PP1-65["$fbb"_"$lpb"]_p"&
./bench_cofactoring_methods -in "Data_PP1-27_"$fbb"_"$lpb"/Data_PP1-27["$fbb"_"$lpb"]" -p -lb $fbb  -out  "Data_PP1-27_"$fbb"_"$lpb"/Data_PP1-27["$fbb"_"$lpb"]_p"&
./bench_cofactoring_methods -in "Data_PM1_"$fbb"_"$lpb"/Data_PM1["$fbb"_"$lpb"]" -p -lb $fbb  -out  "Data_PM1_"$fbb"_"$lpb"/Data_PM1["$fbb"_"$lpb"]_p"&
./bench_cofactoring_methods -in "Data_ECM-M16_"$fbb"_"$lpb"/Data_ECM-M16["$fbb"_"$lpb"]" -p -lb $fbb  -out  "Data_ECM-M16_"$fbb"_"$lpb"/Data_ECM-M16["$fbb"_"$lpb"]_p"&
./bench_cofactoring_methods -in "Data_ECM-M12_"$fbb"_"$lpb"/Data_ECM-M12["$fbb"_"$lpb"]" -p -lb $fbb  -out  "Data_ECM-M12_"$fbb"_"$lpb"/Data_ECM-M12["$fbb"_"$lpb"]_p"&
./bench_cofactoring_methods -in "Data_ECM-B12_"$fbb"_"$lpb"/Data_ECM-B12["$fbb"_"$lpb"]" -p -lb $fbb  -out  "Data_ECM-B12_"$fbb"_"$lpb"/Data_ECM-B12["$fbb"_"$lpb"]_p"


#time
./bench_cofactoring_methods -in "Data_PP1-65_"$fbb"_"$lpb"/Data_PP1-65["$fbb"_"$lpb"]_p" -t  -out  "Data_PP1-65_"$fbb"_"$lpb"/Data_PP1-65["$fbb"_"$lpb"]_pt"&
./bench_cofactoring_methods -in "Data_PP1-27_"$fbb"_"$lpb"/Data_PP1-27["$fbb"_"$lpb"]_p" -t  -out  "Data_PP1-27_"$fbb"_"$lpb"/Data_PP1-27["$fbb"_"$lpb"]_pt"&
./bench_cofactoring_methods -in "Data_PM1_"$fbb"_"$lpb"/Data_PM1["$fbb"_"$lpb"]_p" -t  -out  "Data_PM1_"$fbb"_"$lpb"/Data_PM1["$fbb"_"$lpb"]_pt"&
./bench_cofactoring_methods -in "Data_ECM-M16_"$fbb"_"$lpb"/Data_ECM-M16["$fbb"_"$lpb"]_p" -t  -out  "Data_ECM-M16_"$fbb"_"$lpb"/Data_ECM-M16["$fbb"_"$lpb"]_pt"&
./bench_cofactoring_methods -in "Data_ECM-M12_"$fbb"_"$lpb"/Data_ECM-M12["$fbb"_"$lpb"]_p" -t  -out  "Data_ECM-M12_"$fbb"_"$lpb"/Data_ECM-M12["$fbb"_"$lpb"]_pt"&
./bench_cofactoring_methods -in "Data_ECM-B12_"$fbb"_"$lpb"/Data_ECM-B12["$fbb"_"$lpb"]_p" -t  -out  "Data_ECM-B12_"$fbb"_"$lpb"/Data_ECM-B12["$fbb"_"$lpb"]_pt"
