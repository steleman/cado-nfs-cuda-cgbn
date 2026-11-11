{ lib,
  makeWrapper,
  stdenv,
  stdenvNoCC,
  useNixCC ? !stdenv.hostPlatform.isDarwin,
  cc ? stdenv.cc,
  llvmPackages,

  doCheck ? false,
  ctestCheckHook,

  coreutils,
  cmake, hostname,
  perl,
  python3,
  bc,
  git,
  hwloc, fmt, gmp, mpfr, libmpc, ecm,
  openssh, rsync, gzip,

  CFLAGS ? null,
  CXXFLAGS ? null,
  MPI ? null,
  LDFLAGS ? null,
  FLAGS_SIZE ? null,
  BWC_GF2_ARITHMETIC_BACKENDS ? null,
  BWC_GF2_MATMUL_BACKENDS ? null,
  BWC_GFP_ARITHMETIC_BACKENDS ? null,
  BWC_GFP_MATMUL_BACKENDS ? null,
  BWC_EXTRA_BACKENDS ? null,
  #VALGRIND
  #CHECKS_EXPENSIVE
  #TIMEOUT_SCALE
  ...}:
let runtimeLookedUpBins = [openssh rsync gzip coreutils hostname]; in
stdenvNoCC.mkDerivation {
  pname = "cado-nfs";
  version = "3.0.0";

  src = builtins.fetchGit ./.;

  nativeBuildInputs = [
    cmake
    git
    perl
    bc
    hostname
    makeWrapper
  ] ++ lib.optionals useNixCC [ cc ];

  CFLAGS = CFLAGS;
  CXXFLAGS = CXXFLAGS;
  MPI = MPI;
  LDFLAGS = LDFLAGS;
  FLAGS_SIZE = FLAGS_SIZE;
  BWC_GF2_ARITHMETIC_BACKENDS = BWC_GF2_ARITHMETIC_BACKENDS;
  BWC_GF2_MATMUL_BACKENDS = BWC_GF2_MATMUL_BACKENDS;
  BWC_GFP_ARITHMETIC_BACKENDS = BWC_GFP_ARITHMETIC_BACKENDS;
  BWC_GFP_MATMUL_BACKENDS = BWC_GFP_MATMUL_BACKENDS;
  BWC_EXTRA_BACKENDS = BWC_EXTRA_BACKENDS;

  preConfigure = ''
    patchShebangs --build ./scripts/list_modified_files.sh
  '';

  nativeCheckInputs = [ ctestCheckHook ];

  preCheck = ''
    cmake --build . --target all_test_dependencies
    patchShebangs --host .. # The host and build machines have the same arch
    export HWLOC_XMLFILE=/build/source/ci/placeholder-machine-for-tests.xml
  '';

  doCheck=doCheck;
  ctestFlags="--repeat after-timeout:3";

  postInstall = ''
    wrapProgram $out/bin/cado-nfs.py \
        --set PATH ${lib.makeBinPath runtimeLookedUpBins}

    wrapProgram $out/bin/cado-nfs-client.py \
        --set PATH ${lib.makeBinPath runtimeLookedUpBins}

    patchShebangs --host $out/lib
  ''; # We need to patchShebangs in lib too, because we install scripts there.


  buildInputs = [
    hwloc fmt gmp mpfr libmpc ecm perl
    (python3.withPackages (ps: [ ps.flask ps.requests ]))
  ] ++ runtimeLookedUpBins
    ++ lib.optionals (cc.isClang or false) [ llvmPackages.openmp ];
}
