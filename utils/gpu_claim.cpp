#include "cado.h" // IWYU pragma: keep

#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>
#include <string>

#include <fcntl.h>
#include <sys/file.h>
#include <sys/stat.h>
#include <unistd.h>

#include "gpu_claim.h"

int cado_gpu_claim(const char * tag, int ndev)
{
    static std::mutex mx;
    static std::map<std::string, int> claimed;
    if (ndev <= 0) return 0;
    int n = ndev;
    if (char const * s = getenv("CADO_GPU_SIMULATE_NDEV"))
        if (atoi(s) > 0) n = atoi(s);
    if (n <= 1) return 0;
    std::lock_guard<std::mutex> const dummy(mx);
    auto const it = claimed.find(tag);
    if (it != claimed.end()) return it->second % ndev;

    char const * tmp = getenv("TMPDIR");
    std::string const dir = std::string(tmp && *tmp ? tmp : "/tmp")
        + "/cado-gpu-" + std::to_string((unsigned long) getuid());
    if (mkdir(dir.c_str(), 0700) != 0 && errno != EEXIST) {
        /* no locks: spread by pid, as before */
        int const d = (int) (getpid() % n);
        claimed[tag] = d;
        return d % ndev;
    }
    int result = -1;
    for (int slot = 0; slot < 256 && result < 0; slot++) {
        for (int d = 0; d < n && result < 0; d++) {
            std::string const path = dir + "/" + tag + "." + std::to_string(d)
                + "." + std::to_string(slot) + ".lock";
            int const fd = open(path.c_str(), O_CREAT | O_RDWR | O_CLOEXEC, 0600);
            if (fd < 0) continue;
            if (flock(fd, LOCK_EX | LOCK_NB) == 0) {
                result = d;     /* fd stays open: the lock is ours */
            } else {
                close(fd);
            }
        }
    }
    if (result < 0) result = (int) (getpid() % n);
    claimed[tag] = result;
    if (char const * v = getenv("CADO_GPU_CLAIM_VERBOSE"))
        if (atoi(v))
            fprintf(stderr, "# %s: GPU %d of %d (pid %d)\n", tag, result, n,
                    (int) getpid());
    return result % ndev;
}
