/* 只调用未修改的外部LKH距离函数；不运行搜索，不读取标签。 */
#include "LKH.h"
#include <stddef.h>
#include <stdint.h>
#include <string.h>

int Scale = 1;

__attribute__((visibility("default")))
int gpl_tsplib_distances(int kind, size_t n, const double *xy, int32_t *output)
{
    int (*distance)(Node *, Node *);
    switch (kind) {
    case 1: distance = Distance_EUC_2D; break;
    case 2: distance = Distance_CEIL_2D; break;
    case 3: distance = Distance_ATT; break;
    case 4: distance = Distance_GEO; break;
    default: return -1;
    }
    if (!xy || !output || n < 3 || n > 4096) return -1;
    Node a, b;
    memset(&a, 0, sizeof(a));
    memset(&b, 0, sizeof(b));
    for (size_t i = 0; i < n; ++i) {
        a.X = xy[2*i]; a.Y = xy[2*i+1];
        for (size_t j = 0; j < n; ++j) {
            b.X = xy[2*j]; b.Y = xy[2*j+1];
            /* 接口约定自环为0；GEO原公式只针对两个不同节点。 */
            output[i*n+j] = i == j ? 0 : distance(&a, &b);
        }
    }
    return 0;
}
