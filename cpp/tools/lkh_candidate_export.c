/*
 * GPLSACO研究适配入口。外部LKH 3.0.13保留Keld Helsgaun的研究使用声明。
 * 仅执行原生候选准备，不执行FindTour，不向共同ACO初始化提供POPMUSIC tour。
 */
#include "LKH.h"
#include "Genetic.h"

int main(int argc, char **argv)
{
    Node *n;
    Candidate *candidate;
    FILE *output;
    int i, count, first;
    double started;
    if (argc != 3)
        eprintf("Usage: GPLSACO-LKH-Candidates parameter-file new-output-json");
    ParameterFileName = argv[1];
    ReadParameters();
    if (Optimum != MINUS_INFINITY || InitialTourFileName || InputTourFileName ||
        PiFileName || CandidateFiles || EdgeFiles || MergeTourFiles ||
        POPMUSIC_InitialTour || InitialTourAlgorithm == POPMUSIC)
        eprintf("Candidate-only input must not contain labels, tours or cached priors");
    StartTime = started = GetTime();
    MaxMatrixDimension = 20000;
    MergeWithTour = MergeWithTourIPT;
    ReadProblem();
    if (ProblemType != TSP || Dimension < 3 || Dimension > 10000 ||
        Distance != Distance_EUC_2D || Precision != 1 || Optimum != MINUS_INFINITY ||
        (CandidateSetType != ALPHA && CandidateSetType != POPMUSIC))
        eprintf("Only TSP/EUC_2D, precision 1 and ALPHA/POPMUSIC are supported");
    AllocateStructures();
    CreateCandidateSet();
    /* 即便内部启发式获得了一条tour，也只输出候选成员、原生顺序与先验分数。 */
    output = fopen(argv[2], "wx");
    if (!output)
        eprintf("Cannot exclusively create output");
    fprintf(output, "{\"candidate_export_version\":1,\"dimension\":%d,"
            "\"kind\":\"%s\",\"scale\":%d,\"precision\":%d,"
            "\"seed\":%u,\"maximum_candidates\":%d,"
            "\"native_preparation_cpu_seconds\":%.17g,\"nodes\":[",
            Dimension, CandidateSetType == ALPHA ? "ALPHA" : "POPMUSIC",
            Scale, Precision, Seed, MaxCandidates, GetTime() - started);
    for (i = 1; i <= Dimension; ++i) {
        n = &NodeSet[i];
        count = 0;
        for (candidate = n->CandidateSet; candidate && candidate->To; ++candidate)
            ++count;
        fprintf(output, "%s{\"id\":%d,\"pi\":%d,\"dad\":%d,\"count\":%d,"
                "\"edges\":[", i == 1 ? "" : ",", n->Id, n->Pi,
                n->Dad ? n->Dad->Id : 0, count);
        first = 1;
        for (candidate = n->CandidateSet; candidate && candidate->To; ++candidate) {
            fprintf(output, "%s[%d,%d,%d,%d]", first ? "" : ",",
                    candidate->To->Id, candidate->Alpha, candidate->Cost,
                    Distance(n, candidate->To));
            first = 0;
        }
        fprintf(output, "]}");
    }
    fprintf(output, "]}\n");
    if (fclose(output) != 0)
        eprintf("Candidate output close failed");
    return EXIT_SUCCESS;
}
