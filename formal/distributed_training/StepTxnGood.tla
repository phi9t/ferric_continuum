--------------------------- MODULE StepTxnGood ---------------------------
\* A small, legal instance of the StepTxn contract: two ranks, one logical
\* group, starting from committed version 7. All behaviors satisfy Inv and every
\* step resolves (StepResolves), because Decide is enabled once both ranks are
\* prepared and the commit/abort paths always drain to committed/aborted.
EXTENDS Naturals

GoodRanks == {"r0", "r1"}

VARIABLES phase, overflow, decision, version, optim_mutated

M == INSTANCE StepTxn WITH
  Ranks <- GoodRanks,
  JobGen <- 3,
  Epoch <- 1,
  LogicalStep <- 42,
  AccumWindow <- 4,
  Attempt <- 0,
  InitVersion <- 7,
  phase <- phase,
  overflow <- overflow,
  decision <- decision,
  version <- version,
  optim_mutated <- optim_mutated

Spec == M!Spec
Inv == M!Inv
StepResolves == M!StepResolves

=============================================================================
