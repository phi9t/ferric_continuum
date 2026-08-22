--------------------------- MODULE MeshPlanGood ---------------------------
GoodRanks == {"r00", "r01", "r10", "r11"}
GoodGroups == {"TP0", "TP1", "DP0", "DP1"}
GoodCids == {"tp0", "tp1", "dp0", "dp1"}

VARIABLES pc, posted, completed, aborted

GoodRankCoord ==
  [r \in GoodRanks |->
    CASE r = "r00" -> [dp |-> 0, tp |-> 0]
      [] r = "r01" -> [dp |-> 0, tp |-> 1]
      [] r = "r10" -> [dp |-> 1, tp |-> 0]
      [] r = "r11" -> [dp |-> 1, tp |-> 1]]

GoodOrderedMembers ==
  [g \in GoodGroups |->
    CASE g = "TP0" -> <<"r00", "r01">>
      [] g = "TP1" -> <<"r10", "r11">>
      [] g = "DP0" -> <<"r00", "r10">>
      [] g = "DP1" -> <<"r01", "r11">>]

GoodLocalOrderedMembers == [r \in GoodRanks |-> GoodOrderedMembers]

GoodLaunchPlan ==
  [r \in GoodRanks |->
    CASE r = "r00" -> <<"tp0", "dp0">>
      [] r = "r01" -> <<"tp0", "dp1">>
      [] r = "r10" -> <<"tp1", "dp0">>
      [] r = "r11" -> <<"tp1", "dp1">>]

GoodCGroup ==
  [c \in GoodCids |->
    CASE c = "tp0" -> "TP0"
      [] c = "tp1" -> "TP1"
      [] c = "dp0" -> "DP0"
      [] c = "dp1" -> "DP1"]

GoodCStep == [c \in GoodCids |-> 0]
GoodCEpoch == [c \in GoodCids |-> 0]
GoodCSlot == [c \in GoodCids |-> IF c = "tp0" \/ c = "tp1" THEN 0 ELSE 1]
GoodCKind == [c \in GoodCids |-> IF c = "tp0" \/ c = "tp1" THEN "all_reduce_sum" ELSE "reduce_scatter_sum"]
GoodCTensor == [c \in GoodCids |-> IF c = "tp0" \/ c = "tp1" THEN "activation" ELSE "gradient"]
GoodCShape == [c \in GoodCids |-> "vec4"]
GoodCDType == [c \in GoodCids |-> "f32"]
GoodCReduction == [c \in GoodCids |-> "sum"]
GoodCRoot == [c \in GoodCids |-> "none"]
GoodCInputLayout == [c \in GoodCids |-> IF c = "tp0" \/ c = "tp1" THEN "partial_tp" ELSE "partial_dp"]
GoodCOutputLayout == [c \in GoodCids |-> IF c = "tp0" \/ c = "tp1" THEN "replicate_tp" ELSE "shard_dp"]
GoodCMemberGeneration == [c \in GoodCids |-> 0]

M == INSTANCE MeshPlan WITH
  Ranks <- GoodRanks,
  Groups <- GoodGroups,
  Cids <- GoodCids,
  RankCoord <- GoodRankCoord,
  OrderedMembers <- GoodOrderedMembers,
  LocalOrderedMembers <- GoodLocalOrderedMembers,
  LaunchPlan <- GoodLaunchPlan,
  CGroup <- GoodCGroup,
  CStep <- GoodCStep,
  CEpoch <- GoodCEpoch,
  CSlot <- GoodCSlot,
  CKind <- GoodCKind,
  CTensor <- GoodCTensor,
  CShape <- GoodCShape,
  CDType <- GoodCDType,
  CReduction <- GoodCReduction,
  CRoot <- GoodCRoot,
  CInputLayout <- GoodCInputLayout,
  COutputLayout <- GoodCOutputLayout,
  CMemberGeneration <- GoodCMemberGeneration,
  pc <- pc,
  posted <- posted,
  completed <- completed,
  aborted <- aborted

Spec == M!Spec
Inv == M!Inv
CollectiveResponse == M!CollectiveResponse

=============================================================================
