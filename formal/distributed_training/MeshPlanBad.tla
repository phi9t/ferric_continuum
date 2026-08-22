---------------------------- MODULE MeshPlanBad ----------------------------
BadRanks == {"r00", "r01", "r10", "r11"}
BadGroups == {"TP0", "TP1", "DP0", "DP1"}
BadCids == {"tp0", "tp1", "dp0", "dp1"}

VARIABLES pc, posted, completed, aborted

BadRankCoord ==
  [r \in BadRanks |->
    CASE r = "r00" -> [dp |-> 0, tp |-> 0]
      [] r = "r01" -> [dp |-> 0, tp |-> 1]
      [] r = "r10" -> [dp |-> 1, tp |-> 0]
      [] r = "r11" -> [dp |-> 1, tp |-> 1]]

BadOrderedMembers ==
  [g \in BadGroups |->
    CASE g = "TP0" -> <<"r00", "r01">>
      [] g = "TP1" -> <<"r10", "r11">>
      [] g = "DP0" -> <<"r00", "r10">>
      [] g = "DP1" -> <<"r01", "r11">>]

BadLocalOrderedMembers == [r \in BadRanks |-> BadOrderedMembers]

BadLaunchPlan ==
  [r \in BadRanks |->
    CASE r = "r00" -> <<"tp0", "dp0">>
      [] r = "r01" -> <<"dp1", "tp0">>
      [] r = "r10" -> <<"dp0", "tp1">>
      [] r = "r11" -> <<"tp1", "dp1">>]

BadCGroup ==
  [c \in BadCids |->
    CASE c = "tp0" -> "TP0"
      [] c = "tp1" -> "TP1"
      [] c = "dp0" -> "DP0"
      [] c = "dp1" -> "DP1"]

BadCStep == [c \in BadCids |-> 0]
BadCEpoch == [c \in BadCids |-> 0]
BadCSlot == [c \in BadCids |-> 0]
BadCKind == [c \in BadCids |-> IF c = "tp0" \/ c = "tp1" THEN "all_reduce_sum" ELSE "reduce_scatter_sum"]
BadCTensor == [c \in BadCids |-> IF c = "tp0" \/ c = "tp1" THEN "activation" ELSE "gradient"]
BadCShape == [c \in BadCids |-> "vec4"]
BadCDType == [c \in BadCids |-> "f32"]
BadCReduction == [c \in BadCids |-> "sum"]
BadCRoot == [c \in BadCids |-> "none"]
BadCInputLayout == [c \in BadCids |-> IF c = "tp0" \/ c = "tp1" THEN "partial_tp" ELSE "partial_dp"]
BadCOutputLayout == [c \in BadCids |-> IF c = "tp0" \/ c = "tp1" THEN "replicate_tp" ELSE "shard_dp"]
BadCMemberGeneration == [c \in BadCids |-> 0]

M == INSTANCE MeshPlan WITH
  Ranks <- BadRanks,
  Groups <- BadGroups,
  Cids <- BadCids,
  RankCoord <- BadRankCoord,
  OrderedMembers <- BadOrderedMembers,
  LocalOrderedMembers <- BadLocalOrderedMembers,
  LaunchPlan <- BadLaunchPlan,
  CGroup <- BadCGroup,
  CStep <- BadCStep,
  CEpoch <- BadCEpoch,
  CSlot <- BadCSlot,
  CKind <- BadCKind,
  CTensor <- BadCTensor,
  CShape <- BadCShape,
  CDType <- BadCDType,
  CReduction <- BadCReduction,
  CRoot <- BadCRoot,
  CInputLayout <- BadCInputLayout,
  COutputLayout <- BadCOutputLayout,
  CMemberGeneration <- BadCMemberGeneration,
  pc <- pc,
  posted <- posted,
  completed <- completed,
  aborted <- aborted

Spec == M!Spec
Inv == M!Inv

=============================================================================
