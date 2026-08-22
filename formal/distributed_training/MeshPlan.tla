--------------------------- MODULE MeshPlan ---------------------------
EXTENDS FiniteSets, Naturals, Sequences

CONSTANTS
  Ranks,
  Groups,
  Cids,
  RankCoord,
  OrderedMembers,
  LocalOrderedMembers,
  LaunchPlan,
  CGroup,
  CStep,
  CEpoch,
  CSlot,
  CKind,
  CTensor,
  CShape,
  CDType,
  CReduction,
  CRoot,
  CInputLayout,
  COutputLayout,
  CMemberGeneration

VARIABLES pc, posted, completed, aborted

vars == <<pc, posted, completed, aborted>>

PlanLen(r) == Len(LaunchPlan[r])

Current(r) == LaunchPlan[r][pc[r]]

SeqSet(seq) == {seq[i] : i \in 1 .. Len(seq)}

SeqHasNoDuplicates(seq) ==
  \A i, j \in 1 .. Len(seq) : seq[i] = seq[j] => i = j

Members(g) == SeqSet(OrderedMembers[g])

PostedByAny(c) == \E r \in Ranks : <<r, c>> \in posted

PostedByAllMembers(c) ==
  \A r \in Members(CGroup[c]) : <<r, c>> \in posted

Inflight(c) == PostedByAny(c) /\ c \notin completed /\ c \notin aborted

Busy(r) ==
  \E c \in Cids : Inflight(c) /\ <<r, c>> \in posted

ReadyToComplete(c) ==
  /\ Inflight(c)
  /\ PostedByAllMembers(c)

NoCompleteEnabled ==
  \A c \in Cids : ~ReadyToComplete(c)

AllRanksBusyWithPendingWork ==
  \A r \in Ranks :
    /\ Busy(r)
    /\ pc[r] <= PlanLen(r)

BlockedCycle ==
  /\ posted # {}
  /\ AllRanksBusyWithPendingWork
  /\ NoCompleteEnabled

CanPost(r) ==
  /\ pc[r] <= PlanLen(r)
  /\ ~Busy(r)

Post(r) ==
  LET c == Current(r) IN
  /\ CanPost(r)
  /\ c \in Cids
  /\ r \in Members(CGroup[c])
  /\ posted' = posted \cup {<<r, c>>}
  /\ pc' = [pc EXCEPT ![r] = @ + 1]
  /\ UNCHANGED <<completed, aborted>>

Complete(c) ==
  /\ ReadyToComplete(c)
  /\ completed' = completed \cup {c}
  /\ UNCHANGED <<pc, posted, aborted>>

AbortBlocked ==
  /\ BlockedCycle
  /\ aborted' = aborted \cup {c \in Cids : Inflight(c)}
  /\ UNCHANGED <<pc, posted, completed>>

Terminated ==
  /\ \A r \in Ranks : pc[r] = PlanLen(r) + 1
  /\ \A rc \in posted : rc[2] \in completed \cup aborted

TerminatedStutter ==
  /\ Terminated
  /\ UNCHANGED vars

Init ==
  /\ pc = [r \in Ranks |-> 1]
  /\ posted = {}
  /\ completed = {}
  /\ aborted = {}

Next ==
  \/ \E r \in Ranks : Post(r)
  \/ \E c \in Cids : Complete(c)
  \/ AbortBlocked
  \/ TerminatedStutter

Spec ==
  /\ Init
  /\ [][Next]_vars
  /\ \A r \in Ranks : WF_vars(Post(r))
  /\ \A c \in Cids : WF_vars(Complete(c))

TypeOK ==
  /\ pc \in [Ranks -> Nat]
  /\ posted \subseteq Ranks \X Cids
  /\ completed \subseteq Cids
  /\ aborted \subseteq Cids
  /\ RankCoord \in [Ranks -> [dp: Nat, tp: Nat]]
  /\ OrderedMembers \in [Groups -> Seq(Ranks)]
  /\ LocalOrderedMembers \in [Ranks -> [Groups -> Seq(Ranks)]]
  /\ LaunchPlan \in [Ranks -> Seq(Cids)]
  /\ CGroup \in [Cids -> Groups]
  /\ CStep \in [Cids -> Nat]
  /\ CEpoch \in [Cids -> Nat]
  /\ CSlot \in [Cids -> Nat]
  /\ CKind \in [Cids -> STRING]
  /\ CTensor \in [Cids -> STRING]
  /\ CShape \in [Cids -> STRING]
  /\ CDType \in [Cids -> STRING]
  /\ CReduction \in [Cids -> STRING]
  /\ CRoot \in [Cids -> STRING]
  /\ CInputLayout \in [Cids -> STRING]
  /\ COutputLayout \in [Cids -> STRING]
  /\ CMemberGeneration \in [Cids -> Nat]

UniqueActiveCoordinate ==
  \A r1, r2 \in Ranks : RankCoord[r1] = RankCoord[r2] => r1 = r2

NoGhostMember ==
  /\ \A g \in Groups : Members(g) \subseteq Ranks
  /\ \A r \in Ranks, g \in Groups : SeqSet(LocalOrderedMembers[r][g]) \subseteq Ranks

OrderedLocalRankAgreement ==
  \A g \in Groups : SeqHasNoDuplicates(OrderedMembers[g])

GroupAgreement ==
  \A r \in Ranks, g \in Groups :
    r \in Members(g) => LocalOrderedMembers[r][g] = OrderedMembers[g]

PostedByMember ==
  \A rc \in posted : rc[1] \in Members(CGroup[rc[2]])

CompletedHasAllMembers ==
  \A c \in completed :
    PostedByAllMembers(c)

CollectiveSignature(c) ==
  <<CKind[c], CTensor[c], CShape[c], CDType[c], CReduction[c], CRoot[c],
    CInputLayout[c], COutputLayout[c], CMemberGeneration[c], CEpoch[c],
    CStep[c]>>

NoConflictingPostedSlot ==
  \A c1 \in Cids, c2 \in Cids :
    /\ PostedByAny(c1)
    /\ PostedByAny(c2)
    /\ CGroup[c1] = CGroup[c2]
    /\ CEpoch[c1] = CEpoch[c2]
    /\ CStep[c1] = CStep[c2]
    /\ CSlot[c1] = CSlot[c2]
    => CollectiveSignature(c1) = CollectiveSignature(c2)

NoBlockedCycle == ~BlockedCycle

CollectiveResponse ==
  \A c \in Cids : PostedByAny(c) ~> c \in completed \/ c \in aborted

Inv ==
  /\ TypeOK
  /\ UniqueActiveCoordinate
  /\ NoGhostMember
  /\ OrderedLocalRankAgreement
  /\ GroupAgreement
  /\ PostedByMember
  /\ CompletedHasAllMembers
  /\ NoConflictingPostedSlot
  /\ NoBlockedCycle

=============================================================================
