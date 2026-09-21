---- MODULE MeshPlanBad_TTrace_1789461092 ----
EXTENDS Sequences, TLCExt, Toolbox, Naturals, TLC, MeshPlanBad

_expression ==
    LET MeshPlanBad_TEExpression == INSTANCE MeshPlanBad_TEExpression
    IN MeshPlanBad_TEExpression!expression
----

_trace ==
    LET MeshPlanBad_TETrace == INSTANCE MeshPlanBad_TETrace
    IN MeshPlanBad_TETrace!trace
----

_inv ==
    ~(
        TLCGet("level") = Len(_TETrace)
        /\
        pc = ([r00 |-> 2, r01 |-> 2, r10 |-> 2, r11 |-> 2])
        /\
        aborted = ({})
        /\
        completed = ({})
        /\
        posted = ({<<"r00", "tp0">>, <<"r01", "dp1">>, <<"r10", "dp0">>, <<"r11", "tp1">>})
    )
----

_init ==
    /\ posted = _TETrace[1].posted
    /\ aborted = _TETrace[1].aborted
    /\ completed = _TETrace[1].completed
    /\ pc = _TETrace[1].pc
----

_next ==
    /\ \E i,j \in DOMAIN _TETrace:
        /\ \/ /\ j = i + 1
              /\ i = TLCGet("level")
        /\ posted  = _TETrace[i].posted
        /\ posted' = _TETrace[j].posted
        /\ aborted  = _TETrace[i].aborted
        /\ aborted' = _TETrace[j].aborted
        /\ completed  = _TETrace[i].completed
        /\ completed' = _TETrace[j].completed
        /\ pc  = _TETrace[i].pc
        /\ pc' = _TETrace[j].pc

\* Uncomment the ASSUME below to write the states of the error trace
\* to the given file in Json format. Note that you can pass any tuple
\* to `JsonSerialize`. For example, a sub-sequence of _TETrace.
    \* ASSUME
    \*     LET J == INSTANCE Json
    \*         IN J!JsonSerialize("MeshPlanBad_TTrace_1789461092.json", _TETrace)

=============================================================================

 Note that you can extract this module `MeshPlanBad_TEExpression`
  to a dedicated file to reuse `expression` (the module in the 
  dedicated `MeshPlanBad_TEExpression.tla` file takes precedence 
  over the module `MeshPlanBad_TEExpression` below).

---- MODULE MeshPlanBad_TEExpression ----
EXTENDS Sequences, TLCExt, Toolbox, Naturals, TLC, MeshPlanBad

expression == 
    [
        \* To hide variables of the `MeshPlanBad` spec from the error trace,
        \* remove the variables below.  The trace will be written in the order
        \* of the fields of this record.
        posted |-> posted
        ,aborted |-> aborted
        ,completed |-> completed
        ,pc |-> pc
        
        \* Put additional constant-, state-, and action-level expressions here:
        \* ,_stateNumber |-> _TEPosition
        \* ,_postedUnchanged |-> posted = posted'
        
        \* Format the `posted` variable as Json value.
        \* ,_postedJson |->
        \*     LET J == INSTANCE Json
        \*     IN J!ToJson(posted)
        
        \* Lastly, you may build expressions over arbitrary sets of states by
        \* leveraging the _TETrace operator.  For example, this is how to
        \* count the number of times a spec variable changed up to the current
        \* state in the trace.
        \* ,_postedModCount |->
        \*     LET F[s \in DOMAIN _TETrace] ==
        \*         IF s = 1 THEN 0
        \*         ELSE IF _TETrace[s].posted # _TETrace[s-1].posted
        \*             THEN 1 + F[s-1] ELSE F[s-1]
        \*     IN F[_TEPosition - 1]
    ]

=============================================================================



Parsing and semantic processing can take forever if the trace below is long.
 In this case, it is advised to uncomment the module below to deserialize the
 trace from a generated binary file.

\*
\*---- MODULE MeshPlanBad_TETrace ----
\*EXTENDS IOUtils, TLC, MeshPlanBad
\*
\*trace == IODeserialize("MeshPlanBad_TTrace_1789461092.bin", TRUE)
\*
\*=============================================================================
\*

---- MODULE MeshPlanBad_TETrace ----
EXTENDS TLC, MeshPlanBad

trace == 
    <<
    ([pc |-> [r00 |-> 1, r01 |-> 1, r10 |-> 1, r11 |-> 1],aborted |-> {},completed |-> {},posted |-> {}]),
    ([pc |-> [r00 |-> 2, r01 |-> 1, r10 |-> 1, r11 |-> 1],aborted |-> {},completed |-> {},posted |-> {<<"r00", "tp0">>}]),
    ([pc |-> [r00 |-> 2, r01 |-> 2, r10 |-> 1, r11 |-> 1],aborted |-> {},completed |-> {},posted |-> {<<"r00", "tp0">>, <<"r01", "dp1">>}]),
    ([pc |-> [r00 |-> 2, r01 |-> 2, r10 |-> 2, r11 |-> 1],aborted |-> {},completed |-> {},posted |-> {<<"r00", "tp0">>, <<"r01", "dp1">>, <<"r10", "dp0">>}]),
    ([pc |-> [r00 |-> 2, r01 |-> 2, r10 |-> 2, r11 |-> 2],aborted |-> {},completed |-> {},posted |-> {<<"r00", "tp0">>, <<"r01", "dp1">>, <<"r10", "dp0">>, <<"r11", "tp1">>}])
    >>
----


=============================================================================

---- CONFIG MeshPlanBad_TTrace_1789461092 ----

INVARIANT
    _inv

CHECK_DEADLOCK
    \* CHECK_DEADLOCK off because of PROPERTY or INVARIANT above.
    FALSE

INIT
    _init

NEXT
    _next

CONSTANT
    _TETrace <- _trace

ALIAS
    _expression
=============================================================================
\* Generated on Tue Sep 15 08:31:33 UTC 2026