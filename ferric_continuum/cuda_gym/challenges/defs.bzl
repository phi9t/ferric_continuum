"""Bazel macro for a CUDA gym challenge: student vs reference + grader test."""

load("@rules_cuda//cuda:defs.bzl", "cuda_binary")
load("@rules_python//python:defs.bzl", "py_test")

def cuda_challenge(
        name,
        student_srcs,
        reference_srcs,
        cases,
        reference_deps = None,
        student_deps = None):
    """Defines a challenge's student/reference binaries and grading py_tests.

    Args:
      name: challenge name (e.g. "vector_add").
      student_srcs: sources for the student binary (TODO stubs by default).
      reference_srcs: sources for the complete reference binary.
      cases: the cases.json data file label.
      reference_deps: extra deps for the reference binary only (e.g. cuda_kernels).
        Students never receive these, so they cannot call production kernels.
      student_deps: extra deps for the student binary only (rare).

    Produces:
      :student           cuda_binary linking student_srcs (fails until filled)
      :reference         cuda_binary linking reference_srcs (passes)
      :grade             py_test student vs reference (tagged challenge-unsolved)
      :grade_reference   py_test reference vs itself (CI green self-check)

    GPU-touching targets are tagged `cuda` + `requires-gpu`. Default CPU
    builds exclude the `cuda` tag via `.bazelrc`; `--config=cuda` clears that
    filter. `:grade` also carries Bazel's `manual` tag so wildcards skip the
    expected-fail student grade while an explicit label still runs it.
    """
    harness = ["//ferric_continuum/cuda_gym/challenges/harness:challenge_io"]
    reference_deps = harness + (reference_deps or [])
    student_deps = harness + (student_deps or [])

    cuda_binary(
        name = "student",
        srcs = student_srcs,
        tags = ["cuda", "requires-gpu"],
        deps = student_deps,
    )

    cuda_binary(
        name = "reference",
        srcs = reference_srcs,
        tags = ["cuda", "requires-gpu"],
        deps = reference_deps,
    )

    py_test(
        name = "grade",
        srcs = ["//ferric_continuum/cuda_gym/challenges/harness:grader.py"],
        main = "grader.py",
        args = [
            "--cases",
            "$(location %s)" % cases,
            "--student",
            "$(location :student)",
            "--reference",
            "$(location :reference)",
        ],
        data = [
            cases,
            ":student",
            ":reference",
        ],
        # Expected to fail until student.cu is filled. `manual` keeps it out of
        # wildcards; an explicit label still builds and runs it.
        tags = ["cuda", "requires-gpu", "manual"],
        deps = ["//ferric_continuum/cuda_gym/challenges/harness:grader"],
    )

    py_test(
        name = "grade_reference",
        srcs = ["//ferric_continuum/cuda_gym/challenges/harness:grader.py"],
        main = "grader.py",
        args = [
            "--cases",
            "$(location %s)" % cases,
            "--student",
            "$(location :reference)",
            "--reference",
            "$(location :reference)",
        ],
        data = [
            cases,
            ":reference",
        ],
        tags = ["cuda", "requires-gpu"],
        deps = ["//ferric_continuum/cuda_gym/challenges/harness:grader"],
    )
