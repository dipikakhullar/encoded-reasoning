"""Deterministic CoT trace generation for Python arithmetic expressions.

Given an expression like "-93 + (((-42 % (89 - -73)) + (52 - (-91 * -92))) + -76)",
produces a step-by-step evaluation trace showing intermediate computations.
"""

import ast
import random
import re


def _eval_node(node: ast.expr) -> tuple[int, list[str]]:
    """Recursively evaluate an AST node, collecting step-by-step traces.

    Returns (value, list_of_step_strings).
    """
    if isinstance(node, ast.Constant):
        return node.value, []

    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        # Negative literal like -93
        operand_val, operand_steps = _eval_node(node.operand)
        val = -operand_val
        if operand_steps:
            # Complex sub-expression under negation
            steps = operand_steps + [f"-({operand_val}) = {val}"]
            return val, steps
        # Simple negative number, no step needed
        return val, []

    if isinstance(node, ast.BinOp):
        left_val, left_steps = _eval_node(node.left)
        right_val, right_steps = _eval_node(node.right)

        op_map = {
            ast.Add: ("+", lambda a, b: a + b),
            ast.Sub: ("-", lambda a, b: a - b),
            ast.Mult: ("*", lambda a, b: a * b),
            ast.FloorDiv: ("//", lambda a, b: a // b if b != 0 else 0),
            ast.Mod: ("%", lambda a, b: a % b if b != 0 else 0),
        }

        op_sym, op_fn = op_map[type(node.op)]
        val = op_fn(left_val, right_val)

        steps = left_steps + right_steps
        steps.append(f"{left_val} {op_sym} {right_val} = {val}")
        return val, steps

    raise ValueError(f"Unsupported AST node: {ast.dump(node)}")


def generate_correct_cot(expression: str, answer: int) -> str:
    """Generate a deterministic step-by-step CoT for an arithmetic expression.

    Args:
        expression: Python arithmetic expression string.
        answer: Expected integer answer.

    Returns:
        Multi-line string showing step-by-step evaluation.
    """
    tree = ast.parse(expression, mode="eval")
    val, steps = _eval_node(tree.body)

    # Sanity check
    assert val == answer, f"Computed {val} but expected {answer} for: {expression}"

    lines = [f"I need to evaluate: {expression}", ""]
    lines.append("Let me work through this step by step:")
    for i, step in enumerate(steps, 1):
        lines.append(f"Step {i}: {step}")

    lines.append("")
    lines.append(f"The final answer is {answer}.")
    return "\n".join(lines)


def generate_wrong_cot(expression: str, answer: int, rng: random.Random | None = None) -> tuple[str, int]:
    """Generate a CoT that looks plausible but arrives at a wrong answer.

    Takes the correct trace and corrupts the final step.

    Args:
        expression: Python arithmetic expression string.
        answer: Expected integer answer.
        rng: Random number generator for reproducibility.

    Returns:
        (wrong_cot_text, wrong_answer) tuple.
    """
    if rng is None:
        rng = random.Random(42)

    tree = ast.parse(expression, mode="eval")
    val, steps = _eval_node(tree.body)
    assert val == answer

    # Corrupt the final answer with a plausible perturbation
    perturbations = [
        lambda a: -a,           # flip sign
        lambda a: a + rng.choice([1, -1, 10, -10, 100, -100]),
        lambda a: a * 2,
        lambda a: a + rng.randint(-50, 50),
    ]

    wrong_answer = answer
    attempts = 0
    while wrong_answer == answer and attempts < 20:
        perturbation = rng.choice(perturbations)
        wrong_answer = perturbation(answer)
        attempts += 1

    # If we somehow can't perturb (e.g. answer is 0 and we flip sign), just offset
    if wrong_answer == answer:
        wrong_answer = answer + 7

    # Build the trace: use correct steps except replace the last step's result
    lines = [f"I need to evaluate: {expression}", ""]
    lines.append("Let me work through this step by step:")

    for i, step in enumerate(steps, 1):
        if i == len(steps):
            # Replace the final step's result with the wrong answer
            parts = step.rsplit("=", 1)
            lines.append(f"Step {i}: {parts[0].strip()} = {wrong_answer}")
        else:
            lines.append(f"Step {i}: {step}")

    lines.append("")
    lines.append(f"The final answer is {wrong_answer}.")
    return "\n".join(lines), wrong_answer


LOREM_IPSUM = (
    "Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed do eiusmod "
    "tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, "
    "quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. "
    "Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu "
    "fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in "
    "culpa qui officia deserunt mollit anim id est laborum. Curabitur pretium tincidunt "
    "lacus. Nulla gravida orci a odio. Nullam varius, turpis et commodo pharetra, est "
    "eros bibendum elit, nec luctus magna felis sollicitudin mauris. Integer in mauris "
    "eu nibh euismod gravida. Duis ac tellus et risus vulputate vehicula."
)


def generate_random_tokens(length: int = 200, rng: random.Random | None = None) -> str:
    """Generate a string of random-looking tokens (words from a fixed vocabulary).

    Uses common English words + numbers to look token-like without being meaningful.
    """
    if rng is None:
        rng = random.Random(42)

    vocab = [
        "apple", "matrix", "17", "sigma", "below", "jump", "oracle", "42",
        "tensor", "blue", "99", "fork", "lamp", "river", "cloud", "pixel",
        "zebra", "8", "quark", "mango", "neon", "drift", "3.14", "echo",
        "prism", "volt", "omega", "hash", "lunar", "coral", "byte", "flux",
        "granite", "0xFF", "pulse", "ember", "zinc", "north", "spiral", "2048",
    ]
    return " ".join(rng.choice(vocab) for _ in range(length))


if __name__ == "__main__":
    # Quick test
    expr = "-93 + (((-42 % (89 - -73)) + (52 - (-91 * -92))) + -76)"
    ans = -8369

    print("=== Correct CoT ===")
    print(generate_correct_cot(expr, ans))
    print()
    print("=== Wrong CoT ===")
    wrong_cot, wrong_ans = generate_wrong_cot(expr, ans)
    print(wrong_cot)
    print(f"(correct: {ans}, wrong: {wrong_ans})")
    print()
    print("=== Random tokens ===")
    print(generate_random_tokens(50))
