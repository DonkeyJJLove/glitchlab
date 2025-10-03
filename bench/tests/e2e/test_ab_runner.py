# -*- coding: utf-8 -*-
import pytest
from bench import agent_ms_like


def _run(entrypoint: str, tests: list[dict]):
    task = dict(entrypoint=entrypoint, tests=tests)
    return agent_ms_like.generate_code(task, mode="B")


def test_template_protocol_reverse_str():
    """reverse_str powinno użyć template protocol."""
    task = dict(entrypoint="reverse_str", tests=[dict(args=["abc"], expect="cba")])
    out = agent_ms_like.generate_code(task)
    assert out["metrics"]["protocol"] == "template"
    assert "s[::-1]" in out["code"]


def test_template_protocol_fib():
    """fib powinno użyć template protocol."""
    task = dict(entrypoint="fib", tests=[dict(args=[5], expect=5)])
    out = agent_ms_like.generate_code(task)
    assert out["metrics"]["protocol"] == "template"
    assert "for _ in range" in out["code"]


def test_heuristic_protocol_palindrome():
    """is_palindrome powinno użyć heuristic protocol."""
    out = _run("is_palindrome", [dict(args=["aba"], expect=True)])
    assert out["metrics"]["protocol"] == "heuristic"
    assert "return t == t[::-1]" in out["code"]


def test_heuristic_protocol_factorial():
    """factorial powinno użyć heuristic protocol."""
    out = _run("factorial", [dict(args=[5], expect=120)])
    assert out["metrics"]["protocol"] == "heuristic"
    assert "res *=" in out["code"]


def test_heuristic_protocol_gcd():
    """gcd powinno użyć heuristic protocol."""
    out = _run("gcd", [dict(args=[12, 18], expect=6)])
    assert out["metrics"]["protocol"] == "heuristic"
    assert "while b:" in out["code"]


def test_induction_protocol_reverse():
    """nazwa nieznana, ale testy sugerują odwracanie stringa."""
    out = _run("mystery_func", [dict(args=["abc"], expect="cba")])
    assert out["metrics"]["protocol"] == "induction"
    assert "s[::-1]" in out["code"]


def test_induction_protocol_sum():
    """nazwa nieznana, ale testy sugerują sumowanie listy."""
    out = _run("another_func", [dict(args=[[1, 2, 3]], expect=6)])
    assert out["metrics"]["protocol"] == "induction"
    assert "sum(xs)" in out["code"]


def test_fallback_protocol():
    """nieznana nazwa i brak dopasowania w testach → fallback."""
    out = _run("foo_bar", [dict(args=[1], expect=1)])
    assert out["metrics"]["protocol"] == "fallback"
    assert "raise NotImplementedError" in out["code"]
