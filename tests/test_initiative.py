from familiar_brain.tools.initiative import Tracker
from familiar_brain.tools import initiative


def test_add_sorts_desc_and_next_cycles():
    t = Tracker()
    t.add("Gustave", 14); t.add("Nevron", 18); t.add("Maelle", 9)
    assert t.order() == ["Nevron", "Gustave", "Maelle"]
    assert t.next() == "Nevron" and t.next() == "Gustave" and t.next() == "Maelle" and t.next() == "Nevron"


def test_add_existing_replaces_value():
    t = Tracker(); t.add("A", 5); t.add("A", 12)
    assert t.show() == [("A", 12)]


def test_clear_and_empty_next():
    t = Tracker(); t.add("A", 5); t.clear()
    assert t.order() == [] and t.next() is None


def test_show_on_empty_order_is_readable():
    """Deviation test: the plan's `("Order: " + ...) or "Order is empty."` can never
    reach its fallback, because "Order: " is always truthy."""
    initiative.run({"action": "clear"})
    out = initiative.run({"action": "show"})
    assert out.strip() != "Order:", f"empty order rendered as a dangling label: {out!r}"
    assert "empty" in out.lower()
