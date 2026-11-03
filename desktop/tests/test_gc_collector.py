"""The window's garbage collection (MainThreadCollector in gui/app.py): Python's automatic collection is off (a worker thread that
collected would delete Qt objects outside the main thread), a timer collects on the window's thread, a full collection follows Python's own
rule (only once the old objects grew by a quarter, and counted only every few middle collections), and what the app holds at start is set aside."""
from __future__ import annotations

import gc

import pytest
from PySide6.QtCore import QObject

from audio_cleaner_desktop.gui.app import MainThreadCollector

T0, T1, T2 = gc.get_threshold()  # (700, 10, 10 in CPython 3.12)


class FakeGc:
    """gc.get_count / gc.get_objects / gc.collect / gc.freeze stand-ins that record what the collector does."""

    def __init__(self) -> None:
        self.count = (0, 0, 0)
        self.objects = 1000  # how many objects the oldest generation holds
        self.after_full = None  # how many it holds after a full collection (None: unchanged)
        self.collected: list[int] = []
        self.counted = 0  # calls of gc.get_objects(2) (a full collection is followed by one more: the new starting point)
        self.frozen = 0

    def get_count(self):
        return self.count

    def get_objects(self, generation=None):
        assert generation == 2  # (only the oldest generation is ever asked for)
        self.counted += 1
        return [0] * self.objects

    def collect(self, generation=2):
        self.collected.append(generation)
        if generation == 2 and self.after_full is not None:
            self.objects = self.after_full
        return 0

    def freeze(self):
        self.frozen += 1


@pytest.fixture
def fake(monkeypatch, qapp):
    f = FakeGc()
    for name in ("get_count", "get_objects", "collect", "freeze"):
        monkeypatch.setattr(gc, name, getattr(f, name))
    return f


@pytest.fixture
def collector(fake):
    """A collector made while the gc functions are faked: its timer is stopped (a test calls check() itself), nothing real is frozen."""
    parent = QObject()
    c = MainThreadCollector(parent)
    c._timer.stop()
    fake.collected.clear()
    fake.counted = 0
    yield c
    parent.deleteLater()


def test_automatic_collection_is_off_and_the_app_set_its_start_aside(qapp):
    assert not gc.isenabled()
    assert qapp.collector._timer.isActive() and qapp.collector._timer.interval() == 50
    assert gc.get_freeze_count() > 0  # (the libraries and the app's code are not gone through by a full collection)


def test_a_new_collector_turns_collection_off_collects_once_then_freezes_and_ticks_every_50_ms(fake, qapp):
    parent = QObject()
    gc.enable()  # (the way a library might have left it)
    try:
        c = MainThreadCollector(parent)
        assert not gc.isenabled()
        assert fake.collected == [2] and fake.frozen == 1  # (what is garbage now is collected before the rest is set aside)
        assert fake.counted == 1 and c._old == 1000 and c._middle == 0  # (the old objects it starts from: what there is now)
        assert c._timer.isActive() and c._timer.interval() == 50
        assert c._threshold == (T0, T1, T2)
    finally:
        c._timer.stop()
        gc.disable()
        parent.deleteLater()


def test_nothing_is_done_while_the_youngest_generation_is_below_its_threshold(collector, fake):
    for count in [(0, 0, 0), (T0, 0, 0), (T0, T1 + 5, T2 + 5)]:
        fake.count = count
        collector.check()
    assert fake.collected == [] and fake.counted == 0 and collector._middle == 0


def test_only_the_youngest_generation_is_collected_while_the_middle_one_is_below_its_threshold(collector, fake):
    fake.count = (T0 + 1, T1, T2 + 5)
    for _ in range(3 * T2):
        collector.check()
    assert fake.collected == [0] * (3 * T2) and fake.counted == 0
    assert collector._middle == 0  # (a collection of the youngest generation is not counted as a middle one)


def test_the_middle_generation_is_collected_and_the_old_ones_are_not_counted_while_gen_2_is_below_its_threshold(collector, fake):
    fake.count = (T0 + 1, T1 + 1, T2)
    fake.objects = 10_000  # (grown far past a quarter: it must not matter)
    for _ in range(3 * T2):
        collector.check()
    assert fake.collected == [1] * (3 * T2) and fake.counted == 0


def test_a_full_collection_needs_the_old_objects_to_have_grown_by_more_than_a_quarter_and_comes_only_every_t2_middle_collections(collector, fake):
    fake.count = (T0 + 1, T1 + 1, T2 + 1)
    collector._old = 1000
    fake.objects = 1251  # (251 more: over a quarter)
    for _ in range(T2 - 1):
        collector.check()
    assert fake.collected == [1] * (T2 - 1) and fake.counted == 0  # (not counted yet, however much they grew)
    collector.check()
    assert fake.counted == 2 and fake.collected == [1] * (T2 - 1) + [2]  # (counted, collected, counted again)


def test_exactly_a_quarter_more_is_not_enough_and_the_next_count_comes_t2_collections_later(collector, fake):
    fake.count = (T0 + 1, T1 + 1, T2 + 1)
    collector._old = 1000
    fake.objects = 1250
    for _ in range(T2):
        collector.check()
    assert fake.counted == 1 and fake.collected == [1] * T2 and collector._old == 1000
    for _ in range(T2 - 1):  # (the count starts over: nothing is counted for t2 - 1 more middle collections)
        collector.check()
    assert fake.counted == 1 and 2 not in fake.collected
    fake.objects = 1251
    collector.check()
    assert fake.counted == 3 and fake.collected[-1] == 2


def test_fewer_old_objects_than_before_never_cause_a_full_collection(collector, fake):
    fake.count = (T0 + 1, T1 + 1, T2 + 1)
    collector._old = 1000
    fake.objects = 100
    for _ in range(5 * T2):
        collector.check()
    assert 2 not in fake.collected and fake.counted == 5


def test_the_old_objects_are_counted_again_after_a_full_collection_and_the_next_one_waits_for_a_quarter_more_than_that(collector, fake):
    fake.count = (T0 + 1, T1 + 1, T2 + 1)
    collector._old = 1000
    fake.objects = 2000
    fake.after_full = 1800  # (what the full collection left)
    for _ in range(T2):
        collector.check()
    assert fake.collected[-1] == 2 and collector._old == 1800 and collector._middle == 0
    fake.objects = 1800 + 450  # (a quarter of the new figure, not of the old one: no full collection)
    for _ in range(T2):
        collector.check()
    assert fake.collected.count(2) == 1 and collector._old == 1800
    fake.objects = 1800 + 451
    for _ in range(T2):
        collector.check()
    assert fake.collected.count(2) == 2 and fake.collected[-1] == 2


def test_a_full_collection_that_waited_for_the_old_objects_to_pass_their_threshold_comes_on_the_first_tick_after(collector, fake):
    fake.count = (T0 + 1, T1 + 1, T2)  # (gen 2 not past its threshold: the middle ones are collected, not counted)
    collector._old = 1000
    fake.objects = 1500
    for _ in range(2 * T2):
        collector.check()
    assert 2 not in fake.collected and fake.counted == 0
    fake.count = (T0 + 1, T1 + 1, T2 + 1)
    collector.check()
    assert fake.collected[-1] == 2 and fake.counted == 2


def test_a_real_check_frees_a_cycle_once_enough_objects_were_made(qapp):
    import weakref

    class Cycle:
        pass

    assert not gc.isenabled()
    a, b = Cycle(), Cycle()
    a.other, b.other = b, a
    ref = weakref.ref(a)
    del a, b
    assert ref() is not None  # (automatic collection is off: the cycle stays)
    for _ in range(5):
        keep = [[] for _ in range(T0 + 100)]  # (more new container objects than the threshold)
        qapp.collector.check()
        del keep
        if ref() is None:
            break
    assert ref() is None
