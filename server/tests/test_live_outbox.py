"""Live caption events go out through one ordered queue per session (Session.post): the caption engines and the translator call it from
callbacks that can't wait, a task nobody holds could be collected before it sent, and many tasks could send out of order."""
import asyncio
import json
import time
from types import SimpleNamespace

import pytest

from acw.stream import routes
from acw.stream.features import Features


class Socket:
    """A websocket that records what it is sent; `delays` (seconds, by event number) make some sends slower than later ones."""

    def __init__(self, delays=(), fail_after=None):
        self.sent, self.delays, self.fail_after, self.started = [], list(delays), fail_after, 0

    async def send_text(self, text):
        n = self.started
        self.started += 1
        if self.fail_after is not None and n >= self.fail_after:
            raise RuntimeError('Cannot call "send" once a close message has been sent.')
        await asyncio.sleep(self.delays[n] if n < len(self.delays) else 0)
        self.sent.append(json.loads(text))


def session(ws):
    return routes.Session(ws, SimpleNamespace(user_id="u"), Features(denoise="light"))


def events(n):
    return [{"type": "caption", "n": i} for i in range(n)]


def test_events_posted_from_callbacks_go_out_in_order_on_one_task_the_session_holds():
    async def go():
        ws = Socket(delays=[0.05, 0.03, 0.01, 0, 0.02])  # (the first events are the slowest: separate tasks would finish last-first)
        s = session(ws)
        assert s._outbox is None and s._sender is None  # (nothing until the first event)
        for ev in events(12):
            s.post(ev)
        sender = s._sender
        for _ in range(3):
            s.post({"type": "caption", "n": "later"})
            assert s._sender is sender  # one task for the whole session
        await s.close_outbox()
        return ws.sent, sender

    sent, sender = asyncio.run(go())
    assert [e["n"] for e in sent] == [*range(12), "later", "later", "later"]
    assert sender.cancelled() or sender.done()


def test_closing_lets_what_was_posted_go_out_then_stops_the_sender():
    async def go():
        ws = Socket()
        s = session(ws)
        for ev in events(5):
            s.post(ev)
        t0 = time.monotonic()
        await s.close_outbox()
        return ws.sent, time.monotonic() - t0, s._sender

    sent, took, sender = asyncio.run(go())
    assert [e["n"] for e in sent] == [0, 1, 2, 3, 4] and took < 1.0 and sender.done()


def test_closing_waits_at_most_the_time_it_is_given_for_a_socket_that_does_not_answer():
    async def go():
        ws = Socket(delays=[60])  # (the client does not read)
        s = session(ws)
        for ev in events(3):
            s.post(ev)
        t0 = time.monotonic()
        await s.close_outbox(wait=0.3)
        await asyncio.sleep(0)
        return time.monotonic() - t0, s._sender

    took, sender = asyncio.run(go())
    assert 0.25 <= took < 1.5 and sender.done()  # (stopped, not left running)


def test_closing_does_not_cut_off_the_last_event_while_it_is_being_sent():
    """The queue is empty as soon as the sender has taken the last event, but the event is only out when the socket has taken it."""
    async def go():
        ws = Socket(delays=[0.2])  # one slow send
        s = session(ws)
        s.post(events(1)[0])
        await s.close_outbox()
        await asyncio.sleep(0.4)
        return ws.sent

    assert asyncio.run(go()) == [{"type": "caption", "n": 0}]


def test_a_socket_that_is_closed_stops_the_sender_without_errors_and_posting_goes_on_quietly():
    async def go():
        loop = asyncio.get_running_loop()
        problems = []
        loop.set_exception_handler(lambda _l, ctx: problems.append(ctx))
        ws = Socket(fail_after=2)
        s = session(ws)
        for ev in events(5):
            s.post(ev)
        await asyncio.sleep(0.1)
        done_by_itself = s._sender.done() and not s._sender.cancelled()
        s.post({"type": "caption", "n": "after"})  # (a late caption from the engine: nothing to raise)
        t0 = time.monotonic()
        await s.close_outbox()
        await asyncio.sleep(0.05)
        import gc

        gc.collect()
        return ws.sent, done_by_itself, problems, time.monotonic() - t0, s._sender.exception()

    sent, done_by_itself, problems, took, exc = asyncio.run(go())
    assert [e["n"] for e in sent] == [0, 1] and done_by_itself and problems == [] and exc is None
    assert took < 0.5  # (a finished sender is not waited for)


def test_closing_a_session_that_never_posted_does_nothing():
    async def go():
        s = session(Socket())
        await s.close_outbox()
        return s._sender

    assert asyncio.run(go()) is None


def test_the_captions_and_the_translator_post_through_the_session(monkeypatch):
    async def go():
        ws = Socket()
        s = session(ws)
        offered = []
        s.translator = SimpleNamespace(offer=offered.append)
        emit = routes._event_out(s)
        for ev in events(3):
            emit(ev)
        await s.close_outbox()
        return ws.sent, offered

    sent, offered = asyncio.run(go())
    assert sent == offered == events(3)  # (out in order, and offered to the translator)
    made = []

    class Translator:
        def __init__(self, targets, language, send):
            made.append(send)

        def set_targets(self, targets, language):
            pass

    from acw.stream import live_translate

    monkeypatch.setattr(live_translate, "LiveTranslator", Translator)
    s = session(Socket())
    routes._set_translations(s, Features(denoise="light", captions="local", translate=("es",)))
    assert made == [s.post]  # (the translator's events take the same queue as the captions)


@pytest.mark.parametrize("delay", [0, 0.1])
def test_the_end_of_a_session_lets_the_last_captions_go_out_first(monkeypatch, delay):
    async def go():
        ws = Socket(delays=[delay])
        s = session(ws)
        s.id = "s1"
        order = []

        async def inner(sess):
            sess.post(events(1)[0])  # (the last final caption, forwarded while the session ends)
            order.append("inner")

        monkeypatch.setattr(routes, "_finish_inner", inner)
        await routes._finish(s)
        order.append("finished")
        return ws.sent, order

    sent, order = asyncio.run(go())
    assert order == ["inner", "finished"] and sent == events(1)
