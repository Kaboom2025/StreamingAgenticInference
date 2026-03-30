"""Tests for ObsInjector component."""

import threading
import time

import pytest

from streamagent.engine.injector import ObsInjector
from streamagent.engine.interfaces import Observation


class TestObsInjectorBasic:
    """Test basic functionality of ObsInjector."""

    def test_empty_initially(self):
        """New injector is empty."""
        injector = ObsInjector()
        assert injector.empty() is True

    def test_put_and_get_pending(self):
        """Put one obs, get_pending returns it."""
        injector = ObsInjector()
        obs = Observation(type="test", content="hello")
        injector.put(obs)
        pending = injector.get_pending()
        assert len(pending) == 1
        assert pending[0] == obs

    def test_get_pending_drains_all(self):
        """Put 3 obs, get_pending returns all 3."""
        injector = ObsInjector()
        obs1 = Observation(type="test", content="first")
        obs2 = Observation(type="test", content="second")
        obs3 = Observation(type="test", content="third")
        injector.put(obs1)
        injector.put(obs2)
        injector.put(obs3)
        pending = injector.get_pending()
        assert len(pending) == 3
        assert obs1 in pending
        assert obs2 in pending
        assert obs3 in pending

    def test_get_pending_empty_returns_empty_list(self):
        """No observations returns empty list."""
        injector = ObsInjector()
        pending = injector.get_pending()
        assert pending == []

    def test_empty_after_drain(self):
        """empty() is True after get_pending drains."""
        injector = ObsInjector()
        injector.put(Observation(type="test", content="data"))
        injector.get_pending()
        assert injector.empty() is True

    def test_observation_type_preserved(self):
        """Observation.type and .content are unchanged after round-trip."""
        injector = ObsInjector()
        original = Observation(type="gridworld", content="You are in a room.")
        injector.put(original)
        pending = injector.get_pending()
        retrieved = pending[0]
        assert retrieved.type == original.type
        assert retrieved.content == original.content


class TestObsInjectorCapacity:
    """Test capacity management and overflow behavior."""

    def test_maxsize_parameter(self):
        """maxsize=1 only holds 1 observation."""
        injector = ObsInjector(maxsize=1)
        obs1 = Observation(type="test", content="first")
        injector.put(obs1)
        pending = injector.get_pending()
        assert len(pending) == 1
        assert pending[0] == obs1

    def test_put_full_drops_oldest(self):
        """When full, oldest is dropped, newest kept."""
        injector = ObsInjector(maxsize=2)
        obs1 = Observation(type="test", content="oldest")
        obs2 = Observation(type="test", content="middle")
        obs3 = Observation(type="test", content="newest")
        injector.put(obs1)
        injector.put(obs2)
        # Queue is now full; putting obs3 should drop obs1
        injector.put(obs3)
        pending = injector.get_pending()
        assert len(pending) == 2
        # obs1 should be gone, obs2 and obs3 should remain
        contents = {obs.content for obs in pending}
        assert "oldest" not in contents
        assert "middle" in contents
        assert "newest" in contents

    def test_default_maxsize_is_256(self):
        """Default maxsize is 256 observations."""
        injector = ObsInjector()
        for i in range(256):
            injector.put(Observation(type="test", content=f"obs{i}"))
        pending = injector.get_pending()
        assert len(pending) == 256

    def test_multiple_overflow_drops_oldest_first(self):
        """Multiple overflows drop oldest first."""
        injector = ObsInjector(maxsize=3)
        obs1 = Observation(type="test", content="first")
        obs2 = Observation(type="test", content="second")
        obs3 = Observation(type="test", content="third")
        obs4 = Observation(type="test", content="fourth")
        obs5 = Observation(type="test", content="fifth")

        injector.put(obs1)
        injector.put(obs2)
        injector.put(obs3)
        injector.put(obs4)  # Drops obs1
        injector.put(obs5)  # Drops obs2

        pending = injector.get_pending()
        assert len(pending) == 3
        contents = {obs.content for obs in pending}
        assert "first" not in contents
        assert "second" not in contents
        assert "third" in contents
        assert "fourth" in contents
        assert "fifth" in contents


class TestObsInjectorThreadSafety:
    """Test thread-safe concurrent operations."""

    def test_thread_safe_concurrent_puts(self):
        """10 threads each put 10 obs, all arrive safely."""
        injector = ObsInjector(maxsize=200)
        obs_list = []

        def put_observations(thread_id: int) -> None:
            for i in range(10):
                obs = Observation(
                    type="test",
                    content=f"thread{thread_id}_obs{i}",
                )
                injector.put(obs)
                obs_list.append(obs)

        threads = []
        for i in range(10):
            t = threading.Thread(target=put_observations, args=(i,))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        # All 100 observations should be in the injector
        pending = injector.get_pending()
        assert len(pending) == 100
        # All observations from obs_list should be present
        pending_contents = {obs.content for obs in pending}
        for obs in obs_list:
            assert obs.content in pending_contents

    def test_concurrent_put_and_get(self):
        """Concurrent put and get_pending don't corrupt state."""
        injector = ObsInjector(maxsize=50)
        all_retrieved = []

        def putter() -> None:
            for i in range(20):
                injector.put(Observation(type="test", content=f"item{i}"))

        def getter() -> None:
            for _ in range(3):
                time.sleep(0.001)
                pending = injector.get_pending()
                all_retrieved.extend(pending)

        threads = [
            threading.Thread(target=putter),
            threading.Thread(target=putter),
            threading.Thread(target=getter),
            threading.Thread(target=getter),
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # We should have received a non-zero number of items
        # (exact count depends on timing, but at least some)
        assert len(all_retrieved) > 0

    def test_empty_is_thread_safe(self):
        """empty() check is thread-safe."""
        injector = ObsInjector()

        def check_empty() -> None:
            for _ in range(100):
                injector.empty()

        threads = [threading.Thread(target=check_empty) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Should complete without errors
        assert injector.empty() is True

    def test_sequential_drains_all(self):
        """Sequential get_pending calls return everything then empty."""
        injector = ObsInjector()
        injector.put(Observation(type="test", content="a"))
        injector.put(Observation(type="test", content="b"))
        injector.put(Observation(type="test", content="c"))

        first_drain = injector.get_pending()
        assert len(first_drain) == 3

        second_drain = injector.get_pending()
        assert len(second_drain) == 0
        assert injector.empty() is True


class TestObsInjectorEdgeCases:
    """Test edge cases and special scenarios."""

    def test_empty_string_content(self):
        """Empty string content is preserved."""
        injector = ObsInjector()
        obs = Observation(type="test", content="")
        injector.put(obs)
        pending = injector.get_pending()
        assert len(pending) == 1
        assert pending[0].content == ""

    def test_special_characters_in_content(self):
        """Special characters are preserved."""
        injector = ObsInjector()
        obs = Observation(type="test", content="Hello\nWorld\t<tag>")
        injector.put(obs)
        pending = injector.get_pending()
        assert len(pending) == 1
        assert pending[0].content == "Hello\nWorld\t<tag>"

    def test_unicode_content(self):
        """Unicode content is preserved."""
        injector = ObsInjector()
        obs = Observation(type="test", content="你好 🌍 مرحبا")
        injector.put(obs)
        pending = injector.get_pending()
        assert len(pending) == 1
        assert pending[0].content == "你好 🌍 مرحبا"

    def test_large_content(self):
        """Large content strings are handled."""
        injector = ObsInjector()
        large_content = "x" * 10000
        obs = Observation(type="test", content=large_content)
        injector.put(obs)
        pending = injector.get_pending()
        assert len(pending) == 1
        assert len(pending[0].content) == 10000

    def test_multiple_different_types(self):
        """Different observation types are preserved."""
        injector = ObsInjector()
        obs1 = Observation(type="gridworld", content="room1")
        obs2 = Observation(type="alfworld", content="room2")
        obs3 = Observation(type="custom", content="room3")
        injector.put(obs1)
        injector.put(obs2)
        injector.put(obs3)
        pending = injector.get_pending()
        assert len(pending) == 3
        types = {obs.type for obs in pending}
        assert types == {"gridworld", "alfworld", "custom"}
