# SPDX-License-Identifier: Apache-2.0
"""RadixCache with a persistent lazy eviction heap.

Upstream ``RadixCache.evict`` rebuilds its eviction heap from every
evictable leaf on each call (``list(...)`` + ``heapify``, O(L)). Under
sustained load with few prefix hits the KV pool saturates, ``evict`` then
fires on every allocation shortfall, and the scheduler thread spends its
budget re-heapifying an ever-growing leaf set. This subclass keeps one
min-heap alive across calls: push when a node becomes an evictable leaf,
validate lazily at pop.

Lazy validation is exact for eviction strategies whose priority is a
non-decreasing function of node-local state (LRU/LFU/FIFO all qualify):
a stale recorded priority is a lower bound on the current one, so the heap
top never overtakes a fresher node and re-pushing refreshed nodes preserves
the eviction order.
"""

from __future__ import annotations

import heapq
import time

from sglang.srt.mem_cache.base_prefix_cache import EvictParams, EvictResult
from sglang.srt.mem_cache.radix_cache import RadixCache, TreeNode


class EvictHeapRadixCache(RadixCache):
    def __init__(self, params):
        # Before super().__init__: RadixCache.__init__ calls reset().
        self._evict_heap: list = []
        self._evict_heap_seq = 0
        super().__init__(params)

    def reset(self):
        self._evict_heap.clear()
        super().reset()

    def _evict_heap_push(self, node: TreeNode) -> None:
        self._evict_heap_seq += 1
        heapq.heappush(
            self._evict_heap,
            (self.eviction_strategy.get_priority(node), self._evict_heap_seq, node),
        )

    def _evict_heap_rebuild(self) -> None:
        self._evict_heap = [
            (self.eviction_strategy.get_priority(n), i, n)
            for i, n in enumerate(self.evictable_leaves)
        ]
        self._evict_heap_seq = len(self._evict_heap)
        heapq.heapify(self._evict_heap)

    def _update_leaf_status(self, node: TreeNode) -> None:
        was_evictable = node in self.evictable_leaves
        super()._update_leaf_status(node)
        if not was_evictable and node in self.evictable_leaves:
            self._evict_heap_push(node)

    def evict(self, params: EvictParams) -> EvictResult:
        if self.disable:
            return EvictResult()

        start_time = time.perf_counter()
        num_tokens = params.num_tokens

        # Compact when stale entries dominate, so heap size stays O(leaves).
        if len(self._evict_heap) > max(1024, 4 * len(self.evictable_leaves)):
            self._evict_heap_rebuild()
        if not self._evict_heap and self.evictable_leaves:
            # The push in _update_leaf_status should make this unreachable;
            # rebuild rather than silently under-evict if it is ever violated.
            self._evict_heap_rebuild()

        num_evicted = 0
        while num_evicted < num_tokens and self._evict_heap:
            priority, _seq, x = heapq.heappop(self._evict_heap)

            if x not in self.evictable_leaves:
                # Stale: evicted earlier, locked, or no longer a leaf.
                continue
            current_priority = self.eviction_strategy.get_priority(x)
            if current_priority != priority:
                # Refreshed since push (priorities are non-decreasing):
                # re-insert at the up-to-date priority and keep popping.
                self._evict_heap_push(x)
                continue

            self.token_to_kv_pool_allocator.free(x.value)
            num_evicted += len(x.value)
            # _delete_leaf -> _update_leaf_status(parent) pushes the parent
            # if it just became an evictable leaf.
            self._delete_leaf(x)
            self._record_remove_event(x)

        self.update_eviction_metrics(num_evicted, start_time)
        return EvictResult(num_tokens_evicted=num_evicted)
