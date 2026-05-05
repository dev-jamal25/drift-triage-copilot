"""Worker entry point. Filled in on Day 1."""

import asyncio

import structlog

log = structlog.get_logger()


async def main() -> None:
    log.info("worker.start")
    # TODO Day 1: connect to Redis, consume from queue, dispatch actions
    while True:  # noqa: ASYNC110 — placeholder; replaced by real consumer in Day 1
        await asyncio.sleep(60)


if __name__ == "__main__":
    asyncio.run(main())
