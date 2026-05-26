import time
from env.environment import RagContextOptimizerEnv
import asyncio


async def run_benchmark():
    env = RagContextOptimizerEnv()
    await env.reset()

    # Force _available_chunks to have some content
    print(f"Number of available chunks: {len(env._available_chunks)}")

    # We want to measure the performance of calling _effective_chunk_tokens
    # many times, which internally calls _chunk_map()
    iterations = 100000

    # Get a chunk id to query
    if not env._available_chunks:
        print("No chunks available!")
        return

    chunk_id = env._available_chunks[0].chunk_id

    start_time = time.time()

    for _ in range(iterations):
        env._effective_chunk_tokens(chunk_id)

    end_time = time.time()

    print(
        f"Time taken for {iterations} calls to _effective_chunk_tokens: {end_time - start_time:.4f} seconds"
    )


if __name__ == "__main__":
    asyncio.run(run_benchmark())
