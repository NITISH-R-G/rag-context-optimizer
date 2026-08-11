import asyncio
from playwright.async_api import async_playwright

async def run():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()
        print("Navigating to app...")
        await page.goto("http://localhost:3001")
        print("Waiting for load...")
        await page.wait_for_timeout(3000)
        print("Taking screenshot...")
        await page.screenshot(path="screenshot.png")
        print("Done!")
        await browser.close()

asyncio.run(run())
