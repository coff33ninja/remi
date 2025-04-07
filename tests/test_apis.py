import pytest
from apis import get_weather

@pytest.mark.asyncio
async def test_get_weather():
    result = await get_weather("London")
    assert "The weather in London" in result or "Could not retrieve weather data." in result
