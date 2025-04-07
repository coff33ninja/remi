import pytest
from apis import get_weather, get_news

@pytest.mark.asyncio
async def test_get_weather():
    result = await get_weather("London")
    assert "The weather in London" in result or "Could not retrieve weather data." in result

def test_get_news():
    result = get_news("technology")
    assert "No news found." in result or "1." in result
