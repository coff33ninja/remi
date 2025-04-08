import pytest
from apis import get_weather, get_news

@pytest.mark.asyncio
async def test_get_weather_valid_city():
    result = await get_weather("London", "test_api_key")
    assert "The weather in London" in result or "Error" in result

@pytest.mark.asyncio
async def test_get_weather_invalid_city():
    result = await get_weather("InvalidCity", "test_api_key")
    assert "Could not retrieve weather data" in result or "Error" in result

@pytest.mark.asyncio
async def test_get_weather_missing_key():
    result = await get_weather("London", "")
    assert "Error" in result

def test_get_news():
    result = get_news("technology")
    assert "No news found." in result or "1." in result

def test_get_news_invalid_topic():
    result = get_news("invalid_topic_12345")
    assert "No news found." in result
