from engine.db.session import async_session
from engine.analysis.claude_analyzer import get_analyzer


async def get_db():
    async with async_session() as session:
        yield session


def get_analyzer_dep():
    return get_analyzer()
