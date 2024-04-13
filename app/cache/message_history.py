from langchain_community.chat_message_histories import RedisChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

def get_message_history(session_id: str) -> RedisChatMessageHistory:
    return RedisChatMessageHistory(session_id, url ="redis://redis:6379/0" )

# TODO: Add the exact type of the runnable object
def get_memory_runnable(runnable):
    return RunnableWithMessageHistory(
        runnable,
        get_message_history,
        input_messages_key="question",
        history_messages_key="history",
    )
