# LangChain Streaming Generator

## Background
For most chat applications, we want to stream each token back to the client. LangChain's `callback` support is fantastic for async Web Sockets via FastAPI, and supports this out of the box. 

However, developers migrating from OpenAI's python library may find difficulty in implementing a Python generator along the same lines of the OpenAI library approach.

### OpenAI Streaming Example
Here's an example of the OpenAI library streaming generator, from the [OpenAI Cookbook](https://github.com/openai/openai-cookbook/blob/main/examples/How_to_stream_completions.ipynb)
```py
# Example of an OpenAI ChatCompletion request with stream=True
# https://platform.openai.com/docs/guides/chat

# a ChatCompletion request
response = openai.ChatCompletion.create(
    model='gpt-3.5-turbo',
    messages=[
        {'role': 'user', 'content': "What's 1+1? Answer in one word."}
    ],
    temperature=0,
    stream=True  # this time, we set stream=True
)

for chunk in response:
    print(chunk)
```
Notice how the response is actually a Python generator, and we can easily iterate over each chunk.

### LangChain Streaming Example
LangChain's streaming methodology operates via [callbacks](https://python.langchain.com/en/latest/modules/callbacks/getting_started.html)

Here's an example with callbacks. Note that there is no generator:
```py
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI, ChatAnthropic
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.schema import HumanMessage
# Initialize the callback handler. Each new token will be printed to the screen
class MyCustomHandler(BaseCallbackHandler):

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        print(f"token: {token}")


llm = OpenAI(streaming=True, callbacks=[MyCustomHandler()], temperature=0)
resp = llm("Write me a song about sparkling water.")
```

## Problem
The callbacks approach works best for async websockets. But what if we need a synchronous generator? In Django, for example, `HttpStreamingResponse` requires a generator.


## Solution
We'll be using Python queues and threads, and a reference to [this GitHub issue](https://github.com/hwchase17/langchain/issues/2428#issuecomment-1557583542).

This will allow us to return a generator, similar to the OpenAI library approach.

Here's the solution in its entirety, with comments inline (make sure to have your OpenAI API key set in your environment before running this script):
```py
from threading import Thread
from queue import Queue, Empty
from threading import Thread
from collections.abc import Generator
from langchain.llms import OpenAI
from langchain.callbacks.base import BaseCallbackHandler

# Defined a QueueCallback, which takes as a Queue object during initialization. Each new token is pushed to the queue.
class QueueCallback(BaseCallbackHandler):
    """Callback handler for streaming LLM responses to a queue."""

    def __init__(self, q):
        self.q = q

    def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        self.q.put(token)

    def on_llm_end(self, *args, **kwargs: Any) -> None:
        return self.q.empty()


# Create a function that will return our generator
def stream(input_text) -> Generator:

    # Create a Queue
    q = Queue()
    job_done = object()

    # Initialize the LLM we'll be using
    llm = OpenAI(
        streaming=True, 
        callbacks=[QueueCallback(q)], 
        temperature=0
    )

    # Create a funciton to call - this will run in a thread
    def task():
        resp = llm(input_text)
        q.put(job_done)

    # Create a thread and start the function
    t = Thread(target=task)
    t.start()

    content = ""

    # Get each new token from the queue and yield for our generator
    while True:
        try:
            next_token = q.get(True, timeout=1)
            if next_token is job_done:
                break
            content += next_token
            yield next_token, content
        except Empty:
            continue

if __name__ == "__main__":
    for next_token, content in stream("How cool are LLMs?"):
        print(next_token)
        print(content)

```

And that's it! We now have a generator we can use to stream OpenAI completions via LangChain. This method also works for chains, agents, etc.

## Conclusion
Thanks for reading, and I hope this solution helps you out and removes some headache!
