from langchain.chat_models import init_chat_model

TEMPERATURE = 0.0
NUM_PREDICT = 256
MODEL = "gemma3:12b-it-qat"

model = init_chat_model(
    model=MODEL,
    temperature=TEMPERATURE,
    num_predict=NUM_PREDICT,
    max_new_tokens=NUM_PREDICT,
    use_gpu=True,
    model_provider='ollama',
)

from langchain_core.messages import HumanMessage, SystemMessage

messages = [
    SystemMessage("Translate the following from English into Italian"),
    HumanMessage("hi!"),
]

print(model.invoke(messages))