import sys
from logger import logger
from exception import AppException

from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace


# Create a HuggingFace Endpoint to intitialize llm
def load_hf_llm_model(hf_api_key, model_id="microsoft/Phi-3-mini-4k-instruct", **model_kwargs):
    '''
    Loads a HuggingFace language model and initializes a chat model.

    Args:
        hf_api_key: The HuggingFace API key.
        model_id: The ID of the HuggingFace model to load. Defaults to "microsoft/Phi-3-mini-4k-instruct".
        model_kwargs: 
            task: The task to perform. Defaults to "text-generation".
            max_new_tokens: The maximum number of new tokens to generate. Defaults to 512.
            do_sample: Whether to use sampling. Defaults to False.
            repetition_penalty: The repetition penalty. Defaults to 1.03.
            temperature: The temperature. Defaults to 0.7.
            top_p: The top p value. Defaults to 1.03.
    
    Returns:
        chat: A ChatHuggingFace object initialized with the loaded language model.
    '''

    try:

        llm = HuggingFaceEndpoint(
            repo_id = model_id,
            task = model_kwargs.get('task', "text-generation"),
            max_new_tokens = model_kwargs.get('max_new_tokens', 512),
            do_sample = model_kwargs.get('do_sample', False),
            repetition_penalty = model_kwargs.get('repetition_penalty', 1.03),
            temperature = model_kwargs.get('temperature', 0.7),
            top_p = model_kwargs.get('top_p', 1.03),
            huggingfacehub_api_token = hf_api_key
        )

        chat = ChatHuggingFace(llm=llm, verbose=True)

        logger.info("CREATED HuggingFace Endpoint and Initialized chat model")    
    
    except Exception as e:
        raise AppException(e, sys)
    
    return chat